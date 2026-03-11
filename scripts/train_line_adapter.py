import argparse
import os
import sys
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from fa_planner.models.state_prior import StatePrior
from fa_planner.models.line_adapter import LineAdapter
from fa_planner.utils.config import load_config

def discretize_action(action, bins, a_max):
    edges = np.linspace(-a_max, a_max, bins)
    ix = np.argmin(np.abs(action[0] - edges))
    iy = np.argmin(np.abs(action[1] - edges))
    return ix * bins + iy


class LineAdapterDataset(Dataset):
    def __init__(self, root, context_frames, action_bins, a_max, v_max, gamma, dt):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_frames = context_frames
        self.action_bins = action_bins
        self.a_max = a_max
        self.v_max = v_max
        self.gamma = gamma
        self.dt = dt
        if not self.files:
            raise RuntimeError(f"No token files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        world = data["world_tokens"]
        sim_state = data["sim_state"]
        actions = data["actions"]
        goal = data["goal"]
        t_max = world.shape[0]
        if t_max <= self.context_frames:
            raise RuntimeError("Episode too short for context window")
        t = np.random.randint(self.context_frames - 1, t_max)
        world_seq = world[t - self.context_frames + 1 : t + 1]
        state_t = sim_state[t]
        start = sim_state[0][:2]
        direction = goal - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        action_hint = np.clip(self.a_max * direction, -self.a_max, self.a_max)
        target_token = discretize_action(actions[t], self.action_bins, self.a_max)
        return (
            torch.from_numpy(world_seq.astype(np.int64)),
            torch.from_numpy(action_hint.astype(np.float32)),
            torch.from_numpy(goal.astype(np.float32)),
            torch.tensor(target_token, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--state", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_frames = cfg["state_prior"]["L"]
    action_bins = cfg["action_prior"]["action_bins"]
    action_vocab = cfg["action_prior"]["action_vocab"]
    a_max = cfg["env"]["a_max"]
    v_max = cfg["env"]["v_max"]
    gamma = cfg["env"]["gamma"]
    dt = cfg["env"]["dt"]

    dataset = LineAdapterDataset(
        args.data, context_frames, action_bins, a_max, v_max, gamma, dt
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    state_prior = StatePrior(
        vocab_size=cfg["tokenizer"]["codebook_size"],
        d_model=cfg["state_prior"]["d_model"],
        n_layers=cfg["state_prior"]["n_layers"],
        n_heads=cfg["state_prior"]["n_heads"],
        d_ff=cfg["state_prior"]["d_ff"],
        dropout=cfg["state_prior"]["dropout"],
        context_frames=context_frames,
    ).to(args.device)
    state_prior.load_state_dict(torch.load(args.state, map_location=args.device))
    state_prior.eval()
    for p in state_prior.parameters():
        p.requires_grad = False

    adapter = LineAdapter(
        d_model=cfg["state_prior"]["d_model"],
        action_vocab=action_vocab,
        hidden=cfg["adapter"]["mlp_hidden"] * 2,
    ).to(args.device)

    optim = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0.0
        total_acc = 0.0
        for world_seq, action_hint, goal, target in tqdm(loader):
            world_seq = world_seq.to(args.device)
            action_hint = action_hint.to(args.device)
            goal = goal.to(args.device)
            target = target.to(args.device)
            with torch.no_grad():
                summary = state_prior.hidden_summary(world_seq)
            logits = adapter(summary, action_hint, goal)
            if logits.size(0) > 2:
                loss = torch.nn.functional.cross_entropy(logits[2:], target[2:])
                pred = torch.argmax(logits[2:], dim=-1)
                acc = (pred == target[2:]).float().mean()
            else:
                loss = torch.tensor(0.0, device=args.device)
                acc = torch.tensor(0.0, device=args.device)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_acc += acc.item()
        avg_loss = total_loss / max(1, len(loader))
        avg_acc = total_acc / max(1, len(loader))
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={avg_acc:.4f}")
        scheduler.step()

    torch.save(adapter.state_dict(), args.out)
    out_dir = os.path.dirname(args.out)
    np.save(os.path.join(out_dir, "line_adapter_loss.npy"), np.array(losses, dtype=np.float32))
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_adapter_loss.png"))
    plt.close()
    print(f"Saved LineAdapter to {args.out}")


if __name__ == "__main__":
    main()