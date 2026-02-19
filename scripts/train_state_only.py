import argparse
import os
import sys
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from fa_planner.models.state_prior import StatePrior
from fa_planner.models.state_only import StateOnlyPolicy
from fa_planner.utils.config import load_config


class StateOnlyDataset(Dataset):
    def __init__(self, root, context_frames, action_bins, a_max):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_frames = context_frames
        self.action_bins = action_bins
        self.a_max = a_max
        if not self.files:
            raise RuntimeError(f"No token files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        world = data["world_tokens"]
        actions = data["actions"]
        t_max = world.shape[0]
        if t_max <= self.context_frames:
            raise RuntimeError("Episode too short for context window")
        t = np.random.randint(self.context_frames - 1, t_max)
        world_seq = world[t - self.context_frames + 1 : t + 1]
        act = actions[t]
        edges = np.linspace(-self.a_max, self.a_max, self.action_bins)
        ix = np.argmin(np.abs(act[0] - edges))
        iy = np.argmin(np.abs(act[1] - edges))
        act_token = ix * self.action_bins + iy
        return (
            torch.from_numpy(world_seq.astype(np.int64)),
            torch.tensor(act_token, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--state", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_frames = cfg["state_prior"]["L"]
    action_bins = cfg["action_prior"]["action_bins"]
    a_max = cfg["env"]["a_max"]

    dataset = StateOnlyDataset(args.data, context_frames, action_bins, a_max)
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

    policy = StateOnlyPolicy(
        d_model=cfg["state_prior"]["d_model"],
        action_vocab=cfg["action_prior"]["action_vocab"],
    ).to(args.device)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        for world_seq, act_token in tqdm(loader):
            world_seq = world_seq.to(args.device)
            act_token = act_token.to(args.device)
            with torch.no_grad():
                summary = state_prior.hidden_summary(world_seq)
            logits = policy(summary)
            loss = torch.nn.functional.cross_entropy(logits, act_token)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")

    torch.save(policy.state_dict(), args.out)
    print(f"Saved StateOnlyPolicy to {args.out}")


if __name__ == "__main__":
    main()
