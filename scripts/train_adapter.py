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
from fa_planner.models.action_prior import ActionPrior
from fa_planner.models.adapter import Adapter
from fa_planner.utils.config import load_config


class AlignDataset(Dataset):
    def __init__(self, root, context_steps, action_bins, a_max, codebook_size):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_steps = context_steps
        self.action_bins = action_bins
        self.a_max = a_max
        self.codebook_size = codebook_size
        if not self.files:
            raise RuntimeError(f"No token files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        world = data["world_tokens"]
        ego = data["ego_tokens"]
        actions = data["actions"]
        goal = data["goal"]
        t_max = world.shape[0]
        if t_max <= self.context_steps:
            raise RuntimeError("Episode too short for context window")
        t = np.random.randint(self.context_steps - 1, t_max)
        world_seq = world[t - self.context_steps + 1 : t + 1]
        ego_seq = ego[t - self.context_steps + 1 : t + 1]
        act_seq = actions[t - self.context_steps + 1 : t + 1]
        edges = np.linspace(-self.a_max, self.a_max, self.action_bins)
        idx_x = np.argmin(np.abs(act_seq[:, 0, None] - edges), axis=-1)
        idx_y = np.argmin(np.abs(act_seq[:, 1, None] - edges), axis=-1)
        act_tokens = idx_x * self.action_bins + idx_y
        return (
            torch.from_numpy(world_seq.astype(np.int64)),
            torch.from_numpy(ego_seq.astype(np.int64)),
            torch.from_numpy(act_tokens.astype(np.int64)),
            torch.from_numpy(goal.astype(np.float32)),
        )


def build_action_context_sequence(ego_seq, act_tokens, codebook_size):
    # ego_seq: (L, 64), act_tokens: (L,)
    seq = []
    last = act_tokens.size(0) - 1
    for t in range(ego_seq.size(0)):
        seq.extend(ego_seq[t].reshape(-1).tolist())
        if t < last:
            seq.append(codebook_size + int(act_tokens[t]))
    return torch.tensor(seq, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--state", type=str, required=True)
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_steps = cfg["action_prior"]["L"]
    codebook_size = cfg["tokenizer"]["codebook_size"]
    action_bins = cfg["action_prior"]["action_bins"]
    action_vocab = cfg["action_prior"]["action_vocab"]
    a_max = cfg["env"]["a_max"]
    kl_lambda = cfg["adapter"]["kl_lambda"]

    dataset = AlignDataset(args.data, context_steps, action_bins, a_max, codebook_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    state_prior = StatePrior(
        vocab_size=cfg["tokenizer"]["codebook_size"],
        d_model=cfg["state_prior"]["d_model"],
        n_layers=cfg["state_prior"]["n_layers"],
        n_heads=cfg["state_prior"]["n_heads"],
        d_ff=cfg["state_prior"]["d_ff"],
        dropout=cfg["state_prior"]["dropout"],
        context_frames=cfg["state_prior"]["L"],
    ).to(args.device)
    state_prior.load_state_dict(torch.load(args.state, map_location=args.device))
    state_prior.eval()
    for p in state_prior.parameters():
        p.requires_grad = False

    action_state = torch.load(args.action, map_location=args.device)
    needed_len = context_steps * 64 + (context_steps - 1)
    pos_weight = action_state.get("pos_embed.weight")
    pos_len = pos_weight.shape[0] if pos_weight is not None else 0
    max_seq_len = pos_len if pos_len >= needed_len else needed_len
    action_prior = ActionPrior(
        vocab_size=codebook_size + action_vocab,
        codebook_size=codebook_size,
        d_model=cfg["action_prior"]["d_model"],
        n_layers=cfg["action_prior"]["n_layers"],
        n_heads=cfg["action_prior"]["n_heads"],
        d_ff=cfg["action_prior"]["d_ff"],
        dropout=cfg["action_prior"]["dropout"],
        max_seq_len=max_seq_len,
    ).to(args.device)
    if pos_weight is not None and pos_len != max_seq_len:
        new_weight = pos_weight.new_empty((max_seq_len, pos_weight.shape[1]))
        copy_len = min(pos_len, max_seq_len)
        new_weight[:copy_len] = pos_weight[:copy_len]
        if max_seq_len > pos_len:
            new_weight[copy_len:] = pos_weight[-1:].repeat(max_seq_len - copy_len, 1)
        action_state["pos_embed.weight"] = new_weight
    time_weight = action_state.get("time_embed.weight")
    if time_weight is not None:
        target_steps = max_seq_len // action_prior.step_size + 2
        time_len = time_weight.shape[0]
        if time_len != target_steps:
            new_weight = time_weight.new_empty((target_steps, time_weight.shape[1]))
            copy_len = min(time_len, target_steps)
            new_weight[:copy_len] = time_weight[:copy_len]
            if target_steps > time_len:
                new_weight[copy_len:] = time_weight[-1:].repeat(target_steps - copy_len, 1)
            action_state["time_embed.weight"] = new_weight
    strict = all(
        key in action_state
        for key in ["type_embed.weight", "time_embed.weight", "slot_embed.weight"]
    )
    action_prior.load_state_dict(action_state, strict=strict)
    action_prior.eval()
    for p in action_prior.parameters():
        p.requires_grad = False

    adapter = Adapter(
        d_model=cfg["action_prior"]["d_model"],
        action_vocab=action_vocab,
        hidden=cfg["adapter"]["mlp_hidden"],
    ).to(args.device)

    optim = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0.0
        for world_seq, ego_seq, act_tokens, goal in tqdm(loader):
            world_seq = world_seq.to(args.device)
            goal = goal.to(args.device)

            with torch.no_grad():
                state_summary = state_prior.hidden_summary(world_seq)

                seq_list = []
                for b in range(ego_seq.size(0)):
                    seq_list.append(
                        build_action_context_sequence(ego_seq[b], act_tokens[b], codebook_size)
                    )
                seq = torch.stack(seq_list, dim=0).to(args.device)

                base_logits, hidden = action_prior(seq, prefix_emb=None)
                action_summary = hidden[:, -1, :]
                base_action_logits = base_logits[:, -1, codebook_size:]

            adapter_logits = adapter(state_summary, action_summary, goal)
            target_action = act_tokens[:, -1].to(args.device)
            loss_bc = torch.nn.functional.cross_entropy(adapter_logits, target_action)

            loss = loss_bc
            if kl_lambda > 0:
                logp = torch.log_softmax(adapter_logits, dim=-1)
                logq = torch.log_softmax(base_action_logits, dim=-1)
                kl = torch.sum(torch.exp(logp) * (logp - logq), dim=-1).mean()
                loss = loss + kl_lambda * kl

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")
        scheduler.step()

    torch.save(adapter.state_dict(), args.out)
    out_dir = os.path.dirname(args.out)
    np.save(os.path.join(out_dir, "adapter_loss.npy"), np.array(losses, dtype=np.float32))
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adapter_loss.png"))
    plt.close()
    print(f"Saved Adapter to {args.out}")


if __name__ == "__main__":
    main()
