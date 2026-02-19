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
from fa_planner.models.action_prior import ActionPrior
from fa_planner.utils.config import load_config


class ActionPriorDataset(Dataset):
    def __init__(self, root, context_steps, codebook_size, action_bins, a_max):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_steps = context_steps
        self.codebook_size = codebook_size
        self.action_bins = action_bins
        self.a_max = a_max
        if not self.files:
            raise RuntimeError(f"No token files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        ego = data["ego_tokens"]
        actions = data["actions"]
        t_max = ego.shape[0]
        if t_max <= self.context_steps:
            raise RuntimeError("Episode too short for context window")
        start = np.random.randint(0, t_max - self.context_steps)
        ego_seq = ego[start : start + self.context_steps]
        act_seq = actions[start : start + self.context_steps]
        # discretize actions to tokens
        bins = self.action_bins
        edges = np.linspace(-self.a_max, self.a_max, bins)
        idx_x = np.argmin(np.abs(act_seq[:, 0, None] - edges), axis=-1)
        idx_y = np.argmin(np.abs(act_seq[:, 1, None] - edges), axis=-1)
        act_tokens = idx_x * bins + idx_y
        # build token sequence
        seq = []
        for t in range(self.context_steps):
            seq.extend(ego_seq[t].reshape(-1).tolist())
            seq.append(self.codebook_size + int(act_tokens[t]))
        seq = np.array(seq, dtype=np.int64)
        return torch.from_numpy(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_steps = cfg["action_prior"]["L"]
    codebook_size = cfg["tokenizer"]["codebook_size"]
    action_vocab = cfg["action_prior"]["action_vocab"]
    action_bins = cfg["action_prior"]["action_bins"]
    a_max = cfg["env"]["a_max"]

    dataset = ActionPriorDataset(args.data, context_steps, codebook_size, action_bins, a_max)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = ActionPrior(
        vocab_size=codebook_size + action_vocab,
        d_model=cfg["action_prior"]["d_model"],
        n_layers=cfg["action_prior"]["n_layers"],
        n_heads=cfg["action_prior"]["n_heads"],
        d_ff=cfg["action_prior"]["d_ff"],
        dropout=cfg["action_prior"]["dropout"],
        max_seq_len=context_steps * 65 + 32,
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            inp = batch[:, :-1]
            target = batch[:, 1:]
            logits, _ = model(inp)
            mask = target >= codebook_size
            if mask.any():
                action_logits = logits[mask]
                action_target = target[mask]
                loss = torch.nn.functional.cross_entropy(action_logits, action_target)
            else:
                loss = torch.tensor(0.0, device=args.device)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    out_dir = os.path.dirname(args.out)
    np.save(os.path.join(out_dir, "action_prior_loss.npy"), np.array(losses, dtype=np.float32))
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_prior_loss.png"))
    plt.close()
    print(f"Saved ActionPrior to {args.out}")


if __name__ == "__main__":
    main()
