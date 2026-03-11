import argparse
import os
import sys

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
from fa_planner.utils.config import load_config
from tqdm import tqdm


class StatePriorDataset(Dataset):
    def __init__(self, root, context_frames):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_frames = context_frames
        if not self.files:
            raise RuntimeError(f"No token files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        tokens = data["world_tokens"]
        t_max = tokens.shape[0] - 1
        if t_max < self.context_frames - 1:
            raise RuntimeError("Episode too short for context window")
        t = np.random.randint(self.context_frames - 1, t_max + 1)
        seq = tokens[t - self.context_frames + 1 : t + 1]
        return torch.from_numpy(seq.astype(np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_frames = cfg["state_prior"]["L"]
    dataset = StatePriorDataset(args.data, context_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = StatePrior(
        vocab_size=cfg["tokenizer"]["codebook_size"],
        d_model=cfg["state_prior"]["d_model"],
        n_layers=cfg["state_prior"]["n_layers"],
        n_heads=cfg["state_prior"]["n_heads"],
        d_ff=cfg["state_prior"]["d_ff"],
        dropout=cfg["state_prior"]["dropout"],
        context_frames=context_frames,
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            logits, _ = model(batch)
            flat = batch.view(batch.size(0), -1)
            pred = logits[:, :-1, :]
            target = flat[:, 1:]
            start = max(0, (context_frames - 1) * 64 - 1)
            end = min(start + 64, pred.size(1))
            pred = pred[:, start:end, :].reshape(-1, pred.size(-1))
            target = target[:, start:end].reshape(-1)
            loss = torch.nn.functional.cross_entropy(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    out_dir = os.path.dirname(args.out)
    np.save(os.path.join(out_dir, "state_prior_loss.npy"), np.array(losses, dtype=np.float32))
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_prior_loss.png"))
    plt.close()
    print(f"Saved StatePrior to {args.out}")


if __name__ == "__main__":
    main()
