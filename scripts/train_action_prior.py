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
from fa_planner.models.action_prior import ActionPrior
from fa_planner.utils.config import load_config
from tqdm import tqdm


def discretize_action(action, bins, a_max):
    edges = np.linspace(-a_max, a_max, bins)
    ix = np.argmin(np.abs(action[0] - edges))
    iy = np.argmin(np.abs(action[1] - edges))
    return ix * bins + iy


def estimate_class_weights(files, weight_samples, context_steps, action_bins, a_max, start0_prob):
    counts = np.zeros(action_bins * action_bins, dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(weight_samples):
        fname = files[rng.integers(0, len(files))]
        data = np.load(fname, allow_pickle=True)
        actions = data["actions"]
        t_max = actions.shape[0]
        if t_max <= context_steps:
            continue
        if t_max == context_steps or rng.random() < start0_prob:
            start = 0
        else:
            start = rng.integers(0, t_max - context_steps)
        act = actions[start + context_steps - 1]
        token = discretize_action(act, action_bins, a_max)
        counts[token] += 1
    eps = 1e-6
    weights = counts.sum() / (counts + eps)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 5.0)
    return weights.astype(np.float32), counts


class ActionPriorDataset(Dataset):
    def __init__(self, root, context_steps, codebook_size, action_bins, a_max, start0_prob):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.context_steps = context_steps
        self.codebook_size = codebook_size
        self.action_bins = action_bins
        self.a_max = a_max
        self.start0_prob = start0_prob
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
        if t_max == self.context_steps or np.random.rand() < self.start0_prob:
            start = 0
        else:
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--start0_prob", type=float, default=0.5)
    parser.add_argument("--class_weighting", action="store_true")
    parser.add_argument("--weight_samples", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    context_steps = cfg["action_prior"]["L"]
    codebook_size = cfg["tokenizer"]["codebook_size"]
    action_vocab = cfg["action_prior"]["action_vocab"]
    action_bins = cfg["action_prior"]["action_bins"]
    a_max = cfg["env"]["a_max"]

    dataset = ActionPriorDataset(
        args.data, context_steps, codebook_size, action_bins, a_max, args.start0_prob
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = ActionPrior(
        vocab_size=codebook_size + action_vocab,
        codebook_size=codebook_size,
        d_model=cfg["action_prior"]["d_model"],
        n_layers=cfg["action_prior"]["n_layers"],
        n_heads=cfg["action_prior"]["n_heads"],
        d_ff=cfg["action_prior"]["d_ff"],
        dropout=cfg["action_prior"]["dropout"],
        max_seq_len=context_steps * 65 + 32,
    ).to(args.device)

    weight_tensor = None
    if args.class_weighting:
        weights, counts = estimate_class_weights(
            dataset.files,
            args.weight_samples,
            context_steps,
            action_bins,
            a_max,
            args.start0_prob,
        )
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=args.device)
        center = (action_bins // 2) * action_bins + (action_bins // 2)
        total = counts.sum()
        if total > 0:
            center_ratio = counts[center] / total
            print(f"Action token center ratio: {center_ratio:.3f}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            inp = batch[:, :-1]
            logits, _ = model(inp)
            target = batch[:, 1:]
            pos = torch.arange(target.size(1), device=target.device)
            action_pos = (pos + 1) % 65 == 64
            step_idx = (pos + 1) // 65
            valid_pos = action_pos & (step_idx >= 3)
            action_logits = logits[:, valid_pos, codebook_size:]
            action_target = target[:, valid_pos] - codebook_size
            if action_logits.numel() == 0:
                continue
            loss = torch.nn.functional.cross_entropy(
                action_logits.reshape(-1, action_logits.size(-1)),
                action_target.reshape(-1),
                #weight=weight_tensor,
            )
            pred = torch.argmax(action_logits, dim=-1)
            acc = (pred == action_target).float().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_acc += acc.item()
            total_count += 1
        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_acc / max(1, total_count)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={avg_acc:.4f}")

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
