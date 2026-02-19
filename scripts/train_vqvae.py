import argparse
import os
import sys
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from fa_planner.data.episodes import FrameDataset
from fa_planner.models.vqvae import VQVAE
from fa_planner.utils.config import load_config
from fa_planner.utils.vision import batch_to_tensor


def collate_fn(batch):
    return batch_to_tensor(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = FrameDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = VQVAE(
        in_channels=3,
        hidden=cfg["tokenizer"]["latent_dim"],
        n_codes=cfg["tokenizer"]["codebook_size"],
        code_dim=cfg["tokenizer"]["latent_dim"],
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader):
            batch = batch.to(args.device)
            recon, loss, recon_loss, q_loss, perplexity, _ = model(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} perplexity={perplexity.item():.2f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved VQ-VAE to {args.out}")


if __name__ == "__main__":
    main()
