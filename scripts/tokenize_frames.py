import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from fa_planner.models.vqvae import VQVAE
from fa_planner.utils.config import load_config
from fa_planner.utils.vision import image_to_tensor


def encode_frames(model, frames, device):
    # frames: (T, H, W, C)
    batch = torch.stack([image_to_tensor(f) for f in frames], dim=0).to(device)
    codes = model.encode(batch)
    return codes.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--vqvae", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = VQVAE(
        in_channels=3,
        hidden=cfg["tokenizer"]["latent_dim"],
        n_codes=cfg["tokenizer"]["codebook_size"],
        code_dim=cfg["tokenizer"]["latent_dim"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.vqvae, map_location=args.device))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    files = sorted([f for f in os.listdir(args.data) if f.endswith(".npz")])
    for i, fname in enumerate(files):
        path = os.path.join(args.data, fname)
        data = np.load(path, allow_pickle=True)
        world_tokens = encode_frames(model, data["world_frames"], args.device)
        ego_tokens = encode_frames(model, data["ego_frames"], args.device)
        out_path = os.path.join(args.out, fname)
        np.savez_compressed(
            out_path,
            world_tokens=world_tokens.astype(np.int64),
            ego_tokens=ego_tokens.astype(np.int64),
            actions=data["actions"].astype(np.float32),
            goal=data["goal"].astype(np.float32),
            sim_state=data["sim_state"].astype(np.float32),
            wind=data["wind"].astype(np.float32),
        )
        if (i + 1) % 100 == 0:
            print(f"Tokenized {i+1}/{len(files)}")


if __name__ == "__main__":
    main()
