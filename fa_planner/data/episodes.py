import os
import random
import numpy as np
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")]
        )
        if not self.files:
            raise RuntimeError(f"No npz files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        return {k: data[k] for k in data.files}


class FrameDataset(Dataset):
    def __init__(self, root, frame_keys=("world_frames", "ego_frames")):
        self.root = root
        self.frame_keys = frame_keys
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")]
        )
        if not self.files:
            raise RuntimeError(f"No npz files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        frames = []
        for key in self.frame_keys:
            if key in data:
                arr = data[key]
                t = random.randint(0, arr.shape[0] - 1)
                frames.append(arr[t])
        if not frames:
            raise RuntimeError("No frames found in episode")
        frame = random.choice(frames)
        return frame


class TokenSequenceDataset(Dataset):
    def __init__(self, root, world_key="world_tokens", ego_key="ego_tokens"):
        self.root = root
        self.world_key = world_key
        self.ego_key = ego_key
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")]
        )
        if not self.files:
            raise RuntimeError(f"No npz files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        out = {}
        if self.world_key in data:
            out["world_tokens"] = data[self.world_key]
        if self.ego_key in data:
            out["ego_tokens"] = data[self.ego_key]
        if "actions" in data:
            out["actions"] = data["actions"]
        if "goal" in data:
            out["goal"] = data["goal"]
        if "sim_state" in data:
            out["sim_state"] = data["sim_state"]
        if "wind" in data:
            out["wind"] = data["wind"]
        return out
