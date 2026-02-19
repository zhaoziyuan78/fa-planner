import numpy as np
import torch


def image_to_tensor(img):
    # img: uint8 HWC
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def batch_to_tensor(frames):
    # frames: list of HWC
    return torch.stack([image_to_tensor(f) for f in frames], dim=0)


def tensor_to_image(tensor):
    # tensor: CHW in [0,1]
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr
