import numpy as np


def quantize_action(action, bins, a_max):
    # action: (..., 2)
    action = np.clip(action, -a_max, a_max)
    edges = np.linspace(-a_max, a_max, bins)
    idx = np.argmin(np.abs(action[..., None] - edges), axis=-1)
    return idx


def dequantize_action(idx, bins, a_max):
    edges = np.linspace(-a_max, a_max, bins)
    idx = np.clip(idx, 0, bins - 1)
    return edges[idx]


def action_token_id(ix, iy, bins):
    return ix * bins + iy


def action_token_to_indices(token_id, bins):
    ix = token_id // bins
    iy = token_id % bins
    return ix, iy


def action_token_to_continuous(token_id, bins, a_max):
    ix, iy = action_token_to_indices(token_id, bins)
    ax = dequantize_action(ix, bins, a_max)
    ay = dequantize_action(iy, bins, a_max)
    return np.stack([ax, ay], axis=-1)
