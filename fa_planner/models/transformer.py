import torch
import torch.nn as nn


def causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class DecoderTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        mask = causal_mask(seq_len, x.device)
        return self.transformer(x, mask)
