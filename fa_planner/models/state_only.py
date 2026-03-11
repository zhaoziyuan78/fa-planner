import torch
import torch.nn as nn


class StateOnlyPolicy(nn.Module):
    def __init__(self, d_model, action_vocab):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_vocab),
        )
### TODO: See goal
    def forward(self, state_summary):
        return self.head(state_summary)
