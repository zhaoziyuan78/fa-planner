import torch
import torch.nn as nn


class LineAdapter(nn.Module):
    def __init__(self, d_model, action_vocab, hidden):
        super().__init__()
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model * 3)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_vocab),
        )

    def forward(self, state_summary, action_hint, goal):
        goal_emb = self.goal_mlp(goal)
        action_emb = self.action_mlp(action_hint)
        fused = torch.cat([state_summary, action_emb, goal_emb], dim=-1)
        fused = self.ln(fused)
        return self.mlp(fused)
