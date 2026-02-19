import torch
import torch.nn as nn
from fa_planner.models.transformer import DecoderTransformer


class ScratchPolicy(nn.Module):
    def __init__(self, vocab_size, action_vocab, d_model, n_layers, n_heads, d_ff, dropout, context_frames):
        super().__init__()
        self.vocab_size = vocab_size
        self.action_vocab = action_vocab
        self.d_model = d_model
        self.context_frames = context_frames
        self.tokens_per_frame = 64
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Embedding(context_frames, d_model)
        self.spatial_embed = nn.Embedding(self.tokens_per_frame, d_model)
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_drop = nn.Dropout(dropout)
        self.decoder = DecoderTransformer(d_model, n_layers, n_heads, d_ff, dropout)
        self.head = nn.Linear(d_model, action_vocab)

    def forward(self, tokens, goal):
        # tokens: (B, L, 64) or (B, L, H, W)
        if tokens.dim() == 4:
            b, l, h, w = tokens.shape
            n = h * w
            if n != self.tokens_per_frame:
                raise ValueError("Unexpected tokens_per_frame")
            tokens = tokens.view(b, l, n)
        elif tokens.dim() == 3:
            b, l, n = tokens.shape
            if n != self.tokens_per_frame:
                raise ValueError("Unexpected tokens_per_frame")
        else:
            raise ValueError("Tokens must be 3D or 4D")
        tokens = tokens.view(b, l * n)
        tok_emb = self.token_embed(tokens)
        time_ids = torch.arange(l, device=tokens.device).repeat_interleave(n)
        spatial_ids = torch.arange(n, device=tokens.device).repeat(l)
        time_emb = self.time_embed(time_ids)[None, :, :]
        spatial_emb = self.spatial_embed(spatial_ids)[None, :, :]
        emb = tok_emb + time_emb + spatial_emb
        goal_emb = self.goal_mlp(goal).unsqueeze(1)
        emb = torch.cat([goal_emb, emb], dim=1)
        emb = self.pos_drop(emb)
        hidden = self.decoder(emb)
        pooled = hidden[:, -n:].mean(dim=1)
        logits = self.head(pooled)
        return logits
