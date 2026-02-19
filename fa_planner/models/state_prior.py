import torch
import torch.nn as nn
from fa_planner.models.transformer import DecoderTransformer


class StatePrior(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, context_frames):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_frames = context_frames
        self.tokens_per_frame = 64
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Embedding(context_frames, d_model)
        self.spatial_embed = nn.Embedding(self.tokens_per_frame, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.decoder = DecoderTransformer(d_model, n_layers, n_heads, d_ff, dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def _build_embeddings(self, tokens):
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
        return self.pos_drop(emb)

    def forward(self, tokens):
        emb = self._build_embeddings(tokens)
        hidden = self.decoder(emb)
        logits = self.head(hidden)
        return logits, hidden

    def hidden_summary(self, tokens):
        _, hidden = self.forward(tokens)
        if tokens.dim() == 4:
            b, l, h, w = tokens.shape
            n = h * w
        else:
            b, l, n = tokens.shape
        n = self.tokens_per_frame
        last = hidden[:, (l - 1) * n : l * n]
        return last.mean(dim=1)
