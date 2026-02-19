import torch
import torch.nn as nn
from fa_planner.models.transformer import DecoderTransformer


class ActionPrior(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.decoder = DecoderTransformer(d_model, n_layers, n_heads, d_ff, dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, prefix_emb=None):
        # token_ids: (B, T)
        b, t = token_ids.shape
        tok = self.token_embed(token_ids)
        total_len = t
        if prefix_emb is not None:
            total_len = prefix_emb.size(1) + t
        pos = self.pos_embed(torch.arange(total_len, device=token_ids.device))
        pos = pos.unsqueeze(0).expand(b, -1, -1)
        if prefix_emb is not None:
            emb = torch.cat([prefix_emb, tok], dim=1)
        else:
            emb = tok
        emb = self.drop(emb + pos)
        hidden = self.decoder(emb)
        logits = self.head(hidden)
        return logits, hidden

    @torch.no_grad()
    def sample_next(self, token_ids, prefix_emb=None, temperature=1.0):
        logits, _ = self.forward(token_ids, prefix_emb=prefix_emb)
        next_logits = logits[:, -1] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(1)
