import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, code_dim, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z):
        # z: (B, C, H, W)
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, c)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        codes = dist.argmin(dim=1)
        quant = self.embedding(codes).view(b, h, w, c)
        quant = quant.permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(quant.detach(), z) + self.beta * F.mse_loss(quant, z.detach())
        quant = z + (quant - z).detach()
        avg_probs = torch.mean(F.one_hot(codes, self.n_codes).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        codes = codes.view(b, h, w)
        return quant, loss, perplexity, codes


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=128, n_codes=512, code_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, code_dim, 3, stride=1, padding=1),
        )
        self.quantizer = VectorQuantizer(n_codes, code_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code_dim, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, in_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        quant, q_loss, perplexity, codes = self.quantizer(z)
        recon = self.decoder(quant)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + q_loss
        return recon, loss, recon_loss, q_loss, perplexity, codes

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        _, _, _, codes = self.quantizer(z)
        return codes

    @torch.no_grad()
    def decode(self, codes):
        # codes: (B, H, W)
        quant = self.quantizer.embedding(codes)
        quant = quant.permute(0, 3, 1, 2).contiguous()
        recon = self.decoder(quant)
        return recon
