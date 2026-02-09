"""
Path Autoencoder — Unsupervised learning for path-based clustering
==================================================================
Encoder: CNN backbone (same as TimeSeriesCNN) → latent vector.
Decoder: Linear layers → reconstruct (T, F) shape.
Trained on reconstruction loss; latent space is then clustered (k-means).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from owl.config import INPUT_WINDOW_MINUTES, LATENT_DIM
from owl.models.cnn_model import TimeSeriesCNN


class PathAutoencoder(nn.Module):
    """Encode path → latent, decode → reconstruction. No labels."""

    def __init__(self, in_features: int, t: int = INPUT_WINDOW_MINUTES,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.in_features = in_features
        self.t = t
        self.latent_dim = latent_dim

        self.encoder_net = TimeSeriesCNN(in_features, latent_dim=latent_dim,
                                         num_classes=latent_dim)
        # Remove classifier; we only use encode
        self.encoder_net.classifier = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, t * in_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(z.size(0), self.t, self.in_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
