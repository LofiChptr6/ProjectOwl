"""
Multi-scale 1-D CNN for time-series characterisation
=====================================================
Architecture
------------
1. **Multi-scale convolution block** — parallel 1-D conv branches with
   different kernel sizes capture patterns at multiple temporal resolutions.
2. **Residual stack** — deeper feature extraction with skip connections.
3. **Global average pooling** → latent vector of size ``LATENT_DIM``.
4. **Classification head** → ``NUM_CATEGORIES`` logits.

The ``encode()`` method returns the latent vector for downstream t-SNE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from owl.config import (
    CNN_CHANNELS,
    CNN_KERNEL_SIZES,
    INPUT_WINDOW_MINUTES,
    LATENT_DIM,
    NUM_CATEGORIES,
)
from owl.models.base import BaseTrainer


# ── building blocks ───────────────────────────────────────────────────────

class _MultiScaleBlock(nn.Module):
    """Parallel 1-D convolutions with different kernel sizes, concatenated."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: list[int]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ))
        # project concatenated branches back to out_ch
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch * len(kernel_sizes), out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):                       # (B, C_in, T)
        parts = [branch(x) for branch in self.branches]
        # all parts have the same temporal length (same-padding)
        cat = torch.cat(parts, dim=1)           # (B, out*K, T)
        return self.proj(cat)                    # (B, out, T)


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.conv(x))


# ── main model ────────────────────────────────────────────────────────────

class TimeSeriesCNN(nn.Module):
    """
    Parameters
    ----------
    in_features : number of input features per time-step
    """

    def __init__(self, in_features: int,
                 channels: list[int] | None = None,
                 kernel_sizes: list[int] | None = None,
                 latent_dim: int = LATENT_DIM,
                 num_classes: int = NUM_CATEGORIES):
        super().__init__()
        channels     = channels or CNN_CHANNELS
        kernel_sizes = kernel_sizes or CNN_KERNEL_SIZES

        layers: list[nn.Module] = []

        # first multi-scale block (in_features → channels[0])
        layers.append(_MultiScaleBlock(in_features, channels[0], kernel_sizes))

        # stacked residual + down-sample blocks
        for i in range(1, len(channels)):
            layers.append(nn.Sequential(
                nn.Conv1d(channels[i - 1], channels[i], 3, stride=2, padding=1),
                nn.BatchNorm1d(channels[i]),
                nn.GELU(),
            ))
            layers.append(_ResBlock(channels[i]))

        self.encoder = nn.Sequential(*layers)

        # global average pool → latent
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], latent_dim),
            nn.GELU(),
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)  — batch of time-series windows
            T = INPUT_WINDOW_MINUTES, F = in_features

        Returns
        -------
        logits : (B, num_classes)
        """
        x = x.transpose(1, 2)                  # → (B, F, T) for Conv1d
        z = self.encode_from_conv(x)
        return self.classifier(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector (for t-SNE)."""
        x = x.transpose(1, 2)
        return self.encode_from_conv(x)

    def encode_from_conv(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                     # (B, C_last, T')
        return self.to_latent(h)                # (B, latent_dim)


# ── trainer convenience ───────────────────────────────────────────────────

class CNNTrainer(BaseTrainer):
    model_name = "cnn"

    def __init__(self, in_features: int, **model_kwargs):
        super().__init__()
        self.model = TimeSeriesCNN(in_features, **model_kwargs)
