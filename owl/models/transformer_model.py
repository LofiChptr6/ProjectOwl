"""
Transformer encoder for time-series characterisation
=====================================================
Architecture
------------
1. **Linear projection** of raw features → ``d_model``.
2. **Learnable positional encoding** (since minute-bars have regular spacing).
3. **N Transformer encoder layers** with multi-head self-attention.
4. **CLS token** aggregation → latent vector of ``LATENT_DIM``.
5. **Classification head** → ``NUM_CATEGORIES`` logits.

``encode()`` returns the CLS-aggregated latent for t-SNE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from owl.config import (
    INPUT_WINDOW_MINUTES,
    LATENT_DIM,
    NUM_CATEGORIES,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FF,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
)
from owl.models.base import BaseTrainer


# ── positional encoding ──────────────────────────────────────────────────

class _LearnablePE(nn.Module):
    """Learnable positional embedding for sequences up to *max_len*."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pe      = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── main model ────────────────────────────────────────────────────────────

class TimeSeriesTransformer(nn.Module):
    """
    Parameters
    ----------
    in_features : number of raw features per time-step
    """

    def __init__(
        self,
        in_features: int,
        d_model: int    = TRANSFORMER_D_MODEL,
        nhead: int      = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_ff: int     = TRANSFORMER_DIM_FF,
        dropout: float  = TRANSFORMER_DROPOUT,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CATEGORIES,
        max_seq_len: int = INPUT_WINDOW_MINUTES + 1,
    ):
        super().__init__()

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # positional encoding
        self.pos_enc = _LearnablePE(d_model, max_len=max_seq_len, dropout=dropout)

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # latent projection
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, sector_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)
        sector_idx : optional, ignored (add SectorEmbedding for full support)

        Returns
        -------
        logits : (B, num_classes)
        """
        z = self.encode(x)
        return self.classifier(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS-token latent vector."""
        B = x.size(0)
        x = self.input_proj(x)                          # (B, T, d_model)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)

        x = self.pos_enc(x)
        x = self.transformer(x)

        cls_out = x[:, 0]                               # (B, d_model)
        return self.to_latent(cls_out)                   # (B, latent_dim)


# ── trainer convenience ───────────────────────────────────────────────────

class TransformerTrainer(BaseTrainer):
    model_name = "transformer"

    def __init__(self, in_features: int, **model_kwargs):
        super().__init__()
        self.model = TimeSeriesTransformer(in_features, **model_kwargs)
