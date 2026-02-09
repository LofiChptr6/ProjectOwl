"""
Sector token embedding layer
============================
Maps sector category index → dense vector for conditioning the model.
Index 0 = unknown/padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from owl.data.ticker_config import get_num_sectors


class SectorEmbedding(nn.Module):
    """Embedding layer for sector categories. Padding index 0."""

    def __init__(self, num_sectors: int | None = None, embed_dim: int = 32):
        super().__init__()
        self.num_sectors = num_sectors or get_num_sectors()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=self.num_sectors,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

    def forward(self, sector_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sector_idx : (B,) or (B, 1) — integer sector indices (0 = unknown)

        Returns
        -------
        (B, embed_dim)
        """
        if sector_idx.dim() == 2:
            sector_idx = sector_idx.squeeze(-1)
        return self.embedding(sector_idx.clamp(min=0))
