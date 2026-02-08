"""
t-SNE visualisation utilities
===============================
Reduce high-dimensional latent vectors to 2-D or 3-D via t-SNE,
produce static scatter plots and per-stock trajectory DataFrames
that the video generator can animate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from owl.config import (
    CATEGORY_NAMES,
    NUM_CATEGORIES,
    REPORT_DIR,
    REPORT_DPI,
    TSNE_LEARNING_RATE,
    TSNE_N_ITER,
    TSNE_PERPLEXITY,
)

logger = logging.getLogger(__name__)

# colour palette for categories
_PALETTE = plt.cm.RdYlGn(np.linspace(0.1, 0.9, NUM_CATEGORIES))


# ══════════════════════════════════════════════════════════════════════════
#  Core t-SNE computation
# ══════════════════════════════════════════════════════════════════════════

def compute_tsne(
    latents: np.ndarray,
    n_components: int = 2,
    perplexity: float = TSNE_PERPLEXITY,
    n_iter: int = TSNE_N_ITER,
    learning_rate: float = TSNE_LEARNING_RATE,
    random_state: int = 42,
) -> np.ndarray:
    """Return ``(N, n_components)`` embedding of *latents* ``(N, D)``."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, max(5, len(latents) // 4)),
        max_iter=n_iter,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
    )
    emb = tsne.fit_transform(latents)
    logger.info("t-SNE (%dD) done — KL divergence %.4f",
                n_components, tsne.kl_divergence_)
    return emb


# ══════════════════════════════════════════════════════════════════════════
#  Static scatter plots
# ══════════════════════════════════════════════════════════════════════════

def plot_tsne_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE 2D",
    save_path: str | Path | None = None,
) -> Figure:
    """2-D scatter coloured by category label."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in range(NUM_CATEGORIES):
        mask = labels == cat
        name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else str(cat)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[_PALETTE[cat]], label=name,
            alpha=0.6, s=12, edgecolors="none",
        )
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI)
        logger.info("Saved 2D t-SNE → %s", save_path)
    return fig


def plot_tsne_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE 3D",
    save_path: str | Path | None = None,
    elev: float = 25,
    azim: float = 135,
) -> Figure:
    """3-D scatter coloured by category label."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for cat in range(NUM_CATEGORIES):
        mask = labels == cat
        name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else str(cat)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
            c=[_PALETTE[cat]], label=name,
            alpha=0.6, s=12, edgecolors="none",
        )
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, markerscale=2)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI)
        logger.info("Saved 3D t-SNE → %s", save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Rolling trajectory for a single stock
# ══════════════════════════════════════════════════════════════════════════

def build_rolling_trajectory(
    latents: np.ndarray,
    labels: np.ndarray,
    full_embedding_2d: np.ndarray,
    full_embedding_3d: np.ndarray,
    case_indices: np.ndarray,
    target_case_id: str | int,
) -> dict:
    """Return dict with 2D/3D trajectories for frames in *target_case_id*.

    ``case_indices`` maps each row of *latents* to a case identifier so we
    can slice out the rolling windows that belong to a single stock.
    """
    mask = case_indices == target_case_id
    return {
        "emb_2d": full_embedding_2d[mask],
        "emb_3d": full_embedding_3d[mask],
        "labels": labels[mask],
        "n_frames": int(mask.sum()),
    }
