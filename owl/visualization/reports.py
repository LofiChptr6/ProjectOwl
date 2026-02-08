"""
Report generation
==================
After training, produces image-based report pages:

1. **Category examples** — a grid of time-series snippets from each
   predicted category so you can visually verify the clusters.
2. **Feature importance** — which input features contributed most to
   the model's decisions  (gradient-based attribution for CNNs,
   attention-weight analysis for Transformers).
3. **Training history** — loss & accuracy curves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from owl.config import (
    CATEGORY_NAMES,
    INPUT_WINDOW_MINUTES,
    NUM_CATEGORIES,
    REPORT_DIR,
    REPORT_DPI,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  1. Category example grid
# ══════════════════════════════════════════════════════════════════════════

def plot_category_examples(
    dataset,
    predictions: np.ndarray | None = None,
    n_per_cat: int = 4,
    feature_idx: int = 3,           # default = 'close' after normalisation
    save_path: str | Path | None = None,
) -> Figure:
    """Plot ``n_per_cat`` example time-series from each category.

    Parameters
    ----------
    dataset     : an ``OwlDataset`` (or anything indexable returning (tensor, label))
    predictions : model predictions; if *None*, ground-truth labels are used
    feature_idx : which feature column to plot (index into F dim)
    """
    # bucket indices by category
    buckets: dict[int, list[int]] = {c: [] for c in range(NUM_CATEGORIES)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        cat = int(predictions[idx]) if predictions is not None else int(label)
        if len(buckets[cat]) < n_per_cat:
            buckets[cat].append(idx)
        if all(len(v) >= n_per_cat for v in buckets.values()):
            break

    fig, axes = plt.subplots(
        NUM_CATEGORIES, n_per_cat,
        figsize=(3.5 * n_per_cat, 3 * NUM_CATEGORIES),
        sharex=True, sharey=True,
    )
    if NUM_CATEGORIES == 1:
        axes = axes[np.newaxis, :]
    if n_per_cat == 1:
        axes = axes[:, np.newaxis]

    for cat in range(NUM_CATEGORIES):
        name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else str(cat)
        for j, idx in enumerate(buckets[cat][:n_per_cat]):
            tensor, _ = dataset[idx]
            arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
            ax = axes[cat, j]
            ax.plot(arr[:, feature_idx], linewidth=0.7)
            if j == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)

    fig.suptitle("Category Example Time-Series", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Category examples → %s", save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  2. Feature importance (gradient-based attribution)
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    feature_names: list[str],
    device: torch.device | str = "cpu",
    top_k: int = 25,
    save_path: str | Path | None = None,
) -> Figure:
    """Compute mean absolute gradient w.r.t. input → bar chart of top-*k* features.

    Parameters
    ----------
    model         : trained model (CNN or Transformer)
    sample_batch  : (B, T, F) input tensor
    feature_names : length-F list of column names
    """
    model.eval()
    model.to(device)
    X = sample_batch.to(device).requires_grad_(True)

    logits = model(X)
    # sum of max-class logits as scalar
    logits.max(dim=1).values.sum().backward()

    # mean absolute gradient over batch and time
    grad = X.grad.abs().mean(dim=(0, 1)).cpu().numpy()      # (F,)

    # sort and take top-k
    order = np.argsort(grad)[::-1][:top_k]
    names = [feature_names[i] if i < len(feature_names) else f"f{i}"
             for i in order]
    vals  = grad[order]

    fig, ax = plt.subplots(figsize=(8, max(6, top_k * 0.3)))
    ax.barh(range(len(names)), vals[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |gradient|", fontsize=11)
    ax.set_title("Feature Importance (Gradient Attribution)", fontsize=13)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Feature importance → %s", save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  3. Training history
# ══════════════════════════════════════════════════════════════════════════

def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: str | Path | None = None,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="train", linewidth=1.2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="val", linewidth=1.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # accuracy
    if "train_acc" in history:
        ax2.plot(history["train_acc"], label="train", linewidth=1.2)
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="val", linewidth=1.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Training history → %s", save_path)
    return fig
