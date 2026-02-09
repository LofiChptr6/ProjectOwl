"""
Report generation
==================
After training, produces image-based report pages:

1. **Category examples** — a grid of time-series snippets from each
   predicted category so you can visually verify the clusters.
   **Category overlay** — one subplot per category with multiple curves
   overlaid (input + future, same style as category examples).
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

def _fmt_ts(ts) -> str:
    """Format timestamp for axis label."""
    if ts is None:
        return ""
    try:
        if pd.isna(ts):
            return ""
    except (TypeError, ValueError):
        pass
    return str(ts)[:19]


def plot_category_examples(
    dataset,
    predictions: np.ndarray | None = None,
    n_per_cat: int = 4,
    feature_idx: int = 3,           # default = 'close' after normalisation
    save_path: str | Path | None = None,
) -> Figure:
    """Plot ``n_per_cat`` example time-series from each category.

    Input window (model input) in blue; ~1 day future in orange.
    Subplot titles show symbol and time range.

    Parameters
    ----------
    dataset     : an ``OwlDataset`` with get_sample_for_plot (or indexable)
    predictions : model predictions; if *None*, ground-truth labels are used
    feature_idx : which feature column to plot (index into F dim)
    """
    buckets: dict[int, list[int]] = {c: [] for c in range(NUM_CATEGORIES)}
    for idx in range(len(dataset)):
        out = dataset[idx]
        label = out[1]
        cat = int(predictions[idx]) if predictions is not None else int(label)
        if len(buckets[cat]) < n_per_cat:
            buckets[cat].append(idx)
        if all(len(v) >= n_per_cat for v in buckets.values()):
            break

    has_plot_method = hasattr(dataset, "get_sample_for_plot")

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
            ax = axes[cat, j]

            if has_plot_method:
                inp_arr, tgt_arr, symbol, start_ts, end_ts, _ = dataset.get_sample_for_plot(idx)
                inp_vals = inp_arr[:, feature_idx] if feature_idx < inp_arr.shape[1] else inp_arr[:, 0]
                tgt_vals = tgt_arr[:, feature_idx] if len(tgt_arr) > 0 and feature_idx < tgt_arr.shape[1] else np.array([])
                ax.plot(inp_vals, linewidth=0.8, color="C0", label="input (model)")
                if len(tgt_vals) > 0:
                    ax.plot(np.arange(len(inp_vals), len(inp_vals) + len(tgt_vals)), tgt_vals,
                            linewidth=0.8, color="C1", label="future (actual)")
                title = f"{symbol}" if symbol else ""
                if start_ts is not None and end_ts is not None:
                    title += f"\n{_fmt_ts(start_ts)} → {_fmt_ts(end_ts)}"
                if title:
                    ax.set_title(title, fontsize=8)
            else:
                tensor = dataset[idx][0]
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                ax.plot(arr[:, feature_idx], linewidth=0.7)

            if j == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)

    fig.suptitle("Category Example Time-Series (blue=input, orange=future)", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Category examples → %s", save_path)
    return fig


def plot_category_overlay(
    dataset,
    predictions: np.ndarray | None = None,
    n_per_cat: int = 12,
    feature_idx: int = 3,
    alpha: float = 0.4,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot each category in a grid; each subplot overlays multiple curves.

    One subplot per category. Curves are input (blue) + future (orange),
    same style as category_examples, but multiple samples overlaid per subplot.

    Parameters
    ----------
    dataset     : OwlDataset with get_sample_for_plot
    predictions : model predictions; if None, ground-truth labels are used
    n_per_cat   : number of samples to overlay per category
    feature_idx : which feature column to plot (default = close)
    alpha       : transparency for overlaid curves (0–1)
    """
    buckets: dict[int, list[int]] = {c: [] for c in range(NUM_CATEGORIES)}
    for idx in range(len(dataset)):
        out = dataset[idx]
        label = out[1]
        cat = int(predictions[idx]) if predictions is not None else int(label)
        if len(buckets[cat]) < n_per_cat:
            buckets[cat].append(idx)
        if all(len(v) >= n_per_cat for v in buckets.values()):
            break

    if not hasattr(dataset, "get_sample_for_plot"):
        raise ValueError("dataset must have get_sample_for_plot for overlay plot")

    ncols = int(np.ceil(np.sqrt(NUM_CATEGORIES)))
    nrows = int(np.ceil(NUM_CATEGORIES / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    for cat in range(NUM_CATEGORIES):
        ax = axes.flat[cat]
        name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else str(cat)

        for idx in buckets[cat][:n_per_cat]:
            inp_arr, tgt_arr, _, _, _, _ = dataset.get_sample_for_plot(idx)
            inp_vals = inp_arr[:, feature_idx] if feature_idx < inp_arr.shape[1] else inp_arr[:, 0]
            tgt_vals = tgt_arr[:, feature_idx] if len(tgt_arr) > 0 and feature_idx < tgt_arr.shape[1] else np.array([])
            ax.plot(inp_vals, linewidth=0.7, color="C0", alpha=alpha)
            if len(tgt_vals) > 0:
                ax.plot(
                    np.arange(len(inp_vals), len(inp_vals) + len(tgt_vals)),
                    tgt_vals,
                    linewidth=0.7,
                    color="C1",
                    alpha=alpha,
                )

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Category Overlay (blue=input, orange=future)", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Category overlay → %s", save_path)
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
    sector_idx: torch.Tensor | None = None,
) -> Figure:
    """Compute mean absolute gradient w.r.t. input → bar chart of top-*k* features.

    Parameters
    ----------
    model         : trained model (CNN or Transformer)
    sample_batch  : (B, T, F) input tensor
    feature_names : length-F list of column names
    sector_idx    : optional (B,) sector indices (for CNN with sector embedding)
    """
    model.eval()
    model.to(device)
    X = sample_batch.to(device).requires_grad_(True)
    sector = sector_idx.to(device).long() if sector_idx is not None else None

    logits = model(X, sector_idx=sector)
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

    n = max(
        len(history.get("train_loss", [])),
        len(history.get("val_loss", [])),
        len(history.get("train_acc", [])),
        len(history.get("val_acc", [])),
    )
    if n == 0:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    else:
        x = list(range(1, n + 1))
        kw = {"marker": "o", "markersize": 6, "linewidth": 1.2}
        # loss
        if history.get("train_loss"):
            ax1.plot(x[: len(history["train_loss"])], history["train_loss"], label="train", **kw)
        if history.get("val_loss"):
            ax1.plot(x[: len(history["val_loss"])], history["val_loss"], label="val", **kw)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # accuracy
        if history.get("train_acc"):
            ax2.plot(x[: len(history["train_acc"])], history["train_acc"], label="train", **kw)
        if history.get("val_acc"):
            ax2.plot(x[: len(history["val_acc"])], history["val_acc"], label="val", **kw)
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
