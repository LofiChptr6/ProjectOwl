"""
t-SNE video generator
======================
Produces MP4 videos showing a stock's rolling-window latent trajectory
animated in the 2-D or 3-D t-SNE embedding space.

The "background" is the full set of training points (faded), while the
"foreground" highlights successive rolling windows of a single case,
showing how the stock's behaviour *moves through embedding space* over
the duration of the test period.
"""

from __future__ import annotations

import logging
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

from owl.config import (
    CATEGORY_NAMES,
    NUM_CATEGORIES,
    REPORT_DPI,
    VIDEO_DIR,
    VIDEO_DURATION_SEC,
    VIDEO_FPS,
)

matplotlib.use("Agg")                      # non-interactive backend
logger = logging.getLogger(__name__)

_PALETTE = plt.cm.RdYlGn(np.linspace(0.1, 0.9, NUM_CATEGORIES))


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════

def make_tsne_video_2d(
    bg_embedding: np.ndarray,
    bg_labels: np.ndarray,
    trajectory_2d: np.ndarray,
    trajectory_labels: np.ndarray,
    title: str = "Stock trajectory (2D t-SNE)",
    save_path: str | Path | None = None,
    fps: int = VIDEO_FPS,
    duration_sec: int = VIDEO_DURATION_SEC,
) -> Path:
    """Render a 2-D trajectory video and return the file path."""
    save_path = Path(save_path or VIDEO_DIR / "tsne_2d.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(trajectory_2d)
    total_frames = fps * duration_sec
    step = max(1, n_frames // total_frames)

    writer = imageio.get_writer(str(save_path), fps=fps)

    for i in range(0, n_frames, step):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
        # background
        for cat in range(NUM_CATEGORIES):
            mask = bg_labels == cat
            ax.scatter(bg_embedding[mask, 0], bg_embedding[mask, 1],
                       c=[_PALETTE[cat]], alpha=0.08, s=6, edgecolors="none")

        # trajectory up to current frame
        trail = trajectory_2d[:i + 1]
        ax.plot(trail[:, 0], trail[:, 1],
                color="black", alpha=0.5, linewidth=0.8)
        ax.scatter(trail[-1, 0], trail[-1, 1],
                   c="red", s=80, zorder=5, edgecolors="black", linewidths=0.8)

        ax.set_title(f"{title}  — frame {i}/{n_frames}", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()

        # render to array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        writer.append_data(buf)
        plt.close(fig)

    writer.close()
    logger.info("2D t-SNE video (%d frames) → %s", n_frames // step, save_path)
    return save_path


def make_tsne_video_3d(
    bg_embedding: np.ndarray,
    bg_labels: np.ndarray,
    trajectory_3d: np.ndarray,
    trajectory_labels: np.ndarray,
    title: str = "Stock trajectory (3D t-SNE)",
    save_path: str | Path | None = None,
    fps: int = VIDEO_FPS,
    duration_sec: int = VIDEO_DURATION_SEC,
) -> Path:
    """Render a 3-D rotating trajectory video and return the file path."""
    save_path = Path(save_path or VIDEO_DIR / "tsne_3d.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(trajectory_3d)
    total_frames = fps * duration_sec
    step = max(1, n_frames // total_frames)
    azim_start, azim_end = 0, 360       # full rotation

    writer = imageio.get_writer(str(save_path), fps=fps)

    frame_idx = 0
    for i in range(0, n_frames, step):
        azim = azim_start + (azim_end - azim_start) * (frame_idx / max(total_frames, 1))
        fig = plt.figure(figsize=(9, 7), dpi=80)
        ax = fig.add_subplot(111, projection="3d")

        # background
        for cat in range(NUM_CATEGORIES):
            mask = bg_labels == cat
            ax.scatter(bg_embedding[mask, 0], bg_embedding[mask, 1],
                       bg_embedding[mask, 2],
                       c=[_PALETTE[cat]], alpha=0.08, s=6, edgecolors="none")

        # trajectory
        trail = trajectory_3d[:i + 1]
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                color="black", alpha=0.5, linewidth=0.8)
        ax.scatter(trail[-1, 0], trail[-1, 1], trail[-1, 2],
                   c="red", s=80, zorder=5, edgecolors="black", linewidths=0.8)

        ax.view_init(elev=25, azim=azim)
        ax.set_title(f"{title}  — frame {i}/{n_frames}", fontsize=11)
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        writer.append_data(buf)
        plt.close(fig)
        frame_idx += 1

    writer.close()
    logger.info("3D t-SNE video (%d frames) → %s", frame_idx, save_path)
    return save_path
