#!/usr/bin/env python
"""Generate t-SNE visualisations and trajectory videos.

Usage::

    python scripts/run_tsne.py --model cnn
    python scripts/run_tsne.py --model transformer --video --case-ids abc123 def456
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from owl.config import BATCH_SIZE, NUM_WORKERS, REPORT_DIR, VIDEO_DIR
from owl.data.db import get_case_ids
from owl.data.feeder import OwlDataset
from owl.models.cnn_model import CNNTrainer
from owl.models.transformer_model import TransformerTrainer
from owl.models.tsne_viz import (
    build_rolling_trajectory,
    compute_tsne,
    plot_tsne_2d,
    plot_tsne_3d,
)
from owl.preprocessing.pipeline import PreprocessingPipeline
from owl.visualization.tsne_video import make_tsne_video_2d, make_tsne_video_3d
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualisation")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--checkpoint", default="best",
                        help="Checkpoint tag to load (default: best)")
    parser.add_argument("--table", default="validation_cases")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Cap on number of windows to embed")
    parser.add_argument("--video", action="store_true",
                        help="Also produce trajectory videos")
    parser.add_argument("--case-ids", nargs="*", default=None,
                        help="Specific case IDs for video trajectories")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    # ── dataset ───────────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline()
    dataset  = OwlDataset(args.table, pipeline=pipeline)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
    in_features = dataset.get_feature_dim()

    # ── load model ────────────────────────────────────────────────────────
    if args.model == "cnn":
        trainer = CNNTrainer(in_features)
    else:
        trainer = TransformerTrainer(in_features)
    trainer.model_name = args.model
    trainer.load(tag=args.checkpoint)

    # ── extract latents ───────────────────────────────────────────────────
    latents, labels = trainer.extract_latents(loader)
    if len(latents) > args.max_samples:
        idx = np.random.choice(len(latents), args.max_samples, replace=False)
        latents = latents[idx]
        labels  = labels[idx]

    logging.info("Extracted %d latent vectors (dim=%d)", len(latents), latents.shape[1])

    # ── t-SNE 2D + 3D ────────────────────────────────────────────────────
    emb_2d = compute_tsne(latents, n_components=2)
    emb_3d = compute_tsne(latents, n_components=3)

    out_dir = REPORT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tsne_2d(emb_2d, labels, title=f"{args.model.upper()} t-SNE 2D",
                 save_path=out_dir / "tsne_2d.png")
    plot_tsne_3d(emb_3d, labels, title=f"{args.model.upper()} t-SNE 3D",
                 save_path=out_dir / "tsne_3d.png")

    # ── trajectory videos ─────────────────────────────────────────────────
    if args.video:
        case_ids = args.case_ids or get_case_ids(args.table)[:3]
        # build case-index mapping (which row in latents belongs to which case)
        case_index = []
        for cid, _ in dataset.windows:
            case_index.append(cid)
        case_index = np.array(case_index[:len(latents)])

        vid_dir = VIDEO_DIR / args.model
        vid_dir.mkdir(parents=True, exist_ok=True)

        for cid in case_ids:
            traj = build_rolling_trajectory(
                latents, labels, emb_2d, emb_3d, case_index, cid)
            if traj["n_frames"] < 2:
                logging.warning("Case %s has <2 frames — skipping video", cid)
                continue

            make_tsne_video_2d(
                emb_2d, labels, traj["emb_2d"], traj["labels"],
                title=f"{cid} 2D", save_path=vid_dir / f"{cid}_2d.mp4")
            make_tsne_video_3d(
                emb_3d, labels, traj["emb_3d"], traj["labels"],
                title=f"{cid} 3D", save_path=vid_dir / f"{cid}_3d.mp4")

    logging.info("t-SNE visualisation complete.")


if __name__ == "__main__":
    main()
