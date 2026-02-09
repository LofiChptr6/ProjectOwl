#!/usr/bin/env python
"""Train a CNN or Transformer model on the populated data.

Usage::

    python scripts/train.py --model cnn
    python scripts/train.py --model transformer --epochs 50 --lr 3e-4
    python scripts/train.py --model cnn --dashboard   # also launch live dashboard
"""

import argparse
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from owl.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    REPORT_DIR,
)
from owl.data.db import log_metric
from owl.data.feeder import make_dataloaders
from owl.models.cnn_model import CNNTrainer
from owl.models.transformer_model import TransformerTrainer
from owl.preprocessing.pipeline import PreprocessingPipeline
from owl.visualization.reports import (
    plot_category_examples,
    plot_feature_importance,
    plot_training_history,
)


def _metrics_cb(metrics: dict):
    """Callback passed to the trainer — logs throughput to DB for the dashboard."""
    try:
        log_metric(
            model_name=metrics.get("model", "unknown"),
            epoch=metrics.get("epoch", 0),
            batch=metrics.get("batch", 0),
            metric_name="throughput_rows_per_sec",
            metric_value=metrics.get("throughput_rows_per_sec", 0),
            phase=metrics.get("phase", "train"),
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train ProjectOwl model")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the monitoring dashboard in background")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    # optionally start the dashboard in a background thread
    if args.dashboard:
        from owl.orchestration.dashboard import run_dashboard
        t = threading.Thread(target=run_dashboard, kwargs={"debug": False},
                             daemon=True)
        t.start()
        logging.info("Dashboard started on http://localhost:8050")

    # ── data ──────────────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline()
    train_loader, val_loader = make_dataloaders(
        pipeline=pipeline,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    in_features = train_loader.dataset.get_feature_dim()
    logging.info("Input features: %d", in_features)

    # ── model ─────────────────────────────────────────────────────────────
    if args.model == "cnn":
        trainer = CNNTrainer(in_features)
    else:
        trainer = TransformerTrainer(in_features)

    trainer.model_name = args.model

    # ── train ─────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        metrics_callback=_metrics_cb,
    )

    # ── reports ───────────────────────────────────────────────────────────
    report_dir = REPORT_DIR / args.model
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history, title=f"{args.model.upper()} Training",
        save_path=report_dir / "training_history.png",
    )

    # feature importance (use a sample batch)
    batch = next(iter(val_loader))
    sector_idx = batch[2] if len(batch) > 2 else None
    plot_feature_importance(
        trainer.model, batch[0],
        feature_names=pipeline.feature_columns,
        device=trainer.device,
        save_path=report_dir / "feature_importance.png",
        sector_idx=sector_idx,
    )

    # category examples
    plot_category_examples(
        val_loader.dataset,
        save_path=report_dir / "category_examples.png",
    )

    logging.info("All reports saved to %s", report_dir)
    logging.info("Training complete. Best checkpoint at outputs/checkpoints/")


if __name__ == "__main__":
    main()
