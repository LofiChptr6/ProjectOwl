#!/usr/bin/env python
"""
ProjectOwl — End-to-end example
=================================
Runs the full pipeline in miniature so you can verify every component works:

  1. Init database tables
  2. Populate a small number of random cases
  3. Train a CNN (few epochs)
  4. Generate reports & t-SNE plots
  5. (Optional) launch the live dashboard

Adjust the constants below for a quick smoke-test vs. a longer run.

Usage::

    python run_example.py
    python run_example.py --dashboard     # also start live monitoring
    python run_example.py --model transformer
"""

import argparse
import logging
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
)
logger = logging.getLogger("run_example")


# ── Tuneables for the example run ─────────────────────────────────────────
EXAMPLE_TICKERS  = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
EXAMPLE_N_TRAIN  = 10       # small for quick iteration
EXAMPLE_N_VAL    = 3
EXAMPLE_EPOCHS   = 5
EXAMPLE_BATCH    = 16
EXAMPLE_WORKERS  = 0        # 0 = main-process loading (safer on Windows)


def main():
    parser = argparse.ArgumentParser(description="ProjectOwl end-to-end example")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--skip-populate", action="store_true",
                        help="Skip data population (re-use existing DB data)")
    args = parser.parse_args()

    # ── 0. Optional: dashboard ────────────────────────────────────────────
    if args.dashboard:
        from owl.orchestration.dashboard import run_dashboard
        t = threading.Thread(target=run_dashboard, kwargs={"debug": False},
                             daemon=True)
        t.start()
        logger.info("Dashboard → http://localhost:8050")

    # ── 1. Init database ─────────────────────────────────────────────────
    logger.info("Step 1 — Initialising database tables…")
    from owl.data.db import create_tables
    create_tables()

    # ── 2. Populate ──────────────────────────────────────────────────────
    if not args.skip_populate:
        logger.info("Step 2 — Populating training & validation cases…")
        from owl.data.query_engine import populate_database
        populate_database(
            n_train=EXAMPLE_N_TRAIN,
            n_val=EXAMPLE_N_VAL,
            tickers=EXAMPLE_TICKERS,
        )
    else:
        logger.info("Step 2 — Skipped (--skip-populate)")

    # ── 3. Build DataLoaders ─────────────────────────────────────────────
    logger.info("Step 3 — Building data loaders + preprocessing pipeline…")
    from owl.data.feeder import make_dataloaders
    from owl.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    train_loader, val_loader = make_dataloaders(
        pipeline=pipeline,
        batch_size=EXAMPLE_BATCH,
        num_workers=EXAMPLE_WORKERS,
    )
    in_features = train_loader.dataset.get_feature_dim()
    logger.info("Features per time-step: %d", in_features)
    logger.info("Training windows: %d  |  Validation windows: %d",
                len(train_loader.dataset), len(val_loader.dataset))

    # ── 4. Train ─────────────────────────────────────────────────────────
    logger.info("Step 4 — Training %s model for %d epochs…",
                args.model.upper(), EXAMPLE_EPOCHS)
    from owl.data.db import log_metric

    if args.model == "cnn":
        from owl.models.cnn_model import CNNTrainer
        trainer = CNNTrainer(in_features)
    else:
        from owl.models.transformer_model import TransformerTrainer
        trainer = TransformerTrainer(in_features)

    trainer.model_name = args.model

    def _cb(m):
        try:
            log_metric(args.model, m.get("epoch", 0), m.get("batch", 0),
                       "throughput_rows_per_sec",
                       m.get("throughput_rows_per_sec", 0), m.get("phase", "train"))
        except Exception:
            pass

    history = trainer.fit(
        train_loader, val_loader,
        epochs=EXAMPLE_EPOCHS,
        metrics_callback=_cb,
    )

    # ── 5. Reports ───────────────────────────────────────────────────────
    logger.info("Step 5 — Generating reports…")
    from owl.config import REPORT_DIR
    from owl.visualization.reports import (
        plot_category_examples,
        plot_feature_importance,
        plot_training_history,
    )

    report_dir = REPORT_DIR / args.model
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(history, title=f"{args.model.upper()} Training",
                          save_path=report_dir / "training_history.png")

    sample_batch, _ = next(iter(val_loader))
    plot_feature_importance(
        trainer.model, sample_batch,
        feature_names=pipeline.feature_columns,
        device=trainer.device,
        save_path=report_dir / "feature_importance.png",
    )
    plot_category_examples(val_loader.dataset,
                           save_path=report_dir / "category_examples.png")

    # ── 6. t-SNE ─────────────────────────────────────────────────────────
    logger.info("Step 6 — t-SNE visualisation…")
    import numpy as np
    from owl.models.tsne_viz import compute_tsne, plot_tsne_2d, plot_tsne_3d

    latents, labels = trainer.extract_latents(val_loader)
    if len(latents) > 0:
        emb2 = compute_tsne(latents, n_components=2)
        emb3 = compute_tsne(latents, n_components=3)
        plot_tsne_2d(emb2, labels, save_path=report_dir / "tsne_2d.png")
        plot_tsne_3d(emb3, labels, save_path=report_dir / "tsne_3d.png")
    else:
        logger.warning("No latent vectors — skipping t-SNE")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info("  Example run complete!  Reports in: %s", report_dir)
    logger.info("═══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
