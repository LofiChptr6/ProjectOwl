#!/usr/bin/env python
"""
ProjectOwl — End-to-end example
=================================
Runs the full pipeline (training + reports + t-SNE). Requires the DB to be
populated first via ``python scripts/populate_data.py``.

  1. Check for existing case IDs in DB (exit if empty)
  2. Init tables if needed
  3. Train a CNN (few epochs)
  4. Generate reports & t-SNE plots
  5. Live dashboard (default): worker load vs training ingestion at http://localhost:8050

Usage::

    python scripts/populate_data.py       # run first to populate DB
    python run_example.py
    python run_example.py --no-dashboard # skip live monitoring
    python run_example.py --model transformer
"""

import argparse
import logging
import sys
import threading
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
)
logger = logging.getLogger("run_example")


# ── Tuneables for the example run ─────────────────────────────────────────
EXAMPLE_EPOCHS   = 1
EXAMPLE_BATCH    = 3
EXAMPLE_WORKERS  = 10        # 0 = main-process loading (safer on Windows)


def main():
    parser = argparse.ArgumentParser(description="ProjectOwl end-to-end example")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch live monitoring dashboard (default: True)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Do not launch the dashboard")
    args = parser.parse_args()
    use_dashboard = args.dashboard or not args.no_dashboard

    # ── 0. Live dashboard (worker load vs training ingestion) ─────────────
    if use_dashboard:
        from owl.config import DASHBOARD_PORT
        from owl.orchestration.dashboard import run_dashboard
        url = f"http://localhost:{DASHBOARD_PORT}"
        t = threading.Thread(target=run_dashboard, kwargs={"debug": False},
                             daemon=True)
        t.start()
        logger.info("Dashboard (worker load vs ingestion) → %s", url)
        # Give the server a moment to bind, then open browser
        def _open():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    # ── 1. Check DB has data; init tables if needed ───────────────────────
    from owl.data.db import clear_training_metrics, create_tables, get_case_ids
    from owl.config import TRAINING_TABLE, VALIDATION_TABLE

    create_tables()
    clear_training_metrics()  # fresh charts for this run only
    train_ids = get_case_ids(TRAINING_TABLE)
    val_ids = get_case_ids(VALIDATION_TABLE)
    if not train_ids or not val_ids:
        logger.error(
            "No case data in DB. Populate first: python scripts/populate_data.py"
        )
        sys.exit(1)
    logger.info("Step 1 — Found %d training + %d validation cases", len(train_ids), len(val_ids))

    # ── 2. Build DataLoaders ─────────────────────────────────────────────
    logger.info("Step 2 — Building data loaders + preprocessing pipeline…")
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
    logger.info("Step 3 — Training %s model for %d epochs…",
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
                       "train_loss", m.get("train_loss", 0), m.get("phase", "train"))
            log_metric(args.model, m.get("epoch", 0), m.get("batch", 0),
                       "throughput_rows_per_sec",
                       m.get("throughput_rows_per_sec", 0), m.get("phase", "train"))
            log_metric(args.model, m.get("epoch", 0), m.get("batch", 0),
                       "data_wait_sec", m.get("data_wait_sec", 0), m.get("phase", "train"))
            log_metric(args.model, m.get("epoch", 0), m.get("batch", 0),
                       "compute_sec", m.get("compute_sec", 0), m.get("phase", "train"))
        except Exception:
            pass

    history = trainer.fit(
        train_loader, val_loader,
        epochs=EXAMPLE_EPOCHS,
        metrics_callback=_cb,
    )

    # ── 5. Reports ───────────────────────────────────────────────────────
    logger.info("Step 4 — Generating reports…")
    from owl.config import REPORT_DIR
    from owl.visualization.reports import (
        plot_category_examples,
        plot_category_overlay,
        plot_feature_importance,
        plot_training_history,
    )

    report_dir = REPORT_DIR / args.model
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(history, title=f"{args.model.upper()} Training",
                          save_path=report_dir / "training_history.png")

    batch = next(iter(val_loader))
    sector_idx = batch[2] if len(batch) > 2 else None

    # Save model architecture summary for dashboard
    try:
        from torchinfo import summary
        x = batch[0][:1].to(trainer.device)
        sec = batch[2][:1].to(trainer.device).long() if len(batch) > 2 else None
        s = summary(
            trainer.model,
            input_data=(x,) if sec is None else (x, sec),
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
            verbose=0,
        )
        (report_dir / "model_arch.txt").write_text(str(s), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save model arch: %s", e)

    plot_feature_importance(
        trainer.model, batch[0],
        feature_names=pipeline.feature_columns,
        device=trainer.device,
        save_path=report_dir / "feature_importance.png",
        sector_idx=sector_idx,
    )
    plot_category_examples(val_loader.dataset,
                           save_path=report_dir / "category_examples.png")
    plot_category_overlay(val_loader.dataset,
                          save_path=report_dir / "category_overlay.png")

    # ── 6. Label Gantt (API-fetched, same time axis) ──────────────────────
    try:
        from owl.visualization.gantt import run_label_gantt
        if run_label_gantt(trainer, pipeline, n_symbols=20, days=14, save_path=report_dir / "label_gantt.png"):
            logger.info("Label Gantt → %s", report_dir / "label_gantt.png")
    except Exception as e:
        logger.warning("Skipping Gantt chart (API may be needed): %s", e)

    # ── 7. t-SNE (with trajectory lines) ──────────────────────────────────
    logger.info("Step 5 — t-SNE visualisation…")
    import numpy as np
    from owl.models.tsne_viz import compute_tsne, plot_tsne_2d, plot_tsne_3d

    latents, labels, symbols, timestamps = trainer.extract_latents_with_metadata(
        val_loader.dataset, max_samples=1500
    )
    if len(latents) > 0:
        emb2 = compute_tsne(latents, n_components=2)
        emb3 = compute_tsne(latents, n_components=3)
        plot_tsne_2d(emb2, labels, symbols=symbols, timestamps=timestamps,
                     save_path=report_dir / "tsne_2d.png")
        plot_tsne_3d(emb3, labels, symbols=symbols, timestamps=timestamps,
                     save_path=report_dir / "tsne_3d.png")
    else:
        logger.warning("No latent vectors — skipping t-SNE")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info("  Example run complete!  Reports in: %s", report_dir)
    logger.info("═══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
