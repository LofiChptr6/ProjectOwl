#!/usr/bin/env python
"""Train autoencoder, cluster latent space, save example curves for the labeler GUI.

Usage::

    python scripts/train_clusters.py
    python scripts/train_clusters.py --epochs 20 --clusters 6
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLUSTER_OUTPUT_DIR,
    INPUT_WINDOW_MINUTES,
    NUM_CLUSTERS,
    NUM_WORKERS,
    VALIDATION_TABLE,
)
from owl.data.feeder import OwlDataset
from owl.models.autoencoder import PathAutoencoder
from owl.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train path autoencoder & cluster")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clusters", type=int, default=NUM_CLUSTERS)
    parser.add_argument("--examples-per-cluster", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ─────────────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline()
    dataset = OwlDataset(VALIDATION_TABLE, pipeline=pipeline)
    if len(dataset) == 0:
        logger.error("No data in %s. Populate first.", VALIDATION_TABLE)
        return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=min(NUM_WORKERS, 2), drop_last=True,
    )
    in_features = dataset.get_feature_dim()
    T = dataset[0][0].shape[0]

    # ── Model ────────────────────────────────────────────────────────────
    model = PathAutoencoder(in_features, t=T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ── Train autoencoder ────────────────────────────────────────────────
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n = 0
        for X, _ in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            X = X.to(device).float()
            x_recon, _ = model(X)
            loss = torch.nn.functional.mse_loss(x_recon, X)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
        logger.info("Epoch %d  recon_loss=%.6f", epoch, total_loss / n)

    # ── Extract latents for all samples (no shuffle, so idx = dataset idx) ─
    model.eval()
    extract_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    latents = []
    with torch.no_grad():
        for X, _ in extract_loader:
            z = model.encode(X.to(device).float())
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    # Cap for k-means if huge
    max_samples = min(len(latents), 50_000)
    if len(latents) > max_samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(latents), max_samples, replace=False)
        latents_sub = latents[sel]
    else:
        latents_sub = latents
        sel = np.arange(len(latents))

    # ── K-means ──────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    labels_sub = kmeans.fit_predict(latents_sub)
    if len(latents) > max_samples:
        from sklearn.metrics.pairwise import euclidean_distances
        full_labels = np.empty(len(latents), dtype=np.int64)
        full_labels[sel] = labels_sub
        rest = np.setdiff1d(np.arange(len(latents)), sel)
        dists = euclidean_distances(latents[rest], kmeans.cluster_centers_)
        full_labels[rest] = dists.argmin(axis=1)
    else:
        full_labels = labels_sub

    # ── Pick example curves per cluster ──────────────────────────────────
    examples_per = args.examples_per_cluster
    example_curves = []
    feature_names = pipeline.feature_columns

    for c in range(args.clusters):
        mask = full_labels == c
        if mask.sum() == 0:
            example_curves.append(np.zeros((examples_per, T, in_features), dtype=np.float32))
            continue
        # Pick samples closest to centroid
        centroids = kmeans.cluster_centers_
        dists = np.linalg.norm(latents - centroids[c], axis=1)
        cand = np.where(mask)[0]
        order = np.argsort(dists[cand])[:examples_per]
        chosen = cand[order]

        curves = []
        for i in chosen:
            x, _ = dataset[i]
            curves.append(x.numpy() if hasattr(x, "numpy") else x)
        while len(curves) < examples_per:
            curves.append(curves[-1].copy() if curves else np.zeros((T, in_features)))
        example_curves.append(np.stack(curves[:examples_per]))

    example_curves = np.stack(example_curves)

    # ── Save ─────────────────────────────────────────────────────────────
    CLUSTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "curves": example_curves,
        "feature_names": feature_names,
        "num_clusters": args.clusters,
        "cluster_assignments": full_labels,
    }
    torch.save(out, CLUSTER_OUTPUT_DIR / "cluster_examples.pt")
    torch.save(model.state_dict(), CLUSTER_OUTPUT_DIR / "autoencoder.pt")
    logger.info("Saved to %s", CLUSTER_OUTPUT_DIR)
    logger.info("Run: streamlit run scripts/run_cluster_labeler.py")


if __name__ == "__main__":
    main()
