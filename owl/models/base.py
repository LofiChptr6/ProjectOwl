"""
Base trainer
=============
Shared training / validation loop, checkpoint logic, and metric logging
used by both the CNN and Transformer models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from owl.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    LEARNING_RATE,
    NUM_CATEGORIES,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from owl.data.db import log_metric

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BaseTrainer:
    """Provides ``fit()``, ``evaluate()``, ``save()`` / ``load()``.

    Sub-classes must set ``self.model`` (an ``nn.Module``) and optionally
    ``self.model_name`` before calling ``fit()``.
    """

    model: nn.Module
    model_name: str = "base"

    def __init__(self):
        self.device       = _device()
        self.criterion    = nn.CrossEntropyLoss()
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
        }
        self._metrics_callback = None        # optional external callback

    # ── public API ────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = NUM_EPOCHS,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        metrics_callback=None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Parameters
        ----------
        metrics_callback : optional callable ``(metric_dict) -> None``
            called after every batch with a dict containing keys like
            ``epoch, batch, train_loss, phase, throughput_rows_per_sec``.
        """
        self._metrics_callback = metrics_callback
        self.model.to(self.device)
        optimiser = torch.optim.AdamW(self.model.parameters(),
                                       lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(
                train_loader, optimiser, epoch)
            val_loss, val_acc = self._validate(val_loader, epoch)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "train_acc=%.3f  val_acc=%.3f  (%.1fs)",
                epoch, epochs, train_loss, val_loss,
                train_acc, val_acc, elapsed,
            )

            # checkpoint best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(tag="best")

        self.save(tag="last")
        return self.history

    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return ``(loss, accuracy)`` on *loader*."""
        return self._validate(loader, epoch=0)

    # ── checkpoint ────────────────────────────────────────────────────────

    def save(self, tag: str = "last") -> Path:
        path = CHECKPOINT_DIR / f"{self.model_name}_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "history":     self.history,
        }, path)
        logger.info("Saved checkpoint → %s", path)
        return path

    def load(self, tag: str = "best") -> None:
        path = CHECKPOINT_DIR / f"{self.model_name}_{tag}.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        logger.info("Loaded checkpoint ← %s", path)

    # ── latent extraction (for t-SNE) ─────────────────────────────────────

    @torch.no_grad()
    def extract_latents(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(latents, labels)`` arrays by running the encoder only."""
        self.model.eval()
        self.model.to(self.device)
        latents, labels = [], []
        for batch in loader:
            X, y = batch[0], batch[1]
            X = X.to(self.device)
            z = self.model.encode(X)             # sector not used for latent
            latents.append(z.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        return np.concatenate(latents), np.concatenate(labels)

    @torch.no_grad()
    def extract_latents_with_metadata(
        self, dataset, max_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str], list]:
        """Return (latents, labels, symbols, timestamps) for trajectory plotting.

        Requires dataset with get_sample_for_plot(idx).
        """
        self.model.eval()
        self.model.to(self.device)
        if not hasattr(dataset, "get_sample_for_plot"):
            raise ValueError("Dataset must have get_sample_for_plot for trajectory metadata")
        n = len(dataset)
        if max_samples is not None:
            n = min(n, max_samples)
        latents, labels, symbols, timestamps = [], [], [], []
        for idx in range(n):
            sample = dataset[idx]
            X = sample[0]
            if isinstance(X, torch.Tensor):
                X = X.unsqueeze(0).to(self.device)
            else:
                X = torch.from_numpy(np.asarray(X)).float().unsqueeze(0).to(self.device)
            z = self.model.encode(X)
            latent = z.cpu().numpy()[0]
            lab = sample[1]
            lab_val = lab.item() if hasattr(lab, "item") else int(lab)
            _, _, symbol, start_ts, _, _ = dataset.get_sample_for_plot(idx)
            latents.append(latent)
            labels.append(lab_val)
            symbols.append(str(symbol) if symbol else "")
            timestamps.append(start_ts)
        return (
            np.array(latents),
            np.array(labels),
            symbols,
            timestamps,
        )

    # ── internal ──────────────────────────────────────────────────────────

    def _train_one_epoch(self, loader, optimiser, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        t_after_prev = time.time()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train {epoch}", leave=False)):
            t_iter_start = time.time()
            data_wait_sec = t_iter_start - t_after_prev

            X, y = batch[0], batch[1]
            sector_idx = batch[2].to(self.device).long() if len(batch) > 2 else None
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            logits = self.model(X, sector_idx=sector_idx)
            loss   = self.criterion(logits, y)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == y).sum().item()
            total      += bs

            compute_sec = time.time() - t_iter_start
            t_after_prev = time.time()

            # metrics callback (data_wait_sec = worker fetch delay; compute_sec = GPU time)
            if self._metrics_callback is not None:
                self._metrics_callback({
                    "epoch": epoch, "batch": batch_idx,
                    "train_loss": loss.item(),
                    "phase": "train",
                    "throughput_rows_per_sec": bs / max(compute_sec, 1e-6),
                    "data_wait_sec": data_wait_sec,
                    "compute_sec": compute_sec,
                })

            # DB logging (every 50 batches to avoid overhead)
            if batch_idx % 50 == 0:
                try:
                    log_metric(self.model_name, epoch, batch_idx,
                               "train_loss", loss.item(), "train")
                    log_metric(self.model_name, epoch, batch_idx,
                               "data_wait_sec", data_wait_sec, "train")
                    log_metric(self.model_name, epoch, batch_idx,
                               "compute_sec", compute_sec, "train")
                except Exception:
                    pass

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _validate(self, loader, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            X, y = batch[0], batch[1]
            sector_idx = batch[2].to(self.device).long() if len(batch) > 2 else None
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()
            logits = self.model(X, sector_idx=sector_idx)
            loss   = self.criterion(logits, y)
            bs = X.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == y).sum().item()
            total      += bs

        avg_loss = total_loss / max(total, 1)
        avg_acc  = correct / max(total, 1)
        try:
            log_metric(self.model_name, epoch, 0,
                       "val_loss", avg_loss, "validate")
            log_metric(self.model_name, epoch, 0,
                       "val_acc", avg_acc, "validate")
        except Exception:
            pass
        return avg_loss, avg_acc
