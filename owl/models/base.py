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
        for X, y in loader:
            X = X.to(self.device)
            z = self.model.encode(X)             # sub-classes implement this
            latents.append(z.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        return np.concatenate(latents), np.concatenate(labels)

    # ── internal ──────────────────────────────────────────────────────────

    def _train_one_epoch(self, loader, optimiser, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        t_batch = time.time()

        for batch_idx, (X, y) in enumerate(tqdm(
                loader, desc=f"Train {epoch}", leave=False)):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            logits = self.model(X)
            loss   = self.criterion(logits, y)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == y).sum().item()
            total      += bs

            # metrics callback
            if self._metrics_callback is not None:
                elapsed = time.time() - t_batch
                self._metrics_callback({
                    "epoch": epoch, "batch": batch_idx,
                    "train_loss": loss.item(),
                    "phase": "train",
                    "throughput_rows_per_sec": bs / max(elapsed, 1e-6),
                })
                t_batch = time.time()

            # DB logging (every 50 batches to avoid overhead)
            if batch_idx % 50 == 0:
                try:
                    log_metric(self.model_name, epoch, batch_idx,
                               "train_loss", loss.item(), "train")
                except Exception:
                    pass

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _validate(self, loader, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()
            logits = self.model(X)
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
