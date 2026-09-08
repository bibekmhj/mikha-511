"""Fine-tune loop for the water classifier.

Design points:

* Fixed seeds — Python, NumPy, and torch (CPU + CUDA + MPS) — set once at
  the top so the whole run is reproducible.
* Cosine LR schedule with a short linear warmup; AdamW; ``BCEWithLogits``
  with ``pos_weight`` computed from the train split's class balance.
* Best-by-val-AUROC checkpointing. AUROC is the M6 baseline's headline
  metric; picking the checkpoint that maximizes it keeps the M8 result
  comparable across runs.
* Everything the training run produces (weights + metrics + effective
  config) lives together under ``out_dir`` so the run is self-describing
  after it finishes.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mikha.eval.metrics import auroc, binary_metrics_at_threshold, false_alert_rate_at_recall

from .dataset import FloodImageDataset, compute_pos_weight
from .model import ModelSpec, build_model


@dataclass(frozen=True)
class TrainConfig:
    """Every knob the training run reads. Serialized verbatim into the run dir."""

    # data
    manifest_path: str
    splits_path: str
    images_dir: str
    train_split: str = "train"
    val_split: str = "val"

    # model
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    image_size: int = 224

    # optimizer / schedule
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    # augmentation
    aug_prob: float = 0.5
    aug_severity: float = 0.65
    allowed_augs: tuple[str, ...] = ("rain", "fog", "night", "glare", "jpeg")

    # reproducibility / bookkeeping
    seed: int = 1234
    device: str = "auto"  # "auto" → cuda > mps > cpu
    out_dir: str = "models/finetune_v1"

    # eval
    target_recall: float = 0.9
    default_threshold: float = 0.5

    # runtime
    log_every: int = 25  # batches
    early_stop_patience: int = 5  # epochs without val AUROC improvement

    extras: dict[str, Any] = field(default_factory=dict)


def _pick_device(request: str) -> str:
    import torch

    request = request.lower()
    if request == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return request


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_all(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run model over a loader, return (scores, labels) as numpy arrays."""
    import torch

    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(float(p) for p in probs.ravel())
            labels.extend(float(t) for t in targets.numpy().ravel())
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def train(
    cfg: TrainConfig,
    *,
    on_epoch_end: Callable[[dict], None] | None = None,
) -> dict:
    """Run the full training loop, save artifacts under ``cfg.out_dir``.

    Returns a dict summarizing the run (best val AUROC + path to the best
    checkpoint). Callers get the same summary on disk in ``metrics.json``.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from torch.utils.data import DataLoader

    _seed_everything(cfg.seed)
    device = _pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- data --
    train_ds = FloodImageDataset(
        manifest_path=cfg.manifest_path,
        splits_path=cfg.splits_path,
        images_dir=cfg.images_dir,
        split=cfg.train_split,
        image_size=cfg.image_size,
        aug_prob=cfg.aug_prob,
        aug_severity=cfg.aug_severity,
        base_seed=cfg.seed,
        allowed_augs=cfg.allowed_augs,
    )
    val_ds = FloodImageDataset(
        manifest_path=cfg.manifest_path,
        splits_path=cfg.splits_path,
        images_dir=cfg.images_dir,
        split=cfg.val_split,
        image_size=cfg.image_size,
        aug_prob=0.0,  # val is clean by construction
        base_seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # -- model / optim --
    spec = ModelSpec(backbone=cfg.backbone, pretrained=cfg.pretrained, num_classes=1)
    model = build_model(spec).to(device)

    pos_weight = torch.tensor([compute_pos_weight(train_ds)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup = LinearLR(optim, start_factor=0.1, total_iters=max(1, cfg.warmup_epochs))
    cosine = CosineAnnealingLR(optim, T_max=max(1, cfg.epochs - cfg.warmup_epochs))
    scheduler = SequentialLR(optim, [warmup, cosine], milestones=[cfg.warmup_epochs])

    # -- persist effective config immediately so a crashed run still tells us what it tried --
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {**asdict(cfg), "device_resolved": device, "pos_weight": float(pos_weight.item())},
            indent=2,
        )
    )

    per_epoch: list[dict] = []
    best_auroc = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    ckpt_best = out_dir / "best.pt"

    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_n = 0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images).squeeze(-1)
            loss = loss_fn(logits, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * images.size(0)
            train_n += images.size(0)
            if cfg.log_every > 0 and step % cfg.log_every == 0:
                print(f"    ep{epoch:02d} step{step:04d} loss={loss.item():.4f}")
        scheduler.step()

        # -- validation --
        val_scores, val_labels = _score_all(model, val_loader, device)
        val_auroc = auroc(val_scores, val_labels)
        val_metrics = binary_metrics_at_threshold(
            val_scores, val_labels, threshold=cfg.default_threshold
        )
        val_far, val_far_thr = false_alert_rate_at_recall(
            val_scores, val_labels, target_recall=cfg.target_recall
        )
        rec = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_n),
            "val_auroc": float(val_auroc),
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_far_at_recall": float(val_far),
            "val_far_threshold": float(val_far_thr),
            "lr": scheduler.get_last_lr()[0],
            "wall_s": time.time() - t0,
        }
        per_epoch.append(rec)
        print(
            f"[ep {epoch:02d}] loss={rec['train_loss']:.4f}  "
            f"val AUROC={rec['val_auroc']:.3f}  F1={rec['val_f1']:.3f}  "
            f"FAR@{cfg.target_recall:.0%}={rec['val_far_at_recall']:.3f}  "
            f"({rec['wall_s']:.1f}s)"
        )

        if val_auroc > best_auroc + 1e-6:
            best_auroc = float(val_auroc)
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spec": asdict(spec),
                    "epoch": epoch,
                    "val_auroc": best_auroc,
                    "config": asdict(cfg),
                    "image_size": cfg.image_size,
                },
                ckpt_best,
            )
        else:
            epochs_since_improve += 1

        if on_epoch_end is not None:
            on_epoch_end(rec)

        if epochs_since_improve >= cfg.early_stop_patience:
            print(f"[eval] early stop — no val AUROC improvement for {epochs_since_improve} epochs")
            break

    summary = {
        "best_val_auroc": best_auroc,
        "best_epoch": best_epoch,
        "device": device,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_class_counts": train_ds.class_counts,  # (n_nonflood, n_flood)
        "val_class_counts": val_ds.class_counts,
        "checkpoint": str(ckpt_best),
        "per_epoch": per_epoch,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] best val AUROC={best_auroc:.3f} @ epoch {best_epoch}, checkpoint={ckpt_best}")
    return summary
