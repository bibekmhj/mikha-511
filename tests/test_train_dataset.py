"""Tests for the FloodImageDataset (M8).

Uses a tiny synthetic manifest + splits + JPEG images. Guarded on torch —
skipped cleanly when the model extras aren't installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
import cv2  # noqa: E402

from mikha.train.dataset import FloodImageDataset, compute_pos_weight  # noqa: E402


def _write_image(path: Path, color: int, tag: int) -> str:
    img = np.full((32, 32, 3), color, dtype=np.uint8)
    rng = np.random.default_rng(tag + 1)
    img[0:4, 0:4] = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    row_ids: list[str] = []
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["id", "source", "url", "sha256", "license", "attribution", "image_class", "notes"]
        )
        for idx, cls in enumerate(["flood"] * 6 + ["nonflood"] * 6, start=1):
            row_id = f"mb-{idx:06d}"
            color = 200 if cls == "flood" else 20
            sha = _write_image(images / f"{row_id}.jpg", color, tag=idx)
            w.writerow(
                [
                    row_id,
                    "test_source",
                    f"http://example/{row_id}.jpg",
                    sha,
                    "CC0-1.0",
                    "Test",
                    cls,
                    "synthetic",
                ]
            )
            row_ids.append(row_id)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "meta": {"n_total": len(row_ids)},
                "splits": {
                    "train": {"row_ids": row_ids[:8]},
                    "val": {"row_ids": row_ids[8:10]},
                    "test": {"row_ids": row_ids[10:]},
                },
            }
        )
    )
    return manifest, splits, images


# ---------------------------------------------------------------------------


def test_dataset_length_and_labels(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    ds = FloodImageDataset(
        manifest_path=m, splits_path=s, images_dir=imgs, split="train", aug_prob=0.0
    )
    assert len(ds) == 8
    # first 6 are flood → 6 positives in train (split is row_ids[:8])
    n_neg, n_pos = ds.class_counts
    assert (n_neg, n_pos) == (2, 6)


def test_dataset_yields_normalized_tensor(tmp_path: Path) -> None:
    import torch

    m, s, imgs = _bootstrap(tmp_path)
    ds = FloodImageDataset(
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        split="train",
        image_size=64,
        aug_prob=0.0,
    )
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 64, 64)
    assert x.dtype == torch.float32
    # ImageNet-normalized values should lie roughly in [-3, 3]
    assert float(x.min()) > -3.5 and float(x.max()) < 3.5
    assert y.dtype == torch.float32 and y.item() in (0.0, 1.0)


def test_aug_deterministic_given_seed(tmp_path: Path) -> None:
    """Same (row, epoch, seed) → byte-identical tensor across two dataset instances."""
    m, s, imgs = _bootstrap(tmp_path)
    ds1 = FloodImageDataset(
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        split="train",
        aug_prob=1.0,  # force augmentation
        base_seed=42,
    )
    ds2 = FloodImageDataset(
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        split="train",
        aug_prob=1.0,
        base_seed=42,
    )
    ds1.set_epoch(0)
    ds2.set_epoch(0)
    x1, _ = ds1[0]
    x2, _ = ds2[0]
    assert (x1 - x2).abs().max().item() < 1e-6


def test_aug_off_deterministic(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    ds = FloodImageDataset(
        manifest_path=m, splits_path=s, images_dir=imgs, split="train", aug_prob=0.0
    )
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert (x1 - x2).abs().max().item() == 0.0


def test_epoch_advances_augmentation(tmp_path: Path) -> None:
    """Different epochs → different augmentation seeds → different tensors."""
    m, s, imgs = _bootstrap(tmp_path)
    ds = FloodImageDataset(
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        split="train",
        aug_prob=1.0,
        base_seed=42,
    )
    ds.set_epoch(0)
    x0, _ = ds[0]
    ds.set_epoch(1)
    x1, _ = ds[0]
    # Extremely unlikely to match after a fresh degradation with a different seed.
    assert (x0 - x1).abs().max().item() > 1e-4


def test_pos_weight_matches_class_balance(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    ds = FloodImageDataset(
        manifest_path=m, splits_path=s, images_dir=imgs, split="train", aug_prob=0.0
    )
    n_neg, n_pos = ds.class_counts
    assert compute_pos_weight(ds) == pytest.approx(n_neg / n_pos)


def test_unknown_split_rejected(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        FloodImageDataset(manifest_path=m, splits_path=s, images_dir=imgs, split="nope")


def test_unknown_aug_rejected(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="unknown aug"):
        FloodImageDataset(
            manifest_path=m,
            splits_path=s,
            images_dir=imgs,
            split="train",
            allowed_augs=("hail",),
        )
