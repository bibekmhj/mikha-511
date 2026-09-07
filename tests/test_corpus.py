"""Tests for the evaluation corpus builder.

Uses tmp_path to write a tiny manifest + splits.json + image files, then
enumerates samples and asserts sample counts / order / label mapping.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from mikha.eval.corpus import CLEAN, DEGRADATIONS, build_samples, load_splits, sample_count


def _write_tiny_image(path: Path, tag: int = 0) -> str:
    """Write a small unique image so its sha256 is distinct per row.

    A per-tag deterministic noise block is stamped in the corner so
    JPEG encoding preserves the variation (single-pixel deltas get
    quantized away). PNG would be simpler but the manifest expects
    ``.jpg`` throughout.
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 1] = 128
    rng = np.random.default_rng(tag + 1)
    img[0:8, 0:8] = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(tmp_path: Path, n_flood: int = 3, n_nonflood: int = 2) -> tuple[Path, Path, Path]:
    """Write a synthetic manifest + splits + images for corpus testing."""
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"

    row_ids: list[tuple[str, str]] = []  # (row_id, image_class)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "source", "url", "sha256", "license", "attribution", "image_class", "notes"]
        )
        idx = 1
        for cls, n in (("flood", n_flood), ("nonflood", n_nonflood)):
            for _ in range(n):
                row_id = f"mb-{idx:06d}"
                p = images / f"{row_id}.jpg"
                sha = _write_tiny_image(p, tag=idx)
                writer.writerow(
                    [
                        row_id,
                        "test_source",
                        f"http://example/{row_id}.jpg",
                        sha,
                        "CC0-1.0",
                        "Test attribution",
                        cls,
                        "synthetic",
                    ]
                )
                row_ids.append((row_id, cls))
                idx += 1

    splits = tmp_path / "splits.json"
    all_ids = [rid for rid, _ in row_ids]
    doc = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "manifest_sha256": "unused-in-test",
            "seed": 0,
            "target_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "n_total": len(all_ids),
        },
        "splits": {
            "train": {"row_ids": all_ids[:3], "n_total": 3, "n_flood": 0, "n_nonflood": 0},
            "val": {"row_ids": all_ids[3:4], "n_total": 1, "n_flood": 0, "n_nonflood": 0},
            "test": {"row_ids": all_ids[4:], "n_total": 1, "n_flood": 0, "n_nonflood": 0},
        },
    }
    splits.write_text(json.dumps(doc))
    return manifest, splits, images


# ---------------------------------------------------------------------------


def test_load_splits_returns_sets(tmp_path: Path) -> None:
    _, splits_path, _ = _bootstrap(tmp_path)
    splits = load_splits(splits_path)
    assert set(splits) == {"train", "val", "test"}
    assert all(isinstance(v, set) for v in splits.values())


def test_sample_count_multiplies_by_degradations(tmp_path: Path) -> None:
    m, s, _ = _bootstrap(tmp_path, n_flood=3, n_nonflood=2)
    # 5 rows in total; train=3, val=1, test=1. Default = test only, 6 degradations
    # (clean + 5 aug) → 6 samples.
    n = sample_count(m, s, which_splits=("test",), degradations=DEGRADATIONS)
    assert n == 6
    # All 3 splits → 5 rows × 6 = 30.
    n = sample_count(m, s, which_splits=("train", "val", "test"), degradations=DEGRADATIONS)
    assert n == 30


def test_build_samples_yields_expected_counts(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    samples = list(build_samples(m, s, imgs, which_splits=("train",), degradations=DEGRADATIONS))
    # 3 train rows × 6 degradations = 18
    assert len(samples) == 18
    # Every degradation appears equal number of times
    from collections import Counter

    c = Counter(s.degradation for s in samples)
    for d in DEGRADATIONS:
        assert c[d] == 3


def test_build_samples_respects_limit_per_class(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path, n_flood=3, n_nonflood=2)
    # All rows are in the "train" split; cap flood to 1 and nonflood to 1 per degradation.
    samples = list(build_samples(m, s, imgs, which_splits=("train",), limit_per_class=1))
    # Split has row_ids all_ids[:3] = 3 flood; only 1 flood kept per degradation, 0 nonflood
    # (nonflood rows are at indices 3+). 1 × 6 = 6.
    assert len(samples) == 6


def test_sample_label_maps_from_class(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    samples = list(build_samples(m, s, imgs, which_splits=("train",), degradations=(CLEAN,)))
    for sample in samples:
        assert sample.label == (sample.image_class == "flood")


def test_sample_load_returns_bgr_ndarray(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    samples = list(build_samples(m, s, imgs, which_splits=("train",), degradations=(CLEAN,)))
    img = samples[0].load()
    assert img.shape == (32, 32, 3)


def test_sample_load_applies_degradation(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    samples = list(build_samples(m, s, imgs, which_splits=("train",), degradations=DEGRADATIONS))
    clean = next(x for x in samples if x.degradation == CLEAN).load()
    fogged = next(x for x in samples if x.degradation == "fog").load()
    assert not np.array_equal(clean, fogged), "degradation transform should alter the image"


def test_unknown_split_rejected(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        list(build_samples(m, s, imgs, which_splits=("nope",)))


def test_unknown_degradation_rejected(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="unknown degradation"):
        list(build_samples(m, s, imgs, which_splits=("train",), degradations=("hail",)))
