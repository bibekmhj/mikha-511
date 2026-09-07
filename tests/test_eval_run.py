"""End-to-end tests for the evaluation driver, using a lambda detector."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from mikha.eval import evaluate, write_all
from mikha.eval.corpus import CLEAN


def _write_image(path: Path, color: int, tag: int = 0) -> str:
    """Write a mostly-solid image with a tiny unique noise block.

    The mean brightness stays close to ``color`` (used by the tests'
    lambda detector), but the sha256 differs per row so the manifest
    validator accepts every row. JPEG quantizes away single-pixel
    deltas — a small deterministic 4x4 noise block survives it.
    """
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
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "source", "url", "sha256", "license", "attribution", "image_class", "notes"]
        )
        row_ids = []
        # 6 flood images (bright), 6 nonflood (dark)
        for idx, cls in enumerate(["flood"] * 6 + ["nonflood"] * 6, start=1):
            row_id = f"mb-{idx:06d}"
            color = 200 if cls == "flood" else 20
            sha = _write_image(images / f"{row_id}.jpg", color, tag=idx)
            writer.writerow(
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
                "splits": {"test": {"row_ids": row_ids}},
            }
        )
    )
    return manifest, splits, images


# ---------------------------------------------------------------------------


def test_evaluate_perfect_detector(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)

    # A "detector" that returns brightness — perfectly separates flood (200)
    # from nonflood (20) on clean images.
    def score_fn(img: np.ndarray) -> float:
        return float(img.mean()) / 255.0

    result = evaluate(
        score_fn,
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        which_splits=("test",),
        degradations=(CLEAN,),
        default_threshold=0.5,
    )
    r = result.per_degradation[CLEAN]
    assert r.n_samples == 12
    assert r.n_flood == 6
    assert r.n_nonflood == 6
    assert r.metrics_at_default.f1 == pytest.approx(1.0)
    assert r.auroc == pytest.approx(1.0)
    assert r.far_at_recall == pytest.approx(0.0)


def test_evaluate_random_detector(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    rng = np.random.default_rng(0)

    def score_fn(_img: np.ndarray) -> float:
        return float(rng.random())

    result = evaluate(
        score_fn,
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        which_splits=("test",),
        degradations=(CLEAN,),
    )
    r = result.per_degradation[CLEAN]
    # Random detector: AUROC should be near 0.5 (with 12 samples, wide tolerance)
    assert 0.2 <= r.auroc <= 0.8


def test_evaluate_load_error_is_reported(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    # Delete one image to trigger a load error
    (imgs / "mb-000001.jpg").unlink()

    load_errs: list = []

    def score_fn(_img: np.ndarray) -> float:
        return 0.5

    result = evaluate(
        lambda _img: 0.5,
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        which_splits=("test",),
        degradations=(CLEAN,),
        on_load_error=lambda sample, exc: load_errs.append(sample.row_id),
    )
    assert "mb-000001" in load_errs
    # The other 11 samples should have made it through
    assert result.per_degradation[CLEAN].n_samples == 11


def test_write_all_produces_table_and_json(tmp_path: Path) -> None:
    m, s, imgs = _bootstrap(tmp_path)
    result = evaluate(
        lambda img: float(img.mean()) / 255.0,
        manifest_path=m,
        splits_path=s,
        images_dir=imgs,
        which_splits=("test",),
        degradations=(CLEAN,),
    )
    out = tmp_path / "results"
    written = write_all(result, out, detector_name="mean-brightness")
    assert "table_md" in written
    assert "eval_json" in written
    table = (out / "table1.md").read_text()
    assert "mean-brightness" in table
    assert "clean" in table
    payload = json.loads((out / "eval.json").read_text())
    assert payload["detector"] == "mean-brightness"
    assert "clean" in payload["per_degradation"]
    assert payload["per_degradation"]["clean"]["n_samples"] == 12
