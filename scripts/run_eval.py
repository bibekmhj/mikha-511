#!/usr/bin/env python3
"""One-command evaluation driver.

Reads the manifest + splits, applies Mikha-Aug, runs the M5 detector,
and writes results to ``results/``.

Usage:
    # Full test split, all 6 degradations (clean + 5 aug):
    python scripts/run_eval.py

    # Fast smoke run — 5 flood + 5 nonflood per degradation:
    python scripts/run_eval.py --limit-per-class 5

    # Evaluate on val instead of test:
    python scripts/run_eval.py --splits val

    # Custom operating point:
    python scripts/run_eval.py --threshold 0.10 --target-recall 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mikha.eval import evaluate, sample_count, write_all
from mikha.eval.corpus import DEGRADATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_SPLITS = REPO_ROOT / "bench" / "splits.json"
DEFAULT_IMAGES = REPO_ROOT / "data" / "images"
DEFAULT_OUT = REPO_ROOT / "results"


def _score_from_yolo():
    """Return a scoring callable that lazy-loads YOLOv8-seg."""
    from mikha.ref import YoloDetector

    detector = YoloDetector()

    def score(image) -> float:
        return float(detector.detect(image).mask_area_frac)

    return score, "yolov8n-seg (COCO weights, mask_area_frac score)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mikha-Bench baseline evaluation.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--splits", nargs="+", default=["test"], help="Split names to include.")
    ap.add_argument("--splits-json", default=str(DEFAULT_SPLITS))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--degradations",
        nargs="+",
        default=list(DEGRADATIONS),
        help=f"Subset of {DEGRADATIONS}. Default = all.",
    )
    ap.add_argument("--severity", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Cap flood + nonflood counts per (split, degradation). 0 = no cap.",
    )
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--target-recall", type=float, default=0.9)
    args = ap.parse_args(argv)

    limit = args.limit_per_class if args.limit_per_class > 0 else None

    n_expected = sample_count(
        args.manifest,
        args.splits_json,
        which_splits=args.splits,
        degradations=args.degradations,
    )
    print(
        f"[eval] manifest={args.manifest}  splits={args.splits}  degradations={args.degradations}"
    )
    print(f"[eval] planned samples (no cap): {n_expected}")
    if limit is not None:
        print(
            f"[eval] --limit-per-class {limit} → will actually run at most "
            f"{limit * 2 * len(args.splits) * len(args.degradations)} samples"
        )

    try:
        score_fn, detector_name = _score_from_yolo()
    except ImportError as exc:
        print(f"[fatal] detector deps missing: {exc}", file=sys.stderr)
        print("        pip install -e '.[model]'", file=sys.stderr)
        return 2

    print(f"[eval] detector: {detector_name}")

    load_errors: list[tuple[str, str]] = []

    def _on_load_error(sample, exc):  # noqa: ANN001, ANN202
        load_errors.append((sample.row_id, f"{type(exc).__name__}: {exc}"))

    def _progress(i, total, sample):  # noqa: ANN001, ANN202
        if i % 25 == 0 or i == total:
            print(f"    ─── {i}/{total}  last={sample.row_id} ({sample.degradation})")

    result = evaluate(
        score_fn,
        manifest_path=args.manifest,
        splits_path=args.splits_json,
        images_dir=args.images,
        which_splits=args.splits,
        degradations=args.degradations,
        severity=args.severity,
        seed=args.seed,
        limit_per_class=limit,
        default_threshold=args.threshold,
        target_recall=args.target_recall,
        on_load_error=_on_load_error,
        progress=_progress,
    )

    written = write_all(result, args.out, detector_name=detector_name)

    print("\n=== Results ===")
    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        print(
            f"  {name:10s}  n={r.n_samples:4d}  P={m.precision:.3f}  "
            f"R={m.recall:.3f}  F1={m.f1:.3f}  AUROC={r.auroc:.3f}  "
            f"FAR@{args.target_recall:.0%}={r.far_at_recall:.3f}"
        )
    if load_errors:
        print(f"\n[eval] {len(load_errors)} load errors (first 5):")
        for row_id, err in load_errors[:5]:
            print(f"    {row_id}  {err}")

    print("\n[eval] wrote:")
    for k, v in written.items():
        print(f"    {k:12s}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
