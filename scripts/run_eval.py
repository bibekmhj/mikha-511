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

from mikha.eval import (
    apply_platt,
    evaluate,
    expected_calibration_error,
    fit_platt,
    sample_count,
    write_all,
)
from mikha.eval.corpus import DEGRADATIONS
from mikha.eval.report import CalibrationBlock

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_SPLITS = REPO_ROOT / "bench" / "splits.json"
DEFAULT_IMAGES = REPO_ROOT / "data" / "images"
DEFAULT_OUT = REPO_ROOT / "results"


def _fit_calibration(cal_result, test_result) -> dict:  # noqa: ANN001
    """Fit Platt on cal_result per degradation, apply to test_result.

    Returns ``{degradation_name: CalibrationBlock}``. Degradations with
    fewer than 2 positives or 2 negatives in the calibration split are
    skipped (Platt cannot be fit on a degenerate class balance).
    """
    blocks: dict[str, CalibrationBlock] = {}
    for name, cr in cal_result.per_degradation.items():
        if name not in test_result.per_degradation:
            continue
        tr = test_result.per_degradation[name]

        n_pos = int(cr.labels.sum())
        n_neg = int(cr.labels.size - n_pos)
        if n_pos < 2 or n_neg < 2:
            print(
                f"[warn] skipping Platt for {name}: cal split has {n_pos} flood / {n_neg} nonflood",
                file=sys.stderr,
            )
            continue

        scaler = fit_platt(cr.scores, cr.labels)
        params = scaler.params
        cal_scores_test = apply_platt(tr.scores, scaler)

        ece_before = expected_calibration_error(tr.scores, tr.labels)
        ece_after = expected_calibration_error(cal_scores_test, tr.labels)

        blocks[name] = CalibrationBlock(
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            platt_a=float(params.a),
            platt_b=float(params.b),
            calibrated_scores=cal_scores_test,
        )
    return blocks


def _score_from_yolo():
    """Return a scoring callable that lazy-loads YOLOv8-seg."""
    from mikha.ref import YoloDetector

    detector = YoloDetector()

    def score(image) -> float:
        return float(detector.detect(image).mask_area_frac)

    return score, "yolov8n-seg (COCO weights, mask_area_frac score)"


def _score_from_classifier(weights_path: str, device: str):
    """Return a scoring callable that lazy-loads a fine-tuned water classifier."""
    from mikha.ref.classifier import WaterClassifier

    clf = WaterClassifier(weights_path, device=device)

    def score(image) -> float:
        return float(clf.score(image))

    name = (
        f"water-classifier ({clf.backbone}, ImageNet-pretrained, fine-tuned; "
        f"weights={weights_path})"
    )
    return score, name


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
    ap.add_argument(
        "--detector",
        choices=("yolo", "classifier"),
        default="yolo",
        help=(
            "'yolo' = YOLOv8-seg mask_area_frac (v0.1.0 baseline). "
            "'classifier' = fine-tuned water classifier from --weights (M8/v0.2)."
        ),
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="Checkpoint path for --detector classifier (e.g. models/finetune_v1/best.pt).",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device for --detector classifier: auto (default), cpu, cuda, mps.",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Fit Platt scaling on --calibrate-split (default 'val') per degradation, "
            "then apply to the eval splits and report ECE before + after. "
            "Adds results/calibration.png and ECE columns to table1.md and eval.json."
        ),
    )
    ap.add_argument(
        "--calibrate-split",
        default="val",
        help="Which split to fit Platt scaling on. Default: val.",
    )
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
        if args.detector == "classifier":
            if not args.weights:
                print(
                    "[fatal] --detector classifier requires --weights <checkpoint.pt>",
                    file=sys.stderr,
                )
                return 2
            score_fn, detector_name = _score_from_classifier(args.weights, args.device)
        else:
            score_fn, detector_name = _score_from_yolo()
    except ImportError as exc:
        print(f"[fatal] detector deps missing: {exc}", file=sys.stderr)
        print("        pip install -e '.[model]'", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
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

    calibration_blocks = None
    if args.calibrate:
        if args.calibrate_split in args.splits:
            print(
                f"[warn] --calibrate-split '{args.calibrate_split}' is also in --splits; "
                f"fitting Platt on the same rows we score is optimistic. Consider "
                f"--splits test --calibrate-split val instead.",
                file=sys.stderr,
            )
        print(f"[eval] fitting Platt scaling on split={args.calibrate_split}")
        cal_result = evaluate(
            score_fn,
            manifest_path=args.manifest,
            splits_path=args.splits_json,
            images_dir=args.images,
            which_splits=(args.calibrate_split,),
            degradations=args.degradations,
            severity=args.severity,
            seed=args.seed,
            limit_per_class=limit,
            default_threshold=args.threshold,
            target_recall=args.target_recall,
            on_load_error=_on_load_error,
        )
        calibration_blocks = _fit_calibration(cal_result, result)

    written = write_all(
        result, args.out, detector_name=detector_name, calibration=calibration_blocks
    )

    print("\n=== Results ===")
    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        line = (
            f"  {name:10s}  n={r.n_samples:4d}  P={m.precision:.3f}  "
            f"R={m.recall:.3f}  F1={m.f1:.3f}  AUROC={r.auroc:.3f}  "
            f"FAR@{args.target_recall:.0%}={r.far_at_recall:.3f}"
        )
        if calibration_blocks and name in calibration_blocks:
            cb = calibration_blocks[name]
            line += f"  ECE:{cb.ece_before:.3f}->{cb.ece_after:.3f}"
        print(line)
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
