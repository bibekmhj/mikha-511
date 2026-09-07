"""Evaluation driver.

Ties the corpus (`mikha.eval.corpus`) to a detector (any callable
returning a float score per image) and produces a nested results dict
keyed by degradation class.

Detector agnosticism is deliberate — tests supply a lambda; the CLI
wraps :class:`mikha.ref.YoloDetector`. This lets the eval infrastructure
be tested without pulling torch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .corpus import DEGRADATIONS, EvalSample, build_samples
from .metrics import (
    BinaryMetrics,
    PRCurve,
    auroc,
    binary_metrics_at_threshold,
    false_alert_rate_at_recall,
    pr_curve,
)

# A "detector" for eval is any callable that maps a loaded BGR image to
# one float score in [0, 1]. In practice this is the detector's
# ``mask_area_frac`` (M5 default) or a fine-tuned water head's
# probability (post-fine-tune).
ScoreFn = Callable[[Any], float]


@dataclass(frozen=True)
class DegradationResult:
    """Per-degradation-class summary + retained raw scores for plotting."""

    degradation: str
    n_samples: int
    n_flood: int
    n_nonflood: int
    scores: np.ndarray  # (n,)
    labels: np.ndarray  # (n,) bool
    metrics_at_default: BinaryMetrics
    pr: PRCurve
    far_at_recall: float
    far_threshold: float
    auroc: float


@dataclass(frozen=True)
class EvalResult:
    """Top-level result envelope."""

    per_degradation: dict[str, DegradationResult]
    scores_flat: np.ndarray  # concatenated across degradations
    labels_flat: np.ndarray
    degradation_of_flat: np.ndarray  # matching per-sample degradation names
    default_threshold: float
    target_recall: float


def evaluate(
    score_fn: ScoreFn,
    *,
    manifest_path: str | Path,
    splits_path: str | Path,
    images_dir: str | Path,
    which_splits: Iterable[str] = ("test",),
    degradations: Iterable[str] = DEGRADATIONS,
    severity: float = 0.65,
    seed: int = 1234,
    limit_per_class: int | None = None,
    default_threshold: float = 0.05,
    target_recall: float = 0.9,
    on_load_error: Callable[[EvalSample, Exception], None] | None = None,
    progress: Callable[[int, int, EvalSample], None] | None = None,
) -> EvalResult:
    """Run ``score_fn`` over the full corpus and compute per-degradation metrics.

    ``default_threshold`` is a low-ish mask-area-fraction default. Callers
    can pick their own operating point later — the PR curve is retained
    for that.
    """
    per_deg_scores: dict[str, list[float]] = {d: [] for d in degradations}
    per_deg_labels: dict[str, list[bool]] = {d: [] for d in degradations}
    per_deg_flood: dict[str, int] = {d: 0 for d in degradations}
    per_deg_nonflood: dict[str, int] = {d: 0 for d in degradations}

    samples = list(
        build_samples(
            manifest_path,
            splits_path,
            images_dir,
            which_splits=which_splits,
            degradations=degradations,
            severity=severity,
            seed=seed,
            limit_per_class=limit_per_class,
        )
    )
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        try:
            image = sample.load()
        except Exception as exc:  # noqa: BLE001
            if on_load_error is not None:
                on_load_error(sample, exc)
            continue
        score = float(score_fn(image))
        per_deg_scores[sample.degradation].append(score)
        per_deg_labels[sample.degradation].append(sample.label)
        if sample.label:
            per_deg_flood[sample.degradation] += 1
        else:
            per_deg_nonflood[sample.degradation] += 1
        if progress is not None:
            progress(i, total, sample)

    per_degradation: dict[str, DegradationResult] = {}
    flat_scores: list[float] = []
    flat_labels: list[bool] = []
    flat_deg: list[str] = []

    for d in degradations:
        scores = np.asarray(per_deg_scores[d], dtype=np.float64)
        labels = np.asarray(per_deg_labels[d], dtype=bool)
        if scores.size == 0:
            continue
        m = binary_metrics_at_threshold(scores, labels, default_threshold)
        curve = pr_curve(scores, labels)
        far, far_thresh = false_alert_rate_at_recall(scores, labels, target_recall)
        auc = auroc(scores, labels)
        per_degradation[d] = DegradationResult(
            degradation=d,
            n_samples=int(len(scores)),
            n_flood=per_deg_flood[d],
            n_nonflood=per_deg_nonflood[d],
            scores=scores,
            labels=labels,
            metrics_at_default=m,
            pr=curve,
            far_at_recall=far,
            far_threshold=far_thresh,
            auroc=auc,
        )
        flat_scores.extend(scores.tolist())
        flat_labels.extend(labels.tolist())
        flat_deg.extend([d] * len(scores))

    return EvalResult(
        per_degradation=per_degradation,
        scores_flat=np.asarray(flat_scores, dtype=np.float64),
        labels_flat=np.asarray(flat_labels, dtype=bool),
        degradation_of_flat=np.asarray(flat_deg, dtype=object),
        default_threshold=default_threshold,
        target_recall=target_recall,
    )
