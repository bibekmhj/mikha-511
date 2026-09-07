"""Image-level classification metrics for Mikha-Ref evaluation.

Every function is detector-agnostic: it takes ``scores`` (float in [0, 1])
and ``labels`` (bool, True = flood) and returns numbers. Pure NumPy;
easy to unit-test.

Metrics reported per Mikha-Bench degradation class:

* precision / recall / F1 at a chosen operating threshold
* full precision-recall curve (for PR plots + AUPRC)
* false-alert rate at a fixed recall (the plan's headline metric)
* AUROC (bonus — free once PR is computed)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryMetrics:
    """Precision/recall/F1 at one operating threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_pos: int
    n_neg: int


def binary_metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> BinaryMetrics:
    """Precision / recall / F1 at a chosen score cutoff."""
    scores, labels = _validate(scores, labels)
    preds = scores >= threshold
    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    tn = int(np.sum(~preds & ~labels))
    fn = int(np.sum(~preds & labels))
    n_pos = int(np.sum(labels))
    n_neg = int(np.sum(~labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_pos=n_pos,
        n_neg=n_neg,
    )


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRCurve:
    """Precision-recall trace + area."""

    thresholds: np.ndarray  # (K,) monotonically decreasing
    precisions: np.ndarray  # (K,)
    recalls: np.ndarray  # (K,)
    auprc: float


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> PRCurve:
    """Compute the full precision-recall curve.

    Uses unique score thresholds ordered from high to low. AUPRC is the
    average-precision estimator (weighted by recall step), the same
    quantity scikit-learn calls ``average_precision_score``.
    """
    scores, labels = _validate(scores, labels)
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    y_sorted = labels[order]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return PRCurve(np.array([]), np.array([]), np.array([]), 0.0)

    tp = np.cumsum(y_sorted.astype(np.int64))
    fp = np.cumsum((~y_sorted).astype(np.int64))
    recalls = tp / n_pos
    precisions = tp / np.maximum(tp + fp, 1)

    # Prepend the (recall=0, precision=1) origin so AUPRC integrates correctly.
    recalls_full = np.concatenate([[0.0], recalls])
    precisions_full = np.concatenate([[1.0], precisions])
    # Average precision: sum over recall steps.
    auprc = float(np.sum(np.diff(recalls_full) * precisions_full[1:]))
    return PRCurve(
        thresholds=s_sorted,
        precisions=precisions,
        recalls=recalls,
        auprc=auprc,
    )


def false_alert_rate_at_recall(
    scores: np.ndarray, labels: np.ndarray, target_recall: float = 0.9
) -> tuple[float, float]:
    """Return (FAR, threshold) at the *smallest* threshold whose recall >= target.

    False-alert rate = FP / (FP + TN) — i.e. the fraction of negatives
    that get incorrectly alerted. The plan calls this the headline
    operational metric.

    If no threshold reaches the target recall (rare, only if there are
    no positives), returns (1.0, 0.0).
    """
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")
    scores, labels = _validate(scores, labels)
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    y_sorted = labels[order]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return (1.0 if n_pos == 0 else 0.0, 0.0)

    tp = np.cumsum(y_sorted.astype(np.int64))
    fp = np.cumsum((~y_sorted).astype(np.int64))
    recalls = tp / n_pos
    # Find the first index whose recall meets target.
    reached = np.where(recalls >= target_recall)[0]
    if len(reached) == 0:
        return (1.0, float(s_sorted[-1]))
    idx = int(reached[0])
    far = float(fp[idx]) / n_neg
    return (far, float(s_sorted[idx]))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney U formulation."""
    scores, labels = _validate(scores, labels)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape:
        raise ValueError(f"scores{scores.shape} and labels{labels.shape} shape mismatch")
    if scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    if len(scores) == 0:
        raise ValueError("empty inputs")
    return scores, labels
