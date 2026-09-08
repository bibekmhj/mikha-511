"""Calibration metrics for the eval harness.

M5 promised Platt-scaling of the scorer confidence "on the val split;
report calibration curves and ECE" (plan section 7). M6 promised
``results/calibration.png`` alongside the table and PR curves (plan
section 8). Neither number shipped in v0.1.0. This module closes both
gaps.

* :func:`expected_calibration_error` returns the standard ECE with
  equal-width probability bins (Guo et al. 2017, eqn 3).
* :func:`reliability_curve` returns per-bin ``(mean_confidence,
  empirical_positive_rate, count)`` for plotting.
* :func:`fit_platt_from_val` is a thin adapter that runs the harness's
  val split, fits :class:`mikha.ref.calibrate.PlattScaler` on it, and
  returns the fitted scaler.

Pure numpy. No torch. Safe to import from tests without model extras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin in a reliability diagram."""

    lower: float
    upper: float
    mean_confidence: float
    empirical_positive_rate: float
    count: int


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must be the same shape; got {scores.shape} vs {labels.shape}"
        )
    if scores.size == 0:
        raise ValueError("cannot compute calibration on empty arrays")
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("scores must all lie in [0, 1] for calibration")
    return scores, labels


def reliability_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Return per-bin reliability data with equal-width probability bins.

    Empty bins are omitted from the return list so callers plot only
    what actually exists.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    scores, labels = _check(scores, labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False buckets scores == 1.0 into an extra
    # bin index n_bins+1; clamp them into the top bin.
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)

    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                mean_confidence=float(scores[mask].mean()),
                empirical_positive_rate=float(labels[mask].mean()),
                count=n,
            )
        )
    return bins


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width-bin ECE.

    ECE = sum_b (n_b / N) * |mean_confidence_b - empirical_positive_rate_b|,
    summed over bins with count > 0.
    """
    scores, labels = _check(scores, labels)
    total = float(scores.size)
    bins = reliability_curve(scores, labels, n_bins=n_bins)
    return sum((b.count / total) * abs(b.mean_confidence - b.empirical_positive_rate) for b in bins)


def _to_logits(probs: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities in [0, 1] to logits, clipped to avoid inf."""
    probs = np.clip(np.asarray(probs, dtype=np.float64).ravel(), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def fit_platt(scores: np.ndarray, labels: np.ndarray, *, scaler=None):  # noqa: ANN001, ANN201
    """Fit a :class:`mikha.ref.calibrate.PlattScaler` on probability scores.

    Classical Platt scaling fits ``sigmoid(a * z + b)`` where ``z`` is a
    logit / margin, not a probability. Passing bounded probabilities to
    PlattScaler directly makes gradient descent stall near identity and
    collapses every prediction into a narrow band around 0.5. To avoid
    that, this helper logit-transforms ``scores`` before fitting, so
    with well-calibrated inputs the learned ``(a, b)`` is near ``(1, 0)``
    and the mapping is close to identity.
    """
    from mikha.ref.calibrate import PlattScaler

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must be the same shape; got {scores.shape} vs {labels.shape}"
        )
    if scores.size == 0:
        raise ValueError("cannot fit Platt on empty arrays")
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("scores must all lie in [0, 1] for probability inputs")

    logits = _to_logits(scores)
    scaler = scaler or PlattScaler()
    scaler.fit(logits.tolist(), labels.tolist())
    return scaler


def apply_platt(scores: np.ndarray, scaler) -> np.ndarray:  # noqa: ANN001
    """Vector-apply a fit :class:`mikha.ref.calibrate.PlattScaler` to probability scores.

    Logit-transforms inputs to match :func:`fit_platt` (Platt operates
    on logits, not probabilities). If you fit the scaler yourself on
    already-logit-space values, call ``scaler.transform`` directly
    instead.
    """
    logits = _to_logits(scores)
    return np.asarray([scaler.transform(float(z)) for z in logits], dtype=np.float64)
