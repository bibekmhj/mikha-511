"""Confidence calibration for Mikha-Ref.

The plan calls for Platt scaling on the val split. Val labels are only
produced by Mikha-Bench (M6+), so at M5 this module ships the interface
and an identity fallback: :class:`PlattScaler` accepts a fit but until it
is fit, ``transform`` returns the input unchanged.

Pure Python + math. No numpy dependency for the identity path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationParams:
    """Learned Platt parameters ``p_calibrated = sigmoid(a * s + b)``."""

    a: float
    b: float


class PlattScaler:
    """Platt-scaling calibrator.

    Usage:
        scaler = PlattScaler()
        scaler.fit(scores, labels)   # (list[float], list[bool])
        p = scaler.transform(score)

    Before ``fit`` is called, ``transform`` returns the input score
    unchanged (identity). The fit uses a bounded Newton-style search on
    the log-loss with a small L2 penalty for numerical stability.
    """

    def __init__(self) -> None:
        self._params: CalibrationParams | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        scores: list[float],
        labels: list[bool],
        *,
        max_iter: int = 100,
        lr: float = 0.1,
        l2: float = 1e-3,
    ) -> CalibrationParams:
        """Fit Platt parameters via gradient descent on log-loss."""
        if len(scores) != len(labels):
            raise ValueError("scores and labels must be the same length")
        if not scores:
            raise ValueError("cannot fit on empty data")

        # Bishop's smoothed labels — protects against overconfident sigmoids.
        n_pos = sum(1 for y in labels if y)
        n_neg = len(labels) - n_pos
        y_pos = (n_pos + 1.0) / (n_pos + 2.0) if n_pos else 0.5
        y_neg = 1.0 / (n_neg + 2.0) if n_neg else 0.5

        a, b = 0.0, 0.0
        for _ in range(max_iter):
            grad_a = 2.0 * l2 * a
            grad_b = 2.0 * l2 * b
            for s, y in zip(scores, labels, strict=True):
                target = y_pos if y else y_neg
                z = a * s + b
                p = 1.0 / (1.0 + math.exp(-z)) if z > -50 else 0.0
                err = p - target
                grad_a += err * s
                grad_b += err
            n = len(scores)
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self._params = CalibrationParams(a=a, b=b)
        return self._params

    def transform(self, score: float) -> float:
        """Return the calibrated probability (identity if unfit)."""
        if self._params is None:
            return float(score)
        z = self._params.a * score + self._params.b
        # Numerically safe sigmoid.
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    # ------------------------------------------------------------------

    @property
    def is_fit(self) -> bool:
        return self._params is not None

    @property
    def params(self) -> CalibrationParams | None:
        return self._params
