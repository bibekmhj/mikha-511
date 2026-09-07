"""Mikha-Eval: one-command evaluation harness.

Produces per-degradation-class precision, recall, F1, false-alert rate
at fixed recall, and PR curves. Detector-agnostic — pass any callable
that maps a BGR image to a float score in [0, 1].
"""

from __future__ import annotations

from .corpus import CLEAN, DEGRADATIONS, EvalSample, build_samples, load_splits, sample_count
from .metrics import (
    BinaryMetrics,
    PRCurve,
    auroc,
    binary_metrics_at_threshold,
    false_alert_rate_at_recall,
    pr_curve,
)
from .report import write_all
from .run import DegradationResult, EvalResult, ScoreFn, evaluate

__all__ = [
    "CLEAN",
    "DEGRADATIONS",
    "BinaryMetrics",
    "DegradationResult",
    "EvalResult",
    "EvalSample",
    "PRCurve",
    "ScoreFn",
    "auroc",
    "binary_metrics_at_threshold",
    "build_samples",
    "evaluate",
    "false_alert_rate_at_recall",
    "load_splits",
    "pr_curve",
    "sample_count",
    "write_all",
]
