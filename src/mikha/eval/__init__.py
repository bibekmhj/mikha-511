"""Mikha-Eval: one-command evaluation harness.

Produces per-degradation-class precision, recall, F1, false-alert rate
at fixed recall, and PR curves. Detector-agnostic — pass any callable
that maps a BGR image to a float score in [0, 1].
"""

from __future__ import annotations

from .calibration import (
    ReliabilityBin,
    apply_platt,
    expected_calibration_error,
    fit_platt,
    reliability_curve,
)
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
    "ReliabilityBin",
    "ScoreFn",
    "apply_platt",
    "auroc",
    "binary_metrics_at_threshold",
    "build_samples",
    "evaluate",
    "expected_calibration_error",
    "false_alert_rate_at_recall",
    "fit_platt",
    "load_splits",
    "pr_curve",
    "reliability_curve",
    "sample_count",
    "write_all",
]
