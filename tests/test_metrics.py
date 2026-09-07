"""Tests for mikha.eval.metrics — hand-picked cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from mikha.eval.metrics import (
    auroc,
    binary_metrics_at_threshold,
    false_alert_rate_at_recall,
    pr_curve,
)


def test_perfect_ordering_gives_auroc_1() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([True, True, True, False, False])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_inverse_ordering_gives_auroc_0() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([True, True, True, False, False])
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_random_ordering_near_half() -> None:
    rng = np.random.default_rng(42)
    scores = rng.random(1000)
    labels = rng.random(1000) < 0.5
    assert 0.42 < auroc(scores, labels) < 0.58


def test_binary_metrics_perfect_split() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    m = binary_metrics_at_threshold(scores, labels, threshold=0.5)
    assert m.tp == 2
    assert m.fp == 0
    assert m.fn == 0
    assert m.tn == 2
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_binary_metrics_all_predicted_negative() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([True, True, False, False])
    m = binary_metrics_at_threshold(scores, labels, threshold=0.99)
    assert m.tp == 0
    assert m.fp == 0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_pr_curve_perfect() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    curve = pr_curve(scores, labels)
    # AUPRC for perfect ranking is 1.0
    assert curve.auprc == pytest.approx(1.0)
    # Recall reaches 1.0 at some point
    assert curve.recalls[-1] == pytest.approx(1.0)


def test_far_at_recall_perfect_separation() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    far, thresh = false_alert_rate_at_recall(scores, labels, target_recall=1.0)
    assert far == pytest.approx(0.0)
    assert thresh == pytest.approx(0.8)


def test_far_at_recall_worst_case() -> None:
    # Must alert on every negative to reach 100% recall.
    scores = np.array([0.1, 0.05, 0.9, 0.8])
    labels = np.array([True, True, False, False])
    far, _ = false_alert_rate_at_recall(scores, labels, target_recall=1.0)
    assert far == pytest.approx(1.0)


def test_far_at_partial_recall() -> None:
    # 4 positives with scores [0.9, 0.8, 0.5, 0.4], 2 negatives [0.7, 0.3].
    # Threshold @ 0.8 → recall = 2/4 = 0.5, FP = 0 → FAR = 0.
    # Threshold @ 0.5 → recall = 3/4 = 0.75, FP = 1 (the 0.7 negative) → FAR = 0.5.
    scores = np.array([0.9, 0.8, 0.5, 0.4, 0.7, 0.3])
    labels = np.array([True, True, True, True, False, False])
    far, _ = false_alert_rate_at_recall(scores, labels, target_recall=0.5)
    assert far == pytest.approx(0.0)
    far, _ = false_alert_rate_at_recall(scores, labels, target_recall=0.75)
    assert far == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.1])
def test_far_rejects_bad_target(bad: float) -> None:
    scores = np.array([0.5])
    labels = np.array([True])
    with pytest.raises(ValueError):
        false_alert_rate_at_recall(scores, labels, target_recall=bad)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        binary_metrics_at_threshold(np.array([0.5]), np.array([True, False]), 0.5)


def test_empty_rejected() -> None:
    with pytest.raises(ValueError):
        binary_metrics_at_threshold(np.array([]), np.array([]), 0.5)


def test_only_negatives_auroc_is_half() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([False, False, False])
    assert auroc(scores, labels) == 0.5
