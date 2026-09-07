"""Tests for the Platt-scaling calibrator."""

from __future__ import annotations

import pytest

from mikha.ref.calibrate import PlattScaler


def test_identity_before_fit() -> None:
    scaler = PlattScaler()
    assert not scaler.is_fit
    for x in (0.0, 0.3, 0.7, 1.0):
        assert scaler.transform(x) == pytest.approx(x)


def test_fit_returns_monotone_calibrator() -> None:
    # Separable-ish: positives cluster near high scores, negatives low.
    scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9]
    labels = [False] * 5 + [True] * 5
    scaler = PlattScaler()
    scaler.fit(scores, labels, max_iter=200, lr=0.5)
    assert scaler.is_fit
    prev = -1.0
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p = scaler.transform(s)
        assert 0.0 <= p <= 1.0
        assert p >= prev - 1e-6, "calibrator should be monotone non-decreasing in score"
        prev = p


def test_fit_moves_positives_up_negatives_down() -> None:
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [False, False, False, True, True, True]
    scaler = PlattScaler()
    scaler.fit(scores, labels, max_iter=300, lr=0.5)
    p_low = scaler.transform(0.15)
    p_high = scaler.transform(0.85)
    assert p_high > p_low


def test_fit_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        PlattScaler().fit([0.1, 0.2], [True])


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        PlattScaler().fit([], [])
