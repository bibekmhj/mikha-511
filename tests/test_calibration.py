"""Tests for mikha.eval.calibration (M9).

Hand-picked cases with known answers. Pure numpy, no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from mikha.eval.calibration import (
    apply_platt,
    expected_calibration_error,
    fit_platt,
    reliability_curve,
)
from mikha.ref.calibrate import PlattScaler

# ---------------------------------------------------------------------------
# ECE hand-picked cases


def test_perfectly_calibrated_score_gives_zero_ece() -> None:
    """When every score matches its empirical positive rate, ECE = 0."""
    # 100 samples at score 0.7 with 70% positives -> perfectly calibrated.
    scores = np.full(100, 0.7)
    labels = np.array([True] * 70 + [False] * 30)
    assert expected_calibration_error(scores, labels) == pytest.approx(0.0, abs=1e-9)


def test_all_confident_and_all_wrong_gives_maximum_ece() -> None:
    """Score 0.99, all negatives -> |0.99 - 0.0| = 0.99."""
    scores = np.full(50, 0.99)
    labels = np.zeros(50, dtype=bool)
    assert expected_calibration_error(scores, labels) == pytest.approx(0.99, abs=1e-9)


def test_all_confident_and_all_right_gives_low_ece() -> None:
    """Score 0.99, all positives -> |0.99 - 1.0| = 0.01."""
    scores = np.full(50, 0.99)
    labels = np.ones(50, dtype=bool)
    assert expected_calibration_error(scores, labels) == pytest.approx(0.01, abs=1e-9)


def test_two_bins_averaged_by_count() -> None:
    """Two bins with different counts: ECE weighted by n_b / N."""
    # Bin 0.0-0.1: 4 samples at 0.05, 0 positives  -> gap 0.05, weight 4/10 = 0.4
    # Bin 0.9-1.0: 6 samples at 0.95, 6 positives -> gap 0.05, weight 6/10 = 0.6
    # ECE = 0.4 * 0.05 + 0.6 * 0.05 = 0.05
    scores = np.concatenate([np.full(4, 0.05), np.full(6, 0.95)])
    labels = np.array([False] * 4 + [True] * 6)
    assert expected_calibration_error(scores, labels, n_bins=10) == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# reliability_curve shape and content


def test_reliability_curve_skips_empty_bins() -> None:
    scores = np.array([0.05, 0.95])
    labels = np.array([False, True])
    bins = reliability_curve(scores, labels, n_bins=10)
    # Only the first and last bin have any content.
    assert len(bins) == 2
    assert bins[0].count == 1
    assert bins[0].empirical_positive_rate == 0.0
    assert bins[-1].count == 1
    assert bins[-1].empirical_positive_rate == 1.0


def test_reliability_curve_covers_score_one() -> None:
    """A score of exactly 1.0 must not fall outside the top bin."""
    scores = np.array([1.0, 0.0])
    labels = np.array([True, False])
    bins = reliability_curve(scores, labels, n_bins=5)
    # 2 non-empty bins expected.
    assert sum(b.count for b in bins) == 2


# ---------------------------------------------------------------------------
# Input validation


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([True, False]))


def test_empty_rejected() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([]), np.array([]))


def test_scores_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.5]), np.array([True]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([-0.1]), np.array([True]))


def test_reliability_curve_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError):
        reliability_curve(np.array([0.5]), np.array([True]), n_bins=1)


# ---------------------------------------------------------------------------
# apply_platt integration with PlattScaler


def test_apply_platt_transforms_via_logits() -> None:
    """apply_platt logit-transforms probs first, matching fit_platt's convention."""
    import math

    scaler = PlattScaler()
    # Fit directly in logit space so we know what the scaler's params are.
    logit_scores = [-3.0, -1.0, 0.0, 1.0, 3.0]
    labels = [False, False, False, True, True]
    scaler.fit(logit_scores, labels)

    probs = np.array([1.0 / (1.0 + math.exp(-z)) for z in logit_scores])
    vec = apply_platt(probs, scaler)
    per_sample = np.asarray([scaler.transform(z) for z in logit_scores])
    assert np.allclose(vec, per_sample, atol=1e-6)


def test_apply_platt_preserves_ranking() -> None:
    """Platt is monotone in its input, so AUROC-style rankings are preserved."""
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = rng.random(200) < scores

    scaler = fit_platt(scores, labels)
    calibrated = apply_platt(scores, scaler)

    # Order of samples by score must match order by calibrated score.
    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))


def test_fit_platt_preserves_calibration_of_well_calibrated_scores() -> None:
    """Regression test for the M9 Platt-on-probabilities bug.

    A perfectly calibrated set of probability scores should stay
    approximately calibrated after fit_platt + apply_platt. If Platt is
    fit directly on probabilities instead of logits, the learned map
    squashes everything into a narrow band around 0.5 and blows up ECE.
    That was the shipped M9 behavior; this test guards against a
    regression.
    """
    rng = np.random.default_rng(0)
    n = 800
    # Well-spread probabilities in [0.05, 0.95], labels drawn from them.
    probs = rng.uniform(0.05, 0.95, size=n)
    labels = rng.random(n) < probs

    ece_raw = expected_calibration_error(probs, labels)

    scaler = fit_platt(probs, labels)
    calibrated = apply_platt(probs, scaler)
    ece_after = expected_calibration_error(calibrated, labels)

    # The buggy path produced ece_after ~ 0.25-0.30 on this kind of input
    # (probs squashed into [0.45, 0.55]). The correct path should leave
    # ECE roughly where it started, or improve it.
    assert ece_after < ece_raw + 0.05, (
        f"Platt made ECE meaningfully worse (raw={ece_raw:.3f}, after={ece_after:.3f}); "
        "the shipped M9 bug applied Platt to probabilities instead of logits."
    )
    # And the calibrated scores must actually span [0, 1], not collapse to ~0.5.
    assert calibrated.max() - calibrated.min() > 0.5, (
        f"calibrated scores span only {calibrated.max() - calibrated.min():.3f}; "
        "Platt is squashing to a narrow band (the M9 bug)."
    )


def test_fit_platt_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValueError):
        fit_platt(np.array([1.5]), np.array([True]))
    with pytest.raises(ValueError):
        fit_platt(np.array([-0.1]), np.array([True]))


def test_fit_platt_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_platt(np.array([0.5, 0.5]), np.array([True]))
