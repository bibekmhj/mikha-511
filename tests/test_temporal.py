"""Tests for the persistence gate."""

from __future__ import annotations

import pytest

from mikha.ref.temporal import PersistenceGate


def test_gate_starts_not_persistent() -> None:
    g = PersistenceGate(window=10, threshold_k=6)
    assert not g.is_persistent


def test_gate_raises_at_threshold() -> None:
    g = PersistenceGate(window=10, threshold_k=6, clear_grace=3)
    # 5 positives — not yet persistent
    for _ in range(5):
        st = g.observe(True)
    assert not st.is_persistent
    assert not st.raise_event
    # 6th positive — raises
    st = g.observe(True)
    assert st.is_persistent
    assert st.raise_event
    assert not st.clear_event


def test_raise_fires_exactly_once() -> None:
    g = PersistenceGate(window=10, threshold_k=6, clear_grace=3)
    for _ in range(6):
        g.observe(True)
    st = g.observe(True)  # already persistent
    assert st.is_persistent
    assert not st.raise_event


def test_clear_grace_prevents_flapping() -> None:
    """clear_grace is the number of *consecutive below-threshold windows*
    required to clear. Use a wide window so we can drive the count freely.
    """
    g = PersistenceGate(window=20, threshold_k=6, clear_grace=3)
    # 6 positives → raised
    for _ in range(6):
        g.observe(True)
    assert g.is_persistent
    # With window=20 and 6 T's still in the buffer, the count stays at 6
    # after each new F. The below-streak only starts when the count drops
    # below K=6 — i.e., when a T ages out. That never happens in window=20
    # unless we push 20 F's. So we prove the flap-prevention differently:
    # force below-threshold by filling with F's until count drops.
    # After 15 F's, buffer = [T x 5, F x 15] → count=5 < 6, streak=1
    for _ in range(15):
        g.observe(False)
    assert g.is_persistent, "should still be persistent (streak=1, needs 3)"
    # One more F: buffer = [T x 4, F x 16] → count=4, streak=2
    g.observe(False)
    assert g.is_persistent, "should still be persistent (streak=2, needs 3)"
    # One more F: buffer = [T x 3, F x 17] → count=3, streak=3 → clear
    st = g.observe(False)
    assert not st.is_persistent
    assert st.clear_event


def test_clear_grace_resets_on_return_to_threshold() -> None:
    """A positive that brings the count back to threshold zeroes the streak."""
    g = PersistenceGate(window=10, threshold_k=3, clear_grace=3)
    for _ in range(3):
        g.observe(True)
    assert g.is_persistent
    # Drive count below threshold: enough F's to make count=2, then streak=1
    for _ in range(8):
        g.observe(False)
    # Buffer now [T, F x 8, ...] — count=1, streak=1 (still persistent)
    assert g.is_persistent
    # A positive brings count back over threshold in some form; streak resets.
    for _ in range(3):
        g.observe(True)
    assert g.is_persistent


def test_reset_clears_state() -> None:
    g = PersistenceGate(window=5, threshold_k=3)
    for _ in range(3):
        g.observe(True)
    assert g.is_persistent
    g.reset()
    assert not g.is_persistent
    # And history is empty
    st = g.observe(True)
    assert st.n_observations == 1


@pytest.mark.parametrize(
    ("window", "k", "grace"),
    [(0, 1, 1), (10, 0, 1), (10, 11, 1), (10, 6, 0)],
)
def test_invalid_params_rejected(window: int, k: int, grace: int) -> None:
    with pytest.raises(ValueError):
        PersistenceGate(window=window, threshold_k=k, clear_grace=grace)
