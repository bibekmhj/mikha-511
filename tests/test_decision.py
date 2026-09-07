"""Tests for the decision engine."""

from __future__ import annotations

from mikha.ref.decision import DecisionEngine, DecisionLabel
from mikha.ref.detect import DetectionResult
from mikha.ref.temporal import PersistenceGate


def _det(area: float, conf: float = 0.8) -> DetectionResult:
    return DetectionResult(mask=None, mask_area_frac=area, confidence=conf, n_boxes=1)


def test_below_area_threshold_is_passable() -> None:
    engine = DecisionEngine(area_threshold=0.05)
    for _ in range(20):
        d = engine.observe_detection(_det(0.01))
        assert d.label is DecisionLabel.PASSABLE


def test_persistent_high_confidence_is_flooded() -> None:
    engine = DecisionEngine(
        gate=PersistenceGate(window=5, threshold_k=3),
        area_threshold=0.02,
        high_conf=0.5,
    )
    labels = []
    for _ in range(4):
        labels.append(engine.observe_detection(_det(0.1, conf=0.9)).label)
    assert DecisionLabel.FLOODED in labels


def test_persistent_low_confidence_is_uncertain() -> None:
    engine = DecisionEngine(
        gate=PersistenceGate(window=5, threshold_k=3),
        area_threshold=0.02,
        high_conf=0.9,
    )
    labels = []
    for _ in range(4):
        labels.append(engine.observe_detection(_det(0.1, conf=0.3)).label)
    assert DecisionLabel.FLOODED_UNCERTAIN in labels
    assert DecisionLabel.FLOODED not in labels


def test_raise_event_fires_at_transition() -> None:
    engine = DecisionEngine(
        gate=PersistenceGate(window=5, threshold_k=3, clear_grace=3),
        area_threshold=0.02,
    )
    raised = 0
    for _ in range(6):
        d = engine.observe_detection(_det(0.1, conf=0.8))
        if d.raise_event:
            raised += 1
    assert raised == 1, "raise event should fire exactly once"


def test_offline_clears_and_emits_camera_offline() -> None:
    engine = DecisionEngine(
        gate=PersistenceGate(window=5, threshold_k=3, clear_grace=3),
        area_threshold=0.02,
    )
    for _ in range(3):
        engine.observe_detection(_det(0.1, conf=0.8))
    d = engine.observe_offline()
    assert d.label is DecisionLabel.CAMERA_OFFLINE
    assert d.clear_event  # was persistent → clearing counts as a transition
    assert d.persistence is None
