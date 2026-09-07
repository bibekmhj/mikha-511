"""Mikha-Ref: laptop-runnable reference implementation.

Water-segmentation backbone + temporal-persistence gate + calibrated
confidence + minimal FastAPI/HTMX dashboard for one camera.

Public surface (import from ``mikha.ref``):

* :class:`Decision`, :class:`DecisionEngine`, :class:`DecisionLabel`
* :class:`PersistenceGate`, :class:`PersistenceState`
* :class:`PlattScaler`, :class:`CalibrationParams`
* :class:`EventStore`, :class:`EventRow`
* :class:`SnapshotPoller`, :class:`Snapshot`
* :class:`DetectionResult` (interface); :class:`YoloDetector` (Ultralytics wrapper)
* :func:`create_app` (FastAPI factory)
"""

from __future__ import annotations

from .calibrate import CalibrationParams, PlattScaler
from .decision import Decision, DecisionEngine, DecisionLabel
from .detect import DEFAULT_WEIGHTS, DetectionResult, YoloDetector
from .poller import DEFAULT_HEADERS, Snapshot, SnapshotPoller
from .store import EventRow, EventStore
from .temporal import PersistenceGate, PersistenceState

__all__ = [
    "DEFAULT_HEADERS",
    "DEFAULT_WEIGHTS",
    "CalibrationParams",
    "Decision",
    "DecisionEngine",
    "DecisionLabel",
    "DetectionResult",
    "EventRow",
    "EventStore",
    "PersistenceGate",
    "PersistenceState",
    "PlattScaler",
    "Snapshot",
    "SnapshotPoller",
    "YoloDetector",
]


def create_app(store_path=None):  # noqa: ANN001, ANN202
    """Import ``mikha.ref.api.create_app`` lazily to avoid pulling FastAPI on import."""
    from .api import create_app as _create_app

    return _create_app(store_path)
