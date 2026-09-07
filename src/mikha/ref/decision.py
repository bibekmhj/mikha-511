"""Decision engine for Mikha-Ref.

Combines detector output, temporal-persistence state, and calibrated
confidence into a single operator-facing label:

    passable          — no persistent detection
    flooded_uncertain — persistent but low calibrated confidence
    flooded           — persistent and high calibrated confidence
    camera_offline    — no snapshot returned this cycle

The engine is stateful (it owns the persistence gate + last-seen wall
clock for the offline test). One instance per camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .calibrate import PlattScaler
from .detect import DetectionResult
from .temporal import PersistenceGate, PersistenceState


class DecisionLabel(StrEnum):
    PASSABLE = "passable"
    FLOODED_UNCERTAIN = "flooded_uncertain"
    FLOODED = "flooded"
    CAMERA_OFFLINE = "camera_offline"


@dataclass(frozen=True)
class Decision:
    label: DecisionLabel
    calibrated_confidence: float
    persistence: PersistenceState | None  # None on camera_offline
    mask_area_frac: float  # 0.0 on camera_offline / no detection
    raise_event: bool
    clear_event: bool


class DecisionEngine:
    """Per-camera decision engine.

    Parameters
    ----------
    area_threshold : min fraction of the frame that must be masked to count
                     as a positive frame (fed to the persistence gate).
                     Default 0.02 (2%) — filters ultra-small false positives.
    high_conf      : calibrated confidence at/above which a persistent
                     detection is labelled ``flooded`` (else
                     ``flooded_uncertain``). Default 0.7.
    """

    def __init__(
        self,
        *,
        gate: PersistenceGate | None = None,
        calibrator: PlattScaler | None = None,
        area_threshold: float = 0.02,
        high_conf: float = 0.7,
    ) -> None:
        self._gate = gate or PersistenceGate()
        self._calibrator = calibrator or PlattScaler()
        self._area_threshold = float(area_threshold)
        self._high_conf = float(high_conf)

    # ------------------------------------------------------------------

    def observe_detection(self, det: DetectionResult) -> Decision:
        """Feed a detection from the current frame, return the engine's decision."""
        positive = det.mask_area_frac >= self._area_threshold
        state = self._gate.observe(positive)
        calibrated = self._calibrator.transform(det.confidence)

        if not state.is_persistent:
            label = DecisionLabel.PASSABLE
        elif calibrated >= self._high_conf:
            label = DecisionLabel.FLOODED
        else:
            label = DecisionLabel.FLOODED_UNCERTAIN

        return Decision(
            label=label,
            calibrated_confidence=calibrated,
            persistence=state,
            mask_area_frac=det.mask_area_frac,
            raise_event=state.raise_event,
            clear_event=state.clear_event,
        )

    def observe_offline(self) -> Decision:
        """The snapshot fetch failed / camera returned nothing this cycle."""
        was_persistent = self._gate.is_persistent
        self._gate.reset()
        return Decision(
            label=DecisionLabel.CAMERA_OFFLINE,
            calibrated_confidence=0.0,
            persistence=None,
            mask_area_frac=0.0,
            raise_event=False,
            clear_event=was_persistent,
        )
