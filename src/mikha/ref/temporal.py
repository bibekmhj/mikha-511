"""Temporal persistence gate for Mikha-Ref.

A ring buffer of the last N per-camera frame decisions. The gate emits
"persistent" iff the frame was classified as positive in at least K of the
last N observations. Includes hysteresis so a persistent event stays
raised for a small grace window even after a couple of below-threshold
frames — reflects the operational reality that one glare frame should not
retract a road-flooded alert.

Pure Python + collections.deque. No numpy, no cv2. Fully testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceState:
    """Snapshot of what the gate knows right now."""

    n_observations: int
    n_positive_in_window: int
    is_persistent: bool
    # A "raise" event fires the frame the gate first flips to persistent;
    # a "clear" event fires the frame it flips back. Consumers use these
    # to write one event row per state transition.
    raise_event: bool
    clear_event: bool


class PersistenceGate:
    """K-of-N ring-buffer gate with raise-fast / lower-slow hysteresis.

    Parameters
    ----------
    window      : N — how many recent frames the gate considers.
    threshold_k : K — the count required inside the window to be persistent.
    clear_grace : how many consecutive below-threshold frames are needed
                  before a raised gate clears. Default 3.
    """

    def __init__(self, window: int = 10, threshold_k: int = 6, clear_grace: int = 3) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if threshold_k <= 0 or threshold_k > window:
            raise ValueError("threshold_k must be in (0, window]")
        if clear_grace < 1:
            raise ValueError("clear_grace must be >= 1")

        self._window = int(window)
        self._threshold = int(threshold_k)
        self._clear_grace = int(clear_grace)
        self._history: deque[bool] = deque(maxlen=self._window)
        self._is_persistent = False
        self._below_streak = 0

    # ------------------------------------------------------------------

    def observe(self, positive: bool) -> PersistenceState:
        """Feed one frame's binary observation, return the gate's state."""
        self._history.append(bool(positive))
        n_pos = sum(self._history)
        was_persistent = self._is_persistent

        if not was_persistent:
            if n_pos >= self._threshold:
                self._is_persistent = True
                self._below_streak = 0
        else:
            if n_pos >= self._threshold:
                self._below_streak = 0
            else:
                self._below_streak += 1
                if self._below_streak >= self._clear_grace:
                    self._is_persistent = False
                    self._below_streak = 0

        raise_event = self._is_persistent and not was_persistent
        clear_event = was_persistent and not self._is_persistent

        return PersistenceState(
            n_observations=len(self._history),
            n_positive_in_window=n_pos,
            is_persistent=self._is_persistent,
            raise_event=raise_event,
            clear_event=clear_event,
        )

    # ------------------------------------------------------------------

    @property
    def is_persistent(self) -> bool:
        return self._is_persistent

    def reset(self) -> None:
        """Clear all history — used on camera-offline events."""
        self._history.clear()
        self._is_persistent = False
        self._below_streak = 0
