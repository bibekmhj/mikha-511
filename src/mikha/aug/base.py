"""Shared helpers for Mikha-Aug transforms.

Every degradation transform in this package takes:

    image:    BGR uint8 ``np.ndarray`` of shape (H, W, 3)
    seed:     int — determinism knob; same seed + same severity → same output
    severity: float in [0.0, 1.0] — 0 is a no-op-ish nudge, 1 is a heavy stress case

and returns a BGR uint8 image of the same shape.

Transforms MUST NOT mutate their input.
"""

from __future__ import annotations

import numpy as np


def check_image(image: np.ndarray) -> None:
    """Validate that ``image`` is a BGR uint8 HxWx3 array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image shape must be (H, W, 3), got {image.shape}")


def clip_severity(severity: float) -> float:
    """Clamp severity to [0, 1] and validate it is a real number."""
    if not isinstance(severity, int | float):
        raise TypeError(f"severity must be numeric, got {type(severity).__name__}")
    return float(max(0.0, min(1.0, severity)))


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh, isolated numpy Generator seeded from ``seed``."""
    return np.random.default_rng(int(seed))
