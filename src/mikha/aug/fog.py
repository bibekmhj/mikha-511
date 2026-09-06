"""Fog / spray degradation.

Depth-independent white-veil model: alpha-blend the frame with a light-grey
ground plus a mild global blur. This matches the look of 511 cameras during
heavy fog, sea-spray or wind-driven mist far better than a true depth-aware
haze model, and it is trivially deterministic.

Severity 0.0 → 15% white veil, ksize 3.
Severity 1.0 → 75% white veil, ksize 15.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import check_image, clip_severity, make_rng


def apply_fog(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a fog / spray degradation and return a new BGR uint8 image."""
    check_image(image)
    severity = clip_severity(severity)
    rng = make_rng(seed)

    # Veil colour: light grey with a small, seeded per-image jitter so
    # different cameras/seeds are distinguishable.
    base_grey = 220 + int(rng.integers(-10, 11))
    veil_bgr = np.full_like(image, fill_value=base_grey, dtype=np.uint8)

    alpha = 0.15 + 0.60 * severity  # 0.15 → 0.75
    blended = cv2.addWeighted(image, 1.0 - alpha, veil_bgr, alpha, 0.0)

    # Mild blur to knock the edge sharpness down.
    ksize = max(3, int(3 + 12 * severity) | 1)  # odd
    blurred = cv2.GaussianBlur(blended, (ksize, ksize), 0)

    # Slight contrast reduction: pull everything toward the local mean.
    contrast_scale = 1.0 - 0.30 * severity  # 1.0 → 0.7
    mean = float(np.mean(blurred))
    out = (blurred.astype(np.float32) - mean) * contrast_scale + mean
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
