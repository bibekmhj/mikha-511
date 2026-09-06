"""Glare / lens-flare degradation.

Additive Gaussian blob at a seed-chosen position, biased toward the upper half
of the frame (glare from the sun or a low-angle headlight almost never
originates from below). Followed by a local saturation clip that mimics
sensor blow-out.
"""

from __future__ import annotations

import numpy as np

from .base import check_image, clip_severity, make_rng


def apply_glare(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a glare / lens-flare degradation and return a new BGR uint8 image."""
    check_image(image)
    severity = clip_severity(severity)
    rng = make_rng(seed)

    height, width = image.shape[:2]

    # Choose a flare centre biased toward the upper 60% of the frame.
    cx = int(rng.uniform(0.15 * width, 0.85 * width))
    cy = int(rng.uniform(0.05 * height, 0.60 * height))

    sigma = 0.10 * min(height, width) + severity * 0.25 * min(height, width)
    intensity = 120.0 + 135.0 * severity  # per-channel peak add

    ys, xs = np.mgrid[0:height, 0:width]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    gauss = np.exp(-dist_sq / (2.0 * sigma * sigma))

    # Warm off-white glare (slightly yellow) — [B, G, R].
    tint = np.array([0.85, 0.98, 1.00], dtype=np.float32)
    add = gauss[:, :, None].astype(np.float32) * intensity * tint[None, None, :]

    out = image.astype(np.float32) + add

    # Local saturation clip: everything above 250 gets pinned (blow-out).
    blown = out > 250.0
    out[blown] = 255.0

    return np.clip(out, 0.0, 255.0).astype(np.uint8)
