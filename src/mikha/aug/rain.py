"""Rain-on-lens degradation.

Approximates the effect of water droplets clinging to a traffic-camera housing:
localised, elongated blur patches with a faint bright highlight. Non-droplet
regions are left crisp.

Severity 0.0 → ~15 tiny droplets, occluding <2% of the frame.
Severity 1.0 → ~120 large droplets, occluding ~15% of the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import check_image, clip_severity, make_rng


def apply_rain(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a rain-on-lens degradation and return a new BGR uint8 image."""
    check_image(image)
    severity = clip_severity(severity)
    rng = make_rng(seed)

    height, width = image.shape[:2]
    n_drops = int(15 + severity * 105)  # 15 → 120

    # 1) Build a soft droplet-alpha mask by rasterising N ellipses.
    alpha = np.zeros((height, width), dtype=np.float32)
    min_radius = max(3, int(min(height, width) * (0.01 + 0.005 * severity)))
    max_radius = max(min_radius + 2, int(min(height, width) * (0.02 + 0.03 * severity)))

    for _ in range(n_drops):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(min_radius, max_radius + 1))
        ry = int(rng.integers(min_radius, max_radius + 1))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(alpha, (cx, cy), (rx, ry), angle, 0, 360, color=1.0, thickness=-1)

    # 2) Feather the mask so droplets blend rather than paste.
    blur_ksize = max(3, (min_radius // 2) * 2 + 1)
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 3) Blurred copy of the source represents the light bent through droplets.
    strong_blur_ksize = max(9, (min_radius * 2) | 1)  # ensure odd
    blurred = cv2.GaussianBlur(image, (strong_blur_ksize, strong_blur_ksize), 0)

    # 4) Alpha-composite blurred onto original where droplets live.
    a = alpha[:, :, None]
    composed = (image.astype(np.float32) * (1.0 - a) + blurred.astype(np.float32) * a).astype(
        np.float32
    )

    # 5) Slight bright highlight where the droplet is thickest.
    highlight_strength = 25.0 + 30.0 * severity
    composed = composed + (alpha[:, :, None] ** 2) * highlight_strength

    return np.clip(composed, 0.0, 255.0).astype(np.uint8)
