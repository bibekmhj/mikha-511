"""Night / low-light degradation.

Approximates the look of a 511 traffic camera at night without IR
illumination: gamma darkening, a cool (blue) cast, and a low-level Poisson
noise floor. A separate ``apply_ir_night`` variant simulates a monochromatic
IR camera with a slight green tint, matching the look of cameras that switch
to IR mode after dusk.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import check_image, clip_severity, make_rng


def apply_night(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a night / low-light degradation and return a new BGR uint8 image."""
    check_image(image)
    severity = clip_severity(severity)
    rng = make_rng(seed)

    # 1) Gamma darkening. Severity 0 → gamma 1.5, severity 1 → gamma 4.5.
    gamma = 1.5 + 3.0 * severity
    normalised = image.astype(np.float32) / 255.0
    darkened = np.power(normalised, gamma)

    # 2) Cool colour cast: boost blue slightly, dampen red.
    cast = np.array([1.05, 1.00, 0.85], dtype=np.float32)  # BGR
    darkened *= cast[None, None, :]

    # 3) Poisson-ish noise floor (approximated with Gaussian to stay
    # deterministic without shape gymnastics).
    noise_std = 4.0 + 12.0 * severity
    noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32) / 255.0
    darkened = darkened + noise

    out = np.clip(darkened * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def apply_ir_night(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a simulated IR-mode night camera look (grayscale + green tint)."""
    check_image(image)
    severity = clip_severity(severity)
    rng = make_rng(seed)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # IR is typically brighter than a naive luma; boost slightly.
    grey = np.clip(grey * (1.0 + 0.2 * (1.0 - severity)), 0.0, 1.0)

    # Green-tinted monochrome: B and R attenuated.
    ir = np.stack([grey * 0.25, grey * 1.00, grey * 0.35], axis=-1)

    # A little noise; IR sensors are noisy.
    noise_std = 4.0 + 10.0 * severity
    noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32) / 255.0
    ir = ir + noise

    return np.clip(ir * 255.0, 0.0, 255.0).astype(np.uint8)
