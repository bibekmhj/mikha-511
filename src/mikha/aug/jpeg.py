"""Low-bitrate JPEG artefact degradation.

Round-trips the image through cv2 JPEG encode/decode at a low quality
setting so that block artefacts and colour banding appear the way they do
on the real 511 image pipelines.

Severity 0.0 → quality 35 (mild).
Severity 1.0 → quality 15 (heavy).
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import check_image, clip_severity


def apply_jpeg(image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply low-bitrate JPEG artefacts and return a new BGR uint8 image."""
    # seed is accepted for a uniform interface even though libjpeg is
    # deterministic; we simply ignore it here.
    del seed
    check_image(image)
    severity = clip_severity(severity)

    quality = int(round(35 - severity * 20))  # 35 → 15
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed for jpeg degradation")
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed for jpeg degradation")
    return decoded
