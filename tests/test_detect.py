"""Tests for the YOLOv8-seg detector wrapper.

Model-touching — imports ultralytics + torch. Skipped in the default CI
run (which does not install ``[model]``); runs locally when the model
extras are present.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from mikha.ref.detect import DEFAULT_WEIGHTS, DetectionResult, YoloDetector  # noqa: E402


def _street_image() -> np.ndarray:
    # Use the bundled ultralytics bus.jpg — the same asset our M1 demo used.
    import cv2
    from ultralytics.utils import ASSETS

    img = cv2.imread(f"{ASSETS}/bus.jpg")
    assert img is not None
    return img


def test_default_weights_string_shape() -> None:
    assert isinstance(DEFAULT_WEIGHTS, str)
    assert DEFAULT_WEIGHTS.endswith(".pt")


def test_detect_returns_result_on_real_image() -> None:
    det = YoloDetector()  # auto-download COCO weights
    img = _street_image()
    result = det.detect(img)
    assert isinstance(result, DetectionResult)
    assert result.mask.shape == img.shape[:2]
    assert 0.0 <= result.mask_area_frac <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    # bus.jpg has multiple COCO detections
    assert result.n_boxes > 0


def test_detect_rejects_bad_input() -> None:
    det = YoloDetector()
    with pytest.raises(ValueError):
        det.detect(None)
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10), dtype=np.uint8))  # 2D — not HxWx3
