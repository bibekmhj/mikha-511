"""Segmentation detector wrapper for Mikha-Ref.

Wraps Ultralytics YOLOv8-seg so the rest of the reference implementation
sees a single tiny interface (:func:`detect`) and can be tested without
importing torch.

The v0.1 detector loads pretrained COCO weights. That is deliberate — the
plan defers fine-tuning on Mikha-Bench-train until baseline mIoU on
Mikha-Bench-clean is measured and found <0.6. On COCO weights the mask
returned by the detector is whatever COCO thinks is present (bus, car,
person, ...); the value of this module at M5 is proving the plumbing
end-to-end, not producing a real water detector.

Heavy imports (ultralytics, cv2) are done lazily inside the class so
``import mikha.ref.detect`` is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = "yolov8n-seg.pt"


@dataclass(frozen=True)
class DetectionResult:
    """Single-frame detection output consumed by the temporal gate.

    ``mask``            : bool ndarray (H, W). True = "something" per detector.
    ``mask_area_frac``  : fraction of pixels flagged True (0.0–1.0).
    ``confidence``      : max class confidence across detections (0.0–1.0);
                          0.0 if the frame has no detections.
    ``n_boxes``         : number of raw detections before mask union.
    """

    mask: Any  # np.ndarray[bool]
    mask_area_frac: float
    confidence: float
    n_boxes: int


class YoloDetector:
    """Thin wrapper over Ultralytics YOLOv8-seg.

    Constructing a detector loads the weights (auto-downloads on first use).
    Calling :meth:`detect` returns a :class:`DetectionResult` for one frame.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, imgsz: int = 640) -> None:
        # Import here to keep top-level import free of torch.
        from ultralytics import YOLO

        self._weights = str(weights)
        self._imgsz = int(imgsz)
        self._model = YOLO(self._weights)

    def detect(self, image: Any) -> DetectionResult:
        """Run one forward pass on a BGR HxWx3 uint8 image."""
        import numpy as np

        if image is None or getattr(image, "ndim", 0) != 3:
            raise ValueError("detect expects a BGR HxWx3 image")

        results = self._model.predict(image, imgsz=self._imgsz, verbose=False)
        result = results[0]
        h, w = image.shape[:2]

        # Union all instance masks into a single boolean mask.
        combined = np.zeros((h, w), dtype=bool)
        n_boxes = 0
        confidence = 0.0

        if result.masks is not None and result.boxes is not None:
            n_boxes = int(len(result.boxes))
            # ultralytics returns masks at the model's resized resolution; the
            # .data attribute is (N, h', w'); we resize back to original.
            import cv2  # local import

            masks = result.masks.data.cpu().numpy()  # (N, h', w') float 0/1
            for m in masks:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                combined |= m > 0.5

            if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                confs = result.boxes.conf.cpu().numpy()
                if confs.size:
                    confidence = float(confs.max())

        mask_area_frac = float(combined.mean()) if combined.size else 0.0
        return DetectionResult(
            mask=combined,
            mask_area_frac=mask_area_frac,
            confidence=confidence,
            n_boxes=n_boxes,
        )
