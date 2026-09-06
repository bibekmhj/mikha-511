"""M1 PoC: fetch one traffic-camera snapshot, run a pretrained YOLOv8-seg
model, save an overlay PNG.

At M1 the segmentation model is COCO-pretrained YOLOv8n-seg — NOT a water
detector. The overlay shows whatever COCO classes happen to be present
(bus, car, truck, person). The point is to prove that the
ingestion → inference → overlay pipeline works end to end. A
domain-specific water model replaces it in M5.

Heavy imports (ultralytics, cv2) are done lazily inside functions so that
importing this module for tests does not pull torch into memory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Example FL511 CCTV snapshot URL, from https://fl511.com/cctv . Individual
# camera URLs drift over time — pass --url to point at a currently live one.
DEFAULT_URL = "https://cctvinfo.dot.state.fl.us/image_pull/00_MDX_SR874_000.jpg"

DEFAULT_WEIGHTS = "yolov8n-seg.pt"


@dataclass(frozen=True)
class DemoResult:
    """Return value for run_demo()."""

    source: str
    detections: int
    out_path: Path


def fetch_snapshot(url: str, timeout: float = 15.0) -> Any | None:
    """Fetch bytes at ``url`` and decode as a BGR image. Returns None on failure.

    Kept dependency-light: httpx + numpy + cv2 only. Any network or decode
    error returns None; callers decide whether to fall back.
    """
    import cv2
    import httpx
    import numpy as np

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[fetch] {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    buffer = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        print("[fetch] response was not a decodable image", file=sys.stderr)
    return image


def load_fallback_image() -> tuple[Any, str]:
    """Return the bundled ultralytics sample image and its path.

    Used when the network snapshot cannot be fetched and the caller opted in
    with ``--allow-fallback``. bus.jpg is a street scene, close enough to a
    traffic-camera frame to demonstrate the pipeline.
    """
    import cv2
    from ultralytics.utils import ASSETS

    path = Path(ASSETS) / "bus.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"bundled sample not readable at {path}")
    return image, str(path)


def run_demo(
    url: str = DEFAULT_URL,
    out: str | Path = "demo_out.png",
    weights: str = DEFAULT_WEIGHTS,
    allow_fallback: bool = False,
) -> DemoResult:
    """Fetch → infer → overlay → save. Returns a ``DemoResult`` on success.

    Raises ``RuntimeError`` if the snapshot cannot be fetched and
    ``allow_fallback`` is False.
    """
    import cv2
    from ultralytics import YOLO

    image = fetch_snapshot(url)
    source = url
    if image is None:
        if not allow_fallback:
            raise RuntimeError(
                "snapshot fetch failed; pass allow_fallback=True to use the bundled sample"
            )
        image, source = load_fallback_image()
        print(f"[fallback] using bundled sample: {source}")

    print(f"[model] loading {weights} (auto-downloads if missing)")
    model = YOLO(weights)
    results = model.predict(image, imgsz=640, verbose=False)
    result = results[0]
    overlay = result.plot()  # BGR ndarray with masks + boxes drawn

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), overlay):
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")

    detections = 0 if result.boxes is None else len(result.boxes)
    print(f"[done] source={source}  detections={detections}  overlay={out_path.resolve()}")
    return DemoResult(source=source, detections=detections, out_path=out_path.resolve())
