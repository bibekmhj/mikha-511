"""Inference wrapper around a fine-tuned water classifier.

Loads a checkpoint saved by :mod:`mikha.train.train` and exposes both a
convenient ``.score(image_bgr) -> float`` method and a bare ``score_fn``
callable matching :data:`mikha.eval.run.ScoreFn` so the fine-tuned model
drops straight into ``scripts/run_eval.py``.

Design points:

* Preprocessing is byte-identical to
  :class:`mikha.train.dataset.FloodImageDataset`: resize with
  ``INTER_AREA``, BGR→RGB, ImageNet-normalize. Any drift here silently
  degrades M8 results, so the constants live in one place per module and
  are asserted identical in the classifier test.
* Torch import is lazy so the rest of ``mikha.ref`` remains importable
  without a model install.
* No batching — the eval harness calls ``score_fn`` one image at a time.
  If throughput matters later, wrap in a batching decorator; do not add a
  batch API to this class.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WaterClassifier:
    """Loads once, scores many times."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        image_size: int | None = None,
    ) -> None:
        import torch

        from mikha.train.model import ModelSpec, build_model

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint at {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        spec_dict = ckpt.get("spec", {})
        spec = ModelSpec(
            backbone=spec_dict.get("backbone", "mobilenet_v3_small"),
            pretrained=False,  # loading trained weights, don't refetch ImageNet
            num_classes=int(spec_dict.get("num_classes", 1)),
        )
        model = build_model(spec)
        model.load_state_dict(ckpt["model_state_dict"])

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        model.to(device).eval()

        self._model = model
        self._device = device
        self._image_size = int(image_size or ckpt.get("image_size", 224))
        self._spec = spec
        self._torch = torch

    # -- introspection --

    @property
    def device(self) -> str:
        return self._device

    @property
    def backbone(self) -> str:
        return self._spec.backbone

    @property
    def image_size(self) -> int:
        return self._image_size

    # -- inference --

    def _preprocess(self, image_bgr: np.ndarray) -> Any:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image; got shape {image_bgr.shape}")
        resized = cv2.resize(
            image_bgr, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return self._torch.from_numpy(chw).unsqueeze(0).to(self._device)

    def score(self, image_bgr: np.ndarray) -> float:
        """Return ``P(flood | image)`` in ``[0, 1]``."""
        x = self._preprocess(image_bgr)
        with self._torch.no_grad():
            logits = self._model(x).squeeze()
            prob = self._torch.sigmoid(logits).item()
        return float(prob)

    def as_score_fn(self) -> Callable[[np.ndarray], float]:
        """Bare callable matching ``mikha.eval.run.ScoreFn``."""
        return self.score
