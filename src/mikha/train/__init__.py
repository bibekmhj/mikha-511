"""Mikha-Train — binary water classifier fine-tune (M8 / v0.2).

We fine-tune a small ImageNet-pretrained backbone (MobileNetV3-Small by
default) to output ``P(flood | image)`` on the Mikha-Bench base set. Labels
are image-level ``flood``/``nonflood`` — the manifest carries no per-pixel
masks — so this is honestly a classification task, not segmentation. The
trained model plugs into :mod:`mikha.eval` through :class:`mikha.ref.
classifier.WaterClassifier` as a drop-in ``ScoreFn``.

Everything here depends on ``torch`` and ``torchvision``; import lazily so
the rest of the package (bench, eval, aug) stays torch-free.
"""

from __future__ import annotations

__all__ = ["FloodImageDataset", "TrainConfig", "build_model", "train"]


def __getattr__(name: str):  # noqa: ANN202
    """Lazy re-export so callers only pay the torch import when they need it."""
    if name in {"FloodImageDataset"}:
        from .dataset import FloodImageDataset

        return FloodImageDataset
    if name in {"TrainConfig", "train"}:
        from .train import TrainConfig, train

        return {"TrainConfig": TrainConfig, "train": train}[name]
    if name in {"build_model"}:
        from .model import build_model

        return build_model
    raise AttributeError(name)
