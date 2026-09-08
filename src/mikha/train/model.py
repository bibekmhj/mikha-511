"""Backbone selection for the water classifier.

We pick MobileNetV3-Small by default: ~2.5M parameters, ImageNet weights
ship with torchvision, trains on a laptop CPU in an hour on 397 images
and much faster on Apple MPS or CUDA. Swap through the ``backbone``
config knob; only the ``mobilenet_v3_small`` and ``resnet18`` heads are
wired up because two backbones is enough differentiation for v0.2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model description saved alongside the checkpoint."""

    backbone: str
    pretrained: bool
    num_classes: int = 1  # single logit, sigmoid at inference


def build_model(spec: ModelSpec):  # noqa: ANN202
    """Return a ``torch.nn.Module`` matching ``spec``.

    Raises :class:`ValueError` for unknown backbones. Torch and torchvision
    are imported lazily so the module can be introspected without them.
    """
    import torch.nn as nn
    from torchvision import models

    backbone = spec.backbone.lower()
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if spec.pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, spec.num_classes)
        return model
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if spec.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, spec.num_classes)
        return model
    raise ValueError(f"unknown backbone {spec.backbone!r}; supported: mobilenet_v3_small, resnet18")
