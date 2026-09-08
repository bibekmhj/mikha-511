"""Tests for the backbone builder — shape checks, unknown-backbone guard."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_mobilenet_forward_single_logit() -> None:
    import torch

    from mikha.train.model import ModelSpec, build_model

    spec = ModelSpec(backbone="mobilenet_v3_small", pretrained=False, num_classes=1)
    model = build_model(spec).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)


def test_resnet18_forward_single_logit() -> None:
    import torch

    from mikha.train.model import ModelSpec, build_model

    spec = ModelSpec(backbone="resnet18", pretrained=False, num_classes=1)
    model = build_model(spec).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)


def test_unknown_backbone_rejected() -> None:
    from mikha.train.model import ModelSpec, build_model

    with pytest.raises(ValueError, match="unknown backbone"):
        build_model(ModelSpec(backbone="mystery_net", pretrained=False))
