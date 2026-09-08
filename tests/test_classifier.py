"""Tests for WaterClassifier — checkpoint round-trip + score_fn shape.

Guarded on torch + torchvision. The classifier's preprocessing constants
must stay identical to the training dataset's, so we test that too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_preprocessing_constants_match_dataset() -> None:
    """Same ImageNet mean/std used at train and inference — no silent drift."""
    from mikha.ref import classifier as clf_mod
    from mikha.train import dataset as ds_mod

    assert np.allclose(clf_mod._IMAGENET_MEAN, ds_mod._IMAGENET_MEAN)
    assert np.allclose(clf_mod._IMAGENET_STD, ds_mod._IMAGENET_STD)


def test_checkpoint_roundtrip_and_scoring(tmp_path: Path) -> None:
    import torch

    from mikha.ref.classifier import WaterClassifier
    from mikha.train.model import ModelSpec, build_model

    spec = ModelSpec(backbone="mobilenet_v3_small", pretrained=False, num_classes=1)
    model = build_model(spec)
    ckpt_path = tmp_path / "toy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spec": {"backbone": spec.backbone, "pretrained": False, "num_classes": 1},
            "epoch": 0,
            "val_auroc": 0.0,
            "config": {},
            "image_size": 64,
        },
        ckpt_path,
    )

    clf = WaterClassifier(ckpt_path, device="cpu", image_size=64)
    assert clf.backbone == "mobilenet_v3_small"
    assert clf.image_size == 64
    assert clf.device == "cpu"

    img = np.random.default_rng(0).integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    p = clf.score(img)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

    # score_fn: same shape as ScoreFn (single image → float in [0,1])
    fn = clf.as_score_fn()
    p2 = fn(img)
    assert isinstance(p2, float)
    assert p == pytest.approx(p2)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    from mikha.ref.classifier import WaterClassifier

    with pytest.raises(FileNotFoundError):
        WaterClassifier(tmp_path / "does_not_exist.pt", device="cpu")


def test_bad_input_shape_rejected(tmp_path: Path) -> None:
    import torch

    from mikha.ref.classifier import WaterClassifier
    from mikha.train.model import ModelSpec, build_model

    spec = ModelSpec(backbone="mobilenet_v3_small", pretrained=False)
    model = build_model(spec)
    ckpt_path = tmp_path / "toy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spec": {"backbone": spec.backbone, "pretrained": False, "num_classes": 1},
            "image_size": 64,
        },
        ckpt_path,
    )

    clf = WaterClassifier(ckpt_path, device="cpu", image_size=64)
    with pytest.raises(ValueError, match="HxWx3"):
        clf.score(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="HxWx3"):
        clf.score(np.zeros((32, 32, 4), dtype=np.uint8))
