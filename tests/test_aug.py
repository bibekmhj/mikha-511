"""M2 tests for Mikha-Aug.

Every transform must:

* be deterministic given a seed (same input, same seed → identical bytes);
* preserve shape and dtype;
* actually change something (mean absolute pixel difference above a floor);
* accept and clamp out-of-range severity;
* validate its input.

The tests use a small synthetic image so they are fast (<1s per transform).
"""

from __future__ import annotations

import numpy as np
import pytest

from mikha.aug import BENCH_TRANSFORMS, EXTRA_TRANSFORMS, apply
from mikha.aug.base import check_image, clip_severity


def _synthetic_image(h: int = 96, w: int = 128, seed: int = 42) -> np.ndarray:
    """Reproducible non-uniform test image so a real transform is visible."""
    rng = np.random.default_rng(seed)
    # A gradient + noise + a bright rectangle — gives every transform
    # something to bite on (high-contrast edges, texture, saturation).
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = np.stack([grad, np.roll(grad, w // 3, axis=1), np.roll(grad, w // 2, axis=1)], axis=-1)
    noise = rng.integers(-15, 15, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[h // 4 : h // 2, w // 4 : 3 * w // 4] = 250
    return img


ALL_TO_TEST = {**BENCH_TRANSFORMS, **EXTRA_TRANSFORMS}


@pytest.mark.parametrize("name", sorted(ALL_TO_TEST))
def test_shape_and_dtype_preserved(name: str) -> None:
    img = _synthetic_image()
    out = ALL_TO_TEST[name](img, seed=0, severity=0.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(ALL_TO_TEST))
def test_deterministic_given_seed(name: str) -> None:
    img = _synthetic_image()
    a = ALL_TO_TEST[name](img, seed=123, severity=0.6)
    b = ALL_TO_TEST[name](img, seed=123, severity=0.6)
    assert np.array_equal(a, b), f"{name} is not deterministic given seed"


@pytest.mark.parametrize("name", sorted(ALL_TO_TEST))
def test_input_is_not_mutated(name: str) -> None:
    img = _synthetic_image()
    snapshot = img.copy()
    _ = ALL_TO_TEST[name](img, seed=0, severity=0.5)
    assert np.array_equal(img, snapshot), f"{name} mutated its input"


@pytest.mark.parametrize("name", sorted(ALL_TO_TEST))
def test_actually_changes_the_image(name: str) -> None:
    img = _synthetic_image()
    out = ALL_TO_TEST[name](img, seed=0, severity=0.8)
    diff = float(np.mean(np.abs(out.astype(np.int32) - img.astype(np.int32))))
    # Any real transform should shift mean absolute pixel error above a few LSBs.
    assert diff > 2.0, f"{name} left the image essentially unchanged (mad={diff:.2f})"


@pytest.mark.parametrize("name", sorted(ALL_TO_TEST))
def test_severity_out_of_range_is_clamped(name: str) -> None:
    img = _synthetic_image()
    lo = ALL_TO_TEST[name](img, seed=0, severity=-5.0)
    hi = ALL_TO_TEST[name](img, seed=0, severity=5.0)
    # Both calls should succeed (no ValueError) and return valid images.
    assert lo.dtype == np.uint8
    assert hi.dtype == np.uint8


def test_severity_higher_generally_stronger_effect() -> None:
    """For the four non-JPEG transforms, higher severity should shift pixels more.

    JPEG is excluded because at very small quality deltas the effect is
    non-monotone on smooth synthetic images.
    """
    img = _synthetic_image()
    for name in ("rain", "fog", "night", "glare"):
        mild = ALL_TO_TEST[name](img, seed=7, severity=0.1)
        heavy = ALL_TO_TEST[name](img, seed=7, severity=0.95)
        d_mild = float(np.mean(np.abs(mild.astype(np.int32) - img.astype(np.int32))))
        d_heavy = float(np.mean(np.abs(heavy.astype(np.int32) - img.astype(np.int32))))
        assert d_heavy > d_mild, f"{name}: heavy severity did not increase distortion"


def test_apply_dispatch() -> None:
    img = _synthetic_image()
    out = apply("fog", img, seed=0, severity=0.5)
    assert out.shape == img.shape
    with pytest.raises(KeyError):
        apply("no_such_transform", img)


def test_check_image_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        check_image("not an array")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        check_image(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        check_image(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        check_image(np.zeros((10, 10, 4), dtype=np.uint8))


def test_clip_severity_bounds() -> None:
    assert clip_severity(-1.0) == 0.0
    assert clip_severity(0.5) == 0.5
    assert clip_severity(2.0) == 1.0
    with pytest.raises(TypeError):
        clip_severity("hi")  # type: ignore[arg-type]


def test_registry_has_the_5_bench_transforms() -> None:
    assert set(BENCH_TRANSFORMS) == {"rain", "fog", "night", "glare", "jpeg"}
