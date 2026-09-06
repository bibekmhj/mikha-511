"""Public registry of degradation transforms.

The 5 canonical Mikha-Bench degradation classes are stored under the names
below. These names appear in ``bench/labels.json`` and in the results table
produced by :mod:`mikha.eval`; do NOT rename them without a benchmark bump.

The optional ``ir_night`` transform is exposed for exploration and is NOT
part of the 5-class Mikha-Bench stratification at v0.1.0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .fog import apply_fog
from .glare import apply_glare
from .jpeg import apply_jpeg
from .night import apply_ir_night, apply_night
from .rain import apply_rain

Transform = Callable[[np.ndarray, int, float], np.ndarray]

# Canonical benchmark transforms — order matters for reproducibility.
BENCH_TRANSFORMS: dict[str, Transform] = {
    "rain": apply_rain,
    "fog": apply_fog,
    "night": apply_night,
    "glare": apply_glare,
    "jpeg": apply_jpeg,
}

# Extras exposed for exploration; not part of the v0.1.0 benchmark.
EXTRA_TRANSFORMS: dict[str, Transform] = {
    "ir_night": apply_ir_night,
}

ALL_TRANSFORMS: dict[str, Transform] = {**BENCH_TRANSFORMS, **EXTRA_TRANSFORMS}


def apply(name: str, image: np.ndarray, seed: int = 0, severity: float = 0.5) -> np.ndarray:
    """Apply a named transform. Raises ``KeyError`` for unknown names."""
    try:
        fn = ALL_TRANSFORMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown transform {name!r}; known: {sorted(ALL_TRANSFORMS)}") from exc
    return fn(image, seed, severity)
