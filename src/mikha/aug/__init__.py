"""Mikha-Aug: deterministic, seeded degradation transforms modeling 511-style
camera artifacts.

Public surface (import from ``mikha.aug``):

* :func:`apply_rain`, :func:`apply_fog`, :func:`apply_night`, :func:`apply_glare`,
  :func:`apply_jpeg` — the 5 canonical Mikha-Bench transforms
* :func:`apply_ir_night` — extra, not in the v0.1.0 benchmark
* :data:`BENCH_TRANSFORMS`, :data:`EXTRA_TRANSFORMS`, :data:`ALL_TRANSFORMS`
* :func:`apply` — dispatch by name
"""

from __future__ import annotations

from .fog import apply_fog
from .glare import apply_glare
from .jpeg import apply_jpeg
from .night import apply_ir_night, apply_night
from .pipeline import ALL_TRANSFORMS, BENCH_TRANSFORMS, EXTRA_TRANSFORMS, apply
from .rain import apply_rain

__all__ = [
    "ALL_TRANSFORMS",
    "BENCH_TRANSFORMS",
    "EXTRA_TRANSFORMS",
    "apply",
    "apply_fog",
    "apply_glare",
    "apply_ir_night",
    "apply_jpeg",
    "apply_night",
    "apply_rain",
]
