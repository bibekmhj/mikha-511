"""Mikha-Bench: evaluation corpus construction, loading, and manifest tools.

Labels ship under CC-BY-4.0 (see LICENSE-DATA). Images are represented by
source URL + SHA-256 to respect upstream licenses; the repo does not
redistribute images.

Public surface (import from ``mikha.bench``):

* :class:`ManifestRow`, :class:`ManifestError`
* :data:`ALLOWED_CLASSES`, :data:`ALLOWED_LICENSES`, :data:`COLUMNS`
* :func:`load_manifest`, :func:`save_manifest`, :func:`validate_manifest`, :func:`hash_bytes`
* :mod:`mikha.bench.eu_flood` — European Flood 2013 parser and license normalizer.
"""

from __future__ import annotations

from .manifest import (
    ALLOWED_CLASSES,
    ALLOWED_LICENSES,
    COLUMNS,
    ManifestError,
    ManifestRow,
    hash_bytes,
    load_manifest,
    row_field_names,
    save_manifest,
    validate_manifest,
)
from .split import (
    SplitResult,
    SplitStats,
    extract_group,
    group_stratified_split,
)

__all__ = [
    "ALLOWED_CLASSES",
    "ALLOWED_LICENSES",
    "COLUMNS",
    "ManifestError",
    "ManifestRow",
    "SplitResult",
    "SplitStats",
    "extract_group",
    "group_stratified_split",
    "hash_bytes",
    "load_manifest",
    "row_field_names",
    "save_manifest",
    "validate_manifest",
]
