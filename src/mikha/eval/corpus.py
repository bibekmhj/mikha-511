"""Mikha-Bench evaluation corpus builder.

Iterates over the rows in ``bench/splits.json`` × the 5 canonical
Mikha-Aug degradation classes (plus the clean pass), yielding one
:class:`EvalSample` per (row, degradation) pair. Images are loaded and
transformed lazily so a 3,558-sample corpus doesn't blow the RAM.

Design decisions:

* Base image path is inferred the same way ``fetch_base.py`` writes it:
  ``data/images/<row_id>.<ext>`` where ext is taken from the URL's suffix.
* Sample severity is fixed per-run (default 0.65 — the ~mid-heavy point
  from ``docs/aug_gallery.md``). One severity keeps the results table
  tractable; per-severity sweeps are a follow-up.
* A "clean" pseudo-degradation is included so the baseline number lands
  on the same plot as the degraded numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mikha.aug import BENCH_TRANSFORMS
from mikha.bench import ManifestRow, load_manifest, local_image_path

CLEAN = "clean"
DEGRADATIONS: tuple[str, ...] = (CLEAN, *BENCH_TRANSFORMS.keys())


@dataclass(frozen=True)
class EvalSample:
    """One evaluation sample = one manifest row under one degradation."""

    row_id: str
    split: str  # train / val / test
    source: str
    degradation: str  # clean | rain | fog | night | glare | jpeg
    severity: float
    image_class: str  # flood | nonflood
    image_path: Path
    seed: int  # seeded aug transform → deterministic

    @property
    def label(self) -> bool:
        return self.image_class == "flood"

    def load(self) -> Any:
        """Read the base image from disk, apply the degradation, return BGR ndarray."""
        import cv2  # local import; keeps module import free of cv2

        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image {self.image_path}")
        if self.degradation == CLEAN:
            return img
        fn = BENCH_TRANSFORMS[self.degradation]
        return fn(img, seed=self.seed, severity=self.severity)


# ---------------------------------------------------------------------------


def _local_path(row: ManifestRow, images_dir: Path) -> Path:
    """Thin wrapper preserved for internal callers; delegates to the shared helper."""
    return local_image_path(row, images_dir)


def load_splits(splits_path: str | Path) -> dict[str, set[str]]:
    """Load bench/splits.json → {split_name: {row_id, ...}}."""
    with Path(splits_path).open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return {name: set(payload["row_ids"]) for name, payload in doc["splits"].items()}


def build_samples(
    manifest_path: str | Path,
    splits_path: str | Path,
    images_dir: str | Path,
    *,
    which_splits: Iterable[str] = ("test",),
    degradations: Iterable[str] = DEGRADATIONS,
    severity: float = 0.65,
    seed: int = 1234,
    limit_per_class: int | None = None,
) -> Iterator[EvalSample]:
    """Yield samples for one or more splits.

    ``limit_per_class`` (if given) caps flood + nonflood counts separately
    per (split, degradation) — useful for a --fast smoke run without
    changing the corpus definition itself.
    """
    if not (0.0 <= severity <= 1.0):
        raise ValueError("severity must be in [0, 1]")

    rows = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    images_dir = Path(images_dir)
    which_splits = tuple(which_splits)
    for s in which_splits:
        if s not in splits:
            raise ValueError(f"unknown split {s!r}; known: {sorted(splits)}")

    degradations = tuple(degradations)
    for d in degradations:
        if d not in DEGRADATIONS:
            raise ValueError(f"unknown degradation {d!r}; known: {DEGRADATIONS}")

    for split_name in which_splits:
        wanted_ids = splits[split_name]
        # Preserve manifest ordering for determinism.
        split_rows = [r for r in rows if r.id in wanted_ids]

        for degradation in degradations:
            per_class: dict[str, int] = {"flood": 0, "nonflood": 0}
            for row in split_rows:
                if limit_per_class is not None and per_class[row.image_class] >= limit_per_class:
                    continue
                per_class[row.image_class] += 1
                yield EvalSample(
                    row_id=row.id,
                    split=split_name,
                    source=row.source,
                    degradation=degradation,
                    severity=severity,
                    image_class=row.image_class,
                    image_path=_local_path(row, images_dir),
                    seed=seed,
                )


def sample_count(
    manifest_path: str | Path,
    splits_path: str | Path,
    *,
    which_splits: Iterable[str] = ("test",),
    degradations: Iterable[str] = DEGRADATIONS,
) -> int:
    """Cheap size estimate without loading images."""
    rows = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    ids_in_scope = set().union(*(splits[s] for s in which_splits))
    return sum(1 for r in rows if r.id in ids_in_scope) * len(tuple(degradations))
