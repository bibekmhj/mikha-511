"""Group-aware train / validation / test splitting for Mikha-Bench.

The Mikha-Bench base set is dominated by EF2013 rows whose uploaders are
heavily skewed — a single Wikimedia uploader can contribute 19% of the
corpus. A naive per-row split leaks a photographer's camera and style across
train / val / test, inflating measured accuracy in ways that will not
survive on real 511 traffic-camera imagery.

This module assigns *whole groups* (uploaders, for EF2013; source-scoped
buckets for anything else) to a single split, targeting the requested
ratios and keeping flood / nonflood balanced within each split. Fully
deterministic given the same manifest and seed.

Design decisions worth naming:

* Group granularity is per-uploader for ``eu_flood_2013``. For every other
  source (smoke rows, HydroShare when it lands, US-Gov PD when it lands),
  the group is just the source name — small sources fall into one split,
  which is fine.
* Split algorithm is greedy "load balancing": for each group in
  largest-first order, place it in the split whose current fill ratio
  (``current_size / target_size``) is smallest, tie-broken by resulting
  flood-fraction deviation from the global flood-fraction. This is a
  well-known heuristic and produces reasonable splits without a new
  scikit-learn dependency.
* The public surface is one function, :func:`group_stratified_split`, and
  one helper, :func:`extract_group`. Both are pure — no I/O.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby

from .manifest import ManifestRow

# ---------------------------------------------------------------------------
# Group extraction
# ---------------------------------------------------------------------------

_EF2013_SOURCE = "eu_flood_2013"
# Attribution written by scripts/import_eu_flood.py looks like:
#     "<artist> (Wikimedia Commons, <license>). <descriptionurl>"
# We split at " (Wikimedia" to recover the artist half. Falls back to the
# full attribution string if the sentinel is absent so nothing crashes.
_EF2013_ATTR_SENTINEL = " (Wikimedia"


def extract_group(row: ManifestRow) -> str:
    """Return the leakage-safe group key for ``row``.

    * EF2013 rows: ``"eu_flood_2013:<uploader>"`` (extracted from attribution).
    * All other rows: ``"<source>:<row_id>"`` — each smoke / held-out row is
      effectively its own group, which is the right thing for tiny non-EF
      sources.
    """
    if row.source == _EF2013_SOURCE:
        attr = row.attribution
        head, _, _ = attr.partition(_EF2013_ATTR_SENTINEL)
        uploader = head.strip() or "unknown"
        return f"{_EF2013_SOURCE}:{uploader}"
    return f"{row.source}:{row.id}"


# ---------------------------------------------------------------------------
# Split algorithm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitStats:
    """Per-split summary."""

    name: str
    row_ids: tuple[str, ...]
    n_total: int
    n_flood: int
    n_nonflood: int


@dataclass(frozen=True)
class SplitResult:
    """Outcome of :func:`group_stratified_split`."""

    splits: dict[str, SplitStats]  # keyed by split name
    group_assignments: dict[str, str]  # group_key -> split_name
    seed: int
    target_ratios: dict[str, float]
    n_total: int
    n_flood: int
    n_nonflood: int


def group_stratified_split(
    rows: Iterable[ManifestRow],
    *,
    seed: int = 42,
    target_ratios: dict[str, float] | None = None,
    group_fn=extract_group,
) -> SplitResult:
    """Assign whole groups to splits with balanced flood / nonflood.

    ``target_ratios`` defaults to ``{"train": 0.70, "val": 0.15, "test": 0.15}``
    and must sum to 1.0. Split names are preserved as declared here — the
    caller controls the vocabulary.

    ``group_fn`` maps a row to a group key. Defaults to :func:`extract_group`.
    """
    if target_ratios is None:
        target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    if not target_ratios:
        raise ValueError("target_ratios must be non-empty")
    total_ratio = sum(target_ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"target_ratios must sum to 1.0; got {total_ratio}")

    # Aggregate rows per group. Preserve within-group row order for reproducibility.
    per_group: dict[str, dict] = defaultdict(lambda: {"row_ids": [], "n_flood": 0, "n_nonflood": 0})
    for row in rows:
        g = group_fn(row)
        per_group[g]["row_ids"].append(row.id)
        if row.image_class == "flood":
            per_group[g]["n_flood"] += 1
        elif row.image_class == "nonflood":
            per_group[g]["n_nonflood"] += 1
        else:
            raise ValueError(f"row {row.id}: unexpected image_class {row.image_class!r}")

    n_total = sum(len(g["row_ids"]) for g in per_group.values())
    if n_total == 0:
        raise ValueError("no rows to split")
    n_flood = sum(g["n_flood"] for g in per_group.values())
    n_nonflood = sum(g["n_nonflood"] for g in per_group.values())
    global_flood_ratio = n_flood / n_total

    # Sort groups largest first; break ties by group key so the sort is
    # totally deterministic before we introduce any seeded randomness.
    ordered = sorted(per_group.items(), key=lambda kv: (-len(kv[1]["row_ids"]), kv[0]))

    # Within each same-size bucket, shuffle deterministically so runs with
    # different seeds actually differ, but a fixed seed always reproduces.
    rng = random.Random(seed)
    shuffled: list[tuple[str, dict]] = []
    for _, bucket in groupby(ordered, key=lambda kv: len(kv[1]["row_ids"])):
        chunk = list(bucket)
        rng.shuffle(chunk)
        shuffled.extend(chunk)

    # Greedy load-balancing assignment.
    split_names = list(target_ratios)
    assigned: dict[str, dict] = {
        s: {"row_ids": [], "n_flood": 0, "n_nonflood": 0} for s in split_names
    }
    group_assignments: dict[str, str] = {}

    for group_key, g in shuffled:
        add_flood = g["n_flood"]
        add_nonflood = g["n_nonflood"]
        add_size = len(g["row_ids"])

        best_split = None
        best_score: tuple[float, float, float, str] = (
            float("inf"),
            float("inf"),
            float("inf"),
            "",
        )
        for s in split_names:
            target_size = target_ratios[s] * n_total
            current = len(assigned[s]["row_ids"])
            fill = current / target_size if target_size > 0 else float("inf")

            future_size = current + add_size
            future_fill = future_size / target_size if target_size > 0 else float("inf")
            future_deviation = abs(future_fill - 1.0)

            future_flood = assigned[s]["n_flood"] + add_flood
            future_flood_ratio = (
                future_flood / future_size if future_size > 0 else global_flood_ratio
            )
            balance_penalty = abs(future_flood_ratio - global_flood_ratio)

            # Primary key: fill ratio (put group in emptiest bucket).
            # Secondary: how close to 1.0 we'd end up after adding.
            # Tertiary: flood-balance drift.
            # Final tie-break: split name (alpha) for determinism.
            score = (fill, future_deviation, balance_penalty, s)
            if score < best_score:
                best_score = score
                best_split = s

        assert best_split is not None
        assigned[best_split]["row_ids"].extend(g["row_ids"])
        assigned[best_split]["n_flood"] += add_flood
        assigned[best_split]["n_nonflood"] += add_nonflood
        group_assignments[group_key] = best_split

    # Freeze into typed result.
    splits = {
        s: SplitStats(
            name=s,
            row_ids=tuple(assigned[s]["row_ids"]),
            n_total=len(assigned[s]["row_ids"]),
            n_flood=assigned[s]["n_flood"],
            n_nonflood=assigned[s]["n_nonflood"],
        )
        for s in split_names
    }
    return SplitResult(
        splits=splits,
        group_assignments=group_assignments,
        seed=seed,
        target_ratios=dict(target_ratios),
        n_total=n_total,
        n_flood=n_flood,
        n_nonflood=n_nonflood,
    )
