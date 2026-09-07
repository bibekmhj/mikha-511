"""M4 tests for group-aware train/val/test splitting.

Every test that matters is a property test: run the split against a
synthetic corpus and assert an invariant. Also runs a determinism test
and one small hand-checked case.
"""

from __future__ import annotations

import pytest

from mikha.bench import ManifestRow, group_stratified_split
from mikha.bench.split import extract_group


def _row(
    row_id: str,
    *,
    source: str = "eu_flood_2013",
    attribution: str = "Alice (Wikimedia Commons, CC-BY-4.0). https://x",
    image_class: str = "flood",
    sha_seed: str = "0",
) -> ManifestRow:
    return ManifestRow(
        id=row_id,
        source=source,
        url="https://example.com/x.jpg",
        sha256=(sha_seed * 64)[:64],
        license="CC-BY-4.0",
        attribution=attribution,
        image_class=image_class,
        notes="",
    )


# ---------------------------------------------------------------------------
# Group extraction
# ---------------------------------------------------------------------------


def test_extract_group_from_ef2013_attribution() -> None:
    r = _row(
        "mb-000001",
        source="eu_flood_2013",
        attribution="Matěj Baťha (Wikimedia Commons, CC BY-SA 3.0). https://commons/…",
    )
    assert extract_group(r) == "eu_flood_2013:Matěj Baťha"


def test_extract_group_falls_back_when_attribution_lacks_sentinel() -> None:
    r = _row("mb-000001", source="eu_flood_2013", attribution="Just a name.")
    assert extract_group(r) == "eu_flood_2013:Just a name."


def test_extract_group_for_non_ef_source_uses_row_id() -> None:
    r = _row("mb-000042", source="opencv_extra", attribution="OpenCV project")
    assert extract_group(r) == "opencv_extra:mb-000042"


# ---------------------------------------------------------------------------
# Split algorithm — invariants
# ---------------------------------------------------------------------------


def _synthetic_ef_corpus(
    n_uploaders: int = 30,
    per_uploader: int = 20,
    flood_frac: float = 0.5,
) -> list[ManifestRow]:
    rows = []
    idx = 1
    for u in range(n_uploaders):
        for k in range(per_uploader):
            cls = "flood" if k < int(per_uploader * flood_frac) else "nonflood"
            rows.append(
                _row(
                    f"mb-{idx:06d}",
                    attribution=f"uploader-{u:03d} (Wikimedia Commons, CC-BY-4.0). https://x",
                    image_class=cls,
                    sha_seed=f"{idx:x}",
                )
            )
            idx += 1
    return rows


def test_all_rows_are_assigned_exactly_once() -> None:
    rows = _synthetic_ef_corpus()
    result = group_stratified_split(rows, seed=1)
    all_ids = [rid for s in result.splits.values() for rid in s.row_ids]
    assert len(all_ids) == len(rows)
    assert set(all_ids) == {r.id for r in rows}
    assert len(set(all_ids)) == len(all_ids)  # no duplicates


def test_no_group_leaks_across_splits() -> None:
    rows = _synthetic_ef_corpus()
    result = group_stratified_split(rows, seed=1)
    row_to_group = {r.id: extract_group(r) for r in rows}
    per_split_groups = {
        name: {row_to_group[rid] for rid in s.row_ids} for name, s in result.splits.items()
    }
    names = list(per_split_groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (per_split_groups[a] & per_split_groups[b]), (
                f"group leakage between {a} and {b}"
            )


def test_target_ratios_within_tolerance() -> None:
    """On a well-shaped synthetic corpus the greedy split hits target within ±5pp."""
    rows = _synthetic_ef_corpus(n_uploaders=40, per_uploader=15)
    result = group_stratified_split(rows, seed=1)
    for name, s in result.splits.items():
        actual = s.n_total / result.n_total
        target = result.target_ratios[name]
        assert abs(actual - target) <= 0.05, f"{name}: target {target:.2%}, actual {actual:.2%}"


def test_flood_balance_within_tolerance() -> None:
    rows = _synthetic_ef_corpus(n_uploaders=40, per_uploader=15, flood_frac=0.5)
    result = group_stratified_split(rows, seed=1)
    global_ratio = result.n_flood / result.n_total
    for s in result.splits.values():
        actual = s.n_flood / s.n_total if s.n_total else 0.0
        assert abs(actual - global_ratio) <= 0.10, (
            f"{s.name}: flood ratio {actual:.2%} vs global {global_ratio:.2%}"
        )


def test_deterministic_given_seed() -> None:
    rows = _synthetic_ef_corpus()
    a = group_stratified_split(rows, seed=42)
    b = group_stratified_split(rows, seed=42)
    for name in a.splits:
        assert a.splits[name].row_ids == b.splits[name].row_ids


def test_different_seeds_produce_different_splits() -> None:
    rows = _synthetic_ef_corpus()
    a = group_stratified_split(rows, seed=1)
    b = group_stratified_split(rows, seed=99)
    # Not required to differ on every row, but the overall assignments
    # should not be byte-identical (would defeat the purpose of seeding).
    any_diff = any(a.splits[n].row_ids != b.splits[n].row_ids for n in a.splits)
    assert any_diff


def test_dominant_group_ends_up_in_train() -> None:
    """A group that alone exceeds val/test target size must land in train."""
    rows = _synthetic_ef_corpus(n_uploaders=5, per_uploader=10)
    # Add one dominant uploader with 50 rows (5x larger than others).
    for k in range(50):
        rows.append(
            _row(
                f"mb-{1000 + k:06d}",
                attribution="uploader-DOMINANT (Wikimedia Commons, CC-BY-4.0). https://x",
                image_class="flood" if k % 2 == 0 else "nonflood",
                sha_seed=f"d{k:x}",
            )
        )
    result = group_stratified_split(rows, seed=1)
    assert result.group_assignments["eu_flood_2013:uploader-DOMINANT"] == "train"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_ratios_must_sum_to_one() -> None:
    rows = _synthetic_ef_corpus(n_uploaders=5, per_uploader=5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        group_stratified_split(rows, target_ratios={"train": 0.5, "val": 0.5, "test": 0.5})


def test_empty_ratios_rejected() -> None:
    rows = _synthetic_ef_corpus(n_uploaders=5, per_uploader=5)
    with pytest.raises(ValueError, match="non-empty"):
        group_stratified_split(rows, target_ratios={})


def test_empty_rows_rejected() -> None:
    with pytest.raises(ValueError, match="no rows"):
        group_stratified_split([])


def test_unknown_image_class_rejected() -> None:
    # image_class is validated at ManifestRow construction, so the branch inside
    # group_stratified_split() is defensive only. We document that here.
    pytest.skip("image_class validated at ManifestRow construction; branch defensive only")
