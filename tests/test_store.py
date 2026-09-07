"""Tests for the SQLite event store."""

from __future__ import annotations

from pathlib import Path

from mikha.ref.store import EventStore


def _fresh_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / "events.sqlite")


def test_empty_store_reports_zero(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    assert store.count() == 0
    assert store.recent() == []
    assert store.latest_per_camera() == []


def test_append_and_recent_returns_newest_first(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    for i in range(5):
        store.append(
            camera_id="cam-A",
            label="passable",
            calibrated_conf=0.1 * i,
            mask_area_frac=0.01 * i,
            is_transition=False,
        )
    rows = store.recent(limit=3)
    assert len(rows) == 3
    assert [r.id for r in rows] == [5, 4, 3]


def test_camera_id_filter(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    for cam in ("cam-A", "cam-B", "cam-A"):
        store.append(
            camera_id=cam,
            label="passable",
            calibrated_conf=0.5,
            mask_area_frac=0.05,
            is_transition=False,
        )
    a_rows = store.recent(camera_id="cam-A")
    b_rows = store.recent(camera_id="cam-B")
    assert len(a_rows) == 2
    assert len(b_rows) == 1


def test_latest_per_camera(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    for label in ("passable", "flooded_uncertain", "flooded"):
        store.append(
            camera_id="cam-A",
            label=label,
            calibrated_conf=0.9,
            mask_area_frac=0.5,
            is_transition=True,
        )
    store.append(
        camera_id="cam-B",
        label="passable",
        calibrated_conf=0.1,
        mask_area_frac=0.0,
        is_transition=False,
    )
    latest = {r.camera_id: r.label for r in store.latest_per_camera()}
    assert latest == {"cam-A": "flooded", "cam-B": "passable"}


def test_wipe(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    store.append(
        camera_id="cam-A",
        label="passable",
        calibrated_conf=0.1,
        mask_area_frac=0.0,
        is_transition=False,
    )
    assert store.count() == 1
    store.wipe()
    assert store.count() == 0


def test_transition_flag_roundtrip(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    store.append(
        camera_id="c",
        label="flooded",
        calibrated_conf=0.9,
        mask_area_frac=0.3,
        is_transition=True,
        note="raise",
    )
    (row,) = store.recent(limit=1)
    assert row.is_transition is True
    assert row.note == "raise"
