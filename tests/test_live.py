"""Tests for mikha.bench.live (M10).

Pure in-memory / tmp_path. No network, no torch, no cv2.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mikha.bench.live import (
    LIVE_COLUMNS,
    Label,
    LabelStore,
    LiveManifestError,
    LiveManifestRow,
    append_live_row,
    corpus_stats,
    iter_series,
    load_live_manifest,
    make_row_id,
    save_live_manifest,
    series_key,
)

# ---------------------------------------------------------------------------
# id + series helpers


def test_make_row_id_is_deterministic_and_normalized() -> None:
    a = make_row_id("fl511_00_MDX_SR874_000", "2026-09-08T14:23:45+00:00")
    b = make_row_id("fl511_00_MDX_SR874_000", "2026-09-08T14:23:45Z")
    assert a == b == "live-fl511_00_MDX_SR874_000-20260908T142345"


def test_make_row_id_rejects_bad_camera_id() -> None:
    with pytest.raises(LiveManifestError, match="camera_id"):
        make_row_id("Has Spaces", "2026-09-08T00:00:00Z")


def test_series_key_buckets_by_hour_by_default() -> None:
    k1 = series_key("cam", "2026-09-08T14:00:00Z")
    k2 = series_key("cam", "2026-09-08T14:59:59Z")
    k3 = series_key("cam", "2026-09-08T15:00:00Z")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("cam:")


def test_series_key_custom_bucket() -> None:
    k1 = series_key("cam", "2026-09-08T14:00:00Z", bucket_seconds=300)
    k2 = series_key("cam", "2026-09-08T14:04:59Z", bucket_seconds=300)
    k3 = series_key("cam", "2026-09-08T14:05:00Z", bucket_seconds=300)
    assert k1 == k2 and k1 != k3


# ---------------------------------------------------------------------------
# LiveManifestRow validation


def _row(**overrides: object) -> LiveManifestRow:
    base = {
        "id": "live-cam-20260908T140000",
        "camera_id": "cam",
        "series_id": "cam:1757340000",
        "captured_at": "2026-09-08T14:00:00Z",
        "source": "fl511",
        "url": "https://example.org/snap.jpg",
        "sha256": "a" * 64,
        "local_path": "data/live_511/cam/live-cam-20260908T140000.jpg",
        "notes": "",
    }
    base.update(overrides)
    return LiveManifestRow(**base)  # type: ignore[arg-type]


def test_row_construction_happy_path() -> None:
    r = _row()
    assert r.camera_id == "cam"
    assert r.captured_dt.year == 2026


def test_row_rejects_bad_id_shape() -> None:
    with pytest.raises(LiveManifestError, match="invalid id"):
        _row(id="bad-id")


def test_row_rejects_non_https_url() -> None:
    with pytest.raises(LiveManifestError, match="invalid url"):
        _row(url="ftp://example.org/x.jpg")


def test_row_rejects_absolute_local_path() -> None:
    with pytest.raises(LiveManifestError, match="local_path"):
        _row(local_path="/etc/passwd")


def test_row_rejects_bad_sha256() -> None:
    with pytest.raises(LiveManifestError, match="sha256"):
        _row(sha256="not-a-hash")


# ---------------------------------------------------------------------------
# Load / save / append round-trips


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rows = [
        _row(),
        _row(
            id="live-cam-20260908T140030",
            captured_at="2026-09-08T14:00:30Z",
            local_path="data/live_511/cam/live-cam-20260908T140030.jpg",
        ),
    ]
    path = tmp_path / "manifest.csv"
    save_live_manifest(rows, path)
    loaded = load_live_manifest(path)
    assert [r.id for r in loaded] == [r.id for r in rows]


def test_load_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    save_live_manifest([_row()], path)
    # Manually append a duplicate id to force the loader to fail.
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LIVE_COLUMNS))
        writer.writerow(
            {
                "id": "live-cam-20260908T140000",
                "camera_id": "cam",
                "series_id": "cam:1757340000",
                "captured_at": "2026-09-08T14:00:00Z",
                "source": "fl511",
                "url": "https://example.org/x.jpg",
                "sha256": "b" * 64,
                "local_path": "data/live_511/cam/dup.jpg",
                "notes": "",
            }
        )
    with pytest.raises(LiveManifestError, match="duplicate id"):
        load_live_manifest(path)


def test_append_live_row_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    r = _row()
    append_live_row(r, path)
    append_live_row(r, path)  # should NOT duplicate
    loaded = load_live_manifest(path)
    assert len(loaded) == 1


def test_append_live_row_creates_header(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.csv"
    append_live_row(_row(), path)
    assert path.exists()
    header = path.read_text().splitlines()[0]
    assert header == ",".join(LIVE_COLUMNS)


# ---------------------------------------------------------------------------
# LabelStore


def test_label_store_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "live_labels.json"
    store = LabelStore(p)
    store.set(
        "live-cam-20260908T140000",
        Label(image_class="flood", labeled_by="me", labeled_at="2026-09-08T15:00:00Z"),
    )
    store.set_meta(labeling_protocol_version=1)
    store.save()

    # Reopen a fresh store on the same file
    fresh = LabelStore(p)
    assert "live-cam-20260908T140000" in fresh
    lbl = fresh.get("live-cam-20260908T140000")
    assert lbl is not None
    assert lbl.image_class == "flood"
    assert fresh.meta()["labeling_protocol_version"] == 1


def test_label_store_rejects_bad_class(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.json")
    with pytest.raises(LiveManifestError, match="image_class"):
        store.set(
            "live-cam-20260908T140000",
            Label(image_class="mystery", labeled_by="me", labeled_at="2026-09-08T15:00:00Z"),
        )


def test_label_store_rejects_bad_json_class(tmp_path: Path) -> None:
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps(
            {
                "meta": {},
                "labels": {"live-cam-20260908T140000": {"image_class": "wet"}},
            }
        )
    )
    with pytest.raises(LiveManifestError, match="image_class"):
        LabelStore(p)


def test_label_store_atomic_write_does_not_leave_temp(tmp_path: Path) -> None:
    p = tmp_path / "labels.json"
    store = LabelStore(p)
    store.set(
        "live-cam-20260908T140000",
        Label(image_class="flood", labeled_by="", labeled_at="2026-09-08T15:00:00Z"),
    )
    store.save()
    # No leftover .tmp sibling
    assert not (tmp_path / "labels.json.tmp").exists()


# ---------------------------------------------------------------------------
# iter_series


def _rows_for_camera(cam: str, minutes: list[int]) -> list[LiveManifestRow]:
    """Build rows for one camera at given minute offsets past 14:00 UTC."""
    out = []
    for m in minutes:
        ts = f"2026-09-08T14:{m:02d}:00Z"
        rid = make_row_id(cam, ts)
        out.append(
            LiveManifestRow(
                id=rid,
                camera_id=cam,
                series_id=series_key(cam, ts),
                captured_at=ts,
                source="fl511",
                url=f"https://example.org/{cam}.jpg",
                sha256="a" * 64,
                local_path=f"data/live_511/{cam}/{rid}.jpg",
            )
        )
    return out


def test_iter_series_groups_and_orders() -> None:
    # Camera "a": 3 frames, delivered out of order in the input list.
    a_rows = _rows_for_camera("a", [10, 30, 20])
    # Camera "b": 1 frame.
    b_rows = _rows_for_camera("b", [15])
    series_list = list(iter_series(a_rows + b_rows))
    assert len(series_list) == 2

    a_series = next(s for s in series_list if s.camera_id == "a")
    assert len(a_series) == 3
    minutes = [f.captured_dt.minute for f in a_series.frames]
    assert minutes == sorted(minutes)  # ascending
    assert minutes == [10, 20, 30]


def test_iter_series_require_labels_filters_unlabeled(tmp_path: Path) -> None:
    rows = _rows_for_camera("cam", [0, 30])
    store = LabelStore(tmp_path / "l.json")
    # Only label the first frame; series should be dropped as incomplete.
    store.set(
        rows[0].id,
        Label(image_class="flood", labeled_by="", labeled_at="2026-09-08T15:00:00Z"),
    )
    series_list = list(iter_series(rows, labels=store, require_labels=True))
    assert series_list == []

    # Label both -> series appears.
    store.set(
        rows[1].id,
        Label(image_class="nonflood", labeled_by="", labeled_at="2026-09-08T15:00:00Z"),
    )
    series_list = list(iter_series(rows, labels=store, require_labels=True))
    assert len(series_list) == 1
    assert len(series_list[0]) == 2


def test_iter_series_require_labels_without_store_raises() -> None:
    with pytest.raises(ValueError, match="LabelStore"):
        list(iter_series(_rows_for_camera("cam", [0]), require_labels=True))


# ---------------------------------------------------------------------------
# corpus_stats


def test_corpus_stats_counts_correctly(tmp_path: Path) -> None:
    rows = _rows_for_camera("a", [0, 30]) + _rows_for_camera("b", [0])
    store = LabelStore(tmp_path / "l.json")
    store.set(
        rows[0].id,
        Label(image_class="flood", labeled_by="", labeled_at="2026-09-08T15:00:00Z"),
    )
    store.set(
        rows[2].id,
        Label(image_class="nonflood", labeled_by="", labeled_at="2026-09-08T15:00:00Z"),
    )
    stats = corpus_stats(rows, store)
    assert stats.n_rows == 3
    assert stats.n_cameras == 2
    assert stats.n_series == 2
    assert stats.n_labeled == 2
    assert stats.label_counts["flood"] == 1
    assert stats.label_counts["nonflood"] == 1
    assert stats.label_counts["uncertain"] == 0
