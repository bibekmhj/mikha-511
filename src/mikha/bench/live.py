"""Live 511 held-out corpus loader (M10 / v0.3 scaffold).

Distinct from the redistributable ``data/manifest.csv`` corpus because
FL511 / TX511 snapshots are publicly viewable but their ToS restrict
redistribution. This module ships the schema, the loader, and the label
store; images stay on the local machine (see ``data/live_511/README.md``
and the ``.gitignore`` entries).

Wire layout::

    data/live_511/manifest.csv           # this schema; rows only, gitignored images
    data/live_511/<camera_id>/<file>     # local image files, never committed
    bench/live_labels.json               # hand-labeled ground truth, shippable

M10 is data + tooling only. The persistence-gate ablation that uses
these time series is scheduled for M11 and is deliberately NOT wired
here.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Schema -----------------------------------------------------------------

LIVE_COLUMNS: tuple[str, ...] = (
    "id",
    "camera_id",
    "series_id",
    "captured_at",  # RFC3339 UTC
    "source",  # e.g. "fl511", "tx511"
    "url",  # snapshot URL that was fetched
    "sha256",
    "local_path",  # path under data/live_511/, relative to repo root
    "notes",
)

_ID_RE = re.compile(r"^live-[A-Za-z0-9_.-]{1,64}-\d{8}T\d{6}$")
_CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SERIES_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class LiveManifestError(ValueError):
    """Raised on any live-manifest schema violation."""


@dataclass(frozen=True)
class LiveManifestRow:
    """One live-snapshot row.

    Deliberately separate from :class:`mikha.bench.ManifestRow` because
    live rows carry time-series fields (``camera_id``, ``series_id``,
    ``captured_at``) and reference a local-only image path rather than a
    redistributable URL+license pair. Licensing is not per-row here; ToS
    is per-source and documented in ``data/live_511/README.md``.
    """

    id: str
    camera_id: str
    series_id: str
    captured_at: str
    source: str
    url: str
    sha256: str
    local_path: str
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_live_row(self)

    @property
    def captured_dt(self) -> datetime:
        """Parse ``captured_at`` into a timezone-aware datetime."""
        s = self.captured_at.replace("Z", "+00:00")
        return datetime.fromisoformat(s)


def _validate_live_row(row: LiveManifestRow) -> None:
    if not _ID_RE.match(row.id):
        raise LiveManifestError(f"invalid id {row.id!r} (want 'live-<camera_id>-YYYYMMDDTHHMMSS')")
    if not _CAMERA_ID_RE.match(row.camera_id):
        raise LiveManifestError(f"invalid camera_id {row.camera_id!r}")
    if not _SERIES_ID_RE.match(row.series_id):
        raise LiveManifestError(f"invalid series_id {row.series_id!r}")
    if not _RFC3339.match(row.captured_at):
        raise LiveManifestError(f"invalid captured_at {row.captured_at!r} (want RFC3339 with tz)")
    if not row.source:
        raise LiveManifestError("source must be non-empty")
    if not row.url or not row.url.startswith(("http://", "https://")):
        raise LiveManifestError(f"invalid url {row.url!r}")
    if not _SHA256_RE.match(row.sha256):
        raise LiveManifestError(f"invalid sha256 {row.sha256!r}")
    if not row.local_path or row.local_path.startswith("/"):
        raise LiveManifestError(
            f"invalid local_path {row.local_path!r} (want a repo-relative path)"
        )


def make_row_id(camera_id: str, captured_at: str) -> str:
    """Deterministic id from ``(camera_id, captured_at)``.

    The id is what pairs a manifest row to a labels-JSON entry, so it
    must be stable and independent of file layout. Timestamp is
    normalized to compact ``YYYYMMDDTHHMMSS`` (UTC) form.
    """
    if not _CAMERA_ID_RE.match(camera_id):
        raise LiveManifestError(f"invalid camera_id {camera_id!r}")
    if not _RFC3339.match(captured_at):
        raise LiveManifestError(f"invalid captured_at {captured_at!r}")
    dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    stamp = dt.strftime("%Y%m%dT%H%M%S")
    return f"live-{camera_id}-{stamp}"


def series_key(camera_id: str, captured_at: str, *, bucket_seconds: int = 3600) -> str:
    """Bucket contiguous captures from one camera into a series id.

    Default is one-hour buckets, which is a reasonable stand-in for
    "one event window" on a 30-second poll cadence (about 120 frames per
    series). Callers can override via ``bucket_seconds`` or hand-set
    ``series_id`` directly.
    """
    dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    epoch = int(dt.timestamp())
    bucket = (epoch // max(1, int(bucket_seconds))) * int(bucket_seconds)
    return f"{camera_id}:{bucket}"


# --- Load / save ------------------------------------------------------------


def load_live_manifest(path: str | Path) -> list[LiveManifestRow]:
    """Read the CSV and validate every row. Raises on any schema issue."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise LiveManifestError(f"{path}: missing header row")
        missing = set(LIVE_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise LiveManifestError(f"{path}: missing columns: {sorted(missing)}")
        rows: list[LiveManifestRow] = []
        seen_ids: set[str] = set()
        for raw in reader:
            row = LiveManifestRow(**{k: raw.get(k, "") for k in LIVE_COLUMNS})
            if row.id in seen_ids:
                raise LiveManifestError(f"duplicate id: {row.id}")
            seen_ids.add(row.id)
            rows.append(row)
    return rows


def save_live_manifest(rows: Iterable[LiveManifestRow], path: str | Path) -> None:
    """Atomic write in canonical column order. Refuses invalid rows."""
    rows_list = list(rows)
    for r in rows_list:
        _validate_live_row(r)  # defensive
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LIVE_COLUMNS))
            writer.writeheader()
            for r in rows_list:
                writer.writerow(asdict(r))
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def append_live_row(row: LiveManifestRow, path: str | Path) -> None:
    """Append a single row, creating the header if the file does not exist.

    Idempotent: if ``row.id`` already appears in the file, this is a no-op.
    Not atomic across processes, but safe within a single-process
    collector because the header is written once.
    """
    _validate_live_row(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if "id" in r:
                    existing_ids.add(r["id"])
        if row.id in existing_ids:
            return
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LIVE_COLUMNS))
            writer.writerow(asdict(row))
    else:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LIVE_COLUMNS))
            writer.writeheader()
            writer.writerow(asdict(row))


# --- Labels -----------------------------------------------------------------

_ALLOWED_LABELS = ("flood", "nonflood", "uncertain")


@dataclass(frozen=True)
class Label:
    image_class: str  # one of _ALLOWED_LABELS
    labeled_by: str
    labeled_at: str  # RFC3339
    notes: str = ""


class LabelStore:
    """Load / update ``bench/live_labels.json``.

    Shape mirrors ``bench/splits.json``:
    ``{"meta": {...}, "labels": {row_id: {image_class, labeled_by, ...}}}``.

    Writes go through an atomic replace so an interrupted labeler
    session never leaves a half-written file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._meta: dict[str, Any] = {}
        self._labels: dict[str, Label] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        doc = json.loads(self._path.read_text(encoding="utf-8"))
        self._meta = dict(doc.get("meta", {}))
        raw_labels = doc.get("labels", {}) or {}
        for row_id, entry in raw_labels.items():
            if not isinstance(entry, dict):
                raise LiveManifestError(f"labels[{row_id!r}] must be an object")
            if entry.get("image_class") not in _ALLOWED_LABELS:
                raise LiveManifestError(
                    f"labels[{row_id!r}].image_class must be one of {_ALLOWED_LABELS}"
                )
            self._labels[row_id] = Label(
                image_class=entry["image_class"],
                labeled_by=str(entry.get("labeled_by", "")),
                labeled_at=str(entry.get("labeled_at", "")),
                notes=str(entry.get("notes", "")),
            )

    # -- read --

    def __contains__(self, row_id: str) -> bool:
        return row_id in self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def get(self, row_id: str) -> Label | None:
        return self._labels.get(row_id)

    def labels(self) -> dict[str, Label]:
        return dict(self._labels)

    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    # -- write --

    def set(self, row_id: str, label: Label) -> None:
        if label.image_class not in _ALLOWED_LABELS:
            raise LiveManifestError(
                f"image_class must be one of {_ALLOWED_LABELS}, got {label.image_class!r}"
            )
        self._labels[row_id] = label

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def save(self) -> None:
        """Atomic dump; parent dirs are created if needed."""
        doc = {
            "meta": self._meta,
            "labels": {row_id: asdict(label) for row_id, label in sorted(self._labels.items())},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except BaseException:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
            raise


# --- Series enumeration -----------------------------------------------------


@dataclass(frozen=True)
class LiveSeries:
    """A time-ordered group of frames from one camera + series bucket."""

    camera_id: str
    series_id: str
    frames: tuple[LiveManifestRow, ...]  # sorted by captured_dt ascending

    def __len__(self) -> int:
        return len(self.frames)


def iter_series(
    rows: Iterable[LiveManifestRow],
    *,
    labels: LabelStore | None = None,
    require_labels: bool = False,
) -> Iterator[LiveSeries]:
    """Group rows by ``series_id`` and yield time-ordered series.

    Set ``require_labels=True`` (with ``labels``) to skip any series
    that contains an unlabeled frame -- useful for the M11 gate
    ablation, ignored during collection.
    """
    grouped: dict[tuple[str, str], list[LiveManifestRow]] = {}
    for row in rows:
        grouped.setdefault((row.camera_id, row.series_id), []).append(row)

    for (camera_id, series_id), frames in sorted(grouped.items()):
        frames_sorted = sorted(frames, key=lambda r: r.captured_dt)
        if require_labels:
            if labels is None:
                raise ValueError("require_labels=True but no LabelStore was provided")
            if not all(r.id in labels for r in frames_sorted):
                continue
        yield LiveSeries(
            camera_id=camera_id,
            series_id=series_id,
            frames=tuple(frames_sorted),
        )


# --- Introspection ---------------------------------------------------------


@dataclass(frozen=True)
class LiveCorpusStats:
    n_rows: int
    n_cameras: int
    n_series: int
    n_labeled: int
    label_counts: dict[str, int] = field(default_factory=dict)


def corpus_stats(
    rows: Iterable[LiveManifestRow], labels: LabelStore | None = None
) -> LiveCorpusStats:
    rows_list = list(rows)
    cameras: set[str] = {r.camera_id for r in rows_list}
    series: set[tuple[str, str]] = {(r.camera_id, r.series_id) for r in rows_list}
    n_labeled = 0
    counts: dict[str, int] = {c: 0 for c in _ALLOWED_LABELS}
    if labels is not None:
        for r in rows_list:
            lbl = labels.get(r.id)
            if lbl is None:
                continue
            n_labeled += 1
            counts[lbl.image_class] = counts.get(lbl.image_class, 0) + 1
    return LiveCorpusStats(
        n_rows=len(rows_list),
        n_cameras=len(cameras),
        n_series=len(series),
        n_labeled=n_labeled,
        label_counts=counts,
    )
