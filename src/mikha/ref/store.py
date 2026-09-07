"""SQLite event storage for Mikha-Ref.

One table, ``events``, capturing every state transition and every
periodic heartbeat the poller emits. Deliberately small — the M5
dashboard reads from this table and nothing else.

Schema:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    ts_utc          TEXT     ISO-8601 UTC
    camera_id       TEXT     free-form; matches manifest style
    label           TEXT     one of DecisionLabel.value
    calibrated_conf REAL     0.0–1.0
    mask_area_frac  REAL     0.0–1.0
    is_transition   INTEGER  1 iff raise or clear fired this row
    note            TEXT     free-form (e.g. HTTP error message)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    label TEXT NOT NULL,
    calibrated_conf REAL NOT NULL,
    mask_area_frac REAL NOT NULL,
    is_transition INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events (camera_id);
"""


@dataclass(frozen=True)
class EventRow:
    id: int
    ts_utc: str
    camera_id: str
    label: str
    calibrated_conf: float
    mask_area_frac: float
    is_transition: bool
    note: str


class EventStore:
    """SQLite-backed event log. One instance per running API process."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------

    def append(
        self,
        *,
        camera_id: str,
        label: str,
        calibrated_conf: float,
        mask_area_frac: float,
        is_transition: bool,
        note: str = "",
        ts_utc: str | None = None,
    ) -> int:
        ts = ts_utc or datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events (ts_utc, camera_id, label, calibrated_conf, "
                "mask_area_frac, is_transition, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    camera_id,
                    label,
                    float(calibrated_conf),
                    float(mask_area_frac),
                    1 if is_transition else 0,
                    note,
                ),
            )
            row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("failed to insert event row")
        return int(row_id)

    def recent(self, *, limit: int = 50, camera_id: str | None = None) -> list[EventRow]:
        params: list[object] = []
        sql = "SELECT * FROM events"
        if camera_id is not None:
            sql += " WHERE camera_id = ?"
            params.append(camera_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def latest_per_camera(self) -> list[EventRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.* FROM events e JOIN ("
                "  SELECT camera_id, MAX(id) AS max_id FROM events GROUP BY camera_id"
                ") m ON m.max_id = e.id ORDER BY e.camera_id"
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(n)

    def wipe(self) -> None:
        """Test helper — delete every row."""
        with self._connect() as conn:
            conn.execute("DELETE FROM events")


def _row_to_event(r: Iterable) -> EventRow:
    return EventRow(
        id=r["id"],
        ts_utc=r["ts_utc"],
        camera_id=r["camera_id"],
        label=r["label"],
        calibrated_conf=r["calibrated_conf"],
        mask_area_frac=r["mask_area_frac"],
        is_transition=bool(r["is_transition"]),
        note=r["note"] or "",
    )
