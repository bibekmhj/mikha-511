"""Tests for the FastAPI dashboard app."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mikha.ref.api import create_app  # noqa: E402


def _client(tmp_path: Path) -> TestClient:
    app = create_app(store_path=tmp_path / "events.sqlite")
    return TestClient(app)


def test_health_returns_zero_events(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "n_events": 0}


def test_index_renders_dashboard(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/")
        assert r.status_code == 200
        # dashboard.html mounts htmx and has a table body with hx-get
        assert "Mikha-Ref" in r.text
        assert "hx-get" in r.text


def test_submit_and_events_roundtrip(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        payload = {
            "camera_id": "cam-A",
            "label": "flooded",
            "calibrated_conf": 0.88,
            "mask_area_frac": 0.42,
            "is_transition": True,
            "note": "raise",
        }
        r = c.post("/api/submit", json=payload)
        assert r.status_code == 200
        assert r.json()["id"] == 1

        # HTMX endpoint returns a table body
        r = c.get("/api/events?limit=10")
        assert r.status_code == 200
        assert "flooded" in r.text
        assert "cam-A" in r.text

        # JSON endpoint returns a list of dicts
        r = c.get("/api/events.json?limit=10")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam-A"
        assert rows[0]["is_transition"] is True


def test_submit_rejects_missing_fields(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.post("/api/submit", json={"camera_id": "cam-A"})
        assert r.status_code == 400


def test_limit_out_of_range(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/api/events?limit=0")
        assert r.status_code == 400
        r = c.get("/api/events?limit=99999")
        assert r.status_code == 400


def test_latest_per_camera(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        for label in ("passable", "flooded_uncertain", "flooded"):
            c.post(
                "/api/submit",
                json={
                    "camera_id": "cam-A",
                    "label": label,
                    "calibrated_conf": 0.5,
                    "mask_area_frac": 0.1,
                },
            )
        r = c.get("/api/latest")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["label"] == "flooded"
