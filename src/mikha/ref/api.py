"""FastAPI application for Mikha-Ref.

Serves a minimal HTMX dashboard plus a REST endpoint the poller loop
calls to submit a decision. Keeps state in :class:`EventStore` (SQLite).

Deliberately not concurrent: one process, one camera in v0.1. Multi-
camera + concurrency lands in v0.2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .store import EventStore

UI_DIR = Path(__file__).resolve().parent / "ui"


def create_app(store_path: str | Path | None = None) -> FastAPI:
    """Build a FastAPI app. Store path defaults to $MIKHA_STORE or events.sqlite."""
    path = store_path or os.environ.get("MIKHA_STORE", "events.sqlite")
    store = EventStore(path)

    app = FastAPI(
        title="Mikha-Ref v0.1",
        description="Reference implementation dashboard (one camera).",
    )
    app.state.store = store

    if UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html_path = UI_DIR / "dashboard.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>Mikha-Ref</h1><p>Dashboard template missing.</p>"

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "n_events": app.state.store.count()}

    @app.get("/api/latest")
    def latest() -> JSONResponse:
        rows = app.state.store.latest_per_camera()
        return JSONResponse([_serialize(r) for r in rows])

    @app.get("/api/events", response_class=HTMLResponse)
    def events_htmx(limit: int = 50, camera_id: str | None = None) -> str:
        if limit <= 0 or limit > 500:
            raise HTTPException(status_code=400, detail="limit must be in [1, 500]")
        rows = app.state.store.recent(limit=limit, camera_id=camera_id)
        return _render_events_table(rows)

    @app.get("/api/events.json")
    def events_json(limit: int = 50, camera_id: str | None = None) -> JSONResponse:
        if limit <= 0 or limit > 500:
            raise HTTPException(status_code=400, detail="limit must be in [1, 500]")
        rows = app.state.store.recent(limit=limit, camera_id=camera_id)
        return JSONResponse([_serialize(r) for r in rows])

    @app.post("/api/submit")
    async def submit(request: Request) -> dict[str, Any]:
        payload = await request.json()
        required = ("camera_id", "label", "calibrated_conf", "mask_area_frac")
        missing = [k for k in required if k not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"missing fields: {missing}")
        row_id = app.state.store.append(
            camera_id=str(payload["camera_id"]),
            label=str(payload["label"]),
            calibrated_conf=float(payload["calibrated_conf"]),
            mask_area_frac=float(payload["mask_area_frac"]),
            is_transition=bool(payload.get("is_transition", False)),
            note=str(payload.get("note", "")),
        )
        return {"id": row_id}


def _serialize(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "ts_utc": row.ts_utc,
        "camera_id": row.camera_id,
        "label": row.label,
        "calibrated_conf": round(row.calibrated_conf, 4),
        "mask_area_frac": round(row.mask_area_frac, 4),
        "is_transition": row.is_transition,
        "note": row.note,
    }


def _render_events_table(rows: list[Any]) -> str:
    if not rows:
        return "<tr><td colspan='6'><em>No events yet.</em></td></tr>"
    parts: list[str] = []
    for r in rows:
        badge = _label_badge(r.label)
        tx = "★" if r.is_transition else ""
        parts.append(
            "<tr>"
            f"<td class='ts'>{_escape(r.ts_utc)}</td>"
            f"<td>{_escape(r.camera_id)}</td>"
            f"<td>{badge}</td>"
            f"<td class='num'>{r.calibrated_conf:.2f}</td>"
            f"<td class='num'>{r.mask_area_frac * 100:.1f}%</td>"
            f"<td class='tx'>{tx}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def _label_badge(label: str) -> str:
    colors = {
        "flooded": "#c0392b",
        "flooded_uncertain": "#e67e22",
        "passable": "#27ae60",
        "camera_offline": "#7f8c8d",
    }
    color = colors.get(label, "#333")
    return f"<span class='badge' style='background:{color}'>{_escape(label)}</span>"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Uvicorn entrypoint: `uvicorn mikha.ref.api:app`
app = create_app()
