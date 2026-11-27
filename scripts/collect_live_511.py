#!/usr/bin/env python3
"""FL511 / TX511 snapshot poller (M10 / v0.3 scaffold).

Reads a small YAML config of camera IDs + snapshot URLs, fetches each
on the configured interval, hashes the bytes, writes the image to a
local (gitignored) tree under data/live_511/, and appends a row to
data/live_511/manifest.csv.

Images are NEVER redistributed by this repo. Per plan section 4, we
ship the manifest schema and labels only; the actual bytes stay on
the machine that runs this script under the source operator's own
terms.

Usage:
    # See what would be fetched, no network or writes:
    python scripts/collect_live_511.py --dry-run

    # One-shot fetch of every camera, then exit:
    python scripts/collect_live_511.py --once

    # Poll forever on the configured interval (Ctrl+C to stop):
    python scripts/collect_live_511.py

    # Custom config path:
    python scripts/collect_live_511.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mikha.bench.live import (
    LiveManifestRow,
    append_live_row,
    make_row_id,
    series_key,
)
from mikha.bench.live_hls import HlsCaptureError, capture_frame_from_hls

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "live_cameras.yaml"

# Accepted source_kind values. "jpg" (default) is the original FL511
# behavior: HTTP GET on a direct .jpg URL. "hls" pulls one frame from
# an HLS/M3U8 live stream via ffmpeg (see mikha.bench.live_hls).
_SOURCE_KINDS = ("jpg", "hls")


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    source: str
    url: str
    notes: str = ""
    source_kind: str = "jpg"


@dataclass(frozen=True)
class CollectorConfig:
    interval_seconds: int
    series_bucket_seconds: int
    timeout_seconds: int
    user_agent: str
    manifest_path: Path
    images_root: Path
    cameras: tuple[CameraSpec, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - trivial guard
        raise SystemExit(
            "PyYAML required; install with `pip install pyyaml` or via `.[model]`."
        ) from None
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _load_config(path: Path) -> CollectorConfig:
    raw = _load_yaml(path)
    cameras_raw = raw.get("cameras") or []
    cameras: list[CameraSpec] = []
    for c in cameras_raw:
        if not isinstance(c, dict):
            raise SystemExit(f"[fatal] cameras entry must be a dict, got {c!r}")
        source_kind = str(c.get("source_kind", "jpg"))
        if source_kind not in _SOURCE_KINDS:
            raise SystemExit(
                f"[fatal] camera {c.get('camera_id')!r}: source_kind={source_kind!r} "
                f"not in {_SOURCE_KINDS}"
            )
        cameras.append(
            CameraSpec(
                camera_id=str(c["camera_id"]),
                source=str(c["source"]),
                url=str(c["url"]),
                notes=str(c.get("notes", "")),
                source_kind=source_kind,
            )
        )
    return CollectorConfig(
        interval_seconds=int(raw.get("interval_seconds", 30)),
        series_bucket_seconds=int(raw.get("series_bucket_seconds", 3600)),
        timeout_seconds=int(raw.get("timeout_seconds", 15)),
        user_agent=str(raw.get("user_agent", "Mikha511-Live/0.3 (unset)")),
        manifest_path=REPO_ROOT / str(raw.get("manifest_path", "data/live_511/manifest.csv")),
        images_root=REPO_ROOT / str(raw.get("images_root", "data/live_511")),
        cameras=tuple(cameras),
    )


def _now_utc_rfc3339() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_image(bytes_: bytes, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / filename
    dst.write_bytes(bytes_)
    return dst


def _ext_from_bytes(url: str, body: bytes) -> str:
    # Cheap magic-number sniff; falls back to URL suffix, then .jpg.
    if body.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    for candidate in (".jpg", ".jpeg", ".png"):
        if url.lower().endswith(candidate):
            return candidate
    return ".jpg"


def _record_row(
    camera: CameraSpec,
    cfg: CollectorConfig,
    *,
    body: bytes,
    local_path: Path,
) -> str:
    """Hash bytes, build the manifest row, append it. Returns rel path.

    Shared between the jpg and hls fetchers so both paths write the
    exact same schema. The hash is over the on-disk file bytes, which
    for the hls path are the JPEG that ffmpeg produced.
    """
    import hashlib

    sha256 = hashlib.sha256(body).hexdigest()
    captured_at = _now_utc_rfc3339()
    row_id = make_row_id(camera.camera_id, captured_at)
    # Rename the file to include the row id, so filenames match the
    # existing convention regardless of which fetcher wrote them.
    # Preserve the extension the fetcher chose.
    ext = local_path.suffix or ".jpg"
    final_path = local_path.with_name(f"{row_id}{ext}")
    if final_path != local_path:
        local_path.rename(final_path)
    rel_local = str(final_path.relative_to(REPO_ROOT))

    row = LiveManifestRow(
        id=row_id,
        camera_id=camera.camera_id,
        series_id=series_key(
            camera.camera_id, captured_at, bucket_seconds=cfg.series_bucket_seconds
        ),
        captured_at=captured_at,
        source=camera.source,
        url=camera.url,
        sha256=sha256,
        local_path=rel_local,
        notes=camera.notes,
    )
    append_live_row(row, cfg.manifest_path)
    return rel_local


def _fetch_one_jpg(
    camera: CameraSpec,
    cfg: CollectorConfig,
) -> tuple[str, str]:
    """HTTP GET on a direct .jpg URL. Original FL511 behavior."""
    import httpx

    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        response = httpx.get(
            camera.url,
            headers=headers,
            timeout=cfg.timeout_seconds,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        return "http_error", f"{type(exc).__name__}: {exc}"
    if response.status_code >= 400:
        return "http_error", f"status={response.status_code}"
    body = response.content
    if not body:
        return "http_error", "empty body"

    captured_at_stub = _now_utc_rfc3339()
    stub_row_id = make_row_id(camera.camera_id, captured_at_stub)
    ext = _ext_from_bytes(camera.url, body)
    filename = f"{stub_row_id}{ext}"
    camera_dir = cfg.images_root / camera.camera_id
    local_path = _write_image(body, camera_dir, filename)
    rel_local = _record_row(camera, cfg, body=body, local_path=local_path)
    return "ok", rel_local


def _fetch_one_hls(
    camera: CameraSpec,
    cfg: CollectorConfig,
) -> tuple[str, str]:
    """Grab one frame from an HLS m3u8 stream via ffmpeg (DelDOT et al)."""
    captured_at_stub = _now_utc_rfc3339()
    stub_row_id = make_row_id(camera.camera_id, captured_at_stub)
    camera_dir = cfg.images_root / camera.camera_id
    out_path = camera_dir / f"{stub_row_id}.jpg"
    try:
        result = capture_frame_from_hls(
            camera.url,
            out_path,
            user_agent=cfg.user_agent,
            io_timeout_seconds=cfg.timeout_seconds,
            subprocess_timeout_seconds=max(30, cfg.timeout_seconds * 2),
        )
    except HlsCaptureError as exc:
        return "hls_error", str(exc)
    rel_local = _record_row(camera, cfg, body=result.body, local_path=result.output_path)
    return "ok", rel_local


def _fetch_one(
    camera: CameraSpec,
    cfg: CollectorConfig,
    *,
    dry_run: bool,
) -> tuple[str, str]:
    """Dispatch by source_kind. status is one of:
    "ok" | "dry" | "http_error" | "hls_error".
    """
    if dry_run:
        verb = "would ffmpeg" if camera.source_kind == "hls" else "would GET"
        return "dry", f"{verb} {camera.url}"
    if camera.source_kind == "hls":
        return _fetch_one_hls(camera, cfg)
    return _fetch_one_jpg(camera, cfg)


def _tick(cfg: CollectorConfig, *, dry_run: bool) -> None:
    for cam in cfg.cameras:
        status, detail = _fetch_one(cam, cfg, dry_run=dry_run)
        print(f"[{status:10s}] {cam.camera_id}  {detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live 511 snapshot poller (M10).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched; no network, no writes.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Fetch every camera one time then exit (no polling loop).",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[fatal] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = _load_config(cfg_path)

    if not cfg.cameras:
        print(
            "[warn] config has no cameras. Populate configs/live_cameras.yaml "
            "after reviewing ToS (see data/live_511/README.md).",
            file=sys.stderr,
        )
        return 0

    print(
        f"[live] {len(cfg.cameras)} camera(s)  interval={cfg.interval_seconds}s  "
        f"manifest={cfg.manifest_path}"
    )
    if args.dry_run or args.once:
        _tick(cfg, dry_run=args.dry_run)
        return 0

    try:
        while True:
            _tick(cfg, dry_run=False)
            time.sleep(cfg.interval_seconds)
    except KeyboardInterrupt:
        print("\n[live] stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
