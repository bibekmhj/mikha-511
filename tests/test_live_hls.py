"""Tests for the HLS frame-capture path (M10.1).

The pure-jpg FL511 collector is untouched by M10.1 and its behavior is
already tested in tests/test_live.py plus the existing scripts test
matrix. These tests cover:

* the ffmpeg argv the helper builds (asserts flags and order),
* the four failure modes of capture_frame_from_hls (missing ffmpeg,
  non-zero exit, timeout, empty output),
* dispatch by source_kind in the collector script,
* backwards compat: a yaml entry without source_kind still parses as jpg.

No live streams are hit. The optional end-to-end test that would need
a real M3U8 is left out on purpose - it belongs in the reproducer
docs, not in CI.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mikha.bench.live_hls import (
    HlsCaptureError,
    build_ffmpeg_command,
    capture_frame_from_hls,
    ffmpeg_available,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECT_SCRIPT = REPO_ROOT / "scripts" / "collect_live_511.py"


def _load_collect_module():
    """Import scripts/collect_live_511.py as a module for testing.

    Scripts live outside the src/ tree so pytest cannot import them by
    package path. Load explicitly through importlib.
    """
    spec = importlib.util.spec_from_file_location("collect_live_511", COLLECT_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_live_511"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# build_ffmpeg_command


def test_ffmpeg_command_has_expected_flags(tmp_path: Path) -> None:
    out = tmp_path / "frame.jpg"
    cmd = build_ffmpeg_command(
        "http://video.deldot.gov:1935/live/SCAM013.stream/playlist.m3u8",
        out,
        user_agent="Mikha511-Live/0.3 (mailto:bbkmhj06@gmail.com)",
        io_timeout_seconds=15,
    )
    # Program name first
    assert cmd[0] == "ffmpeg"
    # Overwrite + no-stdin + quiet
    assert "-y" in cmd
    assert "-nostdin" in cmd
    # User-Agent gets propagated through
    ua_idx = cmd.index("-user_agent")
    assert cmd[ua_idx + 1] == "Mikha511-Live/0.3 (mailto:bbkmhj06@gmail.com)"
    # io timeout is in microseconds
    to_idx = cmd.index("-timeout")
    assert cmd[to_idx + 1] == "15000000"
    # Single frame, image2 muxer, quality 2
    fv_idx = cmd.index("-frames:v")
    assert cmd[fv_idx + 1] == "1"
    q_idx = cmd.index("-q:v")
    assert cmd[q_idx + 1] == "2"
    fmt_idx = cmd.index("-f")
    assert cmd[fmt_idx + 1] == "image2"
    # Input URL and output path present
    i_idx = cmd.index("-i")
    assert cmd[i_idx + 1].endswith("playlist.m3u8")
    assert cmd[-1] == str(out)


# ---------------------------------------------------------------------------
# capture_frame_from_hls failure modes (all mocked; no real subprocess)


def test_capture_raises_if_ffmpeg_not_on_path(tmp_path: Path) -> None:
    with (
        patch("mikha.bench.live_hls.ffmpeg_available", return_value=False),
        pytest.raises(HlsCaptureError, match="ffmpeg not found on PATH"),
    ):
        capture_frame_from_hls("http://x/y.m3u8", tmp_path / "out.jpg")


def test_capture_raises_on_nonzero_exit(tmp_path: Path) -> None:
    fake = MagicMock()
    fake.returncode = 1
    fake.stderr = b"Server returned 404 Not Found"
    with (
        patch("mikha.bench.live_hls.ffmpeg_available", return_value=True),
        patch("mikha.bench.live_hls.subprocess.run", return_value=fake),
        pytest.raises(HlsCaptureError, match="ffmpeg exited 1"),
    ):
        capture_frame_from_hls("http://x/y.m3u8", tmp_path / "out.jpg")


def test_capture_raises_on_timeout(tmp_path: Path) -> None:
    with (
        patch("mikha.bench.live_hls.ffmpeg_available", return_value=True),
        patch(
            "mikha.bench.live_hls.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30),
        ),
        pytest.raises(HlsCaptureError, match="timed out"),
    ):
        capture_frame_from_hls(
            "http://x/y.m3u8",
            tmp_path / "out.jpg",
            subprocess_timeout_seconds=30,
        )


def test_capture_raises_when_ffmpeg_writes_no_output(tmp_path: Path) -> None:
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = b""
    # ffmpeg "succeeded" but never touched the output path
    with (
        patch("mikha.bench.live_hls.ffmpeg_available", return_value=True),
        patch("mikha.bench.live_hls.subprocess.run", return_value=fake),
        pytest.raises(HlsCaptureError, match="wrote no data"),
    ):
        capture_frame_from_hls("http://x/y.m3u8", tmp_path / "out.jpg")


def test_capture_returns_bytes_on_success(tmp_path: Path) -> None:
    out = tmp_path / "out.jpg"

    def _fake_run(cmd, capture_output, timeout, check):  # noqa: ANN001, ARG001
        # Simulate ffmpeg writing a JPEG-like file
        out.write_bytes(b"\xff\xd8\xff\xe0fake jpeg body")
        m = MagicMock()
        m.returncode = 0
        m.stderr = b""
        return m

    with (
        patch("mikha.bench.live_hls.ffmpeg_available", return_value=True),
        patch("mikha.bench.live_hls.subprocess.run", side_effect=_fake_run),
    ):
        result = capture_frame_from_hls("http://x/y.m3u8", out)
    assert result.output_path == out
    assert result.body.startswith(b"\xff\xd8\xff")
    assert out.exists()


# ---------------------------------------------------------------------------
# Collector dispatch: source_kind decides which fetcher runs


def test_collector_dispatches_hls_when_source_kind_hls(tmp_path: Path) -> None:
    mod = _load_collect_module()
    cam = mod.CameraSpec(
        camera_id="deldot_scam013",
        source="deldot",
        url="http://video.deldot.gov:1935/live/SCAM013.stream/playlist.m3u8",
        notes="",
        source_kind="hls",
    )
    cfg = mod.CollectorConfig(
        interval_seconds=30,
        series_bucket_seconds=3600,
        timeout_seconds=15,
        user_agent="ua",
        manifest_path=tmp_path / "manifest.csv",
        images_root=tmp_path / "images",
        cameras=(cam,),
    )
    with (
        patch.object(mod, "_fetch_one_hls", return_value=("ok", "hls-path")) as m_hls,
        patch.object(mod, "_fetch_one_jpg", return_value=("ok", "jpg-path")) as m_jpg,
    ):
        status, detail = mod._fetch_one(cam, cfg, dry_run=False)
    assert status == "ok"
    assert detail == "hls-path"
    m_hls.assert_called_once()
    m_jpg.assert_not_called()


def test_collector_dispatches_jpg_by_default(tmp_path: Path) -> None:
    mod = _load_collect_module()
    # No source_kind kwarg -> defaults to "jpg"
    cam = mod.CameraSpec(
        camera_id="fl511_x",
        source="fl511",
        url="https://example.org/x.jpg",
        notes="",
    )
    assert cam.source_kind == "jpg"
    cfg = mod.CollectorConfig(
        interval_seconds=30,
        series_bucket_seconds=3600,
        timeout_seconds=15,
        user_agent="ua",
        manifest_path=tmp_path / "manifest.csv",
        images_root=tmp_path / "images",
        cameras=(cam,),
    )
    with (
        patch.object(mod, "_fetch_one_hls", return_value=("ok", "hls-path")) as m_hls,
        patch.object(mod, "_fetch_one_jpg", return_value=("ok", "jpg-path")) as m_jpg,
    ):
        status, detail = mod._fetch_one(cam, cfg, dry_run=False)
    assert status == "ok"
    assert detail == "jpg-path"
    m_jpg.assert_called_once()
    m_hls.assert_not_called()


def test_collector_dry_run_labels_hls_vs_jpg(tmp_path: Path) -> None:
    mod = _load_collect_module()
    jpg_cam = mod.CameraSpec(camera_id="a", source="fl511", url="https://x/y.jpg", notes="")
    hls_cam = mod.CameraSpec(
        camera_id="b",
        source="deldot",
        url="http://x/y.m3u8",
        notes="",
        source_kind="hls",
    )
    cfg = mod.CollectorConfig(
        interval_seconds=30,
        series_bucket_seconds=3600,
        timeout_seconds=15,
        user_agent="ua",
        manifest_path=tmp_path / "m.csv",
        images_root=tmp_path / "img",
        cameras=(jpg_cam, hls_cam),
    )
    _, jpg_detail = mod._fetch_one(jpg_cam, cfg, dry_run=True)
    _, hls_detail = mod._fetch_one(hls_cam, cfg, dry_run=True)
    assert "would GET" in jpg_detail
    assert "would ffmpeg" in hls_detail


def test_collector_rejects_unknown_source_kind_in_config(tmp_path: Path) -> None:
    """A yaml with source_kind: mystery must fail fast at config load."""
    mod = _load_collect_module()
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "interval_seconds: 30\n"
        "series_bucket_seconds: 3600\n"
        "timeout_seconds: 15\n"
        "user_agent: ua\n"
        "manifest_path: data/live_511/manifest.csv\n"
        "images_root: data/live_511\n"
        "cameras:\n"
        "  - camera_id: bad\n"
        "    source: xyz\n"
        "    source_kind: mystery\n"
        "    url: https://x/y.jpg\n"
    )
    with pytest.raises(SystemExit, match="source_kind"):
        mod._load_config(cfg_path)


# ---------------------------------------------------------------------------
# Utility


def test_ffmpeg_available_is_a_bool() -> None:
    # Whether ffmpeg is present in the test environment or not,
    # the helper must return a plain bool without raising.
    assert isinstance(ffmpeg_available(), bool)
