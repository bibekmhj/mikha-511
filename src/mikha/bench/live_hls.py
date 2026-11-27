"""HLS frame capture for the live 511 corpus (M10.1).

DelDOT and some other 511 operators expose their traffic cameras as
HLS (M3U8) live streams rather than direct JPEG snapshots. This module
grabs a single frame from an HLS URL by shelling out to system
``ffmpeg`` and hands the JPEG bytes back to the same code path the
JPG collector already uses.

ffmpeg is a system dependency, not a pip package. On macOS install
with ``brew install ffmpeg``; on Debian / Ubuntu ``apt install
ffmpeg``. See ``data/live_511/README.md`` for the full setup section.

Design notes:

* Pure subprocess shell-out, no ffmpeg Python bindings. Keeps the
  dependency surface at "ffmpeg exists on PATH".
* One frame per invocation. ``-frames:v 1`` selects the first video
  frame of the current HLS chunk at the moment ffmpeg opens the
  stream. This is the sampling model documented in
  ``docs/live_511.md`` and is what a reproducer needs to know.
* Explicit ``-timeout`` on the input plus a Python-side subprocess
  timeout so a stalled stream cannot hang the collector.
* Writes to the caller's chosen output path, then reads back the
  bytes so the collector can hash them with the same function it
  uses for JPG snapshots.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class HlsCaptureError(RuntimeError):
    """Raised when a single-frame HLS capture fails for any reason.

    Concrete reasons: ffmpeg not installed on PATH, ffmpeg exited
    non-zero, ffmpeg exited zero but wrote no output file, or the
    subprocess timed out. The message names which case fired.
    """


@dataclass(frozen=True)
class HlsCaptureResult:
    """Return value of :func:`capture_frame_from_hls`."""

    output_path: Path
    body: bytes
    ffmpeg_stderr: str  # kept for diagnostics; may be empty


def ffmpeg_available() -> bool:
    """True if ``ffmpeg`` is on PATH. Used by tests and startup checks."""
    return shutil.which("ffmpeg") is not None


def build_ffmpeg_command(
    url: str,
    output_path: Path,
    *,
    user_agent: str,
    io_timeout_seconds: int,
) -> list[str]:
    """Build the ffmpeg argv list. Extracted so tests can assert on it.

    * ``-y`` overwrite the output file if it already exists.
    * ``-nostdin`` never read from stdin (would block a background poll).
    * ``-loglevel error`` keep the log noise down; we only care about
      failures.
    * ``-user_agent`` propagate the collector's UA to the origin so
      the operator sees the same "who is this" string as the JPG path.
    * ``-timeout`` (in microseconds) applies to the input read; a
      stalled HLS chunk gets killed at the ffmpeg level, not just by
      the Python subprocess timeout.
    * ``-frames:v 1`` grab one frame from the first video stream.
    * ``-q:v 2`` MJPEG quality 2 (high; roughly 90-95% JPEG quality).
    * ``-f image2`` force single-image muxer (avoids container guessing).
    """
    return [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-user_agent",
        user_agent,
        "-timeout",
        str(io_timeout_seconds * 1_000_000),
        "-i",
        url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-f",
        "image2",
        str(output_path),
    ]


def capture_frame_from_hls(
    url: str,
    output_path: str | Path,
    *,
    user_agent: str = "Mikha511-Live/0.3 (unset)",
    io_timeout_seconds: int = 15,
    subprocess_timeout_seconds: int = 30,
) -> HlsCaptureResult:
    """Grab one JPEG frame from an HLS ``m3u8`` URL via ``ffmpeg``.

    Raises :class:`HlsCaptureError` on:

    * ffmpeg not installed on PATH;
    * ffmpeg exited non-zero (message includes captured stderr);
    * subprocess timed out after ``subprocess_timeout_seconds``;
    * ffmpeg exited zero but no output file was written (or file is empty).

    The output file is created (parent directories are made as needed)
    and its bytes are read back and returned so the caller can hash
    them and write a manifest row alongside.
    """
    output_path = Path(output_path)
    if not ffmpeg_available():
        raise HlsCaptureError(
            "ffmpeg not found on PATH. Install with `brew install ffmpeg` "
            "on macOS or `apt install ffmpeg` on Debian/Ubuntu. See "
            "data/live_511/README.md for the HLS setup section."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_command(
        url,
        output_path,
        user_agent=user_agent,
        io_timeout_seconds=io_timeout_seconds,
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HlsCaptureError(
            f"ffmpeg timed out after {subprocess_timeout_seconds}s for {url!r}"
        ) from exc

    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.returncode != 0:
        raise HlsCaptureError(
            f"ffmpeg exited {proc.returncode} for {url!r}. stderr: {stderr_text.strip()[:400]}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HlsCaptureError(
            f"ffmpeg exited 0 but wrote no data for {url!r} (expected {output_path})"
        )
    body = output_path.read_bytes()
    return HlsCaptureResult(output_path=output_path, body=body, ffmpeg_stderr=stderr_text)
