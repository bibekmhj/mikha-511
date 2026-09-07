#!/usr/bin/env python3
"""Fetch the Mikha-Bench base images referenced by ``data/manifest.csv``.

For every row:

* download bytes from ``url``;
* verify SHA-256 matches ``sha256``;
* write to ``data/images/<id>.<ext>`` (extension inferred from URL).

Reports OK / SKIP_EXISTS / MISMATCH / HTTP-ERROR per row.
Never overwrites without --force. Never modifies the manifest.

Rate-limit handling: retries on HTTP 429/503 with exponential backoff,
honoring ``Retry-After`` when present. Polite sleep between requests.

Wikimedia's User-Agent policy requires a specific tool name plus contact
info; requests without it get 403. This client sends a compliant UA.
See: https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from mikha.bench import ManifestRow, hash_bytes, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "images"

# Wikimedia (and many other well-behaved hosts) require a tool-specific UA
# with a reach-back. Default httpx UA gets 403 on upload.wikimedia.org.
DEFAULT_UA = "Mikha511/0.1 (https://github.com/bibekmhj/mikha-511; bbkmhj06@gmail.com) httpx"
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_UA,
    "Accept": "image/*,*/*;q=0.8",
    "Api-User-Agent": DEFAULT_UA,
    "From": "bbkmhj06@gmail.com",
}


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single fetch attempt."""

    row: ManifestRow
    status: str  # "ok" | "skip_exists" | "mismatch" | "http_error"
    detail: str


def _ext_from_url(url: str) -> str:
    for candidate in (".jpg", ".jpeg", ".png"):
        if url.lower().endswith(candidate):
            return candidate
    # Default to .jpg — 511 traffic-camera imagery is overwhelmingly JPEG.
    return ".jpg"


def fetch_one(
    row: ManifestRow,
    out_dir: Path,
    *,
    client: httpx.Client,
    force: bool = False,
    timeout: float = 30.0,
    max_retries: int = 5,
) -> FetchResult:
    """Fetch one manifest row's bytes with retry-on-429/503.

    On HTTP 429 or 503 we back off and retry up to ``max_retries`` extra
    times, honoring ``Retry-After`` when the server sets it. Other HTTP
    errors and network errors return a single ``http_error`` result.
    """
    dst = out_dir / f"{row.id}{_ext_from_url(row.url)}"
    if dst.exists() and not force:
        return FetchResult(row, "skip_exists", str(dst))

    last_err: str | None = None
    response: httpx.Response | None = None

    for attempt in range(max_retries + 1):
        try:
            r = client.get(row.url, timeout=timeout, follow_redirects=True)
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30.0, 2.0**attempt))
            continue

        if r.status_code in (429, 503):
            retry_after = r.headers.get("retry-after", "")
            wait = float(retry_after) if retry_after.isdigit() else min(60.0, 2.0**attempt)
            last_err = f"HTTP {r.status_code} (attempt {attempt + 1}/{max_retries + 1})"
            time.sleep(wait)
            continue

        if r.status_code >= 400:
            body_preview = r.text[:120].replace("\n", " ")
            last_err = f"status={r.status_code} body={body_preview!r}"
            break

        response = r
        break

    if response is None:
        return FetchResult(row, "http_error", last_err or "unknown error")

    actual = hash_bytes(response.content)
    if actual != row.sha256:
        return FetchResult(
            row, "mismatch", f"expected {row.sha256} got {actual} ({len(response.content)}b)"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(response.content)
    return FetchResult(row, "ok", str(dst))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch Mikha-Bench base images from the manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for images.")
    ap.add_argument("--force", action="store_true", help="Re-download existing files.")
    ap.add_argument("--dry-run", action="store_true", help="List actions without downloading.")
    ap.add_argument(
        "--sleep-ms",
        type=int,
        default=1000,
        help="Polite delay between successful requests (ms). Default 1000.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry attempts per image on HTTP 429/503 or network errors. Default 5.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, fetch at most this many rows (for smoke-testing). Default 0 = all.",
    )
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    out_dir = Path(args.out)

    if args.dry_run:
        for row in rows:
            dst = out_dir / f"{row.id}{_ext_from_url(row.url)}"
            action = "SKIP" if dst.exists() and not args.force else "FETCH"
            print(f"[{action}] {row.id}  {row.url}")
        return 0

    results: list[FetchResult] = []
    with httpx.Client(headers=DEFAULT_HEADERS, timeout=60.0) as client:
        for i, row in enumerate(rows, 1):
            result = fetch_one(
                row, out_dir, client=client, force=args.force, max_retries=args.max_retries
            )
            results.append(result)
            print(f"[{result.status:12s}] {row.id}  {result.detail}")
            # Polite delay only after we actually hit the network.
            if result.status not in ("skip_exists",):
                time.sleep(args.sleep_ms / 1000.0)
            if i % 25 == 0:
                ok_so_far = sum(1 for r in results if r.status == "ok")
                err_so_far = sum(1 for r in results if r.status == "http_error")
                print(f"    ─── progress {i}/{len(rows)}  ok={ok_so_far}  err={err_so_far}")

    counts = {
        status: sum(1 for r in results if r.status == status)
        for status in ("ok", "skip_exists", "mismatch", "http_error")
    }
    print(
        f"\nSummary: ok={counts['ok']} skipped={counts['skip_exists']} "
        f"mismatch={counts['mismatch']} http_error={counts['http_error']} "
        f"total={len(results)}"
    )
    # Nonzero exit only on hash mismatches — an integrity failure is fatal.
    # HTTP failures are recoverable (network hiccup, upstream down) and do
    # not invalidate the manifest itself.
    return 1 if counts["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
