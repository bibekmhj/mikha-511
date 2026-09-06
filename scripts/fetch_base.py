#!/usr/bin/env python3
"""Fetch the Mikha-Bench base images referenced by ``data/manifest.csv``.

For every row:

* download bytes from ``url``;
* verify SHA-256 matches ``sha256``;
* write to ``data/images/<id>.<ext>`` (extension inferred from URL).

Reports OK / MISMATCH / MISSING per row. Never overwrites without --force.
The network is the only side effect; nothing is committed to git.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from mikha.bench import ManifestRow, hash_bytes, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "images"


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
) -> FetchResult:
    dst = out_dir / f"{row.id}{_ext_from_url(row.url)}"
    if dst.exists() and not force:
        return FetchResult(row, "skip_exists", str(dst))

    try:
        response = client.get(row.url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return FetchResult(row, "http_error", f"{type(exc).__name__}: {exc}")

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
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest)
    out_dir = Path(args.out)

    if args.dry_run:
        for row in rows:
            dst = out_dir / f"{row.id}{_ext_from_url(row.url)}"
            action = "SKIP" if dst.exists() and not args.force else "FETCH"
            print(f"[{action}] {row.id}  {row.url}")
        return 0

    results: list[FetchResult] = []
    with httpx.Client() as client:
        for row in rows:
            result = fetch_one(row, out_dir, client=client, force=args.force)
            results.append(result)
            print(f"[{result.status:12s}] {row.id}  {result.detail}")

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
