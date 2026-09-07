#!/usr/bin/env python3
"""Audit ``data/images/`` against ``data/manifest.csv``.

Answers four questions in one pass, without modifying anything:

1. Do all N manifest rows have a corresponding local file?
2. Are all local files present and readable (nonzero size)?
3. Do all local SHA-256s match the manifest?
4. Which rows would ``fetch_base.py`` have reported as ``skip_exists`` —
   i.e. which files existed on disk before the most recent fetch?

For (4) we use mtime clustering: any file whose mtime falls before a natural
gap in the mtime distribution is treated as pre-existing. This is heuristic
but usually clean-cut because a bulk ``fetch_base.py`` run writes many files
within a few seconds of each other and any pre-existing file has a much
older stat time.

Also reports orphan files in ``data/images/`` that have no manifest row.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from mikha.bench import ManifestRow, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_IMAGES = REPO_ROOT / "data" / "images"


def _local_path(row: ManifestRow, images_dir: Path) -> Path:
    """Mirror the extension-inference logic in scripts/fetch_base.py."""
    lower = row.url.lower()
    for ext in (".jpg", ".jpeg", ".png"):
        if lower.endswith(ext):
            return images_dir / f"{row.id}{ext}"
    return images_dir / f"{row.id}.jpg"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit data/images/ against manifest.csv.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument(
        "--pre-fetch-gap-seconds",
        type=int,
        default=120,
        help="A file whose mtime is more than N seconds older than the median mtime is "
        "treated as pre-existing (skip_exists candidate). Default 120.",
    )
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest)
    images_dir = Path(args.images)
    print(f"[audit] manifest = {args.manifest}  ({len(rows)} rows)")
    print(f"[audit] images   = {images_dir}")

    missing: list[str] = []
    unreadable: list[str] = []
    zero_byte: list[str] = []
    hash_ok: list[tuple[str, Path, float]] = []
    hash_mismatch: list[tuple[str, str, str]] = []

    for row in rows:
        p = _local_path(row, images_dir)
        if not p.exists():
            missing.append(row.id)
            continue
        try:
            st = p.stat()
        except OSError:
            unreadable.append(row.id)
            continue
        if st.st_size == 0:
            zero_byte.append(row.id)
            continue
        actual = _sha256(p)
        if actual != row.sha256:
            hash_mismatch.append((row.id, row.sha256, actual))
            continue
        hash_ok.append((row.id, p, st.st_mtime))

    # Orphan files not represented by any manifest row.
    if images_dir.exists():
        expected = {_local_path(r, images_dir).name for r in rows}
        on_disk = {p.name for p in images_dir.iterdir() if p.is_file()}
        orphans = sorted(on_disk - expected)
    else:
        orphans = []

    print("\n" + "=" * 66)
    print("AUDIT RESULTS")
    print("=" * 66)
    print(f"manifest rows:                  {len(rows)}")
    print(f"local files present + OK hash:  {len(hash_ok)}")
    print(f"missing files:                  {len(missing)}   {missing[:8]}")
    print(f"unreadable stat():              {len(unreadable)} {unreadable[:8]}")
    print(f"zero-byte files:                {len(zero_byte)} {zero_byte[:8]}")
    print(
        f"hash MISMATCH (integrity FAIL): {len(hash_mismatch)}   "
        f"{[m[0] for m in hash_mismatch[:8]]}"
    )
    print(f"orphan files in images/ dir:    {len(orphans)}   {orphans[:8]}")

    # Q4: which rows would have been reported as `skip_exists` in the last fetch?
    if hash_ok:
        mtimes = sorted(t for _, _, t in hash_ok)
        median = statistics.median(mtimes)
        cutoff = median - args.pre_fetch_gap_seconds
        pre_existing = [(row_id, p, t) for row_id, p, t in hash_ok if t < cutoff]
        pre_existing.sort(key=lambda x: x[2])

        print("\n--- mtime distribution (of OK files) ---")
        print(f"  oldest   : {_fmt_mtime(mtimes[0])}")
        print(f"  median   : {_fmt_mtime(median)}")
        print(f"  newest   : {_fmt_mtime(mtimes[-1])}")
        print(f"  cutoff   : {_fmt_mtime(cutoff)}  (median - {args.pre_fetch_gap_seconds}s)")
        print(f"\nPre-existing files (mtime older than cutoff): {len(pre_existing)}")
        print("These are the rows fetch_base.py would report as `skip_exists`:")
        for row_id, p, t in pre_existing:
            print(f"  {row_id}  mtime={_fmt_mtime(t)}  {p.name}")

    hard_fail = bool(missing or unreadable or zero_byte or hash_mismatch)
    verdict = (
        "integrity issues found"
        if hard_fail
        else f"all {len(rows)} files present, readable, hashes verified"
    )
    print(f"\n[audit] {'FAIL' if hard_fail else 'OK'} — {verdict}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
