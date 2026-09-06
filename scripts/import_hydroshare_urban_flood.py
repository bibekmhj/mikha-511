#!/usr/bin/env python3
"""Import the Chen et al. Urban Flood Image Dataset (HydroShare) as the
held-out validation subset.

This importer takes a LOCAL PATH to the archive you downloaded from
HydroShare after accepting its CC-BY-4.0 terms. The archive is not fetched
by this script — HydroShare requires an interactive terms acceptance.

For every image inside the archive it appends a row with:

* ``source = "hydroshare_urban_flood"``
* ``license = "CC-BY-4.0"`` (as declared on the HydroShare resource page)
* ``attribution`` naming Chen et al. and both ODU + WebCOOS sources
* ``image_class`` inferred from the folder path if possible; caller can
  override with ``--flood-glob`` / ``--nonflood-glob``

**Redistribution caveat:** The paper does not explicitly document an ODU or
SECOORA/WebCOOS grant to Chen et al. authorizing CC-BY redistribution.
We include the dataset because HydroShare's uploader declared CC-BY, but we
do NOT rehost the images. The manifest ships URL + SHA-256 pointing at the
HydroShare resource; reproducers download bytes from HydroShare directly.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from mikha.bench import (
    ManifestError,
    ManifestRow,
    hash_bytes,
    load_manifest,
    save_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"

HYDROSHARE_RESOURCE_URL = "https://www.hydroshare.org/resource/24866122a6ee456c8f7c80aa87a9abcb/"
SOURCE_TAG = "hydroshare_urban_flood"
LICENSE_TAG = "CC-BY-4.0"
ATTRIBUTION = (
    "Chen, He, et al. Urban Flood Image Dataset (CUAHSI HydroShare, 2023). "
    "Contains ODU-Norfolk campus flood imagery (2021-08-16) and "
    "WebCOOS-Charleston Hurricane Ian imagery (2022-09-30)."
)


def _next_id(existing: list[ManifestRow]) -> int:
    if not existing:
        return 1
    ids = []
    for row in existing:
        try:
            ids.append(int(row.id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(ids) + 1) if ids else 1


def _iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file())


def _classify(path: Path, flood_globs: list[str], nonflood_globs: list[str]) -> str | None:
    rel = str(path.as_posix())
    for pat in flood_globs:
        if fnmatch.fnmatch(rel, pat):
            return "flood"
    for pat in nonflood_globs:
        if fnmatch.fnmatch(rel, pat):
            return "nonflood"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import HydroShare Urban Flood Image Dataset.")
    ap.add_argument(
        "--src",
        required=True,
        help="Path to the extracted HydroShare archive on your local machine.",
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    ap.add_argument(
        "--flood-glob",
        action="append",
        default=None,
        help=(
            "Glob (POSIX) matched against each image's relative path to tag it as flood. "
            "May be given multiple times. Default: **/flood*/** and **/*flood*.*"
        ),
    )
    ap.add_argument(
        "--nonflood-glob",
        action="append",
        default=None,
        help=(
            "Glob matched against each image's relative path to tag it as nonflood. "
            "Default: **/nonflood*/** and **/*nonflood*.*"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Report what would be added.")
    args = ap.parse_args(argv)

    src_root = Path(args.src).resolve()
    if not src_root.exists():
        print(f"[fatal] source path does not exist: {src_root}", file=sys.stderr)
        return 1

    flood_globs = args.flood_glob or ["**/flood*/**", "**/*flood*.*"]
    nonflood_globs = args.nonflood_glob or ["**/nonflood*/**", "**/*nonflood*.*"]

    manifest_path = Path(args.manifest)
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    existing_hashes = {row.sha256 for row in existing}

    images = _iter_images(src_root)
    print(f"[hydroshare] scanning {src_root} — {len(images)} image files")

    new_rows: list[ManifestRow] = []
    next_int = _next_id(existing)
    skipped_unknown_class = 0
    skipped_dup = 0

    for img in images:
        rel = img.relative_to(src_root)
        cls = _classify(rel, flood_globs, nonflood_globs)
        if cls is None:
            skipped_unknown_class += 1
            continue
        data = img.read_bytes()
        sha = hash_bytes(data)
        if sha in existing_hashes:
            skipped_dup += 1
            continue
        new_rows.append(
            ManifestRow(
                id=f"mb-{next_int:06d}",
                source=SOURCE_TAG,
                url=HYDROSHARE_RESOURCE_URL,
                sha256=sha,
                license=LICENSE_TAG,
                attribution=ATTRIBUTION,
                image_class=cls,
                notes=f"held-out; local_path_in_archive={rel.as_posix()}",
            )
        )
        existing_hashes.add(sha)
        next_int += 1

    print(
        f"[hydroshare] planned rows={len(new_rows)}  "
        f"skipped_unknown_class={skipped_unknown_class}  skipped_dup={skipped_dup}"
    )
    if not new_rows:
        return 0
    if args.dry_run:
        print("[hydroshare] dry-run: manifest unchanged.")
        return 0
    try:
        save_manifest(existing + new_rows, manifest_path)
    except ManifestError as exc:
        print(f"[fatal] refused to save: {exc}", file=sys.stderr)
        return 1
    print(f"[hydroshare] wrote {len(existing) + len(new_rows)} rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
