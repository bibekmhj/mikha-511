#!/usr/bin/env python3
"""Import a hand-curated list of image URLs (e.g. NOAA/USGS/FEMA press pool)
into ``data/manifest.csv``.

Input file format: TSV, one row per image, with the following columns
(header line required):

    url<TAB>license<TAB>image_class<TAB>source<TAB>attribution<TAB>notes

Every row is validated against :data:`mikha.bench.ALLOWED_LICENSES` and
:data:`mikha.bench.ALLOWED_CLASSES` before download.

For each row the script:

* downloads bytes from ``url``,
* computes SHA-256,
* appends a validated :class:`mikha.bench.ManifestRow`.

Rows whose license, class, or URL fields are invalid are refused immediately.
Rows whose bytes cannot be fetched (HTTP error) are skipped and reported.

Curated NOAA/USGS/FEMA lists live at ``data/gov_pd_urls.tsv`` — see
``data/README.md`` for the format and rights-verification protocol.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx

from mikha.bench import (
    ALLOWED_CLASSES,
    ALLOWED_LICENSES,
    ManifestError,
    ManifestRow,
    hash_bytes,
    load_manifest,
    save_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"

REQUIRED_COLS = ("url", "license", "image_class", "source", "attribution", "notes")


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


def _load_url_list(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header row")
        missing = set(REQUIRED_COLS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        rows = [{col: (row.get(col) or "").strip() for col in REQUIRED_COLS} for row in reader]
    # Pre-validate license + class before we ever touch the network.
    for i, row in enumerate(rows, start=2):
        if row["license"] not in ALLOWED_LICENSES:
            raise ValueError(f"{path}:{i}: license {row['license']!r} not in allowed set")
        if row["image_class"] not in ALLOWED_CLASSES:
            raise ValueError(f"{path}:{i}: image_class {row['image_class']!r} not in allowed set")
        if not row["url"].startswith(("http://", "https://")):
            raise ValueError(f"{path}:{i}: url must be http(s)://…")
        if not row["attribution"]:
            raise ValueError(f"{path}:{i}: attribution required (upstream credit)")
        if not row["source"]:
            raise ValueError(f"{path}:{i}: source required")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Import a curated TSV of image URLs into the manifest."
    )
    ap.add_argument("--src", required=True, help="Path to the TSV url list.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    ap.add_argument("--dry-run", action="store_true", help="Do not fetch; print projected count.")
    args = ap.parse_args(argv)

    try:
        rows = _load_url_list(Path(args.src))
    except ValueError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1

    print(f"[url_list] parsed {len(rows)} rows from {args.src}")

    manifest_path = Path(args.manifest)
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    existing_hashes = {row.sha256 for row in existing}
    existing_urls = {row.url for row in existing}

    if args.dry_run:
        print("[url_list] dry-run: not fetching, not writing.")
        return 0

    new_rows: list[ManifestRow] = []
    next_int = _next_id(existing)
    ok = http_error = dup = 0

    with httpx.Client(headers={"User-Agent": "Mikha-511/0.1 (research)"}) as client:
        for row in rows:
            if row["url"] in existing_urls:
                dup += 1
                continue
            try:
                r = client.get(row["url"], timeout=30.0, follow_redirects=True)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                http_error += 1
                print(f"[url_list]   HTTP-ERROR  {row['url']}  {type(exc).__name__}")
                continue
            sha = hash_bytes(r.content)
            if sha in existing_hashes:
                dup += 1
                continue
            new_rows.append(
                ManifestRow(
                    id=f"mb-{next_int:06d}",
                    source=row["source"],
                    url=row["url"],
                    sha256=sha,
                    license=row["license"],
                    attribution=row["attribution"],
                    image_class=row["image_class"],
                    notes=row["notes"],
                )
            )
            existing_hashes.add(sha)
            existing_urls.add(row["url"])
            next_int += 1
            ok += 1

    print(f"[url_list] fetch summary: ok={ok}  http_error={http_error}  dup={dup}")
    if not new_rows:
        return 0 if http_error == 0 else 2
    try:
        save_manifest(existing + new_rows, manifest_path)
    except ManifestError as exc:
        print(f"[fatal] refused to save: {exc}", file=sys.stderr)
        return 1
    print(f"[url_list] wrote {len(existing) + len(new_rows)} rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
