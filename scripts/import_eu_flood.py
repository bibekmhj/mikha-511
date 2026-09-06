#!/usr/bin/env python3
"""Import European Flood 2013 rows into ``data/manifest.csv``.

For each Wikimedia-licensed image whose license is in the accepted set:

* determine its class (``flood`` if in ``relevance/flooding.txt``,
  ``nonflood`` if in ``relevance/irrelevant.txt``, else skipped);
* download the bytes from Wikimedia to compute SHA-256;
* append a validated :class:`mikha.bench.ManifestRow` with the true
  per-image license and attribution.

Idempotent: images already represented in the manifest by SHA-256 are skipped.

Metadata + relevance are pulled from GitHub raw (works everywhere).
Image bytes come from Wikimedia (``upload.wikimedia.org``); if your egress
blocks that host, the script reports HTTP-ERROR for those rows and adds
nothing (a manifest with fake hashes would be worse than an empty one).

Balance control: ``--balance`` caps flood rows to match nonflood rows (or the
other way around) so the produced subset stays ~50/50.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import httpx

from mikha.bench import (
    ManifestError,
    ManifestRow,
    hash_bytes,
    load_manifest,
    save_manifest,
)
from mikha.bench.eu_flood import (
    EFRecord,
    license_summary,
    load_relevance,
    parse_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"

EF2013_METADATA_URL = (
    "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master/metadata.json"
)
EF2013_FLOODING_URL = (
    "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master/relevance/flooding.txt"
)
EF2013_IRRELEVANT_URL = (
    "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master/relevance/irrelevant.txt"
)

SOURCE_TAG = "eu_flood_2013"


def _download_text(url: str, dest: Path, *, client: httpx.Client) -> None:
    r = client.get(url, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def _next_id(existing: list[ManifestRow]) -> int:
    """Return the next mb-NNNNNN integer, one past the max already used."""
    if not existing:
        return 1
    ids = []
    for row in existing:
        try:
            ids.append(int(row.id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(ids) + 1) if ids else 1


def _classify(pageid: str, flood_ids: set[str], nonflood_ids: set[str]) -> str | None:
    """Return ``flood`` / ``nonflood`` for a page id, or None to skip."""
    if pageid in flood_ids:
        return "flood"
    if pageid in nonflood_ids:
        return "nonflood"
    return None


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser(description="Import European Flood 2013 rows into the manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not fetch image bytes or write the manifest; print projected counts only.",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Cap rows added per class (flood, nonflood). Default 500; use 0 for no cap.",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="After capping, further trim so flood == nonflood count.",
    )
    ap.add_argument(
        "--sleep-ms",
        type=int,
        default=250,
        help="Polite delay between Wikimedia downloads (ms). Default 250.",
    )
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    existing_hashes = {row.sha256 for row in existing}
    existing_urls = {row.url for row in existing}

    with tempfile.TemporaryDirectory() as td:
        meta_path = Path(td) / "metadata.json"
        flooding_path = Path(td) / "flooding.txt"
        irrelevant_path = Path(td) / "irrelevant.txt"

        with httpx.Client(headers={"User-Agent": "Mikha-511/0.1 (research)"}) as client:
            print("[eu_flood] fetching EF2013 metadata + relevance from GitHub…")
            _download_text(EF2013_METADATA_URL, meta_path, client=client)
            _download_text(EF2013_FLOODING_URL, flooding_path, client=client)
            _download_text(EF2013_IRRELEVANT_URL, irrelevant_path, client=client)

            records: list[EFRecord] = parse_metadata(meta_path)
            flood_ids: set[str] = load_relevance(flooding_path)
            nonflood_ids: set[str] = load_relevance(irrelevant_path)

            print(
                f"[eu_flood] upstream: {len(records)} records with accepted licenses; "
                f"{len(flood_ids)} flood ids; {len(nonflood_ids)} irrelevant ids"
            )

            # Report the raw license distribution (including rejected) so the
            # user sees what we skipped.
            raw = license_summary(meta_path)
            print("[eu_flood] upstream license distribution (raw):")
            for lic, count in sorted(raw.items(), key=lambda kv: -kv[1]):
                print(f"    {count:5d}  {lic!r}")

            # Bucket candidates by class, preserving upstream ordering.
            flood_candidates: list[EFRecord] = []
            nonflood_candidates: list[EFRecord] = []
            for rec in records:
                cls = _classify(rec.pageid, flood_ids, nonflood_ids)
                if cls == "flood":
                    flood_candidates.append(rec)
                elif cls == "nonflood":
                    nonflood_candidates.append(rec)

            print(
                f"[eu_flood] classified: flood={len(flood_candidates)}  "
                f"nonflood={len(nonflood_candidates)}"
            )

            # Apply per-class cap.
            cap = args.max_per_class if args.max_per_class > 0 else None
            if cap is not None:
                flood_candidates = flood_candidates[:cap]
                nonflood_candidates = nonflood_candidates[:cap]

            # Optional balance step: trim the larger bucket down.
            if args.balance:
                target = min(len(flood_candidates), len(nonflood_candidates))
                flood_candidates = flood_candidates[:target]
                nonflood_candidates = nonflood_candidates[:target]

            plan = [(rec, "flood") for rec in flood_candidates] + [
                (rec, "nonflood") for rec in nonflood_candidates
            ]
            print(f"[eu_flood] planned rows to attempt: {len(plan)}")

            if args.dry_run:
                print("[eu_flood] dry-run: not fetching images, not writing manifest.")
                return 0

            # Fetch each image, hash, and append a row.
            import time

            new_rows: list[ManifestRow] = []
            next_int = _next_id(existing)
            ok = mismatch = http_error = dup_skip = 0

            for rec, cls in plan:
                if rec.url in existing_urls:
                    dup_skip += 1
                    continue
                try:
                    resp = client.get(rec.url, timeout=30.0, follow_redirects=True)
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    http_error += 1
                    print(f"[eu_flood]   HTTP-ERROR  {rec.pageid}  {type(exc).__name__}")
                    time.sleep(args.sleep_ms / 1000.0)
                    continue

                sha = hash_bytes(resp.content)
                if sha in existing_hashes:
                    dup_skip += 1
                    continue

                row = ManifestRow(
                    id=f"mb-{next_int:06d}",
                    source=SOURCE_TAG,
                    url=rec.url,
                    sha256=sha,
                    license=rec.license_spdx,
                    attribution=rec.attribution,
                    image_class=cls,
                    notes=f"pageid={rec.pageid}; title={rec.title}",
                )
                new_rows.append(row)
                existing_hashes.add(sha)
                existing_urls.add(rec.url)
                next_int += 1
                ok += 1
                if ok % 25 == 0:
                    print(f"[eu_flood]   fetched {ok}/{len(plan)}…")
                time.sleep(args.sleep_ms / 1000.0)

            print(
                f"[eu_flood] fetch summary: ok={ok}  http_error={http_error}  "
                f"mismatch={mismatch}  dup_skip={dup_skip}"
            )

            if not new_rows:
                print("[eu_flood] no new rows added; manifest unchanged.")
                return 0 if http_error == 0 else 2

            merged = existing + new_rows
            try:
                save_manifest(merged, manifest_path)
            except ManifestError as exc:
                print(f"[fatal] refused to save: {exc}", file=sys.stderr)
                return 1
            print(f"[eu_flood] wrote {len(merged)} rows to {manifest_path}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
