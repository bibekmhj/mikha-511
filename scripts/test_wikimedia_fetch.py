#!/usr/bin/env python3
"""Small probe: try to fetch 3–5 real EF2013 Wikimedia image URLs and print
per-URL detail (status, final URL after redirects, body preview, byte count,
sha256). Does NOT touch the manifest.

Use this to confirm the Wikimedia User-Agent fix works before spending 30
minutes running the full import.

Usage:
    python scripts/test_wikimedia_fetch.py            # default: first 5 flood ids
    python scripts/test_wikimedia_fetch.py --n 3
    python scripts/test_wikimedia_fetch.py --ua "your custom UA"

Exit code 0 iff every attempted URL returned 200.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import httpx

from mikha.bench import hash_bytes
from mikha.bench.eu_flood import load_relevance, parse_metadata

METADATA_URL = "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master/metadata.json"
FLOODING_URL = (
    "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master/relevance/flooding.txt"
)

DEFAULT_UA = "Mikha511/0.1 (https://github.com/bibekmhj/mikha-511; bbkmhj06@gmail.com) httpx"


def _download(url: str, dest: Path, *, client: httpx.Client) -> None:
    r = client.get(url, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe Wikimedia image fetches for EF2013.")
    ap.add_argument("--n", type=int, default=5, help="How many images to try. Default 5.")
    ap.add_argument("--ua", default=DEFAULT_UA, help="Override the User-Agent string.")
    args = ap.parse_args(argv)

    headers = {
        "User-Agent": args.ua,
        "Accept": "image/*,*/*;q=0.8",
        "Api-User-Agent": args.ua,
        "From": "bbkmhj06@gmail.com",
    }
    print(f"[probe] User-Agent = {args.ua!r}")
    print(f"[probe] attempting {args.n} images")

    with tempfile.TemporaryDirectory() as td:
        meta_p = Path(td) / "metadata.json"
        flood_p = Path(td) / "flooding.txt"

        with httpx.Client(headers=headers, timeout=60.0) as client:
            print("[probe] fetching EF2013 metadata + flooding.txt from GitHub…")
            _download(METADATA_URL, meta_p, client=client)
            _download(FLOODING_URL, flood_p, client=client)

            records = parse_metadata(meta_p)
            flood_ids = load_relevance(flood_p)
            candidates = [r for r in records if r.pageid in flood_ids][: args.n]

            print(f"[probe] selected {len(candidates)} candidate URLs")
            ok = fail = 0
            for rec in candidates:
                print(f"\n[probe] --- {rec.pageid}  {rec.title}")
                print(f"        license upstream: {rec.license_upstream}  → {rec.license_spdx}")
                print(f"        url: {rec.url}")
                try:
                    resp = client.get(rec.url, timeout=60.0, follow_redirects=True)
                except httpx.HTTPError as exc:
                    fail += 1
                    print(f"        NETWORK-ERROR  {type(exc).__name__}: {exc}")
                    continue
                print(f"        status={resp.status_code}  final_url={resp.url}")
                print(f"        response_bytes={len(resp.content)}")
                print(
                    f"        content_type={resp.headers.get('content-type')!r}  "
                    f"content_length={resp.headers.get('content-length')!r}"
                )
                if resp.status_code == 200 and resp.content:
                    sha = hash_bytes(resp.content)
                    print(f"        sha256={sha}")
                    ok += 1
                else:
                    body_preview = resp.text[:400].replace("\n", " ")
                    print(f"        body_preview={body_preview!r}")
                    fail += 1

    print(f"\n[probe] summary: ok={ok}  fail={fail}  total={ok + fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
