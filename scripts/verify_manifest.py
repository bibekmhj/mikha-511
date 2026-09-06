#!/usr/bin/env python3
"""Validate ``data/manifest.csv`` without touching the network.

Runs the full :mod:`mikha.bench.manifest` schema and duplicate checks, then
prints per-class / per-license / per-source counts. Exits nonzero on any
schema or duplicate violation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from mikha.bench import ManifestError, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the base-image manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv.")
    args = ap.parse_args(argv)

    try:
        rows = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"[fatal] manifest invalid: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"[fatal] manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if not rows:
        print("[warn] manifest is empty (0 rows)")
        return 0

    by_class = Counter(r.image_class for r in rows)
    by_license = Counter(r.license for r in rows)
    by_source = Counter(r.source for r in rows)

    print(f"OK — {len(rows)} rows, all valid.")
    print("\nBy class:")
    for k, v in by_class.most_common():
        print(f"  {k:12s} {v}")
    print("\nBy license:")
    for k, v in by_license.most_common():
        print(f"  {k:22s} {v}")
    print("\nBy source:")
    for k, v in by_source.most_common():
        print(f"  {k:22s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
