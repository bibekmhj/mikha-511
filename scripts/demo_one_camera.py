#!/usr/bin/env python3
"""M1 CLI: pull one traffic-camera snapshot and save an overlay PNG.

Usage:
    python scripts/demo_one_camera.py --url <SNAPSHOT_URL> --out demo_out.png
    python scripts/demo_one_camera.py --allow-fallback              # offline demo

Full docs: see :mod:`mikha.ref.demo`.
"""

from __future__ import annotations

import argparse
import sys

from mikha.ref.demo import DEFAULT_URL, DEFAULT_WEIGHTS, run_demo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mikha-511 M1 PoC: one traffic-camera snapshot → one overlay PNG."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Snapshot URL (JPEG/PNG).")
    parser.add_argument("--out", default="demo_out.png", help="Output overlay path.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Ultralytics weights.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If the URL cannot be fetched, run on the bundled ultralytics sample.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_demo(
            url=args.url,
            out=args.out,
            weights=args.weights,
            allow_fallback=args.allow_fallback,
        )
    except RuntimeError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
