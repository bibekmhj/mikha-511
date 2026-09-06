#!/usr/bin/env python3
"""Produce a before/after gallery for the 5 Mikha-Bench degradations.

Writes 20 PNGs to ``docs/aug_samples/`` (5 transforms × 4 severities), plus
a copy of the source image. ``docs/aug_gallery.md`` links to them.

By default the source is the bundled ultralytics sample (``bus.jpg``). Pass
``--src`` to point at any BGR-decodable image on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from mikha.aug import BENCH_TRANSFORMS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "aug_samples"

SEVERITIES: list[tuple[str, float]] = [
    ("s10", 0.10),
    ("s35", 0.35),
    ("s65", 0.65),
    ("s95", 0.95),
]


def _load_default_source() -> tuple[np.ndarray, str]:
    from ultralytics.utils import ASSETS

    path = Path(ASSETS) / "bus.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"bundled sample not readable at {path}")
    return img, str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the Mikha-Aug before/after gallery.")
    ap.add_argument("--src", default=None, help="Source image path. Defaults to bundled bus.jpg.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for PNGs.")
    ap.add_argument("--seed", type=int, default=1234, help="Deterministic seed for transforms.")
    args = ap.parse_args(argv)

    if args.src is None:
        src_img, src_note = _load_default_source()
    else:
        src_img = cv2.imread(args.src)
        if src_img is None:
            print(f"[fatal] could not read {args.src}", file=sys.stderr)
            return 2
        src_note = args.src

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = out_dir / "source.png"
    cv2.imwrite(str(src_path), src_img)
    print(f"[gallery] source: {src_note} → {src_path}")

    written: list[Path] = []
    for name, fn in BENCH_TRANSFORMS.items():
        for tag, severity in SEVERITIES:
            out_path = out_dir / f"{name}_{tag}.png"
            out_img = fn(src_img, seed=args.seed, severity=severity)
            if not cv2.imwrite(str(out_path), out_img):
                print(f"[fatal] cv2.imwrite failed for {out_path}", file=sys.stderr)
                return 3
            written.append(out_path)
            print(f"[gallery] {name} severity={severity:.2f} → {out_path.name}")

    print(f"[gallery] wrote {len(written)} samples to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
