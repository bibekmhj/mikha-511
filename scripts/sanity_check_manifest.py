#!/usr/bin/env python3
"""Post-import sanity check for the base-image manifest.

Runs against a machine that has already run ``fetch_base.py`` — i.e. the
image bytes referenced by every manifest row are present under
``data/images/``. Reports:

* files missing, unreadable, or size-zero
* SHA-256 mismatches vs the manifest
* files that don't decode as valid images (PIL cannot open)
* per-image dimensions/aspect-ratio outliers (very tall/wide, very small)
* per-source dimension distribution
* near-duplicates within each source via perceptual hashing (dHash);
  neighbours with Hamming distance ≤ 5 are flagged as likely duplicates
* an on-disk HTML sample sheet with N random flood + N random nonflood
  images per source, so you can eyeball obvious mislabels

Exits nonzero on any HARD failure (missing / unreadable / hash mismatch /
undecodable). Near-duplicates and small-file warnings are informational.

Usage:
    pip install pillow imagehash            # if not already installed
    python scripts/sanity_check_manifest.py --samples 12
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

from mikha.bench import ManifestRow, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_IMAGES = REPO_ROOT / "data" / "images"
DEFAULT_REPORT = REPO_ROOT / "results" / "sanity_check.html"


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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser(description="Sanity-check the imported base set.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--report", default=str(DEFAULT_REPORT), help="HTML sample-sheet output.")
    ap.add_argument(
        "--samples",
        type=int,
        default=12,
        help="How many random samples per (source, class). Default 12.",
    )
    ap.add_argument(
        "--near-dup-threshold",
        type=int,
        default=5,
        help="Max dHash Hamming distance to flag as near-duplicate. Default 5.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    try:
        from PIL import Image
    except ImportError:
        print("[fatal] Pillow is required. pip install pillow", file=sys.stderr)
        return 2

    try:
        import imagehash

        have_imagehash = True
    except ImportError:
        print("[warn] imagehash not installed; near-duplicate check will be skipped.")
        print("       pip install imagehash to enable it.")
        have_imagehash = False

    rng = random.Random(args.seed)
    rows = load_manifest(args.manifest)
    images_dir = Path(args.images)
    print(f"[sanity] {len(rows)} manifest rows, images dir = {images_dir}")

    missing: list[str] = []
    unreadable: list[str] = []
    zero_byte: list[str] = []
    hash_mismatch: list[tuple[str, str, str]] = []
    undecodable: list[str] = []
    tiny: list[tuple[str, int, int]] = []
    weird_aspect: list[tuple[str, float]] = []

    per_source_hashes: dict[str, list[tuple[str, str, object]]] = defaultdict(list)
    per_source_dims: dict[str, list[tuple[int, int]]] = defaultdict(list)
    per_source_sizes: dict[str, list[int]] = defaultdict(list)
    thumbs: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)

    for i, row in enumerate(rows, 1):
        if i % 100 == 0:
            print(f"[sanity]   scanned {i}/{len(rows)}")
        p = _local_path(row, images_dir)
        if not p.exists():
            missing.append(row.id)
            continue
        try:
            size = p.stat().st_size
        except OSError:
            unreadable.append(row.id)
            continue
        if size == 0:
            zero_byte.append(row.id)
            continue
        per_source_sizes[row.source].append(size)

        actual = _sha256(p)
        if actual != row.sha256:
            hash_mismatch.append((row.id, row.sha256, actual))
            continue

        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                im.load()
                if have_imagehash:
                    im_small = im.convert("RGB").resize((64, 64))
                    ph = imagehash.dhash(im_small)
        except Exception as exc:  # noqa: BLE001
            undecodable.append(f"{row.id}: {type(exc).__name__}: {exc}")
            continue

        per_source_dims[row.source].append((w, h))
        if have_imagehash:
            per_source_hashes[row.source].append((row.id, row.image_class, ph))
        if min(w, h) < 128:
            tiny.append((row.id, w, h))
        ar = w / h if h else 0.0
        if ar > 3.0 or ar < 0.33:
            weird_aspect.append((row.id, ar))

        thumbs[(row.source, row.image_class)].append((row.id, p))

    # Near-duplicate detection (within each source).
    near_dup_pairs: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    if have_imagehash:
        for src, entries in per_source_hashes.items():
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    d = entries[i][2] - entries[j][2]  # Hamming distance
                    if d <= args.near_dup_threshold:
                        near_dup_pairs[src].append((entries[i][0], entries[j][0], d))

    # --- Report ---
    print("\n" + "=" * 66)
    print("SANITY CHECK — RESULTS")
    print("=" * 66)
    print(f"missing files:       {len(missing)}   {missing[:5]}")
    print(f"unreadable files:    {len(unreadable)}   {unreadable[:5]}")
    print(f"zero-byte files:     {len(zero_byte)}   {zero_byte[:5]}")
    print(f"hash mismatches:     {len(hash_mismatch)}   {[m[0] for m in hash_mismatch[:5]]}")
    print(f"undecodable images:  {len(undecodable)}   {undecodable[:3]}")
    print(f"tiny (<128px min):   {len(tiny)}   {tiny[:5]}")
    print(f"extreme aspect >3:1: {len(weird_aspect)}   {weird_aspect[:5]}")

    print("\nPer-source dimensions (median WxH, min WxH, max WxH):")
    for src, dims in per_source_dims.items():
        if not dims:
            continue
        ws = sorted(w for w, _ in dims)
        hs = sorted(h for _, h in dims)
        n = len(dims)
        mw, mh = ws[n // 2], hs[n // 2]
        print(f"  {src:22s} n={n:4d}  median={mw}x{mh}  min={ws[0]}x{hs[0]}  max={ws[-1]}x{hs[-1]}")

    print("\nPer-source file sizes (median KB, min KB, max KB):")
    for src, sizes in per_source_sizes.items():
        ss = sorted(sizes)
        n = len(ss)
        print(
            f"  {src:22s} n={n:4d}  median={ss[n // 2] // 1024}KB  "
            f"min={ss[0] // 1024}KB  max={ss[-1] // 1024}KB"
        )

    if have_imagehash:
        print(f"\nNear-duplicates (dHash Hamming ≤ {args.near_dup_threshold}):")
        for src, pairs in near_dup_pairs.items():
            print(f"  {src:22s} {len(pairs)} pairs flagged")
            for a, b, d in pairs[:6]:
                print(f"    {a} ~ {b}  distance={d}")
            if len(pairs) > 6:
                print(f"    …and {len(pairs) - 6} more")

    # HTML sample sheet
    print(f"\nWriting sample sheet → {args.report}")
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Mikha-511 sanity samples</title>",
        "<style>body{font:14px system-ui;margin:16px;background:#111;color:#eee}",
        "h2{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}",
        ".g{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}",
        ".c{background:#222;padding:6px;border-radius:6px}",
        ".c img{width:100%;height:160px;object-fit:cover;border-radius:4px;display:block}",
        ".c small{color:#aaa;word-break:break-all}</style>",
    ]
    for (src, cls), items in sorted(thumbs.items()):
        rng.shuffle(items)
        picks = items[: args.samples]
        html.append(f"<h2>{src} — {cls} — showing {len(picks)}/{len(items)}</h2><div class='g'>")
        for row_id, p in picks:
            rel = p.resolve().as_uri()
            html.append(f"<div class='c'><img src='{rel}'><small>{row_id}</small></div>")
        html.append("</div>")
    out.write_text("\n".join(html), encoding="utf-8")

    hard = bool(missing or unreadable or zero_byte or hash_mismatch or undecodable)
    print(
        f"\n[sanity] {'FAIL' if hard else 'OK'} — "
        f"open {out} in a browser to eyeball samples for mislabels."
    )
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
