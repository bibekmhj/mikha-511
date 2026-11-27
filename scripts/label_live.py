#!/usr/bin/env python3
"""Hand-labeler for the live 511 corpus (M10 / v0.3 scaffold).

Reads ``data/live_511/manifest.csv``, opens each unlabeled image in
the default viewer, prompts for a class, and appends to
``bench/live_labels.json``. Writes go through an atomic replace so an
interrupted session never leaves a half-written labels file.

Keys:
    f = flood
    n = nonflood
    u = uncertain
    s = skip (leave unlabeled)
    q = save + quit

The image viewer is your OS's default; the script does not embed one.
Point ``--only-camera`` at a single camera to focus a labeling session,
and ``--limit`` to cap the batch size.

Usage:
    python scripts/label_live.py
    python scripts/label_live.py --only-camera fl511_00_MDX_SR874_000
    python scripts/label_live.py --limit 25
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mikha.bench.live import Label, LabelStore, load_live_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "live_511" / "manifest.csv"
DEFAULT_LABELS = REPO_ROOT / "bench" / "live_labels.json"

_PROMPT = "  [f]lood  [n]onflood  [u]ncertain  [s]kip  [q]uit + save > "
_CHOICES = {
    "f": "flood",
    "n": "nonflood",
    "u": "uncertain",
}


def _open_image(path: Path) -> None:
    """Best-effort open in the OS default viewer.

    macOS: ``open``. Linux: ``xdg-open``. Windows: ``start``. Failures
    are printed and the labeler continues (the operator can look at
    the path manually).
    """
    if sys.platform == "darwin":
        cmd = ["open", str(path)]
    elif sys.platform.startswith("win"):
        cmd = ["cmd", "/c", "start", "", str(path)]
    else:
        cmd = ["xdg-open", str(path)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"    (viewer not found; look at {path} manually)")


def _now_utc_rfc3339() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hand-label live 511 snapshots.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--labeled-by", default="", help="Free-text attribution stamp.")
    ap.add_argument("--only-camera", default=None)
    ap.add_argument(
        "--limit", type=int, default=0, help="Max rows to label this session (0 = all)."
    )
    ap.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not open the image in the OS viewer; just print the path.",
    )
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    labels_path = Path(args.labels)
    if not manifest_path.exists():
        print(f"[fatal] no manifest at {manifest_path}", file=sys.stderr)
        return 2

    rows = load_live_manifest(manifest_path)
    if args.only_camera:
        rows = [r for r in rows if r.camera_id == args.only_camera]
    store = LabelStore(labels_path)

    unlabeled = [r for r in rows if r.id not in store]
    if not unlabeled:
        print("[label] no unlabeled rows; nothing to do")
        return 0

    if args.limit and args.limit > 0:
        unlabeled = unlabeled[: args.limit]
    print(f"[label] {len(unlabeled)} unlabeled row(s) (of {len(rows)} total). Store: {labels_path}")

    labeled_by = args.labeled_by or ""
    n_written = 0
    try:
        for i, row in enumerate(unlabeled, 1):
            img_path = REPO_ROOT / row.local_path
            print(f"\n[{i}/{len(unlabeled)}] {row.id}  camera={row.camera_id}  {row.captured_at}")
            print(f"    path: {img_path}")
            if not img_path.exists():
                print("    (image missing on disk; skipping)")
                continue
            if not args.no_viewer:
                _open_image(img_path)
            choice = input(_PROMPT).strip().lower()
            if choice == "q":
                break
            if choice == "s" or choice == "":
                continue
            if choice not in _CHOICES:
                print(f"    (unknown key {choice!r}; skipping)")
                continue
            store.set(
                row.id,
                Label(
                    image_class=_CHOICES[choice],
                    labeled_by=labeled_by,
                    labeled_at=_now_utc_rfc3339(),
                    notes="",
                ),
            )
            n_written += 1
    finally:
        if n_written:
            store.set_meta(
                last_saved_at=_now_utc_rfc3339(),
                labeling_protocol_version=1,
            )
            store.save()
            print(f"\n[label] wrote {n_written} label(s) to {labels_path}")
        else:
            print("\n[label] no labels written this session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
