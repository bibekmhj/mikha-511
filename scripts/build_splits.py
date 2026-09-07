#!/usr/bin/env python3
"""Build ``bench/splits.json`` — group-aware 70/15/15 train/val/test split.

Reads ``data/manifest.csv``, groups rows by uploader (for EF2013 rows) or
per-row for other sources, and writes a JSON artifact recording the
assignments. Deterministic given the manifest and seed.

The output includes the manifest's SHA-256 so the split is provably tied
to one manifest state; if the manifest changes, the split is invalidated
and must be rebuilt.

Does NOT modify data/manifest.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mikha.bench import group_stratified_split, load_manifest
from mikha.bench.split import extract_group

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_OUT = REPO_ROOT / "bench" / "splits.json"


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build group-aware train/val/test split.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    print(f"[split] loaded {len(rows)} rows from {manifest_path}")

    target_ratios = {"train": args.train, "val": args.val, "test": args.test}
    result = group_stratified_split(rows, seed=args.seed, target_ratios=target_ratios)

    # Sanity checks (belt-and-braces; the same properties are enforced by tests).
    all_ids_across_splits = [rid for s in result.splits.values() for rid in s.row_ids]
    assert len(all_ids_across_splits) == len(rows), "row-count mismatch after split"
    assert len(set(all_ids_across_splits)) == len(all_ids_across_splits), "row-id overlap"
    manifest_ids = {r.id for r in rows}
    assert set(all_ids_across_splits) == manifest_ids, "coverage mismatch"

    # No group leakage: build a per-split set of groups and verify no
    # intersection across splits.
    row_id_to_group = {r.id: extract_group(r) for r in rows}
    per_split_groups: dict[str, set[str]] = {
        s.name: {row_id_to_group[rid] for rid in s.row_ids} for s in result.splits.values()
    }
    split_names = list(per_split_groups)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            overlap = per_split_groups[a] & per_split_groups[b]
            if overlap:
                raise SystemExit(
                    f"[fatal] group leakage between {a} and {b}: {sorted(overlap)[:5]} …"
                )

    # Also produce a per-source × per-split table for the human report.
    per_source_per_split: dict[str, Counter] = {}
    src_of = {r.id: r.source for r in rows}
    for s in result.splits.values():
        for rid in s.row_ids:
            per_source_per_split.setdefault(src_of[rid], Counter())[s.name] += 1

    doc = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _manifest_hash(manifest_path),
            "seed": args.seed,
            "target_ratios": target_ratios,
            "n_total": result.n_total,
            "n_flood": result.n_flood,
            "n_nonflood": result.n_nonflood,
            "n_groups": len(result.group_assignments),
            "algorithm": "greedy load-balancing over groups sorted largest-first",
        },
        "splits": {
            s.name: {
                "row_ids": list(s.row_ids),
                "n_total": s.n_total,
                "n_flood": s.n_flood,
                "n_nonflood": s.n_nonflood,
                "actual_ratio": round(s.n_total / result.n_total, 4),
                "flood_ratio": round(s.n_flood / s.n_total if s.n_total else 0.0, 4),
            }
            for s in result.splits.values()
        },
        "group_assignments": dict(sorted(result.group_assignments.items())),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)  # atomic

    # Human summary
    print("\n" + "=" * 66)
    print("SPLIT RESULT")
    print("=" * 66)
    print(f"seed:              {args.seed}")
    print(f"manifest sha256:   {doc['meta']['manifest_sha256']}")
    print(
        f"totals:            n={result.n_total}  flood={result.n_flood}  "
        f"nonflood={result.n_nonflood}  ({result.n_flood / result.n_total * 100:.1f}% flood)"
    )
    print(f"groups:            {len(result.group_assignments)}")
    print()
    print(
        f"{'split':<8} {'target':>7} {'actual':>7} {'n_total':>8} "
        f"{'n_flood':>8} {'n_nonflood':>10} {'flood%':>7}"
    )
    for s in result.splits.values():
        print(
            f"{s.name:<8} {target_ratios[s.name] * 100:>6.1f}% "
            f"{s.n_total / result.n_total * 100:>6.1f}% "
            f"{s.n_total:>8d} {s.n_flood:>8d} {s.n_nonflood:>10d} "
            f"{(s.n_flood / s.n_total * 100 if s.n_total else 0):>6.1f}%"
        )

    print("\nBy source × split:")
    for src, counts in sorted(per_source_per_split.items()):
        row = "  ".join(f"{s}={counts.get(s, 0)}" for s in ("train", "val", "test"))
        print(f"  {src:22s}  {row}")

    print(f"\n[split] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
