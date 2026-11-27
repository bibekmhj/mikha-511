#!/usr/bin/env python3
"""M11 — real-data evaluation of the persistence gate.

Runs the Mikha-Ref pipeline against the live DelDOT corpus assembled by
M10/M10.1, using the hand-labels in ``bench/live_labels.json`` as
ground truth. Produces the paper's Table 2 (real-data version of the
Mikha-Bench table1.md): baseline per-frame classifier vs. baseline +
persistence gate, with precision/recall/F1 and false-alert rate at a
fixed recall.

Design (locked in M11 approval, Nov 2026):

* Score every DelDOT frame in the Nov 26-27 2026 Thanksgiving-nor'easter
  event window for the four flooded-corridor cameras, not just the
  labeled subset. The gate was designed for 30-second-cadence input;
  feeding it the 1-in-10 sampled labels would make its N=10 window span
  50 minutes and change its semantics. Labels are ground truth only.
* Cache per-frame classifier scores to ``results/live_predictions.json``
  so re-runs of the metric sweep are seconds, not minutes.
* ``uncertain`` labels are filtered out of the scored subset (standard
  practice; keep them in the manifest). Reported in the header.
* Four event cameras evaluated per-camera; pooled "overall" is
  reported alongside. FL511 / DE main cams are not evaluated here
  because their labels are pure-nonflood and would skew pooled F1;
  their dataset-level nonflood sanity check lives in the M6 synthetic
  eval.

Nothing else in the repo is touched — no new deps, no re-training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Experiment constants. Deliberately in-file, not CLI, so the script's
# intent stays readable; override via code change if you ever want to
# re-run the experiment over a different storm window.

EVENT_WINDOW_DATES: tuple[str, ...] = ("20261126", "20261127")

EVAL_CAMERAS: tuple[str, ...] = (
    "deldot_scam013_dewey_towers",
    "deldot_scam098_indian_river_inlet_n",
    "deldot_scam169_event_20261126",
    "deldot_pcam4061_event_20261126",
)

# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledRow:
    row_id: str
    camera_id: str
    captured_at: str  # ISO Z
    local_path: str
    label: bool  # True=flood, False=nonflood


def load_labels(labels_path: Path) -> dict[str, str]:
    """Return ``{row_id: 'flood' | 'nonflood'}`` — drops uncertain/skipped."""
    data = json.loads(labels_path.read_text())
    out: dict[str, str] = {}
    for row_id, v in data.get("labels", {}).items():
        cls = v.get("image_class")
        if cls in ("flood", "nonflood"):
            out[row_id] = cls
    return out


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return ``{row_id: manifest_row_dict}``."""
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def _date_prefix(iso_z: str) -> str:
    """'2026-11-26T18:47:43Z' -> '20261126'."""
    return iso_z[:10].replace("-", "")


# ----------------------------------------------------------------------
# Scoring pass


def score_all_frames(
    classifier,  # WaterClassifier (lazy import to keep tests cheap)
    manifest: dict[str, dict],
    cam_ids: tuple[str, ...],
    date_prefixes: tuple[str, ...],
    cache_path: Path | None,
    *,
    cv2_module=None,
) -> dict[str, float | None]:
    """Score every frame in these cameras+dates. Cache to disk.

    ``None`` for a frame means the image on disk could not be decoded.
    """
    cache: dict[str, float | None] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    todo: list[str] = []
    for row_id, row in manifest.items():
        if row["camera_id"] not in cam_ids:
            continue
        if _date_prefix(row["captured_at"]) not in date_prefixes:
            continue
        if row_id in cache:
            continue
        todo.append(row_id)

    logging.info("scoring %d new frames (cache already has %d)", len(todo), len(cache))

    if todo and cv2_module is None:  # pragma: no cover
        import cv2

        cv2_module = cv2

    for i, row_id in enumerate(todo, start=1):
        row = manifest[row_id]
        img = cv2_module.imread(row["local_path"])
        if img is None:
            cache[row_id] = None
        else:
            cache[row_id] = float(classifier.score(img))
        if i % 100 == 0:
            logging.info("  %d/%d scored", i, len(todo))
            if cache_path:
                cache_path.write_text(json.dumps(cache))

    if cache_path:
        cache_path.write_text(json.dumps(cache))
    return cache


# ----------------------------------------------------------------------
# Gate pass


def run_gate_per_camera(
    scores: dict[str, float | None],
    manifest: dict[str, dict],
    cam_ids: tuple[str, ...],
    threshold: float,
    *,
    window: int,
    threshold_k: int,
    clear_grace: int,
    gate_cls,  # PersistenceGate
) -> dict[str, dict]:
    """Return ``{row_id: {raw_score, raw_over_thr, gate_state}}``.

    Frames with ``scores[row_id] is None`` are skipped (dead-cam frames)
    and will not appear in the result dict.
    """
    by_cam: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row_id, row in manifest.items():
        if row["camera_id"] not in cam_ids:
            continue
        s = scores.get(row_id)
        if s is None:
            continue
        by_cam[row["camera_id"]].append((row["captured_at"], row_id))

    out: dict[str, dict] = {}
    for _cam, rows in by_cam.items():
        rows.sort()
        gate = gate_cls(window=window, threshold_k=threshold_k, clear_grace=clear_grace)
        for _, row_id in rows:
            s = scores[row_id]
            assert s is not None  # filtered above
            state = gate.observe(s >= threshold)
            out[row_id] = {
                "raw_score": s,
                "raw_over_thr": bool(s >= threshold),
                "gate_state": bool(state.is_persistent),
                "n_pos_in_window": state.n_positive_in_window,
            }
    return out


# ----------------------------------------------------------------------
# Metrics


def _binary_counts(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, int]:
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def binary_report(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    c = _binary_counts(y_pred, y_true)
    return {**c, **_precision_recall_f1(c)}


def far_at_recall(
    scores: np.ndarray, y_true: np.ndarray, target_recall: float = 0.9
) -> tuple[float, float]:
    """Return (false_alert_rate, chosen_threshold) at >= target_recall.

    Operates on raw scores (baseline). ``false_alert_rate`` = FP / (FP + TN).
    If no threshold achieves target recall, returns (nan, nan).
    """
    if not y_true.any():
        return float("nan"), float("nan")
    # Sort unique thresholds descending so recall is monotone non-decreasing
    thresholds = np.unique(scores)[::-1]
    best_far = float("nan")
    best_thr = float("nan")
    for thr in thresholds:
        pred = scores >= thr
        c = _binary_counts(pred, y_true)
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        if recall >= target_recall:
            denom = c["fp"] + c["tn"]
            far = c["fp"] / denom if denom else float("nan")
            best_far = float(far)
            best_thr = float(thr)
            break
    return best_far, best_thr


def gate_far_at_recall(
    scores: np.ndarray,
    y_true: np.ndarray,
    gate_states_by_thr: dict[float, np.ndarray],
    target_recall: float = 0.9,
) -> tuple[float, float]:
    """Sweep the gate's decision-threshold and report FAR at target recall."""
    if not y_true.any():
        return float("nan"), float("nan")
    for thr in sorted(gate_states_by_thr.keys(), reverse=True):
        pred = gate_states_by_thr[thr]
        c = _binary_counts(pred, y_true)
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        if recall >= target_recall:
            denom = c["fp"] + c["tn"]
            far = c["fp"] / denom if denom else float("nan")
            return float(far), float(thr)
    return float("nan"), float("nan")


def compute_metrics(
    labeled: list[LabeledRow],
    preds: dict[str, dict],
    *,
    manifest: dict[str, dict],
    scores: dict[str, float | None],
    cam_ids: tuple[str, ...],
    gate_cls,
    gate_window: int,
    gate_k: int,
    gate_clear_grace: int,
) -> dict:
    """Metrics on this ``labeled`` slice: baseline + gate, incl. FAR@90%R."""
    if not labeled:
        return {
            "n_total": 0,
            "n_flood": 0,
            "n_nonflood": 0,
            "baseline": None,
            "gate": None,
        }

    y_true = np.array([r.label for r in labeled], dtype=bool)
    y_raw = np.array([preds[r.row_id]["raw_score"] for r in labeled], dtype=float)
    y_base = np.array([preds[r.row_id]["raw_over_thr"] for r in labeled], dtype=bool)
    y_gate = np.array([preds[r.row_id]["gate_state"] for r in labeled], dtype=bool)

    far_b, far_b_thr = far_at_recall(y_raw, y_true, target_recall=0.9)

    # Sweep gate threshold on the subset of cameras present in this slice
    present_cams = tuple(sorted({r.camera_id for r in labeled}))
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    gate_states_by_thr: dict[float, np.ndarray] = {}
    for thr in thresholds:
        preds_thr = run_gate_per_camera(
            scores,
            manifest,
            present_cams,
            float(thr),
            window=gate_window,
            threshold_k=gate_k,
            clear_grace=gate_clear_grace,
            gate_cls=gate_cls,
        )
        states = np.array(
            [preds_thr.get(r.row_id, {}).get("gate_state", False) for r in labeled],
            dtype=bool,
        )
        gate_states_by_thr[float(thr)] = states

    far_g, far_g_thr = gate_far_at_recall(y_raw, y_true, gate_states_by_thr, target_recall=0.9)

    return {
        "n_total": len(labeled),
        "n_flood": int(y_true.sum()),
        "n_nonflood": int((~y_true).sum()),
        "baseline": {
            **binary_report(y_base, y_true),
            "far_at_90_recall": far_b,
            "far_threshold": far_b_thr,
        },
        "gate": {
            **binary_report(y_gate, y_true),
            "far_at_90_recall": far_g,
            "far_threshold": far_g_thr,
        },
    }


# ----------------------------------------------------------------------
# Output artefacts


def write_table(path: Path, results: dict, args) -> None:
    def _fmt_far(m: dict) -> str:
        far = m["far_at_90_recall"]
        thr = m["far_threshold"]
        if far != far:  # NaN
            return "–"
        return f"{far:.3f} (thr={thr:.2f})"

    lines = [
        "# M11 — Real-data evaluation of the persistence gate",
        "",
        f"**Classifier checkpoint:** `{args.checkpoint}`",
        f"**Gate:** PersistenceGate(window={args.window}, k={args.k}, clear_grace={args.clear_grace})",
        f"**Decision threshold (baseline / gate fixed-threshold):** {args.threshold}",
        f"**Event window:** {', '.join(EVENT_WINDOW_DATES)} (Thanksgiving nor'easter 2026)",
        "",
        "## Pooled across evaluation cameras",
        "",
        "| Method | Precision | Recall | F1 | FAR @ 90% recall | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    o = results["overall"]
    for name, key in (("Baseline (per-frame)", "baseline"), ("Proposed (persistence gate)", "gate")):
        m = o[key]
        if m is None:
            lines.append(f"| {name} | – | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {_fmt_far(m)} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    lines += [
        "",
        f"**Labeled rows scored:** {o['n_total']} "
        f"(flood={o['n_flood']}, nonflood={o['n_nonflood']})",
        "",
        "## Per-camera",
        "",
    ]
    for cam in EVAL_CAMERAS:
        if cam not in results:
            continue
        cm = results[cam]
        lines += [
            f"### `{cam}`",
            "",
            f"flood={cm['n_flood']}, nonflood={cm['n_nonflood']}",
            "",
            "| Method | Precision | Recall | F1 | FAR @ 90% R |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, key in (("Baseline", "baseline"), ("Gate", "gate")):
            m = cm[key]
            if m is None:
                lines.append(f"| {name} | – | – | – | – |")
            else:
                lines.append(
                    f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {_fmt_far(m)} |"
                )
        lines.append("")
    path.write_text("\n".join(lines))


def write_transitions(path: Path, preds: dict, manifest: dict, cams: tuple[str, ...]) -> None:
    lines = [
        "# M11 — Gate raise/clear transitions on event cameras",
        "",
        "Every row is a frame where the gate changed state. Raise = first",
        "frame judged persistent-flooded. Clear = first frame that dropped out",
        "after the configured clear-grace window of below-threshold frames.",
        "",
    ]
    for cam in cams:
        lines.append(f"## `{cam}`")
        lines.append("")
        rows = sorted(
            (
                (manifest[rid]["captured_at"], rid, p)
                for rid, p in preds.items()
                if manifest[rid]["camera_id"] == cam
            ),
            key=lambda x: x[0],
        )
        transitions: list[tuple[str, str, float]] = []
        prev_state = False
        for captured_at, _rid, p in rows:
            if p["gate_state"] and not prev_state:
                transitions.append((captured_at, "RAISE", p["raw_score"]))
            elif not p["gate_state"] and prev_state:
                transitions.append((captured_at, "CLEAR", p["raw_score"]))
            prev_state = p["gate_state"]
        if not transitions:
            lines.append("_(no transitions during event window)_")
        else:
            for t, evt, score in transitions:
                lines.append(f"- `{t}`  **{evt}**  (raw_score={score:.3f})")
        lines.append("")
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a .pt WaterClassifier checkpoint (e.g. models/finetune_v1/best.pt)",
    )
    ap.add_argument("--labels", type=Path, default=Path("bench/live_labels.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/live_511/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/live_eval"))
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("results/live_predictions.json"),
        help="Where to cache per-image classifier scores between runs.",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--clear-grace", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    # Imports local to main() so unit tests can import this module
    # without pulling torch / cv2.
    from mikha.ref.classifier import WaterClassifier
    from mikha.ref.temporal import PersistenceGate

    logging.info("loading labels + manifest")
    labels = load_labels(args.labels)
    manifest = load_manifest(args.manifest)
    logging.info("  %d flood/nonflood labels, %d manifest rows", len(labels), len(manifest))

    logging.info("loading classifier from %s", args.checkpoint)
    clf = WaterClassifier(args.checkpoint)
    logging.info("  device=%s backbone=%s", clf.device, clf.backbone)

    logging.info("scoring event-window frames (cache=%s)", args.cache)
    scores = score_all_frames(
        clf, manifest, EVAL_CAMERAS, EVENT_WINDOW_DATES, args.cache
    )

    logging.info("running gate per camera at baseline threshold=%.2f", args.threshold)
    preds = run_gate_per_camera(
        scores,
        manifest,
        EVAL_CAMERAS,
        args.threshold,
        window=args.window,
        threshold_k=args.k,
        clear_grace=args.clear_grace,
        gate_cls=PersistenceGate,
    )

    logging.info("joining labels with predictions")
    labeled: list[LabeledRow] = []
    for row_id, cls in labels.items():
        if row_id not in manifest or row_id not in preds:
            continue
        row = manifest[row_id]
        if row["camera_id"] not in EVAL_CAMERAS:
            continue
        labeled.append(
            LabeledRow(
                row_id=row_id,
                camera_id=row["camera_id"],
                captured_at=row["captured_at"],
                local_path=row["local_path"],
                label=(cls == "flood"),
            )
        )
    logging.info("  %d labeled rows on evaluation cams", len(labeled))

    logging.info("computing metrics (per-cam + pooled)")
    results: dict[str, dict] = {
        "overall": compute_metrics(
            labeled,
            preds,
            manifest=manifest,
            scores=scores,
            cam_ids=EVAL_CAMERAS,
            gate_cls=PersistenceGate,
            gate_window=args.window,
            gate_k=args.k,
            gate_clear_grace=args.clear_grace,
        )
    }
    for cam in EVAL_CAMERAS:
        cam_rows = [r for r in labeled if r.camera_id == cam]
        if not cam_rows:
            continue
        results[cam] = compute_metrics(
            cam_rows,
            preds,
            manifest=manifest,
            scores=scores,
            cam_ids=(cam,),
            gate_cls=PersistenceGate,
            gate_window=args.window,
            gate_k=args.k,
            gate_clear_grace=args.clear_grace,
        )

    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    write_table(args.out_dir / "table1_real.md", results, args)
    write_transitions(
        args.out_dir / "transitions_real.md", preds, manifest, EVAL_CAMERAS
    )
    logging.info("done — see %s", args.out_dir)


if __name__ == "__main__":
    main()
