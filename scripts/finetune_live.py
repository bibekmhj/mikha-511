#!/usr/bin/env python3
"""M11-b — Target-domain fine-tune of the water classifier on the live
DelDOT corpus.

Context: the Sept 2026 Mikha-Bench-finetuned classifier (``best.pt``)
scored ~0.50 AUROC on real nighttime DelDOT HLS frames during the Nov
26-27 Thanksgiving-nor'easter event. The persistence gate cannot
rescue a chance-level base classifier, so this script does standard
small-data domain adaptation: freeze the MobileNetV3 backbone, train
only the final classifier head on the 265 human-labeled real frames
(124 flood + 141 nonflood across 4 event cameras).

Eval protocol — leave-one-camera-out 4-fold cross-validation:
  for each camera c in event cams:
    train head on the OTHER three cameras' labels (from scratch, from
      the base checkpoint)
    eval on camera c
  report per-fold AUROC + F1, mean ± std

After CV, refit on ALL four cameras and save as a new checkpoint at
``models/finetune_live/best.pt`` (never overwrites the Sept 8 Mikha-
Bench head). Point ``run_eval_real.py --checkpoint`` at the new path
and compare.

Deliberately self-contained — reuses ``mikha.ref.classifier`` for
preprocessing constants and the backbone load, but does NOT import
anything from ``mikha.train`` so a broken training module can't block
the inference-side fix.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EVENT_CAMERAS: tuple[str, ...] = (
    "deldot_scam013_dewey_towers",
    "deldot_scam098_indian_river_inlet_n",
    "deldot_scam169_event_20261126",
    "deldot_pcam4061_event_20261126",
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (tested without torch)


@dataclass(frozen=True)
class LabeledRow:
    row_id: str
    camera_id: str
    local_path: str
    label: int  # 1=flood, 0=nonflood


def load_labels(labels_path: Path) -> dict[str, str]:
    data = json.loads(labels_path.read_text())
    return {
        rid: v["image_class"]
        for rid, v in data.get("labels", {}).items()
        if v.get("image_class") in ("flood", "nonflood")
    }


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    with manifest_path.open() as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def assemble_labeled_rows(
    labels: dict[str, str],
    manifest: dict[str, dict],
    cam_ids: tuple[str, ...],
) -> list[LabeledRow]:
    """Join labels + manifest, restrict to ``cam_ids``."""
    out: list[LabeledRow] = []
    for rid, cls in labels.items():
        row = manifest.get(rid)
        if row is None or row["camera_id"] not in cam_ids:
            continue
        out.append(
            LabeledRow(
                row_id=rid,
                camera_id=row["camera_id"],
                local_path=row["local_path"],
                label=1 if cls == "flood" else 0,
            )
        )
    return out


def leave_one_camera_out_folds(
    rows: list[LabeledRow],
    cam_ids: tuple[str, ...],
) -> list[tuple[str, list[LabeledRow], list[LabeledRow]]]:
    """Return ``[(held_cam, train_rows, test_rows), ...]``."""
    by_cam: dict[str, list[LabeledRow]] = defaultdict(list)
    for r in rows:
        by_cam[r.camera_id].append(r)
    folds: list[tuple[str, list[LabeledRow], list[LabeledRow]]] = []
    for held in cam_ids:
        if held not in by_cam:
            continue
        test = list(by_cam[held])
        train = [r for cam, rs in by_cam.items() if cam != held for r in rs]
        if not train or not test:
            continue
        folds.append((held, train, test))
    return folds


def class_balance(rows: list[LabeledRow]) -> tuple[int, int]:
    pos = sum(1 for r in rows if r.label == 1)
    return pos, len(rows) - pos


def _compute_binary_metrics(y_true: list[int], y_pred_prob: list[float], threshold: float = 0.5) -> dict:
    """AUROC + precision/recall/F1 at a fixed threshold. Pure Python.

    AUROC via Mann-Whitney-U formulation so no numpy dependency in tests.
    """
    pos_scores = [s for s, y in zip(y_pred_prob, y_true, strict=True) if y == 1]
    neg_scores = [s for s, y in zip(y_pred_prob, y_true, strict=True) if y == 0]
    if not pos_scores or not neg_scores:
        auroc = float("nan")
    else:
        wins = 0.0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        auroc = wins / (len(pos_scores) * len(neg_scores))

    tp = fp = fn = tn = 0
    for s, y in zip(y_pred_prob, y_true, strict=True):
        pred = s >= threshold
        if pred and y == 1:
            tp += 1
        elif pred and y == 0:
            fp += 1
        elif not pred and y == 1:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "auroc": auroc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": tp + fp + fn + tn,
    }


# ---------------------------------------------------------------------------
# Torch-dependent training (imported lazily so tests stay cheap)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(rows, image_size, torch_module, cv2_module):
    import numpy as np

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # noqa: N806
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # noqa: N806

    class _LiveFrameDataset(torch_module.utils.data.Dataset):
        def __init__(self, rows_list):
            self._rows = list(rows_list)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            r = self._rows[idx]
            img = cv2_module.imread(r.local_path)
            if img is None:
                raise FileNotFoundError(f"could not read {r.local_path}")
            resized = cv2_module.resize(img, (image_size, image_size), interpolation=cv2_module.INTER_AREA)
            rgb = cv2_module.cvtColor(resized, cv2_module.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _MEAN) / _STD
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            tensor = torch_module.from_numpy(chw)
            return tensor, torch_module.tensor([float(r.label)], dtype=torch_module.float32)

    return _LiveFrameDataset(rows)


def _find_classifier_head(model) -> list:
    """Return parameters of the final linear/classifier layer only.

    MobileNetV3 small puts the final classifier at ``model.classifier[-1]``
    (a ``nn.Linear`` with 1 output unit after our train script).
    """
    import torch.nn as nn

    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, nn.Linear):
            logging.info("  unfreezing final linear head: %s %s", name, mod)
            return list(mod.parameters()), name
    raise RuntimeError("no nn.Linear found in model; cannot locate classifier head")


def _freeze_except_head(model) -> tuple[list, str]:
    """Freeze every parameter, then unfreeze the final linear head."""
    for p in model.parameters():
        p.requires_grad = False
    head_params, head_name = _find_classifier_head(model)
    for p in head_params:
        p.requires_grad = True
    return head_params, head_name


def _train_one_fold(
    base_checkpoint: Path,
    train_rows: list[LabeledRow],
    test_rows: list[LabeledRow],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[dict, dict]:
    """Returns (test_metrics, trained_state_dict)."""
    import cv2
    import torch

    from mikha.ref.classifier import WaterClassifier

    _seed_everything(seed)
    clf = WaterClassifier(base_checkpoint)
    model = clf._model
    device = clf.device
    image_size = clf.image_size

    head_params, head_name = _freeze_except_head(model)

    train_ds = _build_dataset(train_rows, image_size, torch, cv2)
    test_ds = _build_dataset(test_rows, image_size, torch, cv2)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Balanced BCE — upweight the minority class so tiny imbalance is handled.
    n_pos = sum(r.label for r in train_rows)
    n_neg = len(train_rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(head_params, lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            n_batches += 1
        logging.info("    epoch %d/%d  train_loss=%.4f", epoch, epochs, running / max(n_batches, 1))

    # Eval on the held-out camera
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits = model(x)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            y_prob.extend(float(p) for p in probs)
            y_true.extend(int(v) for v in y.squeeze(1).cpu().tolist())

    metrics = _compute_binary_metrics(y_true, y_prob, threshold=0.5)
    return metrics, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _save_checkpoint(base_checkpoint: Path, state_dict: dict, out_path: Path) -> None:
    import torch

    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    # Preserve spec and image_size exactly; swap in the retrained weights.
    new = {
        "spec": base.get("spec", {}),
        "image_size": base.get("image_size", 224),
        "model_state_dict": state_dict,
        "retrained_from": str(base_checkpoint),
        "retrained_note": "M11-b head-only finetune on 4 DelDOT event cams (124f+141n)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out_path)


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("models/finetune_v1/best.pt"),
        help="Checkpoint to fine-tune FROM (frozen backbone, re-train head).",
    )
    ap.add_argument("--labels", type=Path, default=Path("bench/live_labels.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/live_511/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("models/finetune_live"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip leave-one-camera-out CV and train directly on all data.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels)
    manifest = load_manifest(args.manifest)
    rows = assemble_labeled_rows(labels, manifest, EVENT_CAMERAS)
    pos, neg = class_balance(rows)
    logging.info("assembled %d labeled rows on %d event cams (flood=%d nonflood=%d)",
                 len(rows), len(EVENT_CAMERAS), pos, neg)

    fold_reports: list[dict] = []
    if not args.skip_cv:
        folds = leave_one_camera_out_folds(rows, EVENT_CAMERAS)
        logging.info("running leave-one-camera-out CV, %d folds", len(folds))
        for i, (held_cam, train_rows, test_rows) in enumerate(folds, start=1):
            tp_tr, tn_tr = class_balance(train_rows)
            tp_te, tn_te = class_balance(test_rows)
            logging.info(
                "  fold %d/%d  held=%s  train=%d (f=%d n=%d)  test=%d (f=%d n=%d)",
                i, len(folds), held_cam,
                len(train_rows), tp_tr, tn_tr,
                len(test_rows), tp_te, tn_te,
            )
            metrics, _ = _train_one_fold(
                args.base_checkpoint, train_rows, test_rows,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
            )
            metrics["held_out_camera"] = held_cam
            fold_reports.append(metrics)
            logging.info(
                "  fold %d  AUROC=%.3f  F1=%.3f  P=%.3f  R=%.3f  (TP=%d FP=%d FN=%d TN=%d)",
                i, metrics["auroc"], metrics["f1"], metrics["precision"], metrics["recall"],
                metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"],
            )

    # Final model: fit on ALL cameras' data
    logging.info("training final model on all %d rows", len(rows))
    _, final_state = _train_one_fold(
        args.base_checkpoint, rows, rows,  # eval on train = report train metrics only
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
    )
    out_path = args.out_dir / "best.pt"
    _save_checkpoint(args.base_checkpoint, final_state, out_path)
    logging.info("saved retrained checkpoint -> %s", out_path)

    # Write summary
    summary = {
        "base_checkpoint": str(args.base_checkpoint),
        "out_checkpoint": str(out_path),
        "labels_used": len(rows),
        "n_flood": pos,
        "n_nonflood": neg,
        "event_cameras": list(EVENT_CAMERAS),
        "cv_folds": fold_reports,
        "cv_mean_auroc": (
            sum(m["auroc"] for m in fold_reports) / len(fold_reports)
            if fold_reports else None
        ),
        "cv_mean_f1": (
            sum(m["f1"] for m in fold_reports) / len(fold_reports)
            if fold_reports else None
        ),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    (args.out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# M11-b — Live corpus head-only fine-tune",
        "",
        f"**Base checkpoint:** `{args.base_checkpoint}`",
        f"**Out checkpoint:** `{out_path}`",
        f"**Labels:** {len(rows)} (flood={pos}, nonflood={neg}) on {len(EVENT_CAMERAS)} event cams",
        f"**Protocol:** freeze backbone, train final linear head only; {args.epochs} epochs, "
        f"lr={args.lr}, batch={args.batch_size}, seed={args.seed}.",
        "",
        "## Leave-one-camera-out cross-validation",
        "",
        "| Held-out camera | n_test | AUROC | F1 | Precision | Recall | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in fold_reports:
        md.append(
            f"| `{m['held_out_camera']}` | {m['n']} | {m['auroc']:.3f} | {m['f1']:.3f} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    if fold_reports:
        md.append(
            f"| **mean** | – | **{summary['cv_mean_auroc']:.3f}** | **{summary['cv_mean_f1']:.3f}** "
            f"| – | – | – | – | – | – |"
        )
    md += [
        "",
        "## Next step",
        "",
        "Re-run M11 with the retrained head:",
        "",
        "```bash",
        "rm results/live_predictions.json   # invalidate score cache",
        f"python scripts/run_eval_real.py --checkpoint {out_path}",
        "```",
        "",
    ]
    (args.out_dir / "cv_summary.md").write_text("\n".join(md))
    logging.info("done — see %s", args.out_dir)


if __name__ == "__main__":
    main()
