"""Tests for ``scripts/finetune_live.py``.

Covers the pure-Python surface: label loading, manifest join, leave-one-
camera-out fold construction, class balance accounting, and the hand-rolled
Mann-Whitney AUROC / binary metrics so the CV summary numbers are
trustworthy without torch.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "finetune_live.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("finetune_live", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["finetune_live"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Label + manifest + row assembly


def test_load_labels_filters_to_flood_and_nonflood(tmp_path: Path) -> None:
    mod = _load_mod()
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({
        "meta": {"labeling_protocol_version": 1},
        "labels": {
            "a": {"image_class": "flood"},
            "b": {"image_class": "nonflood"},
            "c": {"image_class": "uncertain"},
            "d": {"image_class": "skip"},
        },
    }))
    assert mod.load_labels(p) == {"a": "flood", "b": "nonflood"}


def test_assemble_labeled_rows_restricts_to_cams(tmp_path: Path) -> None:
    mod = _load_mod()
    labels = {"r1": "flood", "r2": "nonflood", "r3": "flood"}
    manifest = {
        "r1": {"id": "r1", "camera_id": "A", "local_path": "/tmp/a.jpg"},
        "r2": {"id": "r2", "camera_id": "B", "local_path": "/tmp/b.jpg"},
        "r3": {"id": "r3", "camera_id": "OTHER", "local_path": "/tmp/x.jpg"},
    }
    out = mod.assemble_labeled_rows(labels, manifest, ("A", "B"))
    assert sorted((r.row_id, r.label) for r in out) == [("r1", 1), ("r2", 0)]


# ---------------------------------------------------------------------------
# Fold construction


def test_leave_one_camera_out_folds_builds_correct_splits() -> None:
    mod = _load_mod()
    Row = mod.LabeledRow  # noqa: N806
    rows = [
        Row("rA1", "A", "/A1.jpg", 1),
        Row("rA2", "A", "/A2.jpg", 0),
        Row("rB1", "B", "/B1.jpg", 1),
        Row("rC1", "C", "/C1.jpg", 0),
        Row("rC2", "C", "/C2.jpg", 1),
    ]
    folds = mod.leave_one_camera_out_folds(rows, ("A", "B", "C"))
    assert [f[0] for f in folds] == ["A", "B", "C"]
    # fold held=A → test has only A rows, train has B+C
    held_a, tr_a, te_a = folds[0]
    assert {r.row_id for r in te_a} == {"rA1", "rA2"}
    assert {r.row_id for r in tr_a} == {"rB1", "rC1", "rC2"}


def test_leave_one_camera_out_skips_missing_cam() -> None:
    mod = _load_mod()
    Row = mod.LabeledRow  # noqa: N806
    rows = [Row("x", "A", "/x.jpg", 1), Row("y", "B", "/y.jpg", 0)]
    # C has no rows, should be silently skipped (not break the loop)
    folds = mod.leave_one_camera_out_folds(rows, ("A", "B", "C"))
    assert [f[0] for f in folds] == ["A", "B"]


def test_class_balance_counts() -> None:
    mod = _load_mod()
    Row = mod.LabeledRow  # noqa: N806
    rows = [Row("a", "A", "", 1), Row("b", "A", "", 1), Row("c", "A", "", 0)]
    pos, neg = mod.class_balance(rows)
    assert pos == 2 and neg == 1


# ---------------------------------------------------------------------------
# AUROC + binary metrics


def test_auroc_perfect_separation() -> None:
    mod = _load_mod()
    m = mod._compute_binary_metrics(
        y_true=[1, 1, 1, 0, 0, 0],
        y_pred_prob=[0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
    )
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 3 and m["fp"] == 0


def test_auroc_random() -> None:
    mod = _load_mod()
    m = mod._compute_binary_metrics(
        y_true=[1, 0, 1, 0],
        y_pred_prob=[0.5, 0.5, 0.5, 0.5],
    )
    # all ties → wins = 0.5 per pair → auroc = 0.5
    assert m["auroc"] == 0.5


def test_binary_metrics_counts_at_threshold() -> None:
    mod = _load_mod()
    m = mod._compute_binary_metrics(
        y_true=[1, 1, 0, 0],
        y_pred_prob=[0.9, 0.4, 0.7, 0.1],
        threshold=0.5,
    )
    assert m["tp"] == 1 and m["fp"] == 1
    assert m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_auroc_nan_when_single_class() -> None:
    mod = _load_mod()
    m = mod._compute_binary_metrics(y_true=[0, 0, 0], y_pred_prob=[0.1, 0.2, 0.3])
    assert math.isnan(m["auroc"])
