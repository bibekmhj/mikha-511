"""Tests for ``scripts/run_eval_real.py``.

Isolates the parts that do NOT need torch/cv2 — label parsing, gate
dispatch per camera, metric computation, and FAR@recall sweeps. The
classifier pass is exercised with a dummy stub to keep the suite fast
and dependency-free.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_eval_real.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_eval_real", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_real"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# label loading


def test_load_labels_drops_uncertain_and_skipped(tmp_path: Path) -> None:
    mod = _load_mod()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "meta": {"labeling_protocol_version": 1},
                "labels": {
                    "id_f": {"image_class": "flood"},
                    "id_n": {"image_class": "nonflood"},
                    "id_u": {"image_class": "uncertain"},
                    "id_s": {"image_class": "skip"},
                },
            }
        )
    )
    out = mod.load_labels(labels_path)
    assert out == {"id_f": "flood", "id_n": "nonflood"}


# ---------------------------------------------------------------------------
# gate dispatch


class _StubGate:
    """Minimal stand-in for ``PersistenceGate`` for groupby tests.

    Rule: emit ``is_persistent=True`` iff the last-observed value is True.
    Trivial but lets us assert that the dispatcher sorts by time and
    segregates per camera.
    """

    def __init__(self, *, window: int, threshold_k: int, clear_grace: int) -> None:
        self.window = window
        self.threshold_k = threshold_k
        self.clear_grace = clear_grace
        self._last = False

    def observe(self, positive: bool):
        self._last = bool(positive)
        from types import SimpleNamespace

        return SimpleNamespace(is_persistent=self._last, n_positive_in_window=int(self._last))


def test_run_gate_groups_by_camera_and_sorts_by_time() -> None:
    mod = _load_mod()
    manifest = {
        "r1": {"camera_id": "A", "captured_at": "2026-11-26T12:00:00Z", "local_path": "x"},
        "r2": {"camera_id": "A", "captured_at": "2026-11-26T12:30:00Z", "local_path": "x"},
        "r3": {"camera_id": "B", "captured_at": "2026-11-26T12:30:00Z", "local_path": "x"},
    }
    scores = {"r1": 0.9, "r2": 0.1, "r3": 0.95}
    out = mod.run_gate_per_camera(
        scores,
        manifest,
        ("A", "B"),
        threshold=0.5,
        window=5,
        threshold_k=3,
        clear_grace=2,
        gate_cls=_StubGate,
    )
    # r1 first on cam A → gate sees True; r2 after → False (stub tracks last)
    assert out["r1"]["gate_state"] is True
    assert out["r2"]["gate_state"] is False
    # cam B evaluated independently, not polluted by A's history
    assert out["r3"]["gate_state"] is True
    # dead-score rows are skipped entirely
    scores_with_none = {**scores, "r2": None}
    out2 = mod.run_gate_per_camera(
        scores_with_none,
        manifest,
        ("A", "B"),
        threshold=0.5,
        window=5,
        threshold_k=3,
        clear_grace=2,
        gate_cls=_StubGate,
    )
    assert "r2" not in out2


# ---------------------------------------------------------------------------
# metrics


def test_binary_report_computes_pr_f1() -> None:
    mod = _load_mod()
    y_true = np.array([True, True, True, False, False])
    y_pred = np.array([True, True, False, True, False])
    r = mod.binary_report(y_pred, y_true)
    # tp=2 fp=1 fn=1 tn=1 → p=2/3, r=2/3, f1=2/3
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1 and r["tn"] == 1
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_far_at_recall_picks_least_strict_threshold_meeting_target() -> None:
    mod = _load_mod()
    # Three positives with scores 0.9, 0.7, 0.3 and three negatives at 0.8, 0.5, 0.1
    scores = np.array([0.9, 0.7, 0.3, 0.8, 0.5, 0.1])
    y_true = np.array([True, True, True, False, False, False])
    far, thr = mod.far_at_recall(scores, y_true, target_recall=0.66)
    # Need recall >= 0.66, i.e. >= 2 of 3 positives → threshold 0.7
    # At thr=0.7: pred = [T,T,F,T,F,F] → TP=2 FP=1 FN=1 TN=2 → FAR = 1/3
    assert thr == pytest.approx(0.7)
    assert far == pytest.approx(1 / 3)


def test_far_at_recall_returns_nan_when_target_unachievable() -> None:
    mod = _load_mod()
    # all-negative ground truth
    scores = np.array([0.1, 0.2, 0.3])
    y_true = np.array([False, False, False])
    far, thr = mod.far_at_recall(scores, y_true, target_recall=0.9)
    assert np.isnan(far) and np.isnan(thr)


# ---------------------------------------------------------------------------
# scoring pass — cache hit short-circuits classifier


def test_score_all_frames_uses_cache_and_skips_classifier(tmp_path: Path) -> None:
    mod = _load_mod()

    class _NoCallClassifier:
        def score(self, _img):
            raise AssertionError("classifier should not be called when cache covers all rows")

    manifest = {
        "r1": {
            "camera_id": "A",
            "captured_at": "2026-11-26T12:00:00Z",
            "local_path": str(tmp_path / "not_actually_read.jpg"),
        },
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"r1": 0.42}))
    out = mod.score_all_frames(
        _NoCallClassifier(),
        manifest,
        cam_ids=("A",),
        date_prefixes=("20261126",),
        cache_path=cache_path,
        cv2_module=None,  # not needed because cache hit
    )
    assert out["r1"] == 0.42


def test_score_all_frames_handles_unreadable_image(tmp_path: Path) -> None:
    mod = _load_mod()

    class _DummyClassifier:
        def score(self, _img):
            return 0.7  # should not be called for the None path

    class _DummyCv2:
        @staticmethod
        def imread(_path):
            return None  # simulate a corrupt file on disk

    manifest = {
        "r1": {
            "camera_id": "A",
            "captured_at": "2026-11-26T12:00:00Z",
            "local_path": str(tmp_path / "missing.jpg"),
        },
    }
    cache_path = tmp_path / "cache.json"
    out = mod.score_all_frames(
        _DummyClassifier(),
        manifest,
        cam_ids=("A",),
        date_prefixes=("20261126",),
        cache_path=cache_path,
        cv2_module=_DummyCv2(),
    )
    assert out["r1"] is None
    reloaded = json.loads(cache_path.read_text())
    assert reloaded == {"r1": None}
