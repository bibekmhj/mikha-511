"""Write evaluation artifacts to ``results/``.

Baseline outputs:

* ``results/table1.md`` -- one markdown table with per-degradation P/R/F1/FAR/AUROC.
* ``results/eval.json`` -- full raw scores + params for reproducibility.
* ``results/pr_curves.png`` -- one PR curve per degradation (skipped if
  matplotlib is unavailable).

When calibration info is provided (via ``calibration=`` kwarg), also:

* ``results/calibration.png`` -- reliability diagrams before + after Platt scaling.
* Extra ECE columns in ``table1.md`` and ``eval.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .run import EvalResult


@dataclass(frozen=True)
class CalibrationBlock:
    """Per-degradation calibration info to attach to ``write_all``."""

    ece_before: float
    ece_after: float
    platt_a: float
    platt_b: float
    calibrated_scores: np.ndarray  # same length as test scores


def write_all(
    result: EvalResult,
    out_dir: str | Path,
    *,
    detector_name: str = "unknown",
    calibration: dict[str, CalibrationBlock] | None = None,
) -> dict:
    """Write table1.md + eval.json + pr_curves.png. Returns the paths written.

    When ``calibration`` is provided (keyed by degradation name), the
    markdown and JSON gain ECE-before / ECE-after columns and a
    ``calibration.png`` reliability diagram is emitted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    md_path = out_dir / "table1.md"
    md_path.write_text(
        _render_markdown(result, detector_name=detector_name, calibration=calibration),
        encoding="utf-8",
    )
    written["table_md"] = str(md_path)

    json_path = out_dir / "eval.json"
    json_path.write_text(
        json.dumps(
            _render_json(result, detector_name=detector_name, calibration=calibration), indent=2
        )
    )
    written["eval_json"] = str(json_path)

    plot_path = out_dir / "pr_curves.png"
    if _try_render_pr(result, plot_path, detector_name=detector_name):
        written["pr_png"] = str(plot_path)

    if calibration is not None:
        cal_path = out_dir / "calibration.png"
        if _try_render_calibration(result, calibration, cal_path, detector_name=detector_name):
            written["calibration_png"] = str(cal_path)

    return written


# ---------------------------------------------------------------------------


def _render_markdown(
    result: EvalResult,
    *,
    detector_name: str,
    calibration: dict[str, CalibrationBlock] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Mikha-Bench baseline - `{detector_name}`\n")
    lines.append(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    lines.append(
        f"Default threshold: **{result.default_threshold:.3f}**  "
        f"Target recall for FAR: **{result.target_recall:.2f}**\n"
    )
    lines.append("")

    if calibration:
        header = (
            "| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | "
            f"FAR@Recall {result.target_recall:.0%} | ECE (raw) | ECE (Platt) |"
        )
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = (
            "| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | "
            f"FAR@Recall {result.target_recall:.0%} |"
        )
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)

    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        row = (
            f"| {name} | {r.n_samples} | {r.n_flood} | {r.n_nonflood} | "
            f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | "
            f"{r.auroc:.3f} | {r.far_at_recall:.3f} |"
        )
        if calibration:
            block = calibration.get(name)
            if block is None:
                row += " - | - |"
            else:
                row += f" {block.ece_before:.3f} | {block.ece_after:.3f} |"
        lines.append(row)

    lines.append("")
    lines.append("**Interpretation notes:**")
    lines.append("")
    lines.append(
        "* FAR (false-alert rate) is the fraction of non-flood images "
        "incorrectly alerted at the smallest threshold that reaches the target recall."
    )
    lines.append("* AUROC = 0.5 means detector output is uncorrelated with the flood label.")
    if calibration:
        lines.append(
            "* ECE is the equal-width-bin expected calibration error "
            "(Guo et al. 2017, 10 bins). Lower is better; 0.0 = perfectly calibrated."
        )
        lines.append(
            "* Platt scaling is fit on the val split's scores per degradation, "
            "then applied to the test split's scores. Both ECE numbers are computed on test."
        )
    return "\n".join(lines) + "\n"


def _render_json(
    result: EvalResult,
    *,
    detector_name: str,
    calibration: dict[str, CalibrationBlock] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "detector": detector_name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "default_threshold": result.default_threshold,
        "target_recall": result.target_recall,
        "per_degradation": {},
    }
    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        block: dict[str, Any] = {
            "n_samples": r.n_samples,
            "n_flood": r.n_flood,
            "n_nonflood": r.n_nonflood,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "tn": m.tn,
            "fn": m.fn,
            "auroc": r.auroc,
            "auprc": r.pr.auprc,
            "far_at_recall": r.far_at_recall,
            "far_threshold": r.far_threshold,
            # Raw scores + labels retained so callers can re-plot without rerun.
            "scores": r.scores.tolist(),
            "labels": r.labels.astype(int).tolist(),
        }
        if calibration is not None and name in calibration:
            cb = calibration[name]
            block["calibration"] = {
                "ece_raw": float(cb.ece_before),
                "ece_platt": float(cb.ece_after),
                "platt_a": float(cb.platt_a),
                "platt_b": float(cb.platt_b),
                "calibrated_scores": cb.calibrated_scores.tolist(),
            }
        doc["per_degradation"][name] = block
    return doc


def _try_render_pr(result: EvalResult, out_path: Path, *, detector_name: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in result.per_degradation.items():
        if len(r.pr.recalls) == 0:
            continue
        ax.plot(r.pr.recalls, r.pr.precisions, label=f"{name} (AUPRC={r.pr.auprc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Mikha-Bench PR - {detector_name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _try_render_calibration(
    result: EvalResult,
    calibration: dict[str, CalibrationBlock],
    out_path: Path,
    *,
    detector_name: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    from .calibration import reliability_curve

    names = [n for n in result.per_degradation if n in calibration]
    if not names:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    ax_raw, ax_platt = axes
    ax_raw.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_platt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    for name in names:
        r = result.per_degradation[name]
        cb = calibration[name]

        raw_bins = reliability_curve(r.scores, r.labels, n_bins=10)
        ax_raw.plot(
            [b.mean_confidence for b in raw_bins],
            [b.empirical_positive_rate for b in raw_bins],
            marker="o",
            markersize=4,
            label=f"{name} (ECE={cb.ece_before:.3f})",
        )

        platt_bins = reliability_curve(cb.calibrated_scores, r.labels, n_bins=10)
        ax_platt.plot(
            [b.mean_confidence for b in platt_bins],
            [b.empirical_positive_rate for b in platt_bins],
            marker="o",
            markersize=4,
            label=f"{name} (ECE={cb.ece_after:.3f})",
        )

    for ax, title in ((ax_raw, "Raw scores"), (ax_platt, "After Platt scaling")):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")
    ax_raw.set_ylabel("Empirical positive rate")

    fig.suptitle(f"Mikha-Bench reliability - {detector_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
