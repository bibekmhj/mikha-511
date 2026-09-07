"""Write evaluation artifacts to ``results/``.

Three outputs:

* ``results/table1.md`` — one markdown table with per-degradation P/R/F1/FAR/AUROC.
* ``results/eval.json`` — full raw scores + params for reproducibility.
* ``results/pr_curves.png`` — one PR curve per degradation (skipped if
  matplotlib is unavailable).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .run import EvalResult


def write_all(result: EvalResult, out_dir: str | Path, *, detector_name: str = "unknown") -> dict:
    """Write table1.md + eval.json + pr_curves.png. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    md_path = out_dir / "table1.md"
    md_path.write_text(_render_markdown(result, detector_name=detector_name), encoding="utf-8")
    written["table_md"] = str(md_path)

    json_path = out_dir / "eval.json"
    json_path.write_text(json.dumps(_render_json(result, detector_name=detector_name), indent=2))
    written["eval_json"] = str(json_path)

    plot_path = out_dir / "pr_curves.png"
    if _try_render_pr(result, plot_path, detector_name=detector_name):
        written["pr_png"] = str(plot_path)

    return written


# ---------------------------------------------------------------------------


def _render_markdown(result: EvalResult, *, detector_name: str) -> str:
    lines: list[str] = []
    lines.append(f"# Mikha-Bench baseline — `{detector_name}`\n")
    lines.append(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    lines.append(
        f"Default threshold: **{result.default_threshold:.3f}**  "
        f"Target recall for FAR: **{result.target_recall:.2f}**\n"
    )
    lines.append("")
    lines.append(
        "| Degradation | N | Flood | Non-flood | Precision | Recall | F1 | AUROC | "
        f"FAR@Recall {result.target_recall:.0%} |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    # "clean" first, then bench transforms in canonical order.
    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        lines.append(
            f"| {name} | {r.n_samples} | {r.n_flood} | {r.n_nonflood} | "
            f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | "
            f"{r.auroc:.3f} | {r.far_at_recall:.3f} |"
        )
    lines.append("")
    lines.append("**Interpretation notes:**")
    lines.append("")
    lines.append(
        "* FAR (false-alert rate) is the fraction of non-flood images "
        "incorrectly alerted at the smallest threshold that reaches the target recall."
    )
    lines.append("* AUROC = 0.5 means detector output is uncorrelated with the flood label.")
    lines.append(
        "* On COCO-pretrained weights the detector is not a water detector; low F1/AUROC "
        "here is the plan-expected baseline and triggers the fine-tune step "
        "(gate: mIoU < 0.6)."
    )
    return "\n".join(lines) + "\n"


def _render_json(result: EvalResult, *, detector_name: str) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "detector": detector_name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "default_threshold": result.default_threshold,
        "target_recall": result.target_recall,
        "per_degradation": {},
    }
    for name, r in result.per_degradation.items():
        m = r.metrics_at_default
        doc["per_degradation"][name] = {
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
    ax.set_title(f"Mikha-Bench PR — {detector_name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
