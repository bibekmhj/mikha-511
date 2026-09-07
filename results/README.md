# `results/` - reproducible evaluation outputs

Every artifact in this directory is regenerable by one command:

```bash
python scripts/run_eval.py
```

The v0.1.0 release commits the following:

| File | What it is |
|---|---|
| `table1.md` | Per-degradation-class metrics (P, R, F1, AUROC, AUPRC, FAR@0.9R) rendered as GitHub-flavored markdown. |
| `eval.json` | The full result envelope — per-sample scores and labels for every degradation class — so any downstream metric can be recomputed without re-running inference. |
| `pr_curves.png` | One precision-recall curve per degradation class, AUPRC in the legend. |

## Regenerate

```bash
# Full test split, all 6 degradations (~15 min CPU, ~2 min GPU):
python scripts/run_eval.py

# Fast smoke (~1 min):
python scripts/run_eval.py --limit-per-class 5

# Different operating point:
python scripts/run_eval.py --threshold 0.10 --target-recall 0.8
```

Determinism is fixed by (1) the manifest SHA-256 in `data/manifest.csv`, (2) `bench/splits.json` (bound to that hash), and (3) the eval `--seed` (default 1234, hashed per-row into the degradation RNG). Two runs with the same inputs produce byte-identical outputs.

## Interpretation

See `docs/results.md` for the full discussion of the v0.1.0 baseline (spoiler: COCO weights + `mask_area_frac` produces AUROC < 0.5, which is the plan-expected trigger for the v0.2 fine-tune step).
