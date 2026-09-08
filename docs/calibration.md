# Confidence calibration (M9)

The plan promised Platt scaling on the val split and a `calibration.png`
reliability diagram alongside `table1.md` and `pr_curves.png` (plan
sections 7 and 8). This page describes how to produce them and how to
read what comes out.

## What Platt scaling does

Any raw score in `[0, 1]` can be discriminative (high AUROC) while
being poorly calibrated (a score of 0.9 does not mean 90% chance of
being flood). Platt scaling fits a small sigmoid `p = sigmoid(a * s + b)`
that maps the raw score to a probability with matched frequencies. Two
parameters, so it does nothing to AUROC or FAR@R (both are rank
statistics), but it can move ECE (expected calibration error) toward 0.

## Commands

Fit on val, apply to test:

```bash
python scripts/run_eval.py \
    --detector classifier \
    --weights models/finetune_v1/best.pt \
    --threshold 0.5 \
    --calibrate \
    --out results/finetune_v1_cal
```

Outputs land in `results/finetune_v1_cal/`:

- `table1.md` gains two columns: `ECE (raw)` and `ECE (Platt)`.
- `eval.json` gains a `calibration` block per degradation with `ece_raw`,
  `ece_platt`, `platt_a`, `platt_b`, and the full calibrated score vector.
- `calibration.png` shows one reliability diagram per degradation, side
  by side: raw on the left, Platt-scaled on the right. The dashed diagonal
  is perfect calibration. Points above the diagonal mean the model was
  under-confident on that bin; points below mean over-confident.

## Flags

- `--calibrate` turns on the whole path. Without it, `write_all` behaves
  as it did in v0.2.0 (no ECE columns, no `calibration.png`).
- `--calibrate-split val` picks the split Platt is fit on. Default is
  `val` per the plan. If you set it to a split that is also in
  `--splits`, the driver prints a warning because you are then reporting
  ECE on the same rows Platt saw during fitting.

## Interpreting ECE

ECE is the equal-width-bin form (Guo et al. 2017, 10 bins):

```
ECE = sum_b (n_b / N) * |mean_confidence_b - empirical_positive_rate_b|
```

Lower is better. 0.0 is perfect calibration. A 100-sample benchmark
gives noisy ECE at the ~0.02-0.05 floor even for a perfectly calibrated
scorer, so any improvement smaller than that is not meaningful on the
current 110-image test split.

## What happens when Platt cannot fit

Degradations whose calibration split contains fewer than 2 flood or 2
non-flood samples are skipped. The driver prints a warning and those
degradations get no calibration entry in the table or JSON. This is a
guardrail against a Platt fit collapsing on a degenerate class balance.
