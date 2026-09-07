# Results — v0.1.0 baseline

**Detector:** off-the-shelf YOLOv8n-seg with COCO weights.
**Score:** `mask_area_frac` — the fraction of the frame occupied by any predicted instance mask.
**Corpus:** Mikha-Bench test split (110 images per degradation class), 63 flood + 47 non-flood.
**Command:** `python scripts/run_eval.py` on manifest sha256 `3ee9e6cd…907f4b91`, splits file bound to the same hash.

## Table 1 — per-degradation baseline

| Degradation | N | Precision | Recall | F1 | AUROC | AUPRC | FAR @ 90% Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 110 | 0.125 | 0.063 | 0.084 | **0.159** | 0.42 | 0.872 |
| rain  | 110 | 0.206 | 0.111 | 0.144 | **0.232** | 0.45 | 0.915 |
| fog   | 110 | 0.143 | 0.063 | 0.088 | **0.143** | 0.43 | 0.787 |
| night | 110 | 0.158 | 0.048 | 0.073 | **0.194** | 0.50 | 0.596 |
| glare | 110 | 0.125 | 0.063 | 0.084 | **0.143** | 0.41 | 0.872 |
| jpeg  | 110 | 0.152 | 0.079 | 0.104 | **0.163** | 0.42 | 0.872 |

Reproduce: `python scripts/run_eval.py --threshold 0.05 --target-recall 0.9`.

## What this means

AUROC below 0.5 means the score is **anti-correlated** with the flood label. That is the expected outcome for a COCO-pretrained instance segmenter used as a flood detector via `mask_area_frac`:

- Non-flood traffic-camera frames contain cars, people, trucks, traffic lights — all COCO classes. The segmenter finds them, so the mask area is large.
- Flood frames often have submerged or blocked roadways with no COCO-detectable objects (vehicles avoid the scene, or are hidden under water). Mask area is small.

The signal is therefore backwards, and no threshold on `mask_area_frac` will beat chance on this corpus. `night` gives the highest AUPRC (0.50) because low-light images produce fewer confident COCO detections regardless of class, weakening the anti-correlation.

## Why we still ship this as v0.1

Two reasons.

**First — the harness is what v0.1 delivers.** `mikha.eval` is score-fn agnostic; the baseline exercises every code path (per-degradation stratification, PR / AUROC / FAR@R computation, JSON + Markdown + PNG output). Anyone who trains a better `score_fn` — a fine-tuned water-segmentation head, a CLIP-based water classifier, a hand-crafted color+texture score — can `python scripts/run_eval.py` with their model and directly compare.

**Second — this result is the plan-expected trigger for the fine-tune step.** The implementation plan's kill-switch section says: *"if baseline detector mIoU-clean < 0.5 with pretrained weights and one round of fine-tuning → switch backbone (try SAM 2 with a water prompt) rather than expand data collection."* The v0.1 baseline is well below that. v0.2 will fine-tune a water head on the base set and re-run this table. If the fine-tune does not clear AUROC > 0.7 on clean, the backbone switch is next.

## Threats to validity

- **Corpus scale.** 593 rows is small. It's what a single developer could license-verify individually; it is not what a state DOT should deploy against. The bench is a research instrument, not a certification suite.
- **Class prior.** The corpus is roughly balanced (54 % flood / 46 % non-flood) by design, but real 511 streams are overwhelmingly non-flood. FAR@R is the metric to watch for deployment realism, not F1.
- **Group leakage.** The greedy load-balanced split (`bench/splits.json`) prevents `(source, uploader)`-level leakage but cannot detect "same location shot at different times" — two Wikimedia uploads of the same flooded intersection by different photographers will land in different splits.
- **Degradation realism.** Mikha-Aug approximates 511 conditions synthetically. A future release should compare against a real-511 held-out set (FL 511 / TX 511 nightly captures) with labels-only redistribution, per the roadmap.
- **Single detector.** YOLOv8n-seg is the smallest Ultralytics segmentation model. Larger COCO variants (`yolov8m-seg`, `yolov8x-seg`) will not fix the anti-correlation — the class vocabulary is wrong, not the capacity — but should be measured before claiming fine-tune is necessary.

## Reproducibility

Every table and figure here comes from one command with a pinned seed:

```
python scripts/run_eval.py                # default seed 1234
```

Determinism is enforced at three levels:
1. The manifest is content-addressed by SHA-256; splits are bound to that hash.
2. Degradation transforms hash `(row_id ⊕ eval_seed)` into their own RNG.
3. The detector receives byte-identical input for a given (row, degradation, seed).

`results/eval.json` retains per-sample scores and labels so any downstream metric can be recomputed without re-running inference.
