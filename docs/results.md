# Results

Two model runs live in this repo, plus a calibration pass on top of the second one. The v0.1.0 baseline uses off-the-shelf YOLOv8n-seg with COCO weights, scored by `mask_area_frac`. The v0.2.0 fine-tune uses an ImageNet-pretrained MobileNetV3-Small classifier trained on the base set for one afternoon on a MacBook Air. v0.2.1 adds Platt scaling of the classifier's probabilities, fit on the val split and applied to the test split. Same 110-image test split for all three, same seed, same manifest.

## Table 1 - per-degradation

| Degradation | N | Model | Precision | Recall | F1 | AUROC | AUPRC | FAR @ 90% Recall |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 110 | YOLOv8n-seg (v0.1.0) | 0.125 | 0.063 | 0.084 | 0.159 | 0.42 | 0.872 |
| clean | 110 | classifier (v0.2.0)  | 0.833 | 0.873 | 0.853 | 0.923 | 0.95 | 0.277 |
| rain  | 110 | YOLOv8n-seg (v0.1.0) | 0.206 | 0.111 | 0.144 | 0.232 | 0.45 | 0.915 |
| rain  | 110 | classifier (v0.2.0)  | 0.685 | 0.968 | 0.803 | 0.939 | 0.96 | 0.149 |
| fog   | 110 | YOLOv8n-seg (v0.1.0) | 0.143 | 0.063 | 0.088 | 0.143 | 0.43 | 0.787 |
| fog   | 110 | classifier (v0.2.0)  | 0.778 | 0.889 | 0.830 | 0.901 | 0.94 | 0.426 |
| night | 110 | YOLOv8n-seg (v0.1.0) | 0.158 | 0.048 | 0.073 | 0.194 | 0.50 | 0.596 |
| night | 110 | classifier (v0.2.0)  | 0.740 | 0.905 | 0.814 | 0.882 | 0.90 | 0.404 |
| glare | 110 | YOLOv8n-seg (v0.1.0) | 0.125 | 0.063 | 0.084 | 0.143 | 0.41 | 0.872 |
| glare | 110 | classifier (v0.2.0)  | 0.824 | 0.889 | 0.855 | 0.926 | 0.96 | 0.277 |
| jpeg  | 110 | YOLOv8n-seg (v0.1.0) | 0.152 | 0.079 | 0.104 | 0.163 | 0.42 | 0.872 |
| jpeg  | 110 | classifier (v0.2.0)  | 0.833 | 0.873 | 0.853 | 0.924 | 0.96 | 0.298 |

v0.1.0 was scored at threshold 0.05 (a value that fits `mask_area_frac`). v0.2.0 was scored at threshold 0.5, which is the natural operating point for a sigmoid classifier. AUROC and FAR@R do not depend on the threshold, so the improvement they show is real either way.

## What v0.2.0 changes

AUROC crosses from anti-signal to strong signal on every class. The v0.1.0 baseline produced AUROC below 0.5 everywhere, which means `mask_area_frac` was pointing the wrong direction (COCO segmentation finds cars, people, and traffic lights, which are more common in non-flood traffic-camera frames than in flood frames). The fine-tuned classifier reverses this by learning the actual visual pattern of standing water on roadways.

AUROC uplift by class:

| Degradation | v0.1.0 | v0.2.0 | Change |
|---|---:|---:|---:|
| clean | 0.159 | 0.923 | +0.764 |
| rain  | 0.232 | 0.939 | +0.707 |
| fog   | 0.143 | 0.901 | +0.758 |
| night | 0.194 | 0.882 | +0.688 |
| glare | 0.143 | 0.926 | +0.783 |
| jpeg  | 0.163 | 0.924 | +0.761 |

FAR at 90% recall drops from the 0.60-0.92 range down to 0.15-0.43. Under a normal traffic-camera stream where roads are almost always passable, that gap decides whether alerts are useful or noise.

The success bar written in `docs/finetune.md` was clean AUROC > 0.70 and every degradation AUROC > 0.60. Both are cleared with margin.

## Training details for v0.2.0

Backbone: MobileNetV3-Small, ImageNet-pretrained via torchvision. Single logit head replaces the 1000-class ImageNet classifier. Trained on 397 base-set images (train split), validated on 86 (val split). Test split of 110 is held out and never touched during training.

Loss: BCEWithLogits with `pos_weight = n_neg / n_pos` computed from the train split so class imbalance does not push the model toward the majority class.

Optimizer: AdamW, lr 3e-4, weight decay 1e-4. Linear warmup for 1 epoch then cosine annealing over the remaining epochs. Cap at 20 epochs, early stop patience of 5.

Augmentation during training: with probability 0.5 per sample, apply one of the 5 Mikha-Aug transforms at severity 0.65. The per-sample seed is `sha256(base_seed | row_id | epoch)`, so the augmentation is deterministic across runs with the same `--seed`. The eval seed and the train seed are used in different formulas, so there is no seed overlap between the training augmentations and the eval augmentations for the same row.

Training curve on val:

```
ep 00: loss 0.667  val AUROC 0.661  F1 0.680  FAR@90% 0.882   (warmup)
ep 01: loss 0.496  val AUROC 0.896  F1 0.889  FAR@90% 0.235
ep 02: loss 0.268  val AUROC 0.932  F1 0.899  FAR@90% 0.118   (best)
ep 03: loss 0.150  val AUROC 0.924  F1 0.875  FAR@90% 0.206
ep 04: loss 0.106  val AUROC 0.915  F1 0.887  FAR@90% 0.206
ep 05: loss 0.072  val AUROC 0.912  F1 0.879  FAR@90% 0.235
ep 06: loss 0.087  val AUROC 0.902  F1 0.881  FAR@90% 0.176
ep 07: loss 0.040  val AUROC 0.911  F1 0.876  FAR@90% 0.235   (early stop)
```

Best checkpoint is the ep 02 model. Wall clock was about 25 minutes across 8 epochs on CPU (macOS, M-series, no MPS acceleration used in this run).

Val AUROC (0.932) and test-clean AUROC (0.923) are within 0.01 of each other, which is what a clean split of this size should produce. A leaky split would show val far above test.

## Threats to validity

Corpus size. 397 train + 86 val + 110 test is small. Numbers here move by roughly 0.02-0.03 AUROC across seeds. Do not read the fifth digit.

Class prior. The corpus is close to 50/50 by design, but real 511 streams are overwhelmingly non-flood. FAR at fixed recall is the metric that matters for deployment realism, not F1.

Aug used both at train time and eval time. The same 5 transforms drive both training augmentation and the eval stratification. Per-row seeds differ between the two paths, so there is no exact (row, degradation, seed) overlap, but the model has still seen the general shape of each transform during training. Real held-out imagery (real 511 captures, no synthetic aug) is the honest next validation and is scheduled for v0.3.

Group leakage from co-located photos. The split is group-aware at the uploader level, but two Wikimedia users who both photographed the same flooded intersection from different angles could land in different splits. Manual review would be needed to catch this at scale.

Single backbone. MobileNetV3-Small was picked for speed on a laptop. Larger backbones (ResNet-18 is wired in, larger variants are not) may help, but the current numbers already clear the bar so the case for spending more compute is weak until more data exists.

## Calibration (v0.2.1)

The plan called for Platt scaling of the scorer confidence on the val split with calibration curves and ECE reported alongside the table. That never shipped in v0.1.0 and got the machinery wrong the first time in v0.2.0. v0.2.1 fixes it: the eval driver now takes `--calibrate`, which fits a Platt scaler per degradation on the val split (in logit space, not probability space) and applies it to the test scores.

| Degradation | ECE raw | ECE Platt | Change | Platt a | Platt b |
|---|---:|---:|---:|---:|---:|
| clean | 0.085 | 0.083 | -0.002 | 0.60 | -0.23 |
| rain  | 0.190 | 0.137 | -0.053 | 0.57 | -0.53 |
| fog   | 0.106 | 0.101 | -0.005 | 0.66 | -0.37 |
| night | 0.114 | 0.069 | -0.045 | 0.65 | -0.19 |
| glare | 0.080 | 0.128 | +0.048 | 0.58 | -0.23 |
| jpeg  | 0.077 | 0.097 | +0.020 | 0.60 | -0.25 |

Reproduce: `python scripts/run_eval.py --detector classifier --weights models/finetune_v1/best.pt --threshold 0.5 --calibrate --out results/finetune_v1_cal`. Outputs land in `results/finetune_v1_cal/`, including a `calibration.png` with raw and Platt reliability diagrams side by side.

How to read this. ECE is the equal-width-bin expected calibration error (Guo et al. 2017, 10 bins). Lower is better, 0.0 is perfect calibration. Every Platt `a` sits below 1.0, which means the classifier is systematically overconfident and Platt is softening the logits. That is the usual outcome for BCE-trained classifiers on small training sets. Rankings are preserved because Platt is monotone, so AUROC and FAR at fixed recall are unaffected.

Where Platt helps and where it does not. The two worst-calibrated degradations improve the most (rain drops from 0.190 to 0.137, night drops from 0.114 to 0.069). Clean and fog were already close to their floor, so Platt barely moves them. Glare and jpeg get slightly worse, which is expected: Platt is a two-parameter global fit per degradation, and on a well-calibrated input it cannot beat identity except by luck. The fit is also noisy because the val split has 86 samples split across degradations, roughly 14 per bin. Numbers here move by a few points across seeds. Mean ECE across all six drops from 0.109 to 0.103, so the aggregate direction is right, and the biggest gains show up where they matter.

If you care about individual-degradation calibration more than overall, use the calibrated scorer selectively (rain and night win, others break even at best). If you care about ranking or FAR at fixed recall, calibration is irrelevant and the raw v0.2.0 scores are already the right thing to compare on.

## Reproducibility

Every number in this document comes from these commands, in this order, on the manifest committed at v0.2.0:

```bash
pip install -e '.[dev,model]'
python scripts/fetch_base.py                       # populate data/images/
python scripts/train_water.py                      # writes models/finetune_v1/best.pt
python scripts/run_eval.py                         # writes results/  (v0.1.0 baseline)
python scripts/run_eval.py \
    --detector classifier \
    --weights models/finetune_v1/best.pt \
    --threshold 0.5 \
    --out results/finetune_v1                      # v0.2.0 fine-tune result
python scripts/run_eval.py \
    --detector classifier \
    --weights models/finetune_v1/best.pt \
    --threshold 0.5 \
    --calibrate \
    --out results/finetune_v1_cal                  # v0.2.1 with Platt calibration
```

Determinism holds because:

1. The manifest is content-addressed by SHA-256 and `bench/splits.json` is bound to that hash.
2. Training seeds Python, NumPy, and torch from a single `--seed`.
3. Degradation transforms hash `(base_seed, row_id, epoch)` for training and `(row_id, eval_seed)` for eval into their own RNG streams.
4. Each detector receives byte-identical input for a given (row, degradation, seed).

`results/eval.json` retains every per-sample score and label so any downstream metric (calibration, per-source breakdown, cost-weighted alert curves) can be recomputed without rerunning inference.
