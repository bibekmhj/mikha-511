# Fine-tune (M8 / v0.2)

**Goal:** replace the COCO-pretrained `mask_area_frac` baseline with a task-appropriate scorer, and re-run the v0.1.0 evaluation table for a direct comparison.

## What is (and isn't) fine-tuned

The Mikha-Bench manifest carries **image-level labels** (`flood` / `nonflood`), not per-pixel masks. YOLOv8-seg's segmentation head cannot be fine-tuned without mask supervision, so M8 fine-tunes an **ImageNet-pretrained binary classifier** whose output is `P(flood | image) ∈ [0, 1]`. This value plugs straight into the eval harness's `ScoreFn` interface — the same interface the M6 baseline uses — so the M8 result is directly comparable, not just directionally.

If per-pixel mask labels are added in a future release, the segmentation head becomes the right fine-tune target and this doc is superseded. Until then, the classifier is the honest approach.

## Layout

```
src/mikha/train/
  dataset.py     PyTorch Dataset over manifest+splits with on-the-fly Mikha-Aug
  model.py       MobileNetV3-Small (default) or ResNet18, single-logit head
  train.py       Fixed-seed loop: AdamW + cosine LR + best-val-AUROC checkpoint
src/mikha/ref/
  classifier.py  WaterClassifier — loads checkpoint, exposes as_score_fn()
scripts/
  train_water.py --config configs/finetune_v1.yaml
  run_eval.py    --detector classifier --weights models/finetune_v1/best.pt
configs/
  finetune_v1.yaml   pinned hyperparameters
```

Everything torch-adjacent lives under `mikha.train` (and `mikha.ref.classifier`). The rest of the package — `mikha.aug`, `mikha.bench`, `mikha.eval` — stays torch-free.

## Reproducibility contract

Three fixed-seed layers keep runs byte-identical:

1. **Python / NumPy / torch seeds** set once at the top of `train.train()`.
2. **Per-sample augmentation seed** = `sha256(base_seed | row_id | epoch)`, so any sample's degraded pixels are the same across two runs with the same `--seed`.
3. **CUDA determinism** is best-effort (some ops don't have deterministic implementations); MPS and CPU are fully deterministic.

The full effective config is dumped to `<out_dir>/config.json` on the first epoch so a crashed run still tells us what it tried.

## Commands

Install (adds `torch`, `torchvision`, `pyyaml` via the `model` extra):

```bash
pip install -e ".[dev,model]"
```

Sanity check the config parser without torch or GPU work:

```bash
python scripts/train_water.py --print-config
```

Run the fine-tune with pinned defaults:

```bash
python scripts/train_water.py
# → models/finetune_v1/{best.pt, config.json, metrics.json}
```

Override anything on the CLI:

```bash
python scripts/train_water.py --epochs 30 --lr 1e-4 --batch-size 32 \
    --out-dir models/finetune_v2
```

Re-run the v0.1.0 eval table with the fine-tuned scorer:

```bash
python scripts/run_eval.py --detector classifier \
    --weights models/finetune_v1/best.pt \
    --out results/finetune_v1
# → results/finetune_v1/{table1.md, eval.json, pr_curves.png}
```

The baseline is unchanged and still reproduces with:

```bash
python scripts/run_eval.py          # defaults to --detector yolo
```

## Expected timings (593-row corpus, batch=16, 20 epochs)

| Device | Train time | Notes |
|---|---|---|
| Apple MPS (M-series Mac) | ~25 min | Pass `--device mps` explicitly if `auto` picks CPU on your box. |
| CUDA (RTX-class laptop GPU) | ~5 min | |
| CPU (any modern laptop) | ~2–4 h | Practical for a single retrain, painful for hyperparameter search. |

Val AUROC is logged per epoch; the best-by-val-AUROC checkpoint is saved to `best.pt`. Early stopping triggers after `early_stop_patience` epochs (default 5) with no improvement.

## Success criterion

The plan's kill-switch says: *"if baseline detector mIoU-clean < 0.5 with pretrained weights and one round of fine-tuning → switch backbone (try SAM 2 with a water prompt)."*

For a classifier, the equivalent threshold on the test split is:

- **Clean AUROC > 0.7** — a modest, honest bar for 593 rows.
- **All degradations AUROC > 0.6** — degradation-robust.

If the first `configs/finetune_v1.yaml` run clears both, publish `results/finetune_v1/table1.md` and move on. If it doesn't, the follow-ups in order are: (1) enlarge the corpus (import more from HydroShare / EF2013), (2) try `--backbone resnet18`, (3) switch to SAM 2 with a water text-prompt.

## Artifact layout after a run

```
models/finetune_v1/
├── best.pt          # torch checkpoint {model_state_dict, spec, epoch, val_auroc, config, image_size}
├── config.json      # effective TrainConfig (every knob), plus resolved device + pos_weight
└── metrics.json     # per-epoch train_loss, val_{auroc,precision,recall,f1,far_at_recall}, best summary
```

`best.pt` is a torch checkpoint — do **not** commit it to git. Add `models/` to `.gitignore` and ship it as a release asset attached to the v0.2 tag (or upload to Zenodo alongside the label bundle). The eval harness needs only this file to reproduce the fine-tuned numbers.

## Threats to validity (M8-specific)

- **Sample size.** 397-image train + 86-image val is small. Val AUROC will move ±0.03 across seeds; don't over-index on any single run.
- **Class prior in val.** With 86 val rows the class prior shifts noticeably per split; report both AUROC (prior-free) and FAR@R (prior-sensitive).
- **On-the-fly augmentation during training.** Training-time augmentation uses the same five transforms the eval stratifies on, which risks a subtle form of test contamination if the exact same `(row, seed)` combination appears in both. The eval seed (default 1234) is deliberately different from the train `base_seed` — leave it that way unless you have a reason.
- **Backbone drift.** MobileNetV3-Small was trained on ImageNet-1K, whose class distribution has very few outdoor-water scenes. Expect the first fine-tune to help substantially; expect a large second gain from more data, not from a bigger backbone.
