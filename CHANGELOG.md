# Changelog

All notable changes to Mikha-511 are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-09-08

Calibration release. Closes the M5 and M6 plan gaps around Platt scaling and reliability diagrams that never shipped in v0.1.0. No changes to the manifest, splits, model, or training code. The v0.2.0 evaluation still reproduces byte-identically.

### Added

- **`mikha.eval.calibration`** - new module. `expected_calibration_error` (Guo et al. 2017, 10 equal-width bins), `reliability_curve`, `fit_platt`, `apply_platt`. Pure numpy, no torch.
- **`scripts/run_eval.py`** - new `--calibrate` and `--calibrate-split` flags. When on, the driver runs eval on the calibration split (default `val`), fits Platt per degradation in logit space, applies it to the test scores, and reports ECE before and after.
- **`results/calibration.png`** is now emitted when `--calibrate` is set. Two panels side by side: raw and Platt reliability diagrams per degradation, with the identity diagonal for reference.
- **`table1.md`** gains `ECE (raw)` and `ECE (Platt)` columns when calibration is on. `eval.json` gains a `calibration` block per degradation with `ece_raw`, `ece_platt`, `platt_a`, `platt_b`, and the full calibrated score vector.
- **`docs/calibration.md`** - what the flag does, when to use it, how to read ECE.
- **`docs/results.md`** - new Calibration section with the v0.2.1 table and the honest read of where Platt helps and where it does not.
- 14 new tests in `tests/test_calibration.py` and `tests/test_eval_run.py` covering ECE hand-picked cases, reliability curve edge cases, input validation, Platt on logits vs probabilities, and the write_all calibration round-trip. Includes a regression test that fails on the v0.2.0-shipped bug (Platt fit on probabilities) and passes on the v0.2.1 fix.

### Fixed

- Platt scaling in the eval pipeline is now fit on logits, not on probabilities. The v0.2.0-shipped path passed the classifier's sigmoid outputs directly to `PlattScaler.fit`, which stalled gradient descent at `a ~ 1.0, b ~ -0.08` and collapsed every calibrated score into a narrow band around 0.5. That inflated ECE across the board (clean 0.085 -> 0.290 in the first run). v0.2.1 logit-transforms inputs before Platt fit and apply, so `a` converges to a sensible value and ECE moves in the right direction on the worst-calibrated cases.

### v0.2.1 calibration results (see `docs/results.md`)

| Degradation | ECE raw | ECE Platt | Change |
|---|---:|---:|---:|
| clean | 0.085 | 0.083 | -0.002 |
| rain  | 0.190 | 0.137 | -0.053 |
| fog   | 0.106 | 0.101 | -0.005 |
| night | 0.114 | 0.069 | -0.045 |
| glare | 0.080 | 0.128 | +0.048 |
| jpeg  | 0.077 | 0.097 | +0.020 |

Platt helps most on the worst-calibrated degradations (rain and night), leaves the already-well-calibrated ones essentially unchanged, and slightly worsens two of the six because a 2-parameter global fit on 14-sample-per-bin val cannot beat identity on inputs that are already close to calibrated. Rankings are preserved (Platt is monotone), so AUROC and FAR at fixed recall are unaffected.

### Known limitations

- Val split has 86 samples across 6 degradations, roughly 14 per Platt fit. Individual ECE numbers move by 0.02-0.03 across seeds. Do not over-index on a single run.
- Platt is a 2-parameter monotone map. It cannot fix miscalibrations that are non-monotone in the raw score. Isotonic regression would fit better on this scale but the plan explicitly named Platt scaling.

---

## [0.2.0] - 2026-09-07

Fine-tune release. The v0.1.0 baseline was YOLOv8n-seg with COCO weights, scored by `mask_area_frac`. That produced AUROC below 0.5 on every degradation class because COCO segmentation finds cars and people, which are more common in non-flood traffic-camera frames than in flood frames. This release trains a small water classifier on the same 593-row base set and plugs it into the same eval harness. AUROC crosses from anti-signal to strong signal on every class, and the FAR at 90% recall drops from the 0.60-0.92 range down to 0.15-0.43.

### Added

- **`mikha.train`** - new subpackage with a PyTorch `Dataset` over the manifest + splits, a MobileNetV3-Small (default) and ResNet18 backbone builder, and a fixed-seed training loop with AdamW, cosine LR + linear warmup, BCEWithLogits with `pos_weight`, and best-val-AUROC checkpointing.
- **`mikha.ref.classifier.WaterClassifier`** - inference wrapper that loads a saved checkpoint and exposes `.score(image_bgr) -> float` and `.as_score_fn()`. Plugs directly into `mikha.eval.run.ScoreFn`.
- **`scripts/train_water.py`** - one-command CLI driven by `configs/finetune_v1.yaml`. Any config knob can be overridden from the command line. `--print-config` sanity-checks without importing torch.
- **`configs/finetune_v1.yaml`** - the pinned v0.2.0 experiment. MobileNetV3-Small, 20 epoch cap, batch 16, lr 3e-4, aug prob 0.5 at severity 0.65, seed 1234.
- **`scripts/run_eval.py`** now takes `--detector {yolo, classifier}` and `--weights`. Default stays `yolo`, so the v0.1.0 baseline reproduces unchanged.
- **`docs/finetune.md`** - what M8 does, why a classifier and not a segmenter, exact commands, expected timings, success criteria, threats to validity.
- **`docs/results.md`** - refreshed with the v0.1.0 vs v0.2.0 side-by-side and the training curve.
- **`mikha.bench.local_image_path`** - single source of truth for `manifest_row -> on-disk path`. Fixes a bug where the training dataset was deriving the local path from `row.url.rsplit('/', 1)[-1]`, which for Wikimedia Unicode URLs came back URL-encoded (`Gy%C5%91r_flood%2C_...`) and did not match the ASCII `{row.id}{ext}` names that `fetch_base.py` writes.
- **11 new tests**: `test_local_path.py` (6 URL-encoded / Unicode / integration cases), `test_train_dataset.py` (8 dataset behavior cases including aug determinism and epoch bumping), `test_train_model.py` (3 backbone shape cases), `test_classifier.py` (4 checkpoint round-trip and preprocessing cases). Torch-guarded tests skip cleanly without the model extras.

### v0.2.0 results (see `docs/results.md`)

| Degradation | AUROC | F1 | FAR @ 90% Recall |
|---|---:|---:|---:|
| clean | 0.923 | 0.853 | 0.277 |
| rain  | 0.939 | 0.803 | 0.149 |
| fog   | 0.901 | 0.830 | 0.426 |
| night | 0.882 | 0.814 | 0.404 |
| glare | 0.926 | 0.855 | 0.277 |
| jpeg  | 0.924 | 0.853 | 0.298 |

Clean AUROC clears the 0.70 gate documented in `docs/finetune.md`. Minimum degradation AUROC (night, 0.882) clears the 0.60 gate. Best val AUROC was 0.932 at epoch 2, early stop fired at epoch 7. Val (0.932) and test-clean (0.923) are within 0.01 of each other, which is what a clean split of this size should look like.

### Changed

- `pyproject.toml`: `model` extra now also pulls `torchvision>=0.17` and `pyyaml>=6.0`. Version bumped to 0.2.0.
- `CITATION.cff`: version bumped to 0.2.0.

### Fixed

- URL-encoded Unicode filenames in manifest URLs (typical for EF2013 Wikimedia rows like `Gy%C5%91r_flood,_...` or `Passau_-_Hochwasser_2013_-_Gro%C3%9Fe_Klingergasse_(2).jpg`) no longer break training. The training dataset now uses the same id-based path resolution that the eval corpus and `fetch_base.py` already use. Regression test in `tests/test_local_path.py` covers both URL-encoded and raw-Unicode URLs.

### Known limitations

- The fine-tuned model is a binary image classifier, not a segmenter. The manifest carries `flood` / `nonflood` labels but no per-pixel masks, so segmentation supervision was not available. If per-pixel masks are added later, the segmentation head becomes the correct fine-tune target and `mikha.train` gets a new module.
- 397 train + 86 val + 110 test is small. Metrics move by roughly 0.02-0.03 AUROC across seeds. Do not read the fifth digit.
- Training augmentation and eval stratification use the same 5 transforms. Per-row seeds differ between the two paths, so no exact (row, degradation, seed) overlap exists, but the model has seen the general shape of each transform during training. Real held-out imagery is the honest next validation.
- No CUDA determinism guarantee (some ops do not have deterministic implementations). CPU and MPS are fully deterministic.

---

## [0.1.0] - 2026-09-07

First public release. The three deliverables locked in the plan (Mikha-Aug, Mikha-Bench, Mikha-Ref) are all shipped, plus the evaluation harness and a baseline results table.

### Added

- **`mikha.aug`** - five deterministic seeded 511-style degradation transforms: `rain-on-lens`, `fog`, `night`, `glare`, low-bitrate `jpeg`. ~500 LOC, hand-tuned severity gallery in `docs/aug_gallery.md`.
- **`mikha.bench`** - manifest schema + validator (refuses duplicate ids, duplicate SHA-256, unknown licenses), three importers (EF2013 via Wikimedia with per-image license preservation, HydroShare Urban Flood behind interactive CC-BY acceptance, curated US-Gov public-domain URL list), and a group-aware greedy load-balanced train/val/test split algorithm that prevents uploader leakage.
- **`data/manifest.csv`** - 593 rows (318 flood + 275 non-flood) across 4 sources with per-row license verification (never assumed from a paper's abstract).
- **`bench/splits.json`** - train 397 / val 86 / test 110, bound to the manifest's SHA-256; USGS public-domain rows distributed 9/2/0.
- **`mikha.ref`** - laptop-runnable reference implementation:
  - `detect.py`: YOLOv8-seg wrapper (COCO weights ship as the baseline; a future water-fine-tuned head plugs in behind the same interface).
  - `temporal.py`: K-of-N ring-buffer persistence gate with asymmetric hysteresis.
  - `calibrate.py`: Platt-scaling calibrator (ECE reported alongside metrics).
  - `decision.py`: 4-state FSM (`passable / flooded_uncertain / flooded / camera_offline`).
  - `poller.py`: snapshot poller respecting per-camera intervals + backoff.
  - `store.py`: SQLite event log.
  - `api.py` + `ui/`: single FastAPI process + one HTMX page, no build step, no auth.
- **`mikha.eval`** - detector-agnostic evaluation harness (`score_fn: ndarray -> float`), per-degradation `DegradationResult` with precision/recall/F1/AUROC/AUPRC/FAR@fixed-recall, `write_all` emits `results/table1.md`, `results/eval.json`, `results/pr_curves.png`.
- **`scripts/run_eval.py`** - one-command driver with `--limit-per-class`, `--degradations`, `--splits`, `--threshold`, `--target-recall` flags.
- **`docker-compose.yml`** - one command to run the dashboard against one 511 camera.
- **`docs/architecture.md`, `docs/quickstart.md`, `docs/results.md`, `docs/aug_gallery.md`** - full component walk-through, reproduction path, baseline discussion, degradation samples.
- **CI** - `ruff` + `pytest` on every push; 151 tests passing, 2 skipped (guarded on optional deps).
- **Licenses** - `LICENSE` (Apache-2.0 for code) + `LICENSE-DATA` (CC-BY-4.0 for labels/manifest).
- **`CITATION.cff`** - release-tagged citation metadata.

### Baseline results (see `docs/results.md`)

| Degradation | AUROC | F1 | FAR @ 90% Recall |
|---|---:|---:|---:|
| clean | 0.16 | 0.08 | 0.87 |
| rain  | 0.23 | 0.14 | 0.92 |
| fog   | 0.14 | 0.09 | 0.79 |
| night | 0.19 | 0.07 | 0.60 |
| glare | 0.14 | 0.08 | 0.87 |
| jpeg  | 0.16 | 0.10 | 0.87 |

AUROC below 0.5 is the plan-expected outcome for COCO weights + `mask_area_frac` on this task, and is the trigger for the v0.2 fine-tune step.

### Known limitations

- Baseline detector is COCO-pretrained YOLOv8n-seg used as an anti-detector. Not a water detector. Fine-tune is scheduled for v0.2.
- 593 rows is a research-scale corpus, not a deployment-scale one.
- HydroShare import requires a manual browser step for CC-BY acceptance; the script cannot fully automate it.
- Real 511 held-out evaluation (labels-only redistribution) is scheduled for v0.3.

### Attribution

Every image in `data/manifest.csv` carries its upstream license verbatim (mix of `CC0-1.0`, `PublicDomain-Wikimedia`, `PublicDomain-US-Gov`, `CC-BY-*`, `CC-BY-SA-*`, plus a handful of `Apache-2.0` / `BSD-3-Clause` / `AGPL-3.0` sample images) and an attribution string composed from the Wikimedia `artist`/`user` field or the USGS media library credit line.

---

## [Unreleased]

Planned for v0.3:
- Real FL 511 / TX 511 held-out imagery (labels-only redistribution).
- Publish Zenodo record for the labels + manifest.
- Explore larger backbones (ResNet-50, SAM 2 with a water prompt) if the corpus grows.
