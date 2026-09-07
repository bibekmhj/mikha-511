# Changelog

All notable changes to Mikha-511 are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-07

First public release. The three deliverables locked in the plan (Mikha-Aug, Mikha-Bench, Mikha-Ref) are all shipped, plus the evaluation harness and a baseline results table.

### Added

- **`mikha.aug`** — five deterministic seeded 511-style degradation transforms: `rain-on-lens`, `fog`, `night`, `glare`, low-bitrate `jpeg`. ~500 LOC, hand-tuned severity gallery in `docs/aug_gallery.md`.
- **`mikha.bench`** — manifest schema + validator (refuses duplicate ids, duplicate SHA-256, unknown licenses), three importers (EF2013 via Wikimedia with per-image license preservation, HydroShare Urban Flood behind interactive CC-BY acceptance, curated US-Gov public-domain URL list), and a group-aware greedy load-balanced train/val/test split algorithm that prevents uploader leakage.
- **`data/manifest.csv`** — 593 rows (318 flood + 275 non-flood) across 4 sources with per-row license verification (never assumed from a paper's abstract).
- **`bench/splits.json`** — train 397 / val 86 / test 110, bound to the manifest's SHA-256; USGS public-domain rows distributed 9/2/0.
- **`mikha.ref`** — laptop-runnable reference implementation:
  - `detect.py`: YOLOv8-seg wrapper (COCO weights ship as the baseline; a future water-fine-tuned head plugs in behind the same interface).
  - `temporal.py`: K-of-N ring-buffer persistence gate with asymmetric hysteresis.
  - `calibrate.py`: Platt-scaling calibrator (ECE reported alongside metrics).
  - `decision.py`: 4-state FSM (`passable / flooded_uncertain / flooded / camera_offline`).
  - `poller.py`: snapshot poller respecting per-camera intervals + backoff.
  - `store.py`: SQLite event log.
  - `api.py` + `ui/`: single FastAPI process + one HTMX page, no build step, no auth.
- **`mikha.eval`** — detector-agnostic evaluation harness (`score_fn: ndarray → float`), per-degradation `DegradationResult` with precision/recall/F1/AUROC/AUPRC/FAR@fixed-recall, `write_all` emits `results/table1.md`, `results/eval.json`, `results/pr_curves.png`.
- **`scripts/run_eval.py`** — one-command driver with `--limit-per-class`, `--degradations`, `--splits`, `--threshold`, `--target-recall` flags.
- **`docker-compose.yml`** — one command to run the dashboard against one 511 camera.
- **`docs/architecture.md`, `docs/quickstart.md`, `docs/results.md`, `docs/aug_gallery.md`** — full component walk-through, reproduction path, baseline discussion, degradation samples.
- **CI** — `ruff` + `pytest` on every push; 151 tests passing, 2 skipped (guarded on optional deps).
- **Licenses** — `LICENSE` (Apache-2.0 for code) + `LICENSE-DATA` (CC-BY-4.0 for labels/manifest).
- **`CITATION.cff`** — release-tagged citation metadata.

### Baseline results (see `docs/results.md`)

| Degradation | AUROC | F1 | FAR@0.9R |
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

Planned for v0.2:
- Fine-tune a water-segmentation head on the base set; re-emit `results/table1.md`.
- Add real FL 511 / TX 511 held-out imagery (labels-only redistribution).
- Publish Zenodo record for the labels + manifest.
