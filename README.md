# Mikha-511

> Open-source reference implementation for flooded-road detection on public traffic-camera imagery, released together with the first publicly available degradation-stratified evaluation benchmark for this task.

**Status:** pre-alpha scaffold (M0). No models, no data, no results yet.

---

## Why this exists

Flash flooding is the leading weather-related killer in the United States, and more than half of flood fatalities happen in vehicles that drove onto submerged roads (NWS). Every U.S. state operates a 511 traffic-camera system, and those feeds could in principle be watched during storm events — but no one can watch them all, and existing academic and commercial work reports numbers on clean imagery that don't survive contact with real-world 511 conditions (rain on the lens, night, fog, glare, low-bitrate JPEG artifacts).

Mikha-511 doesn't try to be the state-of-the-art detector. It ships the piece the field is missing: a **public, degradation-stratified evaluation benchmark**, an **open augmentation library** that produces it, and a **minimal, one-laptop reference implementation** that anyone can run and beat.

## What's in the repo

| Package | Role |
|---|---|
| `mikha.aug` | Deterministic, seeded 511-style degradation transforms (rain-on-lens, fog, night, glare, low-bitrate JPEG). |
| `mikha.bench` | Evaluation corpus construction, loading, and manifest tools. Labels ship under CC-BY-4.0; images are represented by source URL + hash to respect upstream licenses. |
| `mikha.ref` | Reference implementation: pretrained YOLOv8-seg water detector + temporal-persistence gate + calibrated confidence + a minimal FastAPI/HTMX dashboard for one camera. |
| `mikha.eval` | One-command evaluation harness producing per-degradation-class precision, recall, F1, mask IoU, and false-alert rate at fixed recall. |

## Quickstart

```bash
git clone https://github.com/bibekmhj/mikha-511.git
cd mikha-511
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"        # model extras pull ultralytics + torch

# Point at any live FL511 (or other 511) camera snapshot URL:
python scripts/demo_one_camera.py \
    --url "https://cctvinfo.dot.state.fl.us/image_pull/00_MDX_SR874_000.jpg" \
    --out demo_out.png

# Or run offline against the bundled ultralytics sample image:
python scripts/demo_one_camera.py --allow-fallback --out demo_out.png
```

At **M1** the segmentation model is COCO-pretrained YOLOv8n-seg — **not** a
water detector yet. The overlay shows whatever COCO classes appear in the
frame. This is deliberate: M1 proves the ingestion → inference → overlay
pipeline works end-to-end. A domain-specific water model replaces it in M5.

## Prior art (honestly cited)

- **ClearObject "ClearFlood"** — proprietary commercial CV on live road/surveillance feeds. Not open source, no benchmark.
- **Rice OpenSafe Fusion** (2024) — multi-source situational-awareness fusion for Houston flooding; broader than a single-camera reference.
- **LSU / Tran-SET USDOT** (2020) — image enhancement + Bayesian filtering on traffic-monitoring cameras; research report, no maintained software.
- **Real-time anticipatory urban flood warning using CCTV and Page–Hinkley change detection** (ScienceDirect, 2026) — single-camera change-detection algorithm.
- **V-FloodNet** (2023) — video segmentation for flood quantification; research prototype.
- **"More eyes on the road"** (Reliability Engineering & System Safety, 2024) — multi-source fusion, not per-camera reference.

Mikha-511's contribution is **infrastructural, not algorithmic**: the benchmark, the augmentation library, and a reproducible baseline these systems can be measured against.

## License

- **Code**: Apache-2.0 (see `LICENSE`).
- **Benchmark labels and manifest**: CC-BY-4.0 (see `LICENSE-DATA`).
- **Images**: not redistributed by this repo. See `data/README.md` for how the base set is reproduced from upstream sources under their own licenses.

## Citation

If you use Mikha-511 or Mikha-Bench in your work, please cite via the `CITATION.cff` in this repository. A Zenodo DOI will be issued at the v0.1.0 release.

## Non-goals

Kept deliberately narrow. No multi-source fusion, no hydrological modeling, no Kubernetes, no auth, no mobile app, no hosted SaaS, no second detector architecture in v0.1.

## Project status roadmap

- [x] M0 — repo scaffold, licenses, README shell, CI, package structure
- [x] M1 — PoC: one FL511 camera → pretrained YOLOv8-seg overlay
- [x] M2 — Mikha-Aug v0 (5 degradations, deterministic, tested)
- [x] M3 — Base image manifest schema + fetch/verify tooling (seed rows committed; production population is the next chunk of work — see `data/README.md`)
- [x] M3.5 — Import tooling for European Flood 2013, HydroShare Urban Flood, and curated US-Gov PD URLs (per-image license preservation; projected ~800–1,200 rows on any machine with normal egress; sandbox cannot fetch upstream so manifest.csv still holds only M3 smoke rows here)
- [ ] M4 — Mikha-Bench v0 (labels + splits + build script)
- [ ] M5 — Mikha-Ref v0 (baseline + persistence gate + dashboard)
- [ ] M6 — Evaluation harness + results table
- [ ] M7 — Docs, Zenodo DOI, v0.1.0 tag

See `docs/quickstart.md` (coming) and the project-level plan for details.
