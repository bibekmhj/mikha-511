# Quickstart

End-to-end setup on a fresh laptop, ~30 minutes on normal home internet.

## 1. Install

```bash
git clone https://github.com/bibekmhj/mikha-511.git
cd mikha-511
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model,api]"
```

Extras:
- `dev` — `pytest`, `ruff`.
- `model` — `ultralytics`, `torch` (for the reference detector). Skip this if you only want the benchmark and eval harness; the tests will `importorskip` cleanly.
- `api` — `fastapi`, `uvicorn`, `jinja2` (for the dashboard).

Verify:

```bash
python -m pytest -q
# 151 passed, 2 skipped in ~1s
```

## 2. Reproduce the base image set

The repo ships `data/manifest.csv` (593 rows) but **not** the images themselves — each row is a URL + SHA-256. Repopulate from upstream:

```bash
python scripts/fetch_base.py --dry-run   # what would be fetched
python scripts/fetch_base.py             # fetch + verify SHA-256
```

Expected time: 5–10 min for the 578 EF2013 (Wikimedia) rows + 11 USGS rows + 4 sample rows. `fetch_base.py` reports `ok / skip / mismatch / http-error` per row; nonzero exit only on a SHA-256 mismatch (that's the one class of failure that invalidates the manifest).

### Adding more rows

Three importers append to `data/manifest.csv`:

```bash
# 1. European Flood 2013 (Wikimedia; per-image license preserved):
python scripts/import_eu_flood.py --max-per-class 500 --balance --sleep-ms 1000

# 2. Curated US-Gov public-domain URL list (17 USC §105):
python scripts/import_url_list.py --src data/gov_pd_urls.tsv

# 3. HydroShare Urban Flood Image Dataset (after accepting CC-BY at HydroShare):
python scripts/import_hydroshare_urban_flood.py --src /path/to/extracted/archive
```

The `--sleep-ms 1000 --max-retries 5` defaults respect Wikimedia's rate-limit policy. Lower values will get 429s. See `data/README.md` for the full row-add checklist.

## 3. Build train/val/test splits

```bash
python scripts/build_splits.py
# writes bench/splits.json bound to the current manifest SHA-256
```

Splits are group-aware: a single Wikimedia uploader with 19 % of the corpus won't leak across train/val/test. Deterministic given the seed (default 1234). Re-running with the same manifest is a no-op.

## 4. Run baseline evaluation

Smoke run in under a minute (5 flood + 5 non-flood per degradation):

```bash
python scripts/run_eval.py --limit-per-class 5
```

Full test split, all 6 degradations (~15 min on CPU, ~2 min on a modest GPU):

```bash
python scripts/run_eval.py
```

Outputs:

```
results/table1.md       Per-degradation P/R/F1/AUROC/AUPRC/FAR@0.9R
results/eval.json       Full scores + labels + metadata
results/pr_curves.png   Per-degradation PR curves
```

Useful flags:

```bash
python scripts/run_eval.py --splits val --degradations clean fog night
python scripts/run_eval.py --threshold 0.10 --target-recall 0.8
python scripts/run_eval.py --seed 7   # deterministic; changes degradation RNG only
```

## 5. Run the live dashboard

```bash
docker compose up
# → http://localhost:8000
```

Or without Docker:

```bash
uvicorn mikha.ref.api:app --port 8000 --reload
# separately, in another shell:
python -m mikha.ref.poller \
    --camera-id fl511_demo \
    --url "https://cctvinfo.dot.state.fl.us/image_pull/00_MDX_SR874_000.jpg" \
    --interval 30
```

The dashboard shows the latest decision (`passable / flooded_uncertain / flooded / camera_offline`), the calibrated probability, the raw `mask_area_frac`, and a 24-hour event log from `events.sqlite`.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` from Wikimedia | Default User-Agent is banned | Already fixed — every downloader sends a compliant UA. If you see this, your egress is being MITM'd. |
| `429 Too Many Requests` from Wikimedia | Sleep too aggressive | Bump `--sleep-ms 1000 --max-retries 5`. |
| `ManifestError: duplicate sha256 across rows` | Two rows point at bytes that hash identically | Genuine — reject the newer row. Never override this check. |
| `mikha.bench.manifest.ManifestError: unknown license` | Added a row with a license not in the vocab | Extend the vocab in `src/mikha/bench/manifest.py` **only** if the license is real and permissive; usually the right answer is to drop the row. |
| Test failures under `pip install` without `.[model]` | Fastapi / ultralytics not installed | Expected — those tests `importorskip` and are counted as skipped, not failed. |
| Fetching EF2013 hangs at ~50 rows | Wikimedia's per-IP throttle | Wait 60 s and re-run — `fetch_base.py` is idempotent. |
| Dashboard shows `camera_offline` immediately | Poller can't reach the snapshot URL | Try the URL in a browser; some 511 cameras rate-limit non-browser UAs. |

## What's NOT in v0.1

- Fine-tuned water-segmentation head (triggered by M6 baseline AUROC ≪ 0.5; scheduled for v0.2).
- Multi-camera correlation, hydrological fusion, RTSP streaming.
- Auth on the dashboard.
- A hosted demo.

See the roadmap in `README.md`.
