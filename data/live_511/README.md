# `data/live_511/` - live 511 held-out corpus (local only)

This directory holds snapshots pulled from public 511 traffic cameras
(FL511, TX511, and equivalents). Per plan section 4, images are
**not** redistributed by this repo. Only the manifest schema and the
hand-labeled JSON in `bench/live_labels.json` are shippable.

If you are cloning this repo, `data/live_511/` will be empty (aside
from this README). Populate it locally with `scripts/collect_live_511.py`
after reading the ToS notes below.

## What lives here

```
data/live_511/
├── README.md                        # this file (shipped)
├── manifest.csv                     # LiveManifestRow schema, one row per snapshot
└── <camera_id>/<row_id>.<ext>       # local image files, gitignored
```

The manifest is committable in principle (it contains only URLs, hashes,
and camera ids) but there is no reason to commit yours to the public
repo. Keep it local unless you have a specific reason to share it.

## Schema

Column order in `manifest.csv`:

| Column       | Meaning                                                            |
|--------------|--------------------------------------------------------------------|
| `id`         | `live-<camera_id>-YYYYMMDDTHHMMSS` (stable, matches labels JSON)   |
| `camera_id`  | Stable per-camera id, e.g. `fl511_00_MDX_SR874_000`                |
| `series_id`  | `<camera_id>:<epoch_bucket>` groups frames from one event window   |
| `captured_at`| RFC3339 UTC timestamp of the fetch                                 |
| `source`     | `fl511`, `tx511`, etc.                                             |
| `url`        | Snapshot URL that was fetched                                      |
| `sha256`     | Hash of the fetched bytes                                          |
| `local_path` | Repo-relative path of the on-disk image (gitignored)               |
| `notes`      | Free-text notes carried over from the collector config             |

Every row is validated on load: id/timestamp shape, hash length, URL
scheme. See `src/mikha/bench/live.py` for the exact rules.

## ToS notes (READ BEFORE COLLECTING)

Traffic-camera snapshots are publicly viewable on state DOT websites,
but the redistribution rules differ by operator. The safe rule is:
**capture for held-out evaluation on your own machine, do not share
the images**.

Points to confirm per operator before pointing the collector at them:

- Does the site's ToS explicitly forbid automated scraping? Some do,
  most do not for reasonable rates.
- What is the polling cadence you can honor without hammering the
  camera? 30 seconds is a reasonable default for storm events.
- Is your `User-Agent` identifying you as a researcher with a contact
  URL? The collector config sets one; do not remove it.
- Are you avoiding personally identifiable information capture? These
  are wide-angle roadway cameras, but incidental license plates or
  faces may appear. Do not publish frames.

**FL511**: publicly viewable at cctvinfo.dot.state.fl.us and 511.fl511.com;
snapshots refresh every ~30 seconds. Review the FDOT terms at the site
before collecting.

**TX511**: publicly viewable at drivetexas.org; TxDOT publishes camera
locations and refresh cadences.

If in doubt, do not add the camera. Data-scale is not the bottleneck
here; label quality is.

## HLS setup for DelDOT (M10.1)

FL511 cameras are direct JPEG snapshots; the collector uses plain
`httpx` and needs nothing else. DelDOT cameras (and some others) are
HLS live streams instead, and the collector uses system `ffmpeg` to
grab a single frame per poll.

Install ffmpeg once per machine:

```bash
# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt install ffmpeg

# verify
ffmpeg -version
```

`ffmpeg` is intentionally a system dependency, not a Python package.
The collector shells out to whatever `ffmpeg` is on your PATH. No pip
install needed; nothing to add to the venv.

In `configs/live_cameras.local.yaml`, tag DelDOT entries with
`source_kind: hls` and point `url` at the M3U8 playlist URL from the
DelDOT FirstMap layer's `M3U8` field:

```yaml
- camera_id: deldot_scam013_dewey_towers
  source: deldot
  source_kind: hls
  url: "http://video.deldot.gov:1935/live/SCAM013.stream/playlist.m3u8"
  notes: "DE 1 at Towers Rd, Dewey Beach. Northern endpoint of Aug 2025 flood closure (WBOC)."
```

Sampling model (important for reproduction and any paper writeup):
the collector invokes ffmpeg once per poll, and ffmpeg emits the
first video frame of the current HLS chunk at the moment it opens the
stream. That is one frame, JPEG, per configured `interval_seconds`.

Failures on the HLS path (ffmpeg missing, exited non-zero, timed out,
or produced no output) are reported as `[hls_error]` lines by the
collector and do not stop the poll loop. Other cameras keep going.

## Labeling protocol

1. Run `scripts/collect_live_511.py --once` to pull a batch, or leave it
   polling during a real event.
2. Run `scripts/label_live.py` to walk the unlabeled rows in an OS
   image viewer. Keys: `f` = flood, `n` = nonflood, `u` = uncertain,
   `s` = skip, `q` = save + quit.
3. Labels land in `bench/live_labels.json` (atomic write). This file
   is committable and shippable under CC-BY-4.0 like `bench/splits.json`.

Definition of `flood` for consistency:

- Standing water covers any drivable portion of the road surface, or
- Water is deep enough that a passenger vehicle would hydroplane / stall.

Standing water only in gutters, ditches, or off-roadway does not count
as `flood`. Wet pavement without standing water does not count.
Use `uncertain` when the frame is too dark, too fogged, or too
compressed to call it either way; those frames stay out of any
downstream eval.

## Reproducing on a fresh machine

```bash
# 1. Configure cameras
cp configs/live_cameras.yaml configs/live_cameras.local.yaml
# edit configs/live_cameras.local.yaml to add cameras you reviewed

# 2. Collect a batch (either --once for a smoke test, or leave running)
python scripts/collect_live_511.py --config configs/live_cameras.local.yaml --once

# 3. Label
python scripts/label_live.py

# 4. When enough labels exist, use them in downstream eval (M11 work).
```

The v0.1.0 base set in `../manifest.csv` is unaffected by anything you
do here. The two corpora are separate on purpose.
