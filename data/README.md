# `data/` — base-image manifest and reproduction

Mikha-511 does **not** redistribute images. This directory carries the
metadata needed to reproduce the Mikha-Bench base set from upstream sources
under their own licenses.

## Files

- `manifest.csv` — the base-image manifest. One row per image.
- `gov_pd_urls.tsv` — hand-curated NOAA / USGS / FEMA / other US-Gov public-domain URL list, consumed by `scripts/import_url_list.py`.
- `images/` — populated locally by `scripts/fetch_base.py`. Gitignored.
- `README.md` — this file.

## Schema (`manifest.csv`)

| Column | Meaning | Constraints |
|---|---|---|
| `id` | Stable local id | `mb-` + 6 zero-padded digits |
| `source` | Upstream provider tag | non-empty, snake_case |
| `url` | Canonical upstream URL | must return the exact bytes hashed here |
| `sha256` | SHA-256 of the fetched bytes | 64 lowercase hex chars |
| `license` | License of the image | must be one of the allowed set below |
| `attribution` | Credit string to include when using this image | non-empty |
| `image_class` | Binary label at the base level | `flood` \| `nonflood` |
| `notes` | Free-text notes | may be empty |

Rows are validated on load by `mikha.bench.load_manifest`; any schema
violation, duplicate `id`, or duplicate `sha256` is refused.

## Allowed licenses

Public-domain / CC0:
`CC0-1.0`, `PublicDomain-US-Gov`, `PublicDomain-Wikimedia`.

Creative Commons Attribution (all live versions used on Wikimedia Commons):
`CC-BY-2.0`, `CC-BY-2.5`, `CC-BY-3.0`, `CC-BY-3.0-AT`, `CC-BY-3.0-DE`,
`CC-BY-4.0`.

Creative Commons Attribution-ShareAlike (all live versions used on Wikimedia
Commons; jurisdictional ports are non-canonical SPDX ids kept for provenance):
`CC-BY-SA-2.0`, `CC-BY-SA-2.5`, `CC-BY-SA-2.5-HU`, `CC-BY-SA-3.0`,
`CC-BY-SA-3.0-AT`, `CC-BY-SA-3.0-DE`, `CC-BY-SA-4.0`.

Permissive software licenses (used only by smoke rows and any test images):
`Apache-2.0`, `MIT`, `BSD-3-Clause`, `AGPL-3.0`.

Escape hatch: `research-use` — must be documented per-source in this file.

## Reproducing the base set

```bash
python scripts/verify_manifest.py             # no network; schema check + counts
python scripts/fetch_base.py --dry-run        # lists what would be fetched
python scripts/fetch_base.py                  # fetches to data/images/, verifies SHA-256
```

`fetch_base.py` reports OK / SKIP / MISMATCH / HTTP-ERROR per row and exits
nonzero only when a SHA-256 mismatches — that is the one class of failure
that invalidates the manifest itself.

## Populating the manifest (importer scripts)

Three importers live under `scripts/`:

### 1. European Flood 2013 — `scripts/import_eu_flood.py`

Pulls `metadata.json` + relevance files from
[cvjena/eu-flood-dataset](https://github.com/cvjena/eu-flood-dataset), keeps
only records whose Wikimedia license is in the accepted vocabulary, downloads
each image from Wikimedia to compute SHA-256, and appends validated rows.

Copyright is preserved per-image: each row carries its true upstream license
(one of `CC0-1.0`, `PublicDomain-Wikimedia`, or `CC-BY(-SA)-*`), and an
attribution string composed from the Wikimedia `artist` (HTML stripped) or
`user` field plus the license string and Wikimedia description URL.

Rejected upstream licenses (no row written): `GFDL`, `Attribution`,
`Copyrighted free use`.

Usage:
```bash
# Project counts without downloading:
python scripts/import_eu_flood.py --dry-run --max-per-class 500 --balance
# Actual import (requires egress to upload.wikimedia.org):
python scripts/import_eu_flood.py --max-per-class 500 --balance --sleep-ms 250
```

Best-case yield from EF2013 alone under `--balance`: **~644 rows**
(322 flood + 322 nonflood — nonflood is the limiting side because upstream
only marks 322 images as irrelevant to all three tasks).

### 2. HydroShare Urban Flood Image Dataset — `scripts/import_hydroshare_urban_flood.py`

Takes a LOCAL path to the archive you downloaded from HydroShare after
accepting its CC-BY-4.0 terms (HydroShare requires an interactive terms
acknowledgement; the script does not fetch it for you).

The manifest rows use the HydroShare resource URL as canonical `url`; we
do NOT rehost the images (see caveat below).

Usage:
```bash
# 1. Download the archive from HydroShare in your browser after accepting CC-BY.
# 2. Extract to a local folder.
python scripts/import_hydroshare_urban_flood.py --src /path/to/extracted/archive
```

**Redistribution caveat:** The Chen et al. paper does not explicitly document
an ODU or SECOORA/WebCOOS grant to Chen et al. authorizing CC-BY
redistribution of the underlying imagery. HydroShare's uploader (Chen et al.)
declared CC-BY-4.0. We include the dataset because HydroShare's declaration
covers research use, and we do NOT rehost the images — the manifest ships
URL + SHA-256 pointing at the HydroShare resource; reproducers download bytes
from HydroShare directly under HydroShare's terms. Anyone wanting to build a
downstream commercial product on top of this dataset should contact ODU and
SECOORA/WebCOOS to confirm rights first.

### 3. US-Gov public domain URL list — `scripts/import_url_list.py`

Reads a hand-curated TSV of image URLs with per-image license and class
labels, downloads each, hashes, and appends rows. Rights **must be verified
per-URL** before adding to the TSV — do not add a row unless you've confirmed
on the source page (NOAA photo library, USGS media library, FEMA multimedia
library, or a NASA/DOD press page) that the image is `PublicDomain-US-Gov`
(17 USC §105) or an equivalent explicit release.

TSV format (`data/gov_pd_urls.tsv`):
```
url	license	image_class	source	attribution	notes
https://…/img.jpg	PublicDomain-US-Gov	flood	fema_multimedia_library	FEMA/name-of-photographer, 2024	Hurricane X flooded arterial in Y, FEMA media library ID Z
```

Usage:
```bash
python scripts/import_url_list.py --src data/gov_pd_urls.tsv
```

**No fabricated rows.** The seed `gov_pd_urls.tsv` in this repo is empty
(header only). Populate it manually with verified public-domain URLs.

## Current state

The committed `manifest.csv` still holds only the 4 smoke rows from M3 (all
`nonflood`, all reachable from `raw.githubusercontent.com`). The three
importers above are the mechanism by which the production base set is
populated on a machine with normal internet egress; this sandbox blocks
Wikimedia, HydroShare, NOAA, USGS, and FEMA, so no production rows have been
added here.

Projected yield when all three importers are run on a machine with normal
egress:

| Source | Class balance | Row estimate |
|---|---|---|
| EF2013 (per-image Wikimedia licenses) with `--balance --max-per-class 500` | 322 flood + 322 nonflood | ~644 |
| HydroShare Urban Flood Image Dataset (held-out) | per uploader-declared CC-BY-4.0, actual class split TBD from archive contents | ~few hundred |
| Curated US-Gov PD list | as populated | 0 in seed; grows with curation |
| **Total realistic v0.1.0** | | **~800–1,200 rows** |

## Adding a row (checklist)

1. Confirm the image's license is in the allowed set. If unsure, do not add it.
2. Fetch the exact bytes you plan to hash from the exact URL you will commit.
3. Compute SHA-256: `python -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" file.jpg`.
4. Choose the next free `mb-NNNNNN` id (strictly increasing; no gaps required — the importers handle this automatically).
5. Append the row to `manifest.csv`.
6. Run `python scripts/verify_manifest.py` — must pass.
7. Run `python scripts/fetch_base.py` — must report `ok` for the new row.
8. Commit `manifest.csv` (and only `manifest.csv`; the image itself stays gitignored under `data/images/`).

Rows that fail any of steps 1–7 do not get committed. There is no
"figure it out later."
