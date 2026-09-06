"""Mikha-Bench base-image manifest schema and helpers.

The manifest (``data/manifest.csv``) is the single source of truth for which
images belong to the base set that Mikha-Aug then transforms into
Mikha-Bench. This module defines:

* :class:`ManifestRow` — the row schema;
* :data:`ALLOWED_CLASSES`, :data:`ALLOWED_LICENSES` — closed-vocab guards;
* :func:`load_manifest`, :func:`save_manifest` — CSV round-trip;
* :func:`validate_manifest` — structural + duplicate checks (no network);
* :func:`hash_bytes` — SHA-256 helper used by both writer and fetcher.

License policy — enforced here, per ``data/README.md``:

* Every row MUST carry an unambiguous ``license`` from :data:`ALLOWED_LICENSES`.
* Ambiguous or unknown licenses are refused at load time, not "figured out later".
* The manifest never carries image bytes. Reproducing the base set requires
  running ``scripts/fetch_base.py`` against upstream URLs, at which point the
  fetched bytes' SHA-256 must match the value in the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Closed vocabularies. Widening these is a benchmark-version bump.
ALLOWED_CLASSES: frozenset[str] = frozenset({"flood", "nonflood"})

# SPDX identifiers we accept, plus two non-SPDX tags for research-use datasets
# that have per-project terms. Rows using these tags MUST have their
# per-source terms documented in ``data/README.md``.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        # Public domain / CC0
        "CC0-1.0",
        "PublicDomain-US-Gov",  # 17 USC §105 works
        "PublicDomain-Wikimedia",  # Wikimedia "Public domain" tag (Marked / expired copyright)
        # Creative Commons Attribution (all live versions used on Wikimedia Commons)
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-3.0-AT",
        "CC-BY-3.0-DE",
        "CC-BY-4.0",
        # Creative Commons Attribution-ShareAlike (all live versions on Wikimedia Commons)
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-2.5-HU",  # Hungarian jurisdiction port; not a canonical SPDX id
        "CC-BY-SA-3.0",
        "CC-BY-SA-3.0-AT",
        "CC-BY-SA-3.0-DE",
        "CC-BY-SA-4.0",
        # Permissive software licenses (used only by smoke rows and any test images)
        "Apache-2.0",
        "MIT",
        "BSD-3-Clause",
        "AGPL-3.0",  # tolerated for smoke rows; discouraged for research content
        # Escape hatch for datasets with per-project terms; MUST be documented in data/README.md
        "research-use",
    }
)

# id format: mb-<6-digit-zero-padded-int>. Deliberately simple and sortable.
ID_RE = re.compile(r"^mb-\d{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Columns in canonical order. save_manifest writes exactly this order.
COLUMNS: tuple[str, ...] = (
    "id",
    "source",
    "url",
    "sha256",
    "license",
    "attribution",
    "image_class",
    "notes",
)


class ManifestError(ValueError):
    """Raised on any schema or integrity violation in the manifest."""


@dataclass(frozen=True)
class ManifestRow:
    """One base-image row.

    Field semantics:

    * ``id`` — stable, sortable local id, e.g. ``mb-000042``.
    * ``source`` — short upstream provider tag, e.g. ``ultralytics_assets``.
    * ``url`` — canonical URL that returns the exact bytes hashed here.
    * ``sha256`` — hex SHA-256 of those bytes.
    * ``license`` — one of :data:`ALLOWED_LICENSES`.
    * ``attribution`` — free-text credit string to include when using this image.
    * ``image_class`` — one of :data:`ALLOWED_CLASSES`.
    * ``notes`` — free-text; may be empty.
    """

    id: str
    source: str
    url: str
    sha256: str
    license: str
    attribution: str
    image_class: str
    notes: str = ""

    def __post_init__(self) -> None:
        # dataclass frozen=True means we use object.__setattr__ only if we
        # need to mutate; here we only validate.
        _validate_row(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_row(row: ManifestRow) -> None:
    """Raise :class:`ManifestError` on any per-row violation."""
    if not ID_RE.match(row.id):
        raise ManifestError(f"{row.id!r}: id must match {ID_RE.pattern}")
    if not row.source or not row.source.strip():
        raise ManifestError(f"{row.id}: source is required")
    if not row.url.startswith(("http://", "https://")):
        raise ManifestError(f"{row.id}: url must be http(s)://…")
    if not SHA256_RE.match(row.sha256):
        raise ManifestError(f"{row.id}: sha256 must be 64 lowercase hex chars")
    if row.license not in ALLOWED_LICENSES:
        raise ManifestError(
            f"{row.id}: license {row.license!r} not in allowed set ({sorted(ALLOWED_LICENSES)})"
        )
    if not row.attribution or not row.attribution.strip():
        raise ManifestError(f"{row.id}: attribution is required (upstream credit)")
    if row.image_class not in ALLOWED_CLASSES:
        raise ManifestError(
            f"{row.id}: image_class {row.image_class!r} not in {sorted(ALLOWED_CLASSES)}"
        )


def validate_manifest(rows: Iterable[ManifestRow]) -> None:
    """Structural + duplicate checks over a whole manifest. Raises on the first defect."""
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for row in rows:
        # Per-row validation already ran in __post_init__; re-run defensively
        # in case a caller constructed rows without dataclass __init__.
        _validate_row(row)
        if row.id in seen_ids:
            raise ManifestError(f"duplicate id: {row.id}")
        seen_ids.add(row.id)
        if row.sha256 in seen_hashes:
            raise ManifestError(f"duplicate sha256 across rows (id={row.id})")
        seen_hashes.add(row.sha256)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def hash_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Load ``path`` (a CSV) into a list of :class:`ManifestRow`.

    Runs :func:`validate_manifest` before returning. Raises
    :class:`ManifestError` on any schema issue.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManifestError(f"{path}: missing header row")
        missing = set(COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ManifestError(f"{path}: missing columns: {sorted(missing)}")
        rows = [ManifestRow(**{k: row.get(k, "") for k in COLUMNS}) for row in reader]
    validate_manifest(rows)
    return rows


def save_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    """Write ``rows`` to ``path`` in canonical column order.

    Runs :func:`validate_manifest` first; refuses to write an invalid manifest.
    """
    rows_list = list(rows)
    validate_manifest(rows_list)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def row_field_names() -> tuple[str, ...]:
    """Introspection helper: canonical row field names in declaration order."""
    return tuple(f.name for f in fields(ManifestRow))
