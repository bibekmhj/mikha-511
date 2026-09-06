"""M3 tests for the base-image manifest schema and helpers.

No network. Manifest I/O uses tmp_path; the committed data/manifest.csv is
loaded once as a fixture-free smoke test to confirm the seed file is valid.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mikha.bench import (
    ALLOWED_CLASSES,
    ALLOWED_LICENSES,
    COLUMNS,
    ManifestError,
    ManifestRow,
    hash_bytes,
    load_manifest,
    row_field_names,
    save_manifest,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_MANIFEST = REPO_ROOT / "data" / "manifest.csv"


VALID_ROW_KWARGS = {
    "id": "mb-000001",
    "source": "opencv_extra",
    "url": "https://example.com/x.png",
    "sha256": "0" * 64,
    "license": "Apache-2.0",
    "attribution": "Test attribution",
    "image_class": "nonflood",
    "notes": "seed",
}


def _row(**overrides: str) -> ManifestRow:
    kwargs = dict(VALID_ROW_KWARGS)
    kwargs.update(overrides)
    return ManifestRow(**kwargs)


# ---------------------------------------------------------------------------
# Row-level validation
# ---------------------------------------------------------------------------


def test_valid_row_constructs() -> None:
    row = _row()
    assert row.id == "mb-000001"
    assert row.license == "Apache-2.0"


@pytest.mark.parametrize(
    "override",
    [
        {"id": "bad"},
        {"id": "mb-1"},
        {"id": "MB-000001"},
    ],
)
def test_bad_id_rejected(override: dict[str, str]) -> None:
    with pytest.raises(ManifestError, match="id"):
        _row(**override)


def test_bad_url_rejected() -> None:
    with pytest.raises(ManifestError, match="url"):
        _row(url="ftp://example.com/x.png")


@pytest.mark.parametrize("sha", ["", "abcd", "z" * 64, "A" * 64])
def test_bad_sha_rejected(sha: str) -> None:
    with pytest.raises(ManifestError, match="sha256"):
        _row(sha256=sha)


def test_unknown_license_rejected() -> None:
    with pytest.raises(ManifestError, match="license"):
        _row(license="WTFPL")


def test_unknown_class_rejected() -> None:
    with pytest.raises(ManifestError, match="image_class"):
        _row(image_class="maybe_flood")


def test_empty_attribution_rejected() -> None:
    with pytest.raises(ManifestError, match="attribution"):
        _row(attribution="   ")


def test_empty_source_rejected() -> None:
    with pytest.raises(ManifestError, match="source"):
        _row(source="")


# ---------------------------------------------------------------------------
# Manifest-level validation
# ---------------------------------------------------------------------------


def test_duplicate_id_rejected() -> None:
    a = _row(id="mb-000001", sha256="a" * 64)
    b = _row(id="mb-000001", sha256="b" * 64)
    with pytest.raises(ManifestError, match="duplicate id"):
        validate_manifest([a, b])


def test_duplicate_hash_rejected() -> None:
    a = _row(id="mb-000001", sha256="a" * 64)
    b = _row(id="mb-000002", sha256="a" * 64)
    with pytest.raises(ManifestError, match="duplicate sha256"):
        validate_manifest([a, b])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    rows = [
        _row(id="mb-000001", sha256="a" * 64),
        _row(id="mb-000002", sha256="b" * 64, image_class="flood"),
    ]
    path = tmp_path / "m.csv"
    save_manifest(rows, path)
    reloaded = load_manifest(path)
    assert reloaded == rows


def test_save_writes_canonical_columns(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    save_manifest([_row(sha256="c" * 64)], path)
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert tuple(header) == COLUMNS


def test_load_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("id,url\nmb-000001,https://x/y.png\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing columns"):
        load_manifest(path)


def test_load_rejects_missing_header(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing header"):
        load_manifest(path)


def test_save_refuses_invalid_manifest(tmp_path: Path) -> None:
    # Two rows sharing a hash → save should refuse.
    rows = [
        _row(id="mb-000001", sha256="a" * 64),
        _row(id="mb-000002", sha256="a" * 64),
    ]
    with pytest.raises(ManifestError):
        save_manifest(rows, tmp_path / "m.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_hash_bytes_is_hex_sha256() -> None:
    # sha256(b"") is well-known.
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert hash_bytes(b"") == empty


def test_row_field_names_matches_columns() -> None:
    assert row_field_names() == COLUMNS


def test_allowed_vocabs_are_frozen() -> None:
    assert isinstance(ALLOWED_CLASSES, frozenset)
    assert isinstance(ALLOWED_LICENSES, frozenset)


# ---------------------------------------------------------------------------
# Committed seed manifest
# ---------------------------------------------------------------------------


def test_committed_seed_manifest_loads_and_validates() -> None:
    """The manifest checked into the repo must always validate."""
    rows = load_manifest(COMMITTED_MANIFEST)
    assert len(rows) >= 1
    for row in rows:
        assert row.image_class in ALLOWED_CLASSES
        assert row.license in ALLOWED_LICENSES
