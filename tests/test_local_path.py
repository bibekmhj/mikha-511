"""Regression tests for mikha.bench.local_image_path.

Guards against the M8 bug where mikha.train.dataset was building local
image paths from ``row.url.rsplit('/', 1)[-1]``, which for Wikimedia
Unicode filenames comes back URL-encoded (``Gy%C5%91r_flood%2C_...``)
and does NOT match the ASCII-only ``{row.id}{ext}`` names that
``scripts/fetch_base.py`` writes to disk. The helper is now the one
place any caller derives on-disk paths from a manifest row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mikha.bench import ManifestRow, local_image_ext, local_image_path


def _row(url: str, id_: str = "mb-000123") -> ManifestRow:
    return ManifestRow(
        id=id_,
        source="wikimedia",
        url=url,
        sha256="a" * 64,
        license="CC-BY-SA-3.0",
        attribution="Test",
        image_class="flood",
        notes="synthetic",
    )


# ---------------------------------------------------------------------------
# ext inference


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.org/foo.jpg", ".jpg"),
        ("https://example.org/foo.JPG", ".jpg"),
        ("https://example.org/foo.jpeg", ".jpeg"),
        ("https://example.org/foo.PNG", ".png"),
        ("https://example.org/foo", ".jpg"),  # unknown → default jpg
        ("https://example.org/foo.webp", ".jpg"),  # unsupported → default jpg
    ],
)
def test_local_image_ext_selection(url: str, expected: str) -> None:
    assert local_image_ext(url) == expected


# ---------------------------------------------------------------------------
# path is id-based, NEVER derived from URL last segment


def test_local_path_is_id_based_ascii() -> None:
    row = _row("https://example.org/plain.jpg", id_="mb-000042")
    p = local_image_path(row, Path("data/images"))
    assert p == Path("data/images/mb-000042.jpg")


def test_url_encoded_unicode_filename_maps_to_id_ascii_path() -> None:
    """The M8 regression: Wikimedia Unicode URLs must NOT leak into the path."""
    row = _row(
        "https://upload.wikimedia.org/wikipedia/commons/1/23/Gy%C5%91r_flood%2C_June_2013_62.JPG",
        id_="mb-000200",
    )
    p = local_image_path(row, "data/images")
    assert p == Path("data/images/mb-000200.jpg")
    # And explicitly: no percent-escapes leaked through
    assert "%" not in str(p)


def test_raw_unicode_filename_also_maps_to_id_ascii_path() -> None:
    """Even a *decoded* Unicode URL must not surface in the local path."""
    row = _row(
        "https://upload.wikimedia.org/wikipedia/commons/1/23/Győr_flood,_June_2013_62.JPG",
        id_="mb-000201",
    )
    p = local_image_path(row, "data/images")
    assert p == Path("data/images/mb-000201.jpg")
    # ASCII-only on disk
    assert str(p).isascii()


def test_local_path_accepts_str_or_path_dir() -> None:
    row = _row("https://example.org/x.png", id_="mb-000009")
    assert local_image_path(row, "data/images") == Path("data/images/mb-000009.png")
    assert local_image_path(row, Path("data/images")) == Path("data/images/mb-000009.png")


# ---------------------------------------------------------------------------
# integration: mikha.eval.corpus and mikha.train.dataset must agree


def test_eval_corpus_local_path_uses_shared_helper() -> None:
    """``mikha.eval.corpus._local_path`` is preserved but must delegate."""
    from mikha.eval.corpus import _local_path

    row = _row(
        "https://upload.wikimedia.org/wikipedia/commons/x/Gr%C3%B6%C3%9Fe.jpg",
        id_="mb-000500",
    )
    assert _local_path(row, Path("data/images")) == local_image_path(row, "data/images")
    assert _local_path(row, Path("data/images")) == Path("data/images/mb-000500.jpg")
