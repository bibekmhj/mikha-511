"""Tests for the European Flood 2013 parser and license normalizer.

No network. Uses a tiny synthetic metadata.json to exercise every code path
including the reject cases (GFDL, Attribution, Copyrighted free use).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mikha.bench import ALLOWED_LICENSES
from mikha.bench.eu_flood import (
    REJECTED_LICENSES,
    WIKIMEDIA_LICENSE_MAP,
    EFRecord,
    _strip_html,
    license_summary,
    load_relevance,
    parse_metadata,
)


def test_every_mapped_spdx_is_allowed() -> None:
    """Every license we map to must already be in ALLOWED_LICENSES."""
    for upstream, spdx in WIKIMEDIA_LICENSE_MAP.items():
        assert spdx in ALLOWED_LICENSES, f"{upstream} → {spdx} but {spdx} not allowed"


def test_rejected_licenses_are_not_mapped() -> None:
    """A rejected upstream license must not appear as a map key."""
    assert REJECTED_LICENSES.isdisjoint(WIKIMEDIA_LICENSE_MAP.keys())


def test_strip_html_basic() -> None:
    assert _strip_html('<a href="x">Alice</a>') == "Alice"
    assert _strip_html("plain text") == "plain text"
    assert _strip_html("") == ""


SYNTH_META = [
    {
        "pageid": 100,
        "title": "File:Good1.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Good1.jpg",
        "license": "CC BY-SA 3.0",
        "user": "Alice",
        "artist": '<a href="x">Alice</a>',
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Good1.jpg",
    },
    {
        "pageid": 101,
        "title": "File:Good2.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2b/Good2.jpg",
        "license": "CC0",
        "user": "Bob",
        "artist": "Bob",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Good2.jpg",
    },
    {
        "pageid": 102,
        "title": "File:Bad_GFDL.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3c/Bad_GFDL.jpg",
        "license": "GFDL",  # rejected
        "user": "Carol",
        "artist": "Carol",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Bad_GFDL.jpg",
    },
    {
        "pageid": 103,
        "title": "File:Ambig_Attribution.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Ambig.jpg",
        "license": "Attribution",  # rejected
        "user": "Dan",
        "artist": "Dan",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Ambig.jpg",
    },
    {
        "pageid": 104,
        "title": "File:Bad_url.jpg",
        "url": "ftp://not-http.example/x.jpg",  # rejected by URL check
        "license": "CC BY 4.0",
        "user": "Eve",
        "artist": "Eve",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Bad_url.jpg",
    },
]


def test_parse_metadata_filters_rejected(tmp_path: Path) -> None:
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(SYNTH_META), encoding="utf-8")
    kept = parse_metadata(p)
    kept_ids = {r.pageid for r in kept}
    # 100 (CC-BY-SA 3.0), 101 (CC0) should pass; 102 (GFDL), 103 (Attribution),
    # 104 (bad URL) should be dropped.
    assert kept_ids == {"100", "101"}


def test_efrecord_attribution_strips_html(tmp_path: Path) -> None:
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(SYNTH_META[:1]), encoding="utf-8")
    (rec,) = parse_metadata(p)
    attr = rec.attribution
    assert "Alice" in attr
    assert "<" not in attr and ">" not in attr
    assert "CC BY-SA 3.0" in attr  # includes upstream license string
    assert "commons.wikimedia.org" in attr


def test_efrecord_attribution_falls_back_to_user(tmp_path: Path) -> None:
    meta = [dict(SYNTH_META[0])]
    meta[0]["artist"] = ""
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    (rec,) = parse_metadata(p)
    assert "Alice" in rec.attribution  # from `user`


def test_license_summary_counts_raw(tmp_path: Path) -> None:
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(SYNTH_META), encoding="utf-8")
    counts = license_summary(p)
    # All 5 upstream licenses appear here, including the rejected ones.
    assert counts["CC BY-SA 3.0"] == 1
    assert counts["CC0"] == 1
    assert counts["GFDL"] == 1
    assert counts["Attribution"] == 1
    assert counts["CC BY 4.0"] == 1


def test_license_summary_rejects_bad_arg() -> None:
    with pytest.raises(TypeError):
        license_summary([EFRecord("1", "t", "https://x/y.jpg", "CC0", "CC0-1.0", "u", "a", "d")])  # type: ignore[arg-type]


def test_load_relevance(tmp_path: Path) -> None:
    p = tmp_path / "flooding.txt"
    p.write_text("100\n101\n\n102\n", encoding="utf-8")
    ids = load_relevance(p)
    assert ids == {"100", "101", "102"}
