"""European Flood 2013 (cvjena/eu-flood-dataset) parser + license normalizer.

The upstream dataset publishes:

* ``metadata.json`` — one array of per-Wikimedia-image records with
  ``pageid``, ``url``, ``license``, ``artist``, ``user``, ``descriptionurl``, …
* ``relevance/flooding.txt`` — image ids relevant to the "flooding" task.
* ``relevance/irrelevant.txt`` — image ids not relevant to any task.
* Image bytes: on Wikimedia (per-record ``url``) and mirrored on archive.org.

Copyright of individual images is held by their Wikimedia uploaders, and each
record carries an unambiguous per-image license string. This module maps
those strings onto the SPDX-style identifiers in
:data:`mikha.bench.ALLOWED_LICENSES` and rejects records whose license is
ambiguous or incompatible.

We do NOT redistribute EF2013 images. The manifest ships URL + SHA-256 only;
reproducers fetch bytes directly from the upstream URL under its license.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Mapping from Wikimedia's license string (as it appears verbatim in
# EF2013's metadata.json) to our accepted vocabulary. Anything NOT in this
# table is refused; that includes "GFDL" (interoperability caveats),
# "Attribution" (ambiguous — could be any-version CC-BY or a custom
# attribution license), and "Copyrighted free use" (custom template).
WIKIMEDIA_LICENSE_MAP: dict[str, str] = {
    # Public domain / CC0
    "CC0": "CC0-1.0",
    "Public domain": "PublicDomain-Wikimedia",
    # Creative Commons Attribution
    "CC BY 2.0": "CC-BY-2.0",
    "CC BY 2.5": "CC-BY-2.5",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY 3.0 at": "CC-BY-3.0-AT",
    "CC BY 3.0 de": "CC-BY-3.0-DE",
    "CC BY 4.0": "CC-BY-4.0",
    # Creative Commons Attribution-ShareAlike
    "CC BY-SA 2.0": "CC-BY-SA-2.0",
    "CC BY-SA 2.5": "CC-BY-SA-2.5",
    "CC BY-SA 2.5 hu": "CC-BY-SA-2.5-HU",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY-SA 3.0 at": "CC-BY-SA-3.0-AT",
    "CC BY-SA 3.0 de": "CC-BY-SA-3.0-DE",
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
}

# Licenses we explicitly reject even though they appear in EF2013.
# Kept as a list to make the intent visible in code review.
REJECTED_LICENSES: frozenset[str] = frozenset(
    {
        "GFDL",  # ShareAlike-like but with copyleft caveats; skip
        "Attribution",  # too vague; could be any of several attribution flavours
        "Copyrighted free use",  # custom Wikimedia template; per-image scrutiny required
    }
)


@dataclass(frozen=True)
class EFRecord:
    """One European Flood 2013 metadata record after license normalization."""

    pageid: str
    title: str
    url: str
    license_upstream: str  # verbatim from Wikimedia
    license_spdx: str  # normalized to our vocabulary
    user: str
    artist_html: str
    description_url: str

    @property
    def attribution(self) -> str:
        """Compose an attribution string that Wikimedia's license requires.

        Strips HTML from the ``artist`` field (Wikimedia embeds a wikilink).
        Falls back to the uploader ``user`` field when ``artist`` is empty.
        """
        artist = _strip_html(self.artist_html).strip() or self.user or "unknown"
        return f"{artist} (Wikimedia Commons, {self.license_upstream}). {self.description_url}"


def _strip_html(text: str) -> str:
    """Cheap HTML-tag stripper; sufficient for Wikimedia artist fields."""
    if not text:
        return ""
    out = []
    depth = 0
    for ch in text:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def parse_metadata(path: str | Path) -> list[EFRecord]:
    """Load EF2013 ``metadata.json`` and return only records with an accepted license.

    Rejected records (unknown or incompatible license) are silently dropped;
    call :func:`license_summary` on the raw JSON to see the full distribution.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    kept: list[EFRecord] = []
    for rec in raw:
        upstream = str(rec.get("license", "")).strip()
        spdx = WIKIMEDIA_LICENSE_MAP.get(upstream)
        if spdx is None:
            continue
        pageid = str(rec.get("pageid", ""))
        url = str(rec.get("url", ""))
        if not pageid or not url.startswith(("http://", "https://")):
            continue
        kept.append(
            EFRecord(
                pageid=pageid,
                title=str(rec.get("title", "")),
                url=url,
                license_upstream=upstream,
                license_spdx=spdx,
                user=str(rec.get("user", "")),
                artist_html=str(rec.get("artist", "")),
                description_url=str(rec.get("descriptionurl", "")),
            )
        )
    return kept


def load_relevance(path: str | Path) -> set[str]:
    """Load an EF2013 relevance file (one image id per line) into a set."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def license_summary(records_or_path: object) -> dict[str, int]:
    """Return {upstream-license-string: count} across records.

    Accepts a path to ``metadata.json`` or an already-loaded list of records.
    Rows are counted BEFORE license filtering, so this shows the full
    upstream picture including rejected licenses.
    """
    if isinstance(records_or_path, str | Path):
        with Path(records_or_path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        it = (str(rec.get("license", "")) for rec in raw)
    else:
        # Not a raw records list — this function is meant for the JSON, not
        # already-filtered EFRecord lists.
        raise TypeError("license_summary expects a path to metadata.json")
    counts: dict[str, int] = {}
    for lic in it:
        counts[lic] = counts.get(lic, 0) + 1
    return counts
