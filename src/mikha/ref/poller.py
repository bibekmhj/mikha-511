"""Snapshot poller for Mikha-Ref.

Fetches one BGR image from a URL, decodes it, and returns bytes + numpy
array. Retries once on 429/503 with a short back-off. No infinite loops
here — the caller drives the polling cadence.

Kept dependency-light: httpx + numpy + cv2. cv2 is imported lazily so
importing this module doesn't require it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

# Reuse the same UA discipline that scripts/fetch_base.py uses so we
# never get 403'd by a well-behaved host.
DEFAULT_UA = "Mikha511/0.1 (https://github.com/bibekmhj/mikha-511; bbkmhj06@gmail.com) httpx"
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_UA,
    "Accept": "image/*,*/*;q=0.8",
}


@dataclass(frozen=True)
class Snapshot:
    """A successful fetch."""

    url: str
    image: Any  # numpy BGR ndarray
    n_bytes: int
    status_code: int


class SnapshotPoller:
    """Single-URL poller. Instantiate once, call :meth:`fetch` per cycle."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 15.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._owns_client = client is None
        self._client = client or httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout)

    # ------------------------------------------------------------------

    def fetch(self) -> Snapshot | None:
        """One fetch attempt (with retry on 429/503). Returns None on failure."""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(self._url, timeout=self._timeout, follow_redirects=True)
            except httpx.HTTPError:
                time.sleep(min(5.0, 2.0**attempt))
                continue
            if response.status_code in (429, 503):
                retry_after = response.headers.get("retry-after", "")
                wait = float(retry_after) if retry_after.isdigit() else min(10.0, 2.0**attempt)
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                return None
            image = _decode_image(response.content)
            if image is None:
                return None
            return Snapshot(
                url=self._url,
                image=image,
                n_bytes=len(response.content),
                status_code=response.status_code,
            )
        return None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SnapshotPoller:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _decode_image(data: bytes) -> Any:
    """Decode raw bytes to a BGR ndarray. Returns None on failure."""
    import cv2
    import numpy as np

    buffer = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
