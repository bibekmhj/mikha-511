"""Tests for the snapshot poller — no live network."""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest

from mikha.ref.poller import SnapshotPoller

pytest.importorskip("cv2")
import cv2  # noqa: E402


def _tiny_jpeg_bytes() -> bytes:
    """Encode a tiny image so cv2.imdecode has something real to chew on."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 1] = 200
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_fetch_success() -> None:
    body = _tiny_jpeg_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    poller = SnapshotPoller("http://example/x.jpg", client=client, max_retries=0)
    snap = poller.fetch()
    assert snap is not None
    assert snap.n_bytes == len(body)
    assert snap.image.shape == (8, 8, 3)


def test_fetch_returns_none_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    poller = SnapshotPoller("http://example/x.jpg", client=client, max_retries=0)
    assert poller.fetch() is None


def test_fetch_retries_on_429_then_succeeds() -> None:
    body = _tiny_jpeg_bytes()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, content=body, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    poller = SnapshotPoller("http://example/x.jpg", client=client, max_retries=2)
    snap = poller.fetch()
    assert snap is not None
    assert calls["n"] == 2


def test_fetch_returns_none_on_undecodable_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not an image")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    poller = SnapshotPoller("http://example/x.jpg", client=client, max_retries=0)
    assert poller.fetch() is None


def test_context_manager_closes_owned_client() -> None:
    # Just proves __enter__/__exit__ don't raise on the code path.
    io.StringIO()  # keep import used
    with SnapshotPoller("http://example/x.jpg", max_retries=0) as p:
        assert p is not None
