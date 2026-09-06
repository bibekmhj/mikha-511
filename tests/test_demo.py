"""M1 tests for the demo module.

These tests are intentionally light-weight and do NOT invoke the model or
the network. They verify:

* the demo module imports cleanly (with model deps present or absent);
* the CLI argparser behaves;
* ``fetch_snapshot`` returns None on network failure instead of raising.

The heavy end-to-end run (which actually loads YOLOv8-seg and writes a PNG)
is verified manually via ``scripts/demo_one_camera.py`` and is not part of
the CI smoke suite; CI does not install the ``model`` extra.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def test_demo_module_imports() -> None:
    from mikha.ref import demo

    assert demo.DEFAULT_URL.startswith("http")
    assert demo.DEFAULT_WEIGHTS.endswith(".pt")
    assert callable(demo.run_demo)
    assert callable(demo.fetch_snapshot)


def test_cli_argparser_defaults() -> None:
    # Load the script as a module so we can test its argparser without
    # invoking main().
    spec = importlib.util.spec_from_file_location("demo_one_camera", SCRIPTS / "demo_one_camera.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    ns = module._parse_args([])
    assert ns.url.startswith("http")
    assert ns.out == "demo_out.png"
    assert ns.allow_fallback is False

    ns = module._parse_args(["--allow-fallback", "--out", "x.png"])
    assert ns.allow_fallback is True
    assert ns.out == "x.png"


def test_fetch_snapshot_returns_none_on_bad_url() -> None:
    # httpx availability isn't guaranteed in every dev env (it IS in ours,
    # but the test guards against a stripped install).
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("cv2")
    from mikha.ref.demo import fetch_snapshot

    # Unroutable RFC5737 address; short timeout so the test stays fast.
    result = fetch_snapshot("http://192.0.2.1/nonexistent.jpg", timeout=1.0)
    assert result is None
    # sanity: httpx is what raised, not a package-not-found
    assert httpx is not None
