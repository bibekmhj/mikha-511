"""M0 smoke tests: verify every public subpackage imports cleanly.

These tests are intentionally trivial. They exist so CI has something green
to enforce from day one, and so refactors that break the package layout are
caught immediately.
"""

from __future__ import annotations


def test_top_level_import() -> None:
    import mikha

    assert isinstance(mikha.__version__, str)
    assert mikha.__version__.count(".") >= 1


def test_subpackages_import() -> None:
    import importlib

    for name in ("mikha.aug", "mikha.bench", "mikha.ref", "mikha.eval"):
        module = importlib.import_module(name)
        # __all__ is intentionally empty at M0
        assert hasattr(module, "__all__")
