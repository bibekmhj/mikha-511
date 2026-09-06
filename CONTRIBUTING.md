# Contributing to Mikha-511

Thanks for your interest. Mikha-511 is deliberately scoped to a small,
single-developer-friendly project (Mikha-Aug + Mikha-Bench + Mikha-Ref).
Contributions that fit that scope are welcome; contributions that expand it
generally are not — please open an issue first if you're unsure.

## Local development

```bash
git clone https://github.com/bibekmhj/mikha-511.git
cd mikha-511
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional; runs ruff on commit
pytest -q
```

## Style

- Python 3.11+
- Formatted and linted with `ruff` (see `pyproject.toml`)
- Type hints on public functions; `from __future__ import annotations` at the top of new modules
- Small pure functions preferred over classes; classes only when state is real
- Every new module gets at least one smoke test

## Data contributions

Please do NOT open PRs that add images. Mikha-511 does not redistribute
images. Contribute by:

- Adding rows to `data/manifest.csv` that point to permissively licensed
  upstream sources with correct SHA-256 hashes.
- Adding or refining labels in `bench/labels.json`.
- Improving `scripts/fetch_base.py` so it can reproduce the base set from the
  manifest.

Any manifest entry must have an unambiguous license note. If it does not, the
PR will not be merged.

## Non-goals (please do not propose)

- New detector architectures beyond swapping the reference backbone
- Multi-camera correlation, hydro-model fusion, RTSP streaming, Kubernetes
- User authentication, roles, or multi-tenant hosting
- A mobile app

These may be worth doing later. They are not v0.1 work.

## Reporting issues

Use GitHub Issues. For anything security- or license-sensitive, please email
the maintainer listed in `CITATION.cff` instead of opening a public issue.
