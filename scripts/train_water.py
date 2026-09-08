#!/usr/bin/env python3
"""Fine-tune the water classifier (M8 / v0.2).

Usage:
    # Default config (configs/finetune_v1.yaml), pinned seed:
    python scripts/train_water.py

    # Override any config knob from the CLI:
    python scripts/train_water.py --epochs 30 --lr 1e-4 --out-dir models/finetune_v2

    # Sanity check without touching torch (config parse only):
    python scripts/train_water.py --print-config
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "finetune_v1.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit(
            "PyYAML required; install with `pip install pyyaml` or via `.[model]`."
        ) from None
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _coerce(value: str, target_type: type) -> Any:
    if target_type is bool:
        return value.lower() in {"1", "true", "yes", "y", "on"}
    if target_type is tuple:
        # comma-separated list, e.g. --allowed-augs rain,fog,night
        return tuple(v.strip() for v in value.split(",") if v.strip())
    return target_type(value)


def _build_parser(field_types: dict[str, type]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fine-tune Mikha-511 water classifier.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    ap.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit (no torch import).",
    )
    for name in field_types:
        if name == "extras":
            continue
        ap.add_argument(f"--{name.replace('_', '-')}", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    # We import TrainConfig by traversing the module attribute so this
    # script also works when torch is missing (for --print-config).
    from mikha.train.train import TrainConfig

    field_types: dict[str, type] = {}
    for f in fields(TrainConfig):
        if f.name == "extras":
            continue
        # dataclass tuple[str, ...] fields expose type as generic; unwrap origin
        if getattr(f.type, "__origin__", None) is tuple or f.type is tuple:
            field_types[f.name] = tuple
        else:
            field_types[f.name] = f.type if isinstance(f.type, type) else type(f.default)

    ap = _build_parser(field_types)
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg_dict = _load_yaml(cfg_path)

    for name, type_ in field_types.items():
        cli_val = getattr(args, name)
        if cli_val is not None:
            cfg_dict[name] = _coerce(cli_val, type_)

    # tuple fields loaded from yaml come in as lists — normalize
    if "allowed_augs" in cfg_dict and isinstance(cfg_dict["allowed_augs"], list):
        cfg_dict["allowed_augs"] = tuple(cfg_dict["allowed_augs"])

    try:
        cfg = TrainConfig(**cfg_dict)
    except TypeError as exc:
        raise SystemExit(f"invalid config: {exc}") from None

    if args.print_config:
        import json
        from dataclasses import asdict

        print(json.dumps(asdict(cfg), indent=2, default=list))
        return 0

    from mikha.train.train import train

    print(f"[train] config={cfg_path}  out={cfg.out_dir}  device={cfg.device}")
    summary = train(cfg)
    print(f"[train] done — best val AUROC={summary['best_val_auroc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
