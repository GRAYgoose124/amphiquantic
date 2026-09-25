"""Read experiment config.yaml helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(experiment_root: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    path = root / "config.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def read_step_count(
    experiment_root: str | Path,
    step_name: str,
    default: int,
) -> int:
    cfg = load_config(experiment_root)
    steps = cfg.get("steps", {})
    value = steps.get(step_name, default)
    return int(value)
