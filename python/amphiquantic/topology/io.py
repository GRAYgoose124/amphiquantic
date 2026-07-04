"""Read and write .aqtop topology files (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AQTOP_VERSION = 1


def load_topology(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != AQTOP_VERSION:
        raise ValueError(f"Unsupported .aqtop version: {data.get('version')}")
    return data


def save_topology(path: str | Path, topology: dict[str, Any]) -> None:
    topology = dict(topology)
    topology["version"] = AQTOP_VERSION
    with open(path, "w") as f:
        json.dump(topology, f, indent=2)
