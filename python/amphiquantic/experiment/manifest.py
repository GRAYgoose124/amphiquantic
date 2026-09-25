"""Experiment manifest and run provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _git_rev(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(
    experiment_root: Path,
    *,
    experiment_id: str,
    protocol_name: str,
    config: dict[str, Any],
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or experiment_root.parent.parent
    data = {
        "id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_name": protocol_name,
        "git_rev": _git_rev(repo_root),
        "config_hash": _config_hash(config),
        "config": config,
    }
    path = experiment_root / "manifest.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def read_manifest(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "manifest.yaml"
    return yaml.safe_load(path.read_text())


def write_run_json(
    run_dir: Path,
    *,
    experiment_id: str,
    steps: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    data = {
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps or [],
        "artifacts": artifacts or {},
    }
    path = run_dir / "run.json"
    path.write_text(json.dumps(data, indent=2))
    return path
