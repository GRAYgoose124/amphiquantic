"""Experiment directory layout helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def resolve_experiment_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not (root / "manifest.yaml").is_file():
        raise FileNotFoundError(f"Not an experiment directory (missing manifest.yaml): {root}")
    return root


def new_run_dir(experiment_root: Path, datestamp: str | None = None) -> Path:
    stamp = datestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = experiment_root / "outputs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_run_layout(run_dir: Path) -> dict[str, Path]:
    """Create standard subdirs under a run output folder."""
    subdirs = {
        "topology": run_dir / "topology",
        "trajectories": run_dir / "trajectories",
        "docking": run_dir / "docking",
        "render": run_dir / "render",
        "logs": run_dir / "logs",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (subdirs["render"] / "frames").mkdir(parents=True, exist_ok=True)
    return subdirs
