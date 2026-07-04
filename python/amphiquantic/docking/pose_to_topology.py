"""Apply docked pose coordinates to ligand .aqtop."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from amphiquantic.topology.io import load_topology, save_topology


def _parse_pdbqt_coords(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
        elif line.startswith("ROOT") or line.startswith("ENDROOT"):
            continue
        elif line.startswith("TORSDOF"):
            break
    return coords


def _parse_sdf_coords(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        return []
    n_atoms = int(lines[3].split()[0])
    coords = []
    for line in lines[4 : 4 + n_atoms]:
        parts = line.split()
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return coords


def parse_pose_coords(pose_path: str | Path) -> list[tuple[float, float, float]]:
    path = Path(pose_path)
    suffix = path.suffix.lower()
    if suffix == ".pdbqt":
        return _parse_pdbqt_coords(path)
    if suffix in {".sdf", ".mol"}:
        return _parse_sdf_coords(path)
    if suffix == ".pdb":
        coords = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        return coords
    raise ValueError(f"Unsupported pose format: {path}")


def apply_pose_to_topology(
    ligand_top: dict[str, Any],
    pose_coords: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Update ligand atom positions from docked pose; preserve FF terms."""
    top = dict(ligand_top)
    atoms = list(top["atoms"])
    if len(pose_coords) != len(atoms):
        raise ValueError(
            f"Pose has {len(pose_coords)} atoms but topology has {len(atoms)}"
        )
    for i, (x, y, z) in enumerate(pose_coords):
        a = dict(atoms[i])
        a["position"] = [x, y, z]
        atoms[i] = a
    top["atoms"] = atoms
    return top


def pose_to_topology(
    ligand_aqtop: str | Path,
    pose_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    top = load_topology(ligand_aqtop)
    coords = parse_pose_coords(pose_path)
    updated = apply_pose_to_topology(top, coords)
    save_topology(output, updated)
    return updated
