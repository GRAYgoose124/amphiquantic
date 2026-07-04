"""Solvate .aqtop systems with TIP3P water and periodic box."""

from __future__ import annotations

import math
from typing import Any

from amphiquantic.topology.complex_builder import merge_topologies
from amphiquantic.topology.tip3p import tip3p_molecule_at


def _solute_bounds(atoms: list[dict[str, Any]], padding: float) -> tuple[float, float, float, float, float, float]:
    xs = [a["position"][0] for a in atoms]
    ys = [a["position"][1] for a in atoms]
    zs = [a["position"][2] for a in atoms]
    return (
        min(xs) - padding,
        max(xs) + padding,
        min(ys) - padding,
        max(ys) + padding,
        min(zs) - padding,
        max(zs) + padding,
    )


def solvate_topology(
    topology: dict[str, Any],
    padding: float = 1.0,
    spacing: float = 3.0,
) -> dict[str, Any]:
    """Add TIP3P waters on a grid; set orthorhombic PBC box."""
    atoms = topology["atoms"]
    if not atoms:
        return topology

    x0, x1, y0, y1, z0, z1 = _solute_bounds(atoms, padding)
    lx = x1 - x0
    ly = y1 - y0
    lz = z1 - z0

    waters: list[dict[str, Any]] = []
    mol_id = max((a.get("molecule_id", 0) for a in atoms), default=0) + 1
    solute_positions = [a["position"] for a in atoms]

    x = x0
    while x < x1:
        y = y0
        while y < y1:
            z = z0
            while z < z1:
                origin = (x, y, z)
                too_close = any(
                    math.dist(origin, sp) < 2.5 for sp in solute_positions
                )
                if not too_close:
                    waters.append(tip3p_molecule_at(origin, mol_id))
                    mol_id += 1
                z += spacing
            y += spacing
        x += spacing

    if not waters:
        out = dict(topology)
        out["box"] = {"lx": lx, "ly": ly, "lz": lz, "pbc": True}
        return out

    merged = merge_topologies([topology] + waters)
    merged["box"] = {"lx": lx, "ly": ly, "lz": lz, "pbc": True}
    merged["metadata"] = dict(merged.get("metadata", {}))
    merged["metadata"]["solvent"] = "TIP3P"
    return merged
