"""Implicit solvation topology preparation (GB OBC-II)."""

from __future__ import annotations

from typing import Any

from amphiquantic.topology.io import load_topology, save_topology

# Bondi radii (Å) × 0.9 HCT scale
BONDI_RADII: dict[str, float] = {
    "H": 1.20 * 0.9,
    "C": 1.70 * 0.9,
    "N": 1.55 * 0.9,
    "O": 1.52 * 0.9,
    "S": 1.80 * 0.9,
    "P": 1.80 * 0.9,
    "F": 1.47 * 0.9,
    "CL": 1.75 * 0.9,
    "BR": 1.85 * 0.9,
    "I": 1.98 * 0.9,
    "NA": 2.27 * 0.9,
    "MG": 1.73 * 0.9,
    "CA": 2.31 * 0.9,
    "ZN": 1.39 * 0.9,
    "FE": 1.0 * 0.9,
}


def assign_born_radii(topology: dict[str, Any]) -> dict[str, Any]:
    for atom in topology["atoms"]:
        elem = str(atom.get("element", "C")).upper()
        atom["born_r"] = BONDI_RADII.get(elem, BONDI_RADII.get(elem[:1], 1.5 * 0.9))
    return topology


def prepare_implicit_topology(
    topology: dict[str, Any],
    *,
    model: str = "gb_obc2",
    dielectric: float = 78.5,
) -> dict[str, Any]:
    top = assign_born_radii(dict(topology))
    meta = top.setdefault("metadata", {})
    meta["solvation_model"] = model
    meta["dielectric"] = dielectric
    top["box"] = {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False}
    return top


def prepare_implicit_file(input_path: str, output_path: str, **kwargs: Any) -> None:
    top = load_topology(input_path)
    prepared = prepare_implicit_topology(top, **kwargs)
    save_topology(output_path, prepared)
