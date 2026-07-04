"""TIP3P water template for solvation (.aqtop fragment)."""

from __future__ import annotations

from typing import Any

from amphiquantic.topology.io import AQTOP_VERSION

# TIP3P geometry (Angstrom) and parameters (kJ/mol, nm scaled to engine units)
TIP3P_TEMPLATE: dict[str, Any] = {
    "version": AQTOP_VERSION,
    "metadata": {"source": "tip3p_template", "openff_version": None, "smiles": None},
    "box": {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False},
    "atoms": [
        {
            "element": "O",
            "name": "OW",
            "mass": 15.999,
            "charge": -0.834,
            "sigma": 3.166,
            "epsilon": 0.650,
            "position": [0.0, 0.0, 0.0],
            "residue_id": 0,
            "molecule_id": 0,
        },
        {
            "element": "H",
            "name": "HW1",
            "mass": 1.008,
            "charge": 0.417,
            "sigma": 0.0,
            "epsilon": 0.0,
            "position": [0.9572, 0.0, 0.0],
            "residue_id": 0,
            "molecule_id": 0,
        },
        {
            "element": "H",
            "name": "HW2",
            "mass": 1.008,
            "charge": 0.417,
            "sigma": 0.0,
            "epsilon": 0.0,
            "position": [-0.2399872, 0.92662721, 0.0],
            "residue_id": 0,
            "molecule_id": 0,
        },
    ],
    "bonds": [
        {"i": 0, "j": 1, "k": 462750.4, "r0": 0.9572},
        {"i": 0, "j": 2, "k": 462750.4, "r0": 0.9572},
    ],
    "angles": [{"i": 1, "j": 0, "k": 2, "k_theta": 836.8, "theta0": 1.824218134}],
    "dihedrals": [],
    "impropers": [],
    "exclusions": [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
}


def tip3p_molecule_at(origin: tuple[float, float, float], molecule_id: int) -> dict[str, Any]:
    """Return a TIP3P water molecule translated to origin."""
    import copy

    mol = copy.deepcopy(TIP3P_TEMPLATE)
    ox, oy, oz = origin
    for atom in mol["atoms"]:
        atom["position"] = [
            atom["position"][0] + ox,
            atom["position"][1] + oy,
            atom["position"][2] + oz,
        ]
        atom["molecule_id"] = molecule_id
    return mol
