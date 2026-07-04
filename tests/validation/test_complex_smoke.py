"""Complex workflow smoke tests."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
def test_complex_builder_and_solvate(tmp_path):
    from amphiquantic.topology.complex_builder import build_complex
    from amphiquantic.topology.io import save_topology
    from amphiquantic.topology.solvate import solvate_topology
    from amphiquantic.topology.tip3p import tip3p_molecule_at

    solute = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 0, "ly": 0, "lz": 0, "pbc": False},
        "atoms": [
            {
                "element": "C",
                "name": "CA",
                "mass": 12,
                "charge": 0,
                "sigma": 0.34,
                "epsilon": 0.36,
                "position": [0, 0, 0],
                "residue_id": 0,
                "molecule_id": 0,
            }
        ],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": [],
    }
    p1 = tmp_path / "solute.aqtop"
    save_topology(p1, solute)
    solvated = solvate_topology(solute, padding=2.0, spacing=3.0)
    assert len(solvated["atoms"]) > 3
    assert solvated["box"]["pbc"] is True
