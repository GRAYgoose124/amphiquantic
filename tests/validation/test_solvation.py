"""TIP3P solvation workflow tests."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
def test_tip3p_template():
    from amphiquantic.topology.tip3p import TIP3P_TEMPLATE, tip3p_molecule_at

    assert len(TIP3P_TEMPLATE["atoms"]) == 3
    mol = tip3p_molecule_at((1.0, 2.0, 3.0), 5)
    assert mol["atoms"][0]["position"][0] == pytest.approx(1.0)


@pytest.mark.tier_a
def test_solvate_adds_waters():
    from amphiquantic.topology.solvate import solvate_topology

    top = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 0, "ly": 0, "lz": 0, "pbc": False},
        "atoms": [
            {
                "element": "C",
                "name": "C1",
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
    solvated = solvate_topology(top, padding=2.0, spacing=2.5)
    assert solvated["box"]["pbc"] is True
    assert len(solvated["atoms"]) > 1
