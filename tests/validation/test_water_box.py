"""Water box validation tests."""

from __future__ import annotations

import pytest

from amphiquantic.validation.physics import density_g_cm3


@pytest.mark.tier_a
def test_tip3p_equilibrium_geometry_energy():
    from amphiquantic.topology.tip3p import TIP3P_TEMPLATE

    assert len(TIP3P_TEMPLATE["bonds"]) == 2
    assert len(TIP3P_TEMPLATE["angles"]) == 1
    assert abs(TIP3P_TEMPLATE["atoms"][0]["charge"] + 0.834) < 0.01


@pytest.mark.tier_a
@pytest.mark.rust
def test_water_box_nominal_density(built_extension, force_backend_cpu, tmp_path):
    from amphiquantic.topology.io import save_topology
    from amphiquantic.topology.solvate import solvate_topology

    solute = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 0, "ly": 0, "lz": 0, "pbc": False},
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": [],
    }
    solvated = solvate_topology(solute, padding=1.2, spacing=0.31)
    rho = density_g_cm3(solvated)
    assert 0.85 <= rho <= 1.15, f"nominal water density {rho:.3f} g/cm³ out of range"


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.slow
def test_water_box_minimize(built_extension, force_backend_cpu, tmp_path):
    from amphiquantic.topology.io import save_topology
    from amphiquantic.topology.solvate import solvate_topology
    from rustquantic import minimize_topology

    solute = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 0, "ly": 0, "lz": 0, "pbc": False},
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": [],
    }
    solvated = solvate_topology(solute, padding=1.5, spacing=3.5)
    path = tmp_path / "water.aqtop"
    save_topology(path, solvated)
    _, energy, iters = minimize_topology(str(path), None, 5, 0.001)
    assert iters >= 1
    assert energy == energy
    rho = density_g_cm3(solvated)
    assert 0.85 <= rho <= 1.15
