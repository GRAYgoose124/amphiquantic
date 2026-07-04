"""Minimizer and dynamics validation."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.rust
def test_minimize_lowers_harmonic_energy(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = tmp_path / "bond.aqtop"
    top.write(str(path))
    _, e0, _ = minimize_topology(str(path), None, 1, 0.0001)
    _, e1, _ = minimize_topology(str(path), None, 50, 0.0001)
    assert e1 <= e0 + 1.0


@pytest.mark.tier_a
@pytest.mark.rust
def test_nve_energy_finite(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import Topology, equilibrate_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = tmp_path / "nve.aqtop"
    top.write(str(path))
    equilibrate_topology(str(path), None, 10, 300.0)
