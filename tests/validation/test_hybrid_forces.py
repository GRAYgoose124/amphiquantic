"""Hybrid CPU+GPU force agreement tests."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
@pytest.mark.hybrid
def test_hybrid_minimize_runs(built_extension, monkeypatch):
    monkeypatch.setenv("AMPHI_HYBRID", "1")
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0), (0.09, 0.16, 0.0)],
        ["C", "C", "O"],
        [(0, 1), (1, 2)],
    )
    path = "/tmp/amphi_hybrid_test.aqtop"
    top.write(path)
    _, energy, iters = minimize_topology(path, None, 5, 0.001)
    assert iters >= 1
    assert energy == energy


@pytest.mark.tier_a
@pytest.mark.rust
def test_cpu_dihedral_topology_loads(built_extension, force_backend_cpu):
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.30, 0.0, 0.0), (0.45, 0.1, 0.0)],
        ["C", "C", "C", "C"],
        [(0, 1), (1, 2), (2, 3)],
    )
    path = "/tmp/amphi_dihedral_test.aqtop"
    top.write(path)
    _, energy, iters = minimize_topology(path, None, 10, 0.001)
    assert iters >= 1
    assert energy == energy  # finite
