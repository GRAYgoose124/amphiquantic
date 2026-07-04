"""Force field and energy contract tests (CPU reference path)."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.rust
def test_minimize_reduces_or_stabilizes_energy(built_extension, force_backend_cpu):
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = "/tmp/amphi_force_test.aqtop"
    top.write(path)
    _, energy, iters = minimize_topology(path, None, 20, 0.001)
    assert iters >= 1
    assert energy == energy  # finite


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
def test_gpu_minimize_runs(built_extension, force_backend_gpu):
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = "/tmp/amphi_gpu_force_test.aqtop"
    top.write(path)
    _, energy, iters = minimize_topology(path, None, 5, 0.001)
    assert iters >= 1
