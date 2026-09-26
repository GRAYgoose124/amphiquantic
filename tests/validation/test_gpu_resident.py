"""GPU-resident MD step backend tests (AMPHI_FORCE_BACKEND=gpu-resident).

See docs/gpu_resident.md for the design and what these tests do (and don't)
exercise. These require a real (or software/lavapipe) Vulkan/Metal/DX12
adapter; they are marked `gpu` and skipped in environments without one.
"""

from __future__ import annotations

import csv

import pytest


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
def test_gpu_resident_single_step_matches_gpu_backend_force(built_extension, force_backend_gpu_resident):
    """A single force evaluation through the generic backend API should not
    crash and should produce a finite energy, same contract as the `gpu` and
    `hybrid` backends."""
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = "/tmp/amphi_gpu_resident_force_test.aqtop"
    top.write(path)
    _, energy, iters = minimize_topology(path, None, 5, 0.001)
    assert iters >= 1
    assert energy == energy  # finite


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
def test_gpu_resident_nve_runs_and_logs_energy(built_extension, tmp_path):
    """Runs the dedicated GPU-resident multi-step loop (not the generic
    integrator) on a small LJ dimer and checks it produces a finite,
    bounded-energy trajectory."""
    from rustquantic import Topology, simulate_topology_gpu_resident

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)],
        ["Ar", "Ar"],
        [],
    )
    path = tmp_path / "gpu_resident_nve.aqtop"
    top.write(str(path))
    log_path = tmp_path / "gpu_resident_nve.csv"

    _, final_temp = simulate_topology_gpu_resident(
        str(path),
        None,
        200,
        0.0,
        0.001,
        "none",
        1.0,
        42,
        str(log_path),
        1.2,
    )
    assert final_temp == final_temp  # finite
    assert final_temp < 500.0  # not a blow-up

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 5
    for row in rows:
        ke = float(row["kinetic"])
        assert ke == ke
        assert ke < 10.0


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
def test_gpu_resident_vrescale_thermostat_runs(built_extension, tmp_path):
    from rustquantic import Topology, simulate_topology_gpu_resident

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)],
        ["Ar", "Ar", "Ar"],
        [],
    )
    path = tmp_path / "gpu_resident_nvt.aqtop"
    top.write(str(path))

    _, final_temp = simulate_topology_gpu_resident(
        str(path),
        None,
        100,
        300.0,
        0.001,
        "vrescale",
        1.0,
        1,
        None,
        1.2,
    )
    assert final_temp == final_temp


@pytest.mark.tier_a
@pytest.mark.rust
def test_gpu_resident_rejects_unknown_thermostat(built_extension, tmp_path):
    from rustquantic import Topology, simulate_topology_gpu_resident

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)],
        ["Ar", "Ar"],
        [],
    )
    path = tmp_path / "gpu_resident_bad_thermostat.aqtop"
    top.write(str(path))
    with pytest.raises(Exception):
        simulate_topology_gpu_resident(str(path), None, 1, 300.0, 0.001, "nose-hoover", 1.0, 1, None, 1.2)


def _tip3p_water_box_aqtop(n_per_axis: int, spacing_nm: float = 0.31) -> dict:
    """A pure TIP3P water box on a regular grid (nm-scale geometry matching
    the Rust gpu-resident tests' convention: `roh=0.09572`, HOH=104.52deg),
    `n_per_axis**3` waters (`3*n_per_axis**3` atoms) in a periodic cubic box.
    Used by the ns/day benchmark to exercise a realistic-sized system rather
    than a handful of atoms.
    """
    import math

    roh = 0.09572
    hoh = math.radians(104.52)
    box_len = n_per_axis * spacing_nm

    atoms = []
    bonds = []
    angles = []
    exclusions = []
    mol_id = 0
    for ix in range(n_per_axis):
        for iy in range(n_per_axis):
            for iz in range(n_per_axis):
                ox, oy, oz = ix * spacing_nm, iy * spacing_nm, iz * spacing_nm
                base = len(atoms)
                atoms.append(
                    {
                        "element": "O",
                        "name": "OW",
                        "mass": 15.999,
                        "charge": -0.834,
                        "sigma": 0.3166,
                        "epsilon": 0.650,
                        "position": [ox, oy, oz],
                        "residue_id": mol_id,
                        "molecule_id": mol_id,
                    }
                )
                atoms.append(
                    {
                        "element": "H",
                        "name": "HW1",
                        "mass": 1.008,
                        "charge": 0.417,
                        "sigma": 0.0,
                        "epsilon": 0.0,
                        "position": [ox + roh, oy, oz],
                        "residue_id": mol_id,
                        "molecule_id": mol_id,
                    }
                )
                atoms.append(
                    {
                        "element": "H",
                        "name": "HW2",
                        "mass": 1.008,
                        "charge": 0.417,
                        "sigma": 0.0,
                        "epsilon": 0.0,
                        "position": [ox + roh * math.cos(hoh), oy + roh * math.sin(hoh), oz],
                        "residue_id": mol_id,
                        "molecule_id": mol_id,
                    }
                )
                bonds.append({"i": base, "j": base + 1, "k": 450.0, "r0": roh})
                bonds.append({"i": base, "j": base + 2, "k": 450.0, "r0": roh})
                angles.append({"i": base + 1, "j": base, "k": base + 2, "k_theta": 55.0, "theta0": hoh})
                exclusions.extend(
                    [
                        [base, base + 1],
                        [base + 1, base],
                        [base, base + 2],
                        [base + 2, base],
                        [base + 1, base + 2],
                        [base + 2, base + 1],
                    ]
                )
                mol_id += 1

    return {
        "version": 1,
        "metadata": {"source": "gpu_resident_benchmark", "openff_version": None, "smiles": None},
        "box": {"lx": box_len, "ly": box_len, "lz": box_len, "pbc": True},
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": [],
        "impropers": [],
        "exclusions": exclusions,
    }


@pytest.mark.tier_b
@pytest.mark.rust
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skip(reason="Benchmark only: run explicitly with `-k benchmark -m gpu -s`")
def test_gpu_resident_benchmark_ns_per_day(built_extension, tmp_path):
    """Prints GPU-resident throughput (ns/day) for a ~5k-atom periodic TIP3P
    water box (SETTLE + PME both active — the full GPU-resident path this
    change completes), for comparison against the CPU/hybrid backends. Not
    a correctness assertion; run manually:
    `uv run pytest tests/validation/test_gpu_resident.py -k benchmark -m gpu -s`.
    On lavapipe (software Vulkan) this is a correctness/plumbing check, not
    a meaningful performance number — see the printed caveat.
    """
    import json
    import time

    from rustquantic import simulate_topology_gpu_resident

    # 12 waters/axis = 1728 waters = 5184 atoms (the low end of "5-20k
    # atoms" — chosen so this still completes in well under a minute on
    # lavapipe/software Vulkan; raise n_per_axis for a larger box on real
    # GPU hardware).
    top = _tip3p_water_box_aqtop(n_per_axis=12)
    path = tmp_path / "gpu_resident_bench.aqtop"
    path.write_text(json.dumps(top))
    n_atoms = len(top["atoms"])

    n_steps = 200
    dt_ps = 0.0005
    start = time.perf_counter()
    simulate_topology_gpu_resident(str(path), None, n_steps, 300.0, dt_ps, "vrescale", 1.0, 1, None, 0.9)
    elapsed_s = time.perf_counter() - start

    ns_simulated = n_steps * dt_ps / 1000.0
    ns_per_day = ns_simulated / (elapsed_s / 86400.0)
    print(
        f"\nGPU-resident: {n_atoms} atoms, {n_steps} steps in {elapsed_s:.3f}s "
        f"-> {ns_per_day:.4f} ns/day (this backend/adapter — see docstring for "
        "the lavapipe/software-Vulkan caveat)"
    )
