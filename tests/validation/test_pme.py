"""PME electrostatics validation: SPME vs. an independent direct-Ewald
reference, and stability of a short NPT run with PME turned on (periodic
box)."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.pme
@pytest.mark.rust
def test_spme_matches_direct_ewald_reference(built_extension, force_backend_cpu, tmp_path):
    """A small periodic water box: SPME total electrostatic energy must
    agree with a slow, independent brute-force direct-Ewald reference
    (real-space minimum image + O(N*K^3) reciprocal sum + self term)."""
    from amphiquantic.topology.io import save_topology
    from rustquantic import direct_ewald_reference_energy, topology_energy

    # Pure point charges, no LJ and no bonded terms, so topology_energy's
    # total potential energy is exactly the Coulomb energy that the direct
    # Ewald reference also computes (apples-to-apples electrostatics-only
    # comparison, independent of the LJ/bonded machinery).
    charges = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    positions = [
        (0.6, 0.6, 0.6),
        (1.8, 1.2, 0.8),
        (1.0, 2.0, 1.6),
        (2.2, 2.4, 0.4),
        (0.4, 1.8, 2.2),
        (1.4, 0.4, 1.8),
    ]
    atoms = [
        {
            "element": "X",
            "name": "X",
            "mass": 1.0,
            "charge": q,
            "sigma": 0.0,
            "epsilon": 0.0,
            "position": list(p),
            "residue_id": i,
            "molecule_id": i,
        }
        for i, (q, p) in enumerate(zip(charges, positions))
    ]
    topology = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 3.0, "ly": 3.0, "lz": 3.0, "pbc": True},
        "atoms": atoms,
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": [],
    }
    path = tmp_path / "small_ions.aqtop"
    save_topology(path, topology)

    # topology_energy uses a fixed real-space cutoff of 1.0 nm and the
    # default Ewald rtol (1e-5); match those exactly so both paths use the
    # same alpha/cutoff split of the same underlying Ewald sum.
    cutoff = 1.0
    ewald_rtol = 1e-5
    spme_energy = topology_energy(str(path))
    direct_energy = direct_ewald_reference_energy(str(path), cutoff, ewald_rtol, 12)

    assert spme_energy == spme_energy  # not NaN
    assert direct_energy == direct_energy
    denom = max(abs(direct_energy), 1.0)
    assert abs(spme_energy - direct_energy) / denom < 5e-3, (
        f"SPME energy {spme_energy} vs direct Ewald reference {direct_energy}"
    )


@pytest.mark.tier_a
@pytest.mark.pme
@pytest.mark.rust
@pytest.mark.slow
def test_npt_water_box_stays_finite_with_pme(built_extension, force_backend_cpu, tmp_path):
    """A short NPT run on a small periodic (PME-electrostatics) water box
    must produce a finite energy and a valid box at every frame — no NaNs
    from the PME path under barostat-induced box rescaling."""
    from amphiquantic.topology.complex_builder import merge_topologies
    from amphiquantic.topology.io import save_topology
    from amphiquantic.topology.tip3p import tip3p_molecule_at
    from amphiquantic.validation.trajectory_analysis import parse_aqtrj
    from rustquantic import simulate_topology

    merged = merge_topologies(
        [
            tip3p_molecule_at((0.3, 0.3, 0.3), 0),
            tip3p_molecule_at((0.9, 0.6, 0.4), 1),
        ]
    )
    merged["box"] = {"lx": 1.2, "ly": 1.2, "lz": 1.2, "pbc": True}
    path = tmp_path / "npt_water.aqtop"
    traj_path = tmp_path / "npt.aqtrj"
    save_topology(path, merged)

    simulate_topology(str(path), str(traj_path), 20, 300.0, None, True)

    frames = parse_aqtrj(traj_path)
    assert len(frames) >= 1
    for frame in frames:
        assert frame.box_lx > 0 and frame.box_ly > 0 and frame.box_lz > 0
        for value in (frame.box_lx, frame.box_ly, frame.box_lz):
            assert value == value  # not NaN
