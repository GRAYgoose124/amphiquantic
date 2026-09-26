"""Tests for the pure-Python trajectory analysis toolkit (amphiquantic.analysis)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from amphiquantic.analysis import (
    Selection,
    select,
    load_aqtrj,
    kabsch,
    superpose,
    rmsd,
    rmsd_series,
    rmsf,
    radius_of_gyration,
    radius_of_gyration_series,
    distance_series,
    angle_series,
    dihedral_series,
    radial_distribution_function,
    coordination_number,
    unwrap_pbc,
    mean_squared_displacement,
    diffusion_coefficient,
    hydrogen_bonds,
    hbond_occupancy,
    block_average,
    block_average_error,
    statistical_inefficiency,
    autocorrelation,
    equilibration_time,
)

ROOT = Path(__file__).resolve().parents[2]


def _random_rotation(rng):
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    theta = rng.uniform(0, 2 * np.pi)
    k = np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]
    )
    r = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)
    return r


# ---------------------------------------------------------------------------
# Kabsch / RMSD / RMSF
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_kabsch_recovers_exact_rotation():
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(30, 3))
    r_true = _random_rotation(rng)
    t = np.array([3.0, -2.0, 1.5])
    rotated = pts @ r_true + t

    fitted = superpose(rotated, pts)
    assert np.allclose(fitted, pts, atol=1e-9)


@pytest.mark.tier_a
def test_rmsd_of_rotated_translated_copy_is_zero():
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(25, 3))
    r_true = _random_rotation(rng)
    rotated = pts @ r_true + np.array([10.0, 0.0, -5.0])

    assert rmsd(rotated, pts, fit=True) < 1e-8
    # Without fitting, a rigid-body transform should generally show nonzero RMSD.
    assert rmsd(rotated, pts, fit=False) > 1.0


@pytest.mark.tier_a
def test_rmsd_series_against_reference_frame():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(10, 3))
    n_frames = 5
    positions = np.empty((n_frames, 10, 3))
    for i in range(n_frames):
        r_true = _random_rotation(rng)
        positions[i] = ref @ r_true + rng.normal(size=3)

    series = rmsd_series(positions, reference=ref, fit=True)
    assert series.shape == (n_frames,)
    assert np.all(series < 1e-8)


@pytest.mark.tier_a
def test_rmsf_zero_for_static_structure():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(12, 3))
    positions = np.tile(ref, (8, 1, 1))
    values = rmsf(positions, fit=True)
    assert values.shape == (12,)
    assert np.allclose(values, 0.0, atol=1e-10)


@pytest.mark.tier_a
def test_rmsf_positive_for_fluctuating_atom():
    rng = np.random.default_rng(2)
    n_frames, n_atoms = 200, 4
    base = rng.normal(size=(n_atoms, 3))
    positions = np.tile(base, (n_frames, 1, 1))
    # Perturb only atom 0 with noise each frame.
    positions[:, 0, :] += rng.normal(scale=0.5, size=(n_frames, 3))
    values = rmsf(positions, fit=False)
    assert values[0] > values[1]
    assert values[0] > 0.3


@pytest.mark.tier_a
def test_selection_by_index_and_name(tmp_path):
    topology = {
        "atoms": [
            {"name": "C1", "element": "C", "residue_id": 0},
            {"name": "H1", "element": "H", "residue_id": 0},
            {"name": "H2", "element": "H", "residue_id": 1},
        ]
    }
    sel = select(topology, element="H")
    assert list(sel.indices) == [1, 2]
    sel2 = select(topology, index="0,2")
    assert list(sel2.indices) == [0, 2]
    sel3 = select(topology, resid=1)
    assert list(sel3.indices) == [2]


# ---------------------------------------------------------------------------
# Radius of gyration / geometric series
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_radius_of_gyration_two_point_dumbbell():
    d = 2.0
    pts = np.array([[-d / 2, 0, 0], [d / 2, 0, 0]])
    rg = radius_of_gyration(pts)
    assert np.isclose(rg, d / 2)


@pytest.mark.tier_a
def test_radius_of_gyration_square_matches_analytic():
    # Square centered at origin, side length L, 4 equal point masses at the
    # corners: each corner is at distance L/sqrt(2) from the centroid, so
    # Rg^2 = mean(dist^2) = L^2 / 2.
    l = 2.0
    h = l / 2
    pts = np.array([[-h, -h, 0], [-h, h, 0], [h, -h, 0], [h, h, 0]])
    rg = radius_of_gyration(pts)
    assert np.isclose(rg, np.sqrt(l**2 / 2.0), rtol=1e-10)


@pytest.mark.tier_a
def test_radius_of_gyration_mass_weighted():
    pts = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    masses = np.array([1.0, 1.0])
    rg_equal = radius_of_gyration(pts, masses)
    masses_heavy = np.array([100.0, 1.0])
    rg_heavy = radius_of_gyration(pts, masses_heavy)
    # Heavier mass pulls the COM toward it, reducing Rg relative to equal masses.
    assert rg_heavy < rg_equal


@pytest.mark.tier_a
def test_distance_angle_dihedral_known_geometry():
    # Right angle at origin: i=(1,0,0), j=(0,0,0), k=(0,1,0)
    positions = np.array(
        [[[1.0, 0, 0], [0.0, 0, 0], [0.0, 1, 0], [0.0, 1, 1]]]
    )
    dist = distance_series(positions, 0, 1)
    assert np.isclose(dist[0], 1.0)

    ang = angle_series(positions, 0, 1, 2)
    assert np.isclose(ang[0], 90.0)

    # Dihedral for a planar zig-zag (all in a plane) is 0; here we build a
    # simple known case: p0,p1,p2,p3 forming a 90-degree torsion.
    positions2 = np.array([[[0.0, 1, 0], [0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 1]]])
    dih = dihedral_series(positions2, 0, 1, 2, 3)
    assert np.isclose(abs(dih[0]), 90.0, atol=1e-6)


# ---------------------------------------------------------------------------
# RDF
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_rdf_ideal_gas_converges_to_one():
    rng = np.random.default_rng(11)
    box = np.array([12.0, 12.0, 12.0])
    n_atoms = 300
    n_frames = 40
    frames = rng.uniform(0, 12.0, size=(n_frames, n_atoms, 3))
    r, g = radial_distribution_function(frames, box, r_max=5.0, n_bins=50)
    tail = g[r > 2.0]
    assert np.isclose(tail.mean(), 1.0, atol=0.05)
    # Near r=0 with random uniform points there is no structure either,
    # but statistics are noisier at small shell volume; just check no NaNs.
    assert np.all(np.isfinite(g))


@pytest.mark.tier_a
def test_rdf_two_group_cross_correlation():
    # Two atoms held at a fixed separation across all frames should produce
    # a sharp peak in g(r) near that distance.
    box = np.array([20.0, 20.0, 20.0])
    n_frames = 50
    sep = 1.0
    a_positions = np.zeros((n_frames, 1, 3))
    b_positions = np.tile(np.array([sep, 0, 0]), (n_frames, 1, 1))
    frames = np.concatenate([a_positions, b_positions], axis=1)
    r, g = radial_distribution_function(
        frames, box, indices_a=np.array([0]), indices_b=np.array([1]), r_max=2.0, n_bins=40
    )
    peak_r = r[np.argmax(g)]
    assert abs(peak_r - sep) < (r[1] - r[0]) * 1.5


@pytest.mark.tier_a
def test_coordination_number_matches_uniform_density_estimate():
    rng = np.random.default_rng(5)
    box = np.array([10.0, 10.0, 10.0])
    n_atoms = 500
    n_frames = 20
    frames = rng.uniform(0, 10.0, size=(n_frames, n_atoms, 3))
    r, g = radial_distribution_function(frames, box, r_max=3.0, n_bins=60)
    density = n_atoms / np.prod(box)
    r_cut = 2.0
    n_coord = coordination_number(r, g, density, r_cut)
    expected = density * (4.0 / 3.0) * np.pi * r_cut**3
    assert np.isclose(n_coord, expected, rtol=0.15)


# ---------------------------------------------------------------------------
# MSD / diffusion
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_msd_brownian_motion_slope_matches_einstein_relation():
    rng = np.random.default_rng(99)
    d_true = 0.4
    dt = 0.02
    n_frames = 3000
    n_atoms = 80
    steps = rng.normal(scale=np.sqrt(2 * d_true * dt), size=(n_frames, n_atoms, 3))
    traj = np.cumsum(steps, axis=0)

    msd = mean_squared_displacement(traj)
    d_est, slope, intercept = diffusion_coefficient(msd, dt)

    assert np.isclose(d_est, d_true, rtol=0.15)
    # 3D Einstein relation: MSD(t) = 6*D*t
    assert np.isclose(slope, 6 * d_true, rtol=0.15)


@pytest.mark.tier_a
def test_unwrap_pbc_removes_periodic_jumps():
    box = np.array([10.0, 10.0, 10.0])
    # A particle drifting steadily in +x that gets wrapped back into the box.
    true_x = np.array([9.0, 9.5, 10.2, 10.8, 11.5]) % 10.0
    wrapped = np.zeros((5, 1, 3))
    wrapped[:, 0, 0] = true_x
    unwrapped = unwrap_pbc(wrapped, box)
    diffs = np.diff(unwrapped[:, 0, 0])
    # Real displacement per step is always ~+0.5 to +0.7, never a large jump.
    assert np.all(diffs > 0)
    assert np.all(diffs < 2.0)


@pytest.mark.tier_a
def test_msd_zero_at_lag_zero():
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(50, 5, 3))
    msd = mean_squared_displacement(traj)
    assert msd[0] == 0.0


# ---------------------------------------------------------------------------
# Hydrogen bonds
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_hydrogen_bond_detected_for_linear_geometry():
    # D at origin, H along +x at bond length 0.1 nm, A further along +x
    # (linear, D-H...A = 180 deg), within distance cutoff.
    positions = np.array(
        [[[0.0, 0, 0], [0.1, 0, 0], [0.4, 0, 0]]]
    )  # donor=0, hydrogen=1, acceptor=2
    hbonds = hydrogen_bonds(
        positions,
        donors=np.array([0]),
        hydrogens=np.array([1]),
        acceptors=np.array([2]),
        distance_cutoff=0.35,
        angle_cutoff=30.0,
    )
    assert len(hbonds[0]) == 1
    assert hbonds[0][0].donor == 0
    assert hbonds[0][0].acceptor == 2


@pytest.mark.tier_a
def test_hydrogen_bond_rejected_when_bent():
    # H-A distance is short but D-H...A angle is far from linear (90 deg).
    positions = np.array([[[0.0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0]]])
    hbonds = hydrogen_bonds(
        positions,
        donors=np.array([0]),
        hydrogens=np.array([1]),
        acceptors=np.array([2]),
        distance_cutoff=0.35,
        angle_cutoff=30.0,
    )
    assert len(hbonds[0]) == 0


@pytest.mark.tier_a
def test_hydrogen_bond_rejected_when_too_far():
    positions = np.array([[[0.0, 0, 0], [0.1, 0, 0], [2.0, 0, 0]]])
    hbonds = hydrogen_bonds(
        positions,
        donors=np.array([0]),
        hydrogens=np.array([1]),
        acceptors=np.array([2]),
        distance_cutoff=0.35,
        angle_cutoff=30.0,
    )
    assert len(hbonds[0]) == 0


@pytest.mark.tier_a
def test_hbond_occupancy_fraction():
    # Bond present in frames 0,1,2 out of 4.
    from amphiquantic.analysis.hbonds import HBond

    per_frame = [
        [HBond(0, 1, 2)],
        [HBond(0, 1, 2)],
        [HBond(0, 1, 2)],
        [],
    ]
    occ = hbond_occupancy(per_frame)
    assert occ[(0, 1, 2)] == 0.75


# ---------------------------------------------------------------------------
# Energy log statistics
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_statistical_inefficiency_uncorrelated_is_near_one():
    rng = np.random.default_rng(21)
    series = rng.normal(size=2000)
    g = statistical_inefficiency(series)
    assert 0.9 <= g <= 1.5


@pytest.mark.tier_a
def test_statistical_inefficiency_correlated_is_larger():
    rng = np.random.default_rng(22)
    n = 4000
    x = np.zeros(n)
    phi = 0.9
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    g = statistical_inefficiency(x)
    # AR(1) with phi=0.9 has correlation time tau ~ 1/(1-phi) => g ~ (1+phi)/(1-phi) ~ 19
    assert g > 5


@pytest.mark.tier_a
def test_autocorrelation_at_zero_lag_is_one():
    rng = np.random.default_rng(4)
    series = rng.normal(size=500)
    c = autocorrelation(series)
    assert np.isclose(c[0], 1.0)


@pytest.mark.tier_a
def test_block_average_reduces_length_and_preserves_mean():
    series = np.arange(100.0)
    blocks = block_average(series, 10)
    assert blocks.shape == (10,)
    assert np.isclose(blocks.mean(), series.mean())


@pytest.mark.tier_a
def test_block_average_error_returns_positive_errors():
    rng = np.random.default_rng(6)
    series = rng.normal(size=2000)
    sizes, errors = block_average_error(series)
    assert len(sizes) > 0
    assert np.all(errors > 0)


@pytest.mark.tier_a
def test_equilibration_time_detects_drifting_start():
    rng = np.random.default_rng(8)
    n = 1000
    drift = np.linspace(50, 0, 200)
    noise = rng.normal(scale=1.0, size=n)
    series = np.concatenate([drift, np.zeros(n - 200)]) + noise
    t_eq = equilibration_time(series)
    assert t_eq > 50  # should discard at least a good chunk of the drifting part
    assert t_eq < n


# ---------------------------------------------------------------------------
# .aqtrj loading
# ---------------------------------------------------------------------------


def _write_aqtrj(path: Path, n_frames: int, n_atoms: int, rng) -> np.ndarray:
    positions = rng.normal(size=(n_frames, n_atoms, 3))
    with open(path, "w") as fh:
        fh.write("# aqtrj v1 step natoms epot ekin box_lx box_ly box_lz pbc\n")
        for f in range(n_frames):
            fh.write(f"FRAME {f} {n_atoms} {-100.0 - f:.8f} {10.0 + f:.8f} 3.0 3.0 3.0 1\n")
            for a in range(n_atoms):
                x, y, z = positions[f, a]
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    return positions


@pytest.mark.tier_a
def test_load_aqtrj_roundtrips_positions(tmp_path):
    rng = np.random.default_rng(55)
    path = tmp_path / "test.aqtrj"
    positions = _write_aqtrj(path, n_frames=6, n_atoms=5, rng=rng)
    traj = load_aqtrj(path)
    assert traj.n_frames == 6
    assert traj.n_atoms == 5
    assert np.allclose(traj.positions, positions, atol=1e-5)
    assert np.allclose(traj.box[0], [3.0, 3.0, 3.0])
    assert traj.epot[0] == pytest.approx(-100.0)
    assert traj.ekin[-1] == pytest.approx(15.0)


@pytest.mark.tier_a
def test_load_aqtrj_with_topology(tmp_path):
    rng = np.random.default_rng(56)
    path = tmp_path / "methane.aqtrj"
    _write_aqtrj(path, n_frames=3, n_atoms=5, rng=rng)
    top_path = ROOT / "tests" / "fixtures" / "methane.aqtop"
    traj = load_aqtrj(path, topology=top_path)
    assert traj.topology is not None
    assert list(traj.atom_field("element")) == ["C", "H", "H", "H", "H"]
    masses = traj.masses()
    assert masses[0] == pytest.approx(12.011)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_analyze_cli_exists(repo_root: Path):
    assert (repo_root / "bin" / "analyze").is_file()


@pytest.mark.tier_a
def test_analyze_cli_rmsd(tmp_path, repo_root: Path):
    rng = np.random.default_rng(77)
    traj_path = tmp_path / "run.aqtrj"
    _write_aqtrj(traj_path, n_frames=10, n_atoms=6, rng=rng)
    out_csv = tmp_path / "rmsd.csv"

    proc = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "analyze"), "rmsd", str(traj_path), "-o", str(out_csv)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_csv.is_file()
    lines = out_csv.read_text().strip().splitlines()
    assert lines[0] == "step,time,rmsd_nm"
    assert len(lines) == 11  # header + 10 frames


@pytest.mark.tier_a
def test_analyze_cli_rg_and_rdf(tmp_path, repo_root: Path):
    rng = np.random.default_rng(78)
    traj_path = tmp_path / "run.aqtrj"
    _write_aqtrj(traj_path, n_frames=8, n_atoms=10, rng=rng)

    rg_csv = tmp_path / "rg.csv"
    proc = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "analyze"), "rg", str(traj_path), "-o", str(rg_csv)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert rg_csv.is_file()

    rdf_csv = tmp_path / "rdf.csv"
    proc = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "analyze"), "rdf", str(traj_path), "-o", str(rdf_csv), "--r-max", "1.0"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert rdf_csv.is_file()


@pytest.mark.tier_a
def test_analyze_cli_energy(tmp_path, repo_root: Path):
    rng = np.random.default_rng(79)
    traj_path = tmp_path / "run.aqtrj"
    _write_aqtrj(traj_path, n_frames=50, n_atoms=4, rng=rng)
    out_csv = tmp_path / "energy.csv"

    proc = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "analyze"), "energy", str(traj_path), "-o", str(out_csv), "--field", "epot"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_csv.is_file()
    assert "statistical_inefficiency" in proc.stdout
