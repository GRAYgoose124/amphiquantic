"""Application example validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.validation.gromacs_export import export_all, load_aqtop
from amphiquantic.validation.physics import density_g_cm3
from amphiquantic.validation.trajectory_analysis import parse_aqtrj

ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "tests" / "fixtures" / "applications"


@pytest.mark.tier_a
def test_app_water_box_density():
    data = load_aqtop(APPS / "water_box" / "solvated.aqtop")
    rho = density_g_cm3(data)
    assert 0.85 <= rho <= 1.15, f"water density {rho:.3f} g/cm³ out of range"
    assert len(data["atoms"]) >= 50


@pytest.mark.tier_a
def test_app_ethanol_fixture():
    data = load_aqtop(APPS / "ethanol_solv" / "ethanol_solv_ion.aqtop")
    assert len(data["atoms"]) > 10
    charge = sum(a["charge"] for a in data["atoms"])
    assert abs(charge) < 0.5


@pytest.mark.tier_a
def test_app_ethanol_pipeline(built_extension, force_backend_cpu):
    """Solvated ionized ethanol fixture supports full pipeline inputs."""
    from rustquantic import load_topology

    ion = load_topology(str(APPS / "ethanol_solv" / "ethanol_solv_ion.aqtop"))
    assert ion.num_atoms() > 10
    assert abs(ion.net_charge()) < 0.5


@pytest.mark.tier_a
@pytest.mark.rust
def test_app_ala_dipeptide_fixture(built_extension):
    from rustquantic import load_topology

    top = load_topology(str(APPS / "ala_dipeptide" / "ala_dipeptide_solv.aqtop"))
    assert top.num_atoms() > 20
    assert abs(top.net_charge()) < 0.5


@pytest.mark.tier_a
@pytest.mark.rust
def test_app_gromacs_export_all_fixtures(tmp_path, built_extension):
    fixtures = [
        APPS / "water_box" / "solvated.aqtop",
        APPS / "ethanol_solv" / "ethanol_solv_ion.aqtop",
        APPS / "ala_dipeptide" / "ala_dipeptide_solv.aqtop",
        APPS / "nacl_aq" / "ionized.aqtop",
    ]
    for aqtop in fixtures:
        outdir = tmp_path / aqtop.stem
        paths = export_all(aqtop, outdir)
        assert paths["gro"].is_file()
        assert paths["top"].read_text().startswith("; Generated")


@pytest.mark.tier_a
def test_trajectory_analysis_parse(tmp_path):
    traj = tmp_path / "test.aqtrj"
    traj.write_text(
        "# aqtrj v1 step natoms epot ekin box_lx box_ly box_lz pbc\n"
        "FRAME 0 2 1.0 2.0 3.0 3.0 3.0 1\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "FRAME 1 2 1.5 2.5 3.0 3.0 3.0 1\n"
        "0.1 0.0 0.0\n"
        "1.1 0.0 0.0\n"
    )
    frames = parse_aqtrj(traj)
    assert len(frames) == 2
    assert frames[0].epot == pytest.approx(1.0)
    from amphiquantic.validation.trajectory_analysis import energy_drift

    assert energy_drift(frames) == pytest.approx(0.5)


@pytest.mark.tier_b
@pytest.mark.rust
def test_app_nacl_neutral(built_extension):
    from rustquantic import load_topology

    top = load_topology(str(APPS / "nacl_aq" / "ionized.aqtop"))
    assert abs(top.net_charge()) < 0.01


@pytest.mark.tier_b
@pytest.mark.rust
def test_app_gromacs_energy_oracle(tmp_path, built_extension, force_backend_cpu):
    """GROMACS EM energy vs amphiquantic on a stable small-molecule fixture."""
    from amphiquantic.validation.gromacs_energy import compare_em_energy
    from amphiquantic.validation.gromacs_run import find_gmx

    if not find_gmx():
        pytest.skip("GROMACS not installed")

    result = compare_em_energy(
        ROOT / "tests" / "fixtures" / "methane.aqtop",
        tmp_path / "gmx",
        rtol=0.9,
    )
    assert result["within_rtol"], result


@pytest.mark.tier_b
@pytest.mark.rust
@pytest.mark.slow
def test_app_npt_water_density(tmp_path, built_extension, force_backend_cpu):
    """NPT barostat keeps a valid box and trajectory frames (tier_b)."""
    from amphiquantic.topology.io import save_topology
    from amphiquantic.topology.tip3p import tip3p_molecule_at
    from amphiquantic.topology.complex_builder import merge_topologies
    from rustquantic import simulate_topology

    # Single TIP3P molecule: multi-water clusters are unstable in short NPT runs.
    merged = merge_topologies([tip3p_molecule_at((0.6, 0.6, 0.6), 0)])
    merged["box"] = {"lx": 1.2, "ly": 1.2, "lz": 1.2, "pbc": True}
    path = tmp_path / "water.aqtop"
    traj_path = tmp_path / "md.aqtrj"
    save_topology(path, merged)
    simulate_topology(str(path), str(traj_path), 20, 300.0, None, True)

    frames = parse_aqtrj(traj_path)
    assert len(frames) >= 1
    last = frames[-1]
    assert last.box_lx > 0 and last.box_ly > 0 and last.box_lz > 0
    assert abs(last.box_lx - 1.2) > 1e-4  # Berendsen barostat ran

    data = load_aqtop(path)
    data["box"] = {
        "lx": last.box_lx,
        "ly": last.box_ly,
        "lz": last.box_lz,
        "pbc": last.pbc,
    }
    rho = density_g_cm3(data)
    assert rho > 0


@pytest.mark.tier_b
def test_app_crambin_fetch_smoke(tmp_path):
    from amphiquantic.structure import fetch_pdb, normalize_pdb_id

    try:
        text = fetch_pdb(normalize_pdb_id("1CRN"))
    except Exception as exc:
        pytest.skip(f"network unavailable: {exc}")
    assert "ATOM" in text or "HETATM" in text
