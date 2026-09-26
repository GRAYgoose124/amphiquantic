"""Trajectory format, checkpoint and CLI-flag coverage for the DCD/PDB/XYZ
writers, checkpoint round trip, and the amphiquantic.trajectory Python
readers added alongside the neighbor-list/trajectory Rust work."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _make_dipeptide_topology(path, tmp_path):
    from rustquantic import Topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.30, 0.0, 0.0)],
        ["C", "C", "C"],
        [(0, 1), (1, 2)],
    )
    top.write(str(path))
    return top


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_writes_dcd_and_python_reader_loads_it(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    traj = tmp_path / "out.dcd"
    simulate_topology(
        str(path),
        str(traj),
        5,
        300.0,
        None,
        False,
        traj_format="dcd",
    )
    assert traj.exists()

    sys.path.insert(0, str((path.parents[2] / "python")))
    from amphiquantic.trajectory import load_dcd

    data = load_dcd(str(traj))
    assert data["natoms"] == 3
    assert data["positions"].shape == (5, 3, 3)
    assert data["box"].shape == (5, 3)


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_writes_pdb_multimodel(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    traj = tmp_path / "out.pdb"
    simulate_topology(str(path), str(traj), 3, 300.0, None, False, traj_format="pdb")

    text = traj.read_text()
    assert text.count("MODEL") == 3
    assert text.count("ENDMDL") == 3


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_writes_xyz(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    traj = tmp_path / "out.xyz"
    simulate_topology(str(path), str(traj), 2, 300.0, None, False, traj_format="xyz")

    lines = traj.read_text().splitlines()
    assert lines[0] == "3"


@pytest.mark.tier_a
@pytest.mark.rust
def test_default_aqtrj_format_still_works(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    traj = tmp_path / "out.aqtrj"
    simulate_topology(str(path), str(traj), 2, 300.0, None, False)

    text = traj.read_text()
    assert text.startswith("# aqtrj")


@pytest.mark.tier_a
@pytest.mark.rust
def test_checkpoint_round_trip_via_simulate(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    ckpt = tmp_path / "state.ckpt"
    simulate_topology(str(path), None, 5, 300.0, None, False, checkpoint_out=str(ckpt))
    assert ckpt.exists()

    sys.path.insert(0, str((path.parents[2] / "python")))
    from amphiquantic.trajectory import load_checkpoint

    data = load_checkpoint(str(ckpt))
    assert data["positions"].shape == (3, 3)
    assert data["velocities"].shape == (3, 3)
    assert data["step"] == 5

    # Resuming from the checkpoint should not error and should keep going
    # from where it left off (best-effort: just check it completes).
    simulate_topology(str(path), None, 2, 300.0, None, False, checkpoint_in=str(ckpt))


@pytest.mark.tier_a
@pytest.mark.rust
def test_energy_log_csv_columns(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    path = tmp_path / "chain.aqtop"
    _make_dipeptide_topology(path, tmp_path)

    log = tmp_path / "energy.csv"
    simulate_topology(str(path), None, 4, 300.0, None, False, energy_log=str(log))

    lines = log.read_text().splitlines()
    assert lines[0] == "step,time_ps,potential,kinetic,total,temperature"
    assert len(lines) == 5  # header + 4 steps


@pytest.mark.tier_a
def test_simulate_cli_accepts_traj_format_and_checkpoint_flags(repo_root, tmp_path):
    """--help should list the new flags without invoking the Rust build."""
    result = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "simulate"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--traj-format" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--energy-log" in result.stdout


@pytest.mark.tier_a
def test_equilibrate_cli_accepts_traj_format_and_checkpoint_flags(repo_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(repo_root / "bin" / "equilibrate"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--traj-format" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--energy-log" in result.stdout
