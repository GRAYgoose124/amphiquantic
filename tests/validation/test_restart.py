"""Restart-from-trajectory tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_restart_from_trajectory(tmp_path, built_extension, force_backend_cpu):
    from rustquantic import equilibrate_topology, load_topology, simulate_topology

    src = ROOT / "tests" / "fixtures" / "complex" / "dipeptide_solv.aqtop"
    top_path = tmp_path / "start.aqtop"
    top_path.write_text(src.read_text())
    before = load_topology(str(top_path)).to_pdb_types()[0]
    traj = tmp_path / "eq.aqtrj"
    equilibrate_topology(str(top_path), str(traj), 5, 300.0, None, None)
    out_top, _final_temp, _final_pressure = simulate_topology(
        str(top_path), str(tmp_path / "md.aqtrj"), 3, 300.0, str(traj), False
    )
    after = out_top.to_pdb_types()[0]
    assert after != before


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_cli_restart_flag(tmp_path, built_extension, force_backend_cpu):
    src = ROOT / "tests" / "fixtures" / "complex" / "dipeptide_solv.aqtop"
    top = tmp_path / "t.aqtop"
    top.write_text(src.read_text())
    eq_traj = tmp_path / "eq.aqtrj"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "equilibrate"),
            str(top),
            "--steps",
            "3",
            "--trajectory",
            str(eq_traj),
        ],
        check=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "simulate"),
            str(top),
            "--steps",
            "2",
            "--restart",
            str(eq_traj),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.tier_a
@pytest.mark.rust
def test_equilibrate_restrain_flag(tmp_path, built_extension, force_backend_cpu):
    src = ROOT / "tests" / "fixtures" / "methane.aqtop"
    top = tmp_path / "m.aqtop"
    top.write_text(src.read_text())
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "equilibrate"),
            str(top),
            "--steps",
            "2",
            "--restrain",
            "--restraint-k",
            "500",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
