"""End-to-end experiment protocol tests (no Vina required)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from amphiquantic.experiment.create import create_experiment

ROOT = Path(__file__).resolve().parents[2]
ALA_PDB = ROOT / "tests" / "fixtures" / "complex" / "ala_dipeptide.pdb"
METHANE = ROOT / "tests" / "fixtures" / "methane.aqtop"


@pytest.mark.tier_b
@pytest.mark.rust
@pytest.mark.complex
def test_experiment_implicit_no_dock(tmp_path, built_extension, force_backend_cpu):
    """Run implicit experiment steps without docking (AMPHI_REQUIRE_DOCK=0)."""
    exp = create_experiment(
        "protein_ligand_implicit",
        experiments_dir=tmp_path,
        receptor=ALA_PDB,
        ligand_aqtop=METHANE,
    )
    env = {
        **os.environ,
        "AMPHI_REQUIRE_DOCK": "0",
        "PATH": os.environ.get("PATH", ""),
    }
    proc = subprocess.run(
        ["bash", str(exp / "protocol.sh")],
        cwd=exp,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:] + proc.stdout[-3000:]
    runs = list((exp / "outputs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "topology" / "minimized.aqtop").is_file()
    assert (run_dir / "trajectories" / "md.aqtrj").is_file()
    render_dir = run_dir / "render"
    dock_mp4 = render_dir / "dock_preview.mp4"
    dock_frames = render_dir / "dock_frames"
    movie_mp4 = render_dir / "movie.mp4"
    md_frames = render_dir / "frames"
    assert dock_mp4.is_file() or any(dock_frames.glob("frame_*.png"))
    assert movie_mp4.is_file() or any(md_frames.glob("frame_*.png"))


@pytest.mark.tier_b
@pytest.mark.rust
@pytest.mark.complex
def test_explicit_experiment_e2e(tmp_path, built_extension, force_backend_cpu):
    """Run explicit TIP3P experiment with short MD (no Vina required)."""
    exp = create_experiment(
        "protein_ligand_explicit",
        experiments_dir=tmp_path,
        receptor=ALA_PDB,
        ligand_aqtop=METHANE,
        config_overrides={
            "ligand_resname": None,
            "parameterization": {"receptor": "bootstrap", "ligand": "bootstrap"},
            "steps": {"minimize": 50, "equilibrate": 100, "simulate": 50},
        },
    )
    env = {
        **os.environ,
        "AMPHI_REQUIRE_DOCK": "0",
        "AMPHI_SKIP_RENDER": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    proc = subprocess.run(
        ["bash", str(exp / "protocol.sh")],
        cwd=exp,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:] + proc.stdout[-4000:]
    runs = list((exp / "outputs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "topology" / "solvated.aqtop").is_file()
    assert (run_dir / "topology" / "minimized.aqtop").is_file()
    assert (run_dir / "trajectories" / "md.aqtrj").is_file()
    assert (run_dir / "topology" / "complex.aqtop").is_file()
