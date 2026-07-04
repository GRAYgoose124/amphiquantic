"""Workflow and CLI contract tests for agents."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tier_a
def test_bin_scripts_exist(repo_root: Path):
    for name in (
        "parameterize",
        "solvate",
        "ionize",
        "minimize",
        "equilibrate",
        "simulate",
        "plot_pdb",
    ):
        path = repo_root / "bin" / name
        assert path.is_file(), f"missing bin/{name}"


@pytest.mark.tier_a
def test_workflow_steps_exist(repo_root: Path):
    minimal = repo_root / "scripts" / "steps" / "minimal"
    for step in (
        "parameterize.sh",
        "ionize.sh",
        "solvate.sh",
        "minimize.sh",
        "equilibrate.sh",
        "simulate.sh",
    ):
        assert (minimal / step).is_file(), f"missing {step}"


@pytest.mark.tier_a
@pytest.mark.rust
def test_bootstrap_parameterize_cli(tmp_path, built_extension, ala_pdb: Path):
    out = tmp_path / "out.aqtop"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "parameterize"), str(ala_pdb), "-o", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.tier_a
@pytest.mark.rust
def test_agent_verify_script_dry(repo_root: Path):
    script = repo_root / "scripts" / "agent_verify.sh"
    assert script.stat().st_mode & 0o111, "agent_verify.sh must be executable"
