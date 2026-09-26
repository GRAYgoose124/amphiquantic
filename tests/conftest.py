"""Shared pytest fixtures and markers for amphiquantic agent TDD."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_PKG = ROOT / "python"
FIXTURES = ROOT / "tests" / "fixtures"


def pytest_configure(config):
    for name, doc in {
        "tier_a": "Fast CI tests — always run in agent_verify",
        "tier_b": "Manual / GROMACS validation — optional",
        "rust": "Requires rustquantic maturin extension",
        "gpu": "Requires wgpu GPU backend",
        "openff": "Requires openff-toolkit optional extra",
        "slow": "Skipped by agent_verify --fast",
        "hybrid": "Requires hybrid CPU+GPU parallel path",
        "pme": "PME-enabled tests (slow)",
        "docking": "Docking orchestration tests",
        "complex": "Full complex fixtures (slow, tier_b)",
    }.items():
        config.addinivalue_line("markers", f"{name}: {doc}")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def built_extension(repo_root):
    """Build rustquantic once per session.

    Uses `--inexact` so this does not strip optional extras (e.g. the
    `openff` extra's openff-toolkit/rdkit/python-constraint) that were
    already installed into the environment before pytest started —
    otherwise `-m openff` tests would non-deterministically fail with
    ModuleNotFoundError the first time this fixture (or `uv run maturin
    develop`) resynced the environment mid-session.
    """
    subprocess.run(
        ["uv", "sync", "--inexact"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "maturin",
            "develop",
            "--manifest-path",
            "rust/Cargo.toml",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def force_backend_cpu(monkeypatch):
    monkeypatch.setenv("AMPHI_FORCE_BACKEND", "cpu")


@pytest.fixture
def force_backend_gpu(monkeypatch):
    monkeypatch.setenv("AMPHI_FORCE_BACKEND", "gpu")


@pytest.fixture
def force_backend_gpu_resident(monkeypatch):
    monkeypatch.setenv("AMPHI_FORCE_BACKEND", "gpu-resident")


@pytest.fixture
def ala_pdb(tmp_path, repo_root) -> Path:
    src = FIXTURES / "ala.pdb"
    if src.exists():
        dest = tmp_path / "ala.pdb"
        dest.write_text(src.read_text())
        return dest
    dest = tmp_path / "ala.pdb"
    dest.write_text(
        "ATOM      1  N   ALA A   1      64.000  64.000  64.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      63.000  63.000  63.000  1.00  0.00           C\n"
    )
    return dest
