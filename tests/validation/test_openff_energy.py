"""OpenFF and CPU/GPU force validation tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tier_a
@pytest.mark.rust
def test_topology_roundtrip(tmp_path, built_extension):
    from rustquantic import Topology, load_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = tmp_path / "test.aqtop"
    top.write(str(path))
    loaded = load_topology(str(path))
    assert loaded.num_atoms() == 2


@pytest.mark.tier_a
def test_methane_fixture_atom_count(repo_root: Path):
    import json

    data = json.loads((repo_root / "tests" / "fixtures" / "methane.aqtop").read_text())
    assert len(data["atoms"]) == 5
    assert len(data["bonds"]) == 4


@pytest.mark.openff
@pytest.mark.tier_a
def test_openff_methane_parameterize(tmp_path, built_extension):
    pytest.importorskip("openff.toolkit")
    from amphiquantic.topology import parameterize_from_smiles

    out = tmp_path / "methane.aqtop"
    top = parameterize_from_smiles("C", out)
    assert top["version"] == 1
    assert len(top["atoms"]) >= 1
    assert top["metadata"]["openff_version"]


@pytest.mark.tier_a
def test_bootstrap_parameterize(tmp_path, ala_pdb: Path):
    from amphiquantic.topology.bootstrap_assign import parameterize_from_pdb_bootstrap

    out = tmp_path / "ala.aqtop"
    top = parameterize_from_pdb_bootstrap(ala_pdb, out)
    assert len(top["atoms"]) == 4


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.gpu
def test_cpu_gpu_force_parity(built_extension, repo_root: Path):
    import os
    import subprocess
    import sys

    path = repo_root / "tests" / "fixtures" / "methane.aqtop"
    env_cpu = os.environ.copy()
    env_cpu["AMPHI_FORCE_BACKEND"] = "cpu"
    env_gpu = os.environ.copy()
    env_gpu["AMPHI_FORCE_BACKEND"] = "gpu"

    def run_energy(env):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                f"from rustquantic import topology_energy; print(topology_energy('{path}'))",
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

    e_cpu = run_energy(env_cpu)
    e_gpu = run_energy(env_gpu)
    assert e_cpu.returncode == 0, e_cpu.stderr
    assert e_gpu.returncode == 0, e_gpu.stderr
    e_cpu_val = float(e_cpu.stdout.strip())
    e_gpu_val = float(e_gpu.stdout.strip())
    denom = max(abs(e_cpu_val), abs(e_gpu_val), 1.0)
    assert abs(e_cpu_val - e_gpu_val) / denom < 0.5


@pytest.mark.tier_a
def test_peptide_sequence_smiles():
    from amphiquantic.topology.peptide import sequence_to_smiles, tag_residue_ids

    smi = sequence_to_smiles(["ALA", "GLY"])
    assert "N" in smi
    top = tag_residue_ids(
        {"atoms": [{"residue_id": 0}, {"residue_id": 0}]},
        ["ALA", "GLY"],
    )
    assert top["metadata"]["residue_sequence"] == ["ALA", "GLY"]
