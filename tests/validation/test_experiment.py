"""Experiment framework tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.experiment.create import create_experiment
from amphiquantic.experiment.layout import ensure_run_layout, new_run_dir, resolve_experiment_root
from amphiquantic.experiment.manifest import read_manifest


@pytest.mark.tier_a
def test_create_experiment_layout(tmp_path):
    receptor = Path(__file__).resolve().parents[1] / "fixtures" / "complex" / "ala_dipeptide.pdb"
    exp = create_experiment(
        "protein_ligand_implicit",
        experiments_dir=tmp_path / "experiments",
        receptor=receptor,
        ligand="CCO",
    )
    assert (exp / "manifest.yaml").is_file()
    assert (exp / "protocol.sh").is_file()
    assert (exp / "config.yaml").is_file()
    assert (exp / "inputs" / "receptor.pdb").is_file()
    assert (exp / "inputs" / "ligand.smi").is_file()
    manifest = read_manifest(exp)
    assert manifest["protocol_name"] == "protein_ligand_implicit"
    assert manifest["config"]["solvation_model"] == "gb_obc2"


@pytest.mark.tier_a
def test_run_dir_layout(tmp_path):
    exp = create_experiment(
        "protein_ligand_implicit",
        experiments_dir=tmp_path,
        receptor=Path(__file__).resolve().parents[1] / "fixtures" / "ala.pdb",
        ligand="C",
    )
    run = new_run_dir(exp)
    sub = ensure_run_layout(run)
    assert sub["topology"].is_dir()
    assert sub["trajectories"].is_dir()
    resolve_experiment_root(exp)


@pytest.mark.tier_a
def test_explicit_config_steps(tmp_path):
    from amphiquantic.experiment.config import read_step_count

    exp = create_experiment(
        "protein_ligand_explicit",
        experiments_dir=tmp_path,
        receptor=Path(__file__).resolve().parents[1] / "fixtures" / "complex" / "ala_dipeptide.pdb",
        ligand="CCO",
        config_overrides={"steps": {"minimize": 42, "simulate": 99}},
    )
    assert read_step_count(exp, "minimize", 1000) == 42
    assert read_step_count(exp, "simulate", 500000) == 99
    assert read_step_count(exp, "equilibrate", 10000) == 10000


@pytest.mark.tier_a
def test_create_experiment_ligand_aqtop(tmp_path):
    receptor = Path(__file__).resolve().parents[1] / "fixtures" / "complex" / "ala_dipeptide.pdb"
    ligand = Path(__file__).resolve().parents[1] / "fixtures" / "methane.aqtop"
    exp = create_experiment(
        "protein_ligand_implicit",
        experiments_dir=tmp_path,
        receptor=receptor,
        ligand_aqtop=ligand,
    )
    assert (exp / "inputs" / "ligand.aqtop").is_file()
    assert not (exp / "inputs" / "ligand.smi").exists()
