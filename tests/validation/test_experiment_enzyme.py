"""Enzyme complex experiment template tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.experiment.create import create_experiment
from amphiquantic.experiment.manifest import read_manifest


@pytest.mark.tier_b
@pytest.mark.rust
def test_enzyme_complex_experiment_create(tmp_path):
    receptor = Path(__file__).resolve().parents[1] / "fixtures" / "complex" / "ala_dipeptide.pdb"
    exp = create_experiment(
        "enzyme_complex_implicit",
        experiments_dir=tmp_path,
        receptor=receptor,
        ligand="CCO",
    )
    assert (exp / "scripts" / "parameterize_assembly.sh").is_file()
    manifest = read_manifest(exp)
    assert manifest["protocol_name"] == "enzyme_complex_implicit"
