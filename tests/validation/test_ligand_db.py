"""Tests for ligand database helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.ligand import extract_hetatm_residue, resolve_ligand_query, save_ligand_smiles

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.tier_a
def test_resolve_literal_smiles():
    smiles, source = resolve_ligand_query("CCO")
    assert smiles == "CCO"
    assert source == "smiles:literal"


@pytest.mark.tier_a
def test_extract_hetatm_residue():
    pdb = (FIXTURES / "ligand_het.pdb").read_text()
    block = extract_hetatm_residue(pdb, "LIG")
    assert block.count("HETATM") == 2
    assert "LIG" in block


@pytest.mark.tier_a
def test_smiles_has_complete_stereo():
    from amphiquantic.ligand.db import smiles_has_complete_stereo

    assert smiles_has_complete_stereo("CCO")
    assert smiles_has_complete_stereo("C[C@@H](O)N")


@pytest.mark.tier_a
@pytest.mark.openff
def test_smiles_is_openff_ready():
    from amphiquantic.ligand.db import smiles_is_openff_ready

    assert smiles_is_openff_ready("CCO")
    # HET-derived FK506 SMILES (undefined stereo for OpenFF)
    fk506_het = (
        "CCC[C@@H]1CC(C)C[C@H](C)C[C@H](OC)[C@H]2O[C@@](O)(C(O)C(O)N3CCCC[C@H]3C(O)O"
        "[C@H](C(C)C[C@@H]3CC[C@@H](O)[C@H](OC)C3)[C@H](C)[C@@H](O)CC1O)[C@H](C)C[C@@H]2OC"
    )
    assert not smiles_is_openff_ready(fk506_het)


@pytest.mark.tier_a
def test_save_ligand_smiles(tmp_path):
    out = tmp_path / "lig.smi"
    save_ligand_smiles("c1ccccc1", out)
    assert out.read_text().strip() == "c1ccccc1"


@pytest.mark.tier_b
def test_pubchem_name_live():
    smiles, source = resolve_ligand_query("ethanol")
    assert "C" in smiles
    assert source.startswith("pubchem:name:")
