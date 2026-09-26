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
    # `smiles_is_openff_ready` falls back to a text-only heuristic
    # (`smiles_has_complete_stereo`) when openff-toolkit (or a
    # cheminformatics backend for it, e.g. RDKit) is not installed; that
    # heuristic cannot detect an *implicit* stereocentre with no `@`/`@@`
    # marker at all, so the case below only exercises real OpenFF-backed
    # stereo perception and needs the optional `openff` extra
    # (`uv sync --extra openff`) to be meaningful.
    pytest.importorskip("openff.toolkit")
    from amphiquantic.ligand.db import smiles_is_openff_ready

    assert smiles_is_openff_ready("CCO")
    # A chiral centre (4 distinct substituents: CH3, F, Cl, Br) with no
    # stereo marker in the SMILES: genuinely undefined stereochemistry, so
    # OpenFF's Molecule.from_smiles refuses it with
    # UndefinedStereochemistryError. (Note: the SMILES formerly used here
    # for this case, a HET-derived FK506 string, actually carries full
    # `@`/`@@` stereo markers on every centre — real RDKit-backed OpenFF
    # parses it fine, so it was not actually testing the undefined-stereo
    # path; it only "worked" while this test ran without a cheminformatics
    # toolkit installed, via the text-heuristic fallback in
    # `smiles_is_openff_ready`.)
    undefined_stereo_smiles = "CC(F)(Cl)Br"
    assert not smiles_is_openff_ready(undefined_stereo_smiles)


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
