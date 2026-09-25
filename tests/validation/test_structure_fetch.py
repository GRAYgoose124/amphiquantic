"""Tests for PDBx/mmCIF parsing and RCSB fetch helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.structure import mmcif_to_pdb, normalize_pdb_id, parse_atom_site

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.tier_a
def test_normalize_pdb_id():
    assert normalize_pdb_id("1crn") == "1CRN"
    with pytest.raises(ValueError):
        normalize_pdb_id("bad")


@pytest.mark.tier_a
def test_parse_mmcif_fixture():
    text = (FIXTURES / "ala_mini.cif").read_text()
    atoms = parse_atom_site(text)
    assert len(atoms) == 4
    assert atoms[0].resname == "ALA"
    assert atoms[0].element == "N"


@pytest.mark.tier_a
def test_mmcif_to_pdb_roundtrip_coords():
    text = (FIXTURES / "ala_mini.cif").read_text()
    pdb = mmcif_to_pdb(text)
    assert "ATOM" in pdb
    assert "64.000" in pdb
    assert pdb.strip().endswith("END")


@pytest.mark.tier_a
def test_prepare_apo_strips_het():
    from amphiquantic.structure.prepare_receptor import prepare_apo_pdb

    holo = (FIXTURES / "ligand_het.pdb").read_text()
    apo = prepare_apo_pdb(holo)
    assert "ATOM" in apo
    assert "HETATM" not in apo
    assert "LIG" not in apo


@pytest.mark.tier_b
def test_fetch_mmcif_live():
    from amphiquantic.structure import fetch_mmcif

    cif = fetch_mmcif("1CRN")
    assert "_atom_site" in cif or "loop_" in cif
    atoms = parse_atom_site(cif)
    assert len(atoms) > 40
