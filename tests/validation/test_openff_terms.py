"""OpenFF Interchange term export tests."""

from __future__ import annotations

import pytest


@pytest.mark.openff
@pytest.mark.tier_a
def test_openff_interchange_has_angles():
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openff.interchange")
    from openff.toolkit import Molecule

    from amphiquantic.topology.openff_assign import parameterize_from_smiles

    top = parameterize_from_smiles("CCO", "/tmp/amphi_ethanol.aqtop")
    assert len(top["bonds"]) >= 2
    # Sage ethanol should have angles when Interchange works
    assert "angles" in top
    assert "dihedrals" in top
    assert "impropers" in top


@pytest.mark.openff
def test_protein_ff_import():
    from amphiquantic.topology import protein_ff

    assert hasattr(protein_ff, "parameterize_protein_pdb")
