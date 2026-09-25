"""Docking binding-site and integration tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALA_PDB = ROOT / "tests" / "fixtures" / "complex" / "ala_dipeptide.pdb"


@pytest.mark.tier_a
def test_auto_docking_box_centroid():
    from amphiquantic.docking.binding_site import auto_docking_box

    box = auto_docking_box(ALA_PDB)
    center = box["center"]
    assert all(abs(c) < 50 for c in center)
    assert box["size"][0] == pytest.approx(20.0)


@pytest.mark.tier_a
def test_docking_box_from_reference_ligand(tmp_path):
    from amphiquantic.docking.binding_site import auto_docking_box

    ref = tmp_path / "ref.pdb"
    ref.write_text(
        "ATOM      1  CA  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\n"
    )
    box = auto_docking_box(ALA_PDB, reference_ligand=ref)
    assert box["center"][0] == pytest.approx(1.0)
    assert box["center"][1] == pytest.approx(2.0)
    assert box["center"][2] == pytest.approx(3.0)


@pytest.mark.tier_b
@pytest.mark.docking
def test_dock_ligand_vina_smoke(tmp_path):
    if not shutil.which("vina") and not shutil.which("smina"):
        pytest.skip("vina/smina not installed")
    pytest.importorskip("rdkit")
    pytest.importorskip("meeko")

    from amphiquantic.docking.ligand import dock_ligand

    out = tmp_path / "poses.pdbqt"
    dock_ligand(ALA_PDB, "CCO", out)
    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.tier_a
@pytest.mark.docking
def test_pose_to_topology_pdbqt_expands_hydrogens(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("meeko")

    from amphiquantic.docking.pose_to_topology import pose_to_topology
    from amphiquantic.topology.bootstrap_assign import parameterize_from_smiles_bootstrap
    from amphiquantic.topology.io import load_topology

    lig = tmp_path / "ligand.aqtop"
    posed = tmp_path / "posed.aqtop"
    pose = tmp_path / "poses.pdbqt"
    pose.write_text(_minimal_cco_pdbqt())

    parameterize_from_smiles_bootstrap("CCO", lig)
    n_atoms = len(load_topology(lig)["atoms"])
    assert n_atoms == 9

    pose_to_topology(lig, pose, posed)
    updated = load_topology(posed)
    assert len(updated["atoms"]) == n_atoms
    assert updated["atoms"][0]["position"] != load_topology(lig)["atoms"][0]["position"]


def _minimal_cco_pdbqt() -> str:
    return """MODEL 1
REMARK SMILES CCO
REMARK SMILES IDX 1 1 2 2 3 3
REMARK H PARENT 3 4
ROOT
ATOM      1  C   UNL     1       0.595   3.241   3.396  1.00  0.00     0.034 C
ATOM      2  C   UNL     1       0.024   1.852   3.281  1.00  0.00     0.152 C
ENDROOT
BRANCH   2   3
ATOM      3  O   UNL     1       1.040   0.931   3.418  1.00  0.00    -0.397 OA
ATOM      4  H   UNL     1       1.524   0.711   2.600  1.00  0.00     0.210 HD
ENDBRANCH   2   3
TORSDOF 1
ENDMDL
"""
