"""Docking orchestration smoke tests (no external Vina required)."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.docking
def test_geometric_assembly(tmp_path):
    from amphiquantic.docking.assembly import geometric_merge

    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    out = tmp_path / "complex.pdb"
    a.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    b.write_text(
        "ATOM      1  CA  GLY B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    geometric_merge(a, b, out)
    assert out.exists()
    assert "ATOM" in out.read_text()


@pytest.mark.tier_a
def test_complex_builder_merge(tmp_path):
    from amphiquantic.topology.complex_builder import build_complex
    from amphiquantic.topology.io import save_topology

    t1 = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 0, "ly": 0, "lz": 0, "pbc": False},
        "atoms": [{"element": "C", "name": "C1", "mass": 12, "charge": 0,
                   "sigma": 0.34, "epsilon": 0.36, "position": [0, 0, 0],
                   "residue_id": 0, "molecule_id": 0}],
        "bonds": [], "angles": [], "dihedrals": [], "impropers": [], "exclusions": [],
    }
    t2 = dict(t1)
    t2["atoms"] = [dict(t1["atoms"][0])]
    t2["atoms"][0]["position"] = [1, 0, 0]
    p1 = tmp_path / "a.aqtop"
    p2 = tmp_path / "b.aqtop"
    save_topology(p1, t1)
    save_topology(p2, t2)
    merged = build_complex([p1, p2], tmp_path / "merged.aqtop")
    assert len(merged["atoms"]) == 2
