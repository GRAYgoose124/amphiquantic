"""Ionize preserves .aqtop force field terms."""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.rust
def test_ionize_preserves_bonded_terms(built_extension, tmp_path):
    from amphiquantic.topology.io import load_topology, save_topology
    from amphiquantic.topology.tip3p import tip3p_molecule_at
    from amphiquantic.topology.complex_builder import merge_topologies
    from rustquantic import ionize_topology, load_topology as load_top_rust

    water = tip3p_molecule_at((0.0, 0.0, 0.0), 0)
    water["box"] = {"lx": 5.0, "ly": 5.0, "lz": 5.0, "pbc": True}
    charged = dict(water)
    charged["atoms"] = [dict(a) for a in water["atoms"]]
    charged["atoms"][0]["charge"] = 1.0
    merged = merge_topologies([water, charged])
    inp = tmp_path / "charged.aqtop"
    save_topology(inp, merged)

    n_angles = len(merged["angles"])
    n_bonds = len(merged["bonds"])
    out = tmp_path / "neutral.aqtop"
    ionize_topology(str(inp), str(out))
    result = load_topology(out)
    assert len(result["angles"]) == n_angles
    assert len(result["bonds"]) == n_bonds
    assert abs(load_top_rust(str(out)).net_charge()) < 0.5
