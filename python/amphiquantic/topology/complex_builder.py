"""Merge protein, ligand, and cofactor .aqtop topologies into one complex."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amphiquantic.topology.io import load_topology, save_topology


def merge_topologies(topologies: list[dict[str, Any]]) -> dict[str, Any]:
    if not topologies:
        raise ValueError("No topologies to merge")
    merged = dict(topologies[0])
    merged["atoms"] = list(topologies[0]["atoms"])
    merged["bonds"] = list(topologies[0].get("bonds", []))
    merged["angles"] = list(topologies[0].get("angles", []))
    merged["dihedrals"] = list(topologies[0].get("dihedrals", []))
    merged["impropers"] = list(topologies[0].get("impropers", []))
    merged["exclusions"] = list(topologies[0].get("exclusions", []))
    offset = len(topologies[0]["atoms"])

    for mol_id, top in enumerate(topologies[1:], start=1):
        for atom in top["atoms"]:
            a = dict(atom)
            a["molecule_id"] = mol_id
            merged["atoms"].append(a)
        for key in ("bonds", "angles", "dihedrals", "impropers"):
            for term in top.get(key, []):
                t = dict(term)
                for field in ("i", "j", "k", "l"):
                    if field in t:
                        t[field] = t[field] + offset
                merged[key].append(t)
        for ex in top.get("exclusions", []):
            merged["exclusions"].append([ex[0] + offset, ex[1] + offset])
        offset += len(top["atoms"])

    merged["metadata"] = dict(merged.get("metadata", {}))
    merged["metadata"]["merged_molecules"] = len(topologies)
    return merged


def build_complex(
    components: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    tops = [load_topology(p) for p in components]
    merged = merge_topologies(tops)
    save_topology(output, merged)
    return merged
