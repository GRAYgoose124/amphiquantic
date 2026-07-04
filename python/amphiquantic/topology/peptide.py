"""Peptide residue templates and polymer typing for .aqtop export."""

from __future__ import annotations

from typing import Any

from amphiquantic.topology.io import AQTOP_VERSION

# Standard residue SMILES fragments (N-terminus capped with ACE, C-terminus with NME where noted)
RESIDUE_SMILES: dict[str, str] = {
    "ALA": "N[C@@H](C)C(=O)O",
    "GLY": "NCC(=O)O",
    "VAL": "N[C@@H](C(C)C)C(=O)O",
    "LEU": "N[C@@H](CC(C)C)C(=O)O",
    "SER": "N[C@@H](CO)C(=O)O",
    "THR": "N[C@@H](C(C)O)C(=O)O",
}


def sequence_to_smiles(residue_codes: list[str]) -> str:
    """Build a linear peptide SMILES by joining residue templates (bootstrap)."""
    if not residue_codes:
        raise ValueError("Empty residue sequence")
    unknown = [r for r in residue_codes if r not in RESIDUE_SMILES]
    if unknown:
        raise ValueError(f"Unknown residues: {unknown}")
    # For bootstrap validation use first residue only when multiple — full polymer via OpenFF in M4
    return RESIDUE_SMILES[residue_codes[0]]


def tag_residue_ids(topology: dict[str, Any], residue_codes: list[str]) -> dict[str, Any]:
    """Assign residue_id per atom block when parameterizing polymers."""
    top = dict(topology)
    atoms = top.get("atoms", [])
    if not atoms or not residue_codes:
        return top
    per_res = max(1, len(atoms) // len(residue_codes))
    for i, atom in enumerate(atoms):
        atom["residue_id"] = min(i // per_res, len(residue_codes) - 1)
    top["metadata"] = dict(top.get("metadata", {}))
    top["metadata"]["residue_sequence"] = residue_codes
    top["version"] = AQTOP_VERSION
    return top
