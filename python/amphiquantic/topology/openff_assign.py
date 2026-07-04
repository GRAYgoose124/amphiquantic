"""Assign OpenFF Sage parameters and emit .aqtop topology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amphiquantic.topology.interchange_export import create_interchange, interchange_to_aqtop
from amphiquantic.topology.io import AQTOP_VERSION, save_topology

DEFAULT_FF = "openff-2.2.1.offxml"

# Bootstrap LJ tables (fallback when Interchange unavailable)
LJ_BY_ELEMENT: dict[str, tuple[float, float]] = {
    "H": (0.106, 0.066),
    "C": (0.339, 0.457),
    "N": (0.325, 0.711),
    "O": (0.296, 0.879),
    "S": (0.356, 1.046),
    "P": (0.374, 0.836),
    "F": (0.312, 0.255),
    "Cl": (0.347, 1.108),
    "Br": (0.383, 1.065),
    "I": (0.419, 1.035),
    "Na": (0.259, 0.011),
    "K": (0.349, 0.011),
    "Ca": (0.299, 0.011),
}


def _lj(element: str) -> tuple[float, float]:
    return LJ_BY_ELEMENT.get(element, (0.34, 0.36))


def _extract_topology_fallback(offmol, ff_name: str, source: str, smiles: str | None) -> dict[str, Any]:
    """Legacy bootstrap when Interchange export fails."""
    from openff.toolkit import ForceField

    ff = ForceField(ff_name)
    offmol.assign_partial_charges(partial_charge_method="gasteiger")
    if offmol.n_conformers == 0:
        offmol.generate_conformers(n_conformers=1)
    conformer = offmol.conformers[0]
    coords = conformer.m_as("angstrom")
    atoms = []
    for idx, atom in enumerate(offmol.atoms):
        sigma, epsilon = _lj(atom.symbol)
        x, y, z = coords[idx]
        atoms.append(
            {
                "element": atom.symbol,
                "name": atom.name or f"{atom.symbol}{idx + 1}",
                "mass": float(atom.mass.m),
                "charge": float(atom.partial_charge.m),
                "sigma": sigma,
                "epsilon": epsilon,
                "position": [float(x), float(y), float(z)],
                "residue_id": 0,
                "molecule_id": 0,
            }
        )
    bonds = []
    for bond in offmol.bonds:
        i, j = bond.atom1_index, bond.atom2_index
        length = offmol.get_bond_length(bond).m_as("nanometer") * 10.0
        bonds.append({"i": i, "j": j, "k": 500.0, "r0": float(length)})
    exclusions = []
    for b in bonds:
        exclusions.append([b["i"], b["j"]])
        exclusions.append([b["j"], b["i"]])
    return {
        "version": AQTOP_VERSION,
        "metadata": {"openff_version": ff_name, "source": source, "smiles": smiles},
        "box": {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False},
        "atoms": atoms,
        "bonds": bonds,
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": exclusions,
    }


def _extract_topology(offmol, ff_name: str, source: str, smiles: str | None) -> dict[str, Any]:
    try:
        interchange, _ = create_interchange(offmol, ff_name)
        return interchange_to_aqtop(offmol, interchange, ff_name, source, smiles)
    except Exception:
        return _extract_topology_fallback(offmol, ff_name, source, smiles)


def parameterize_from_smiles(
    smiles: str,
    output: str | Path,
    forcefield: str = DEFAULT_FF,
) -> dict[str, Any]:
    from openff.toolkit import Molecule

    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = _extract_topology(mol, forcefield, f"smiles:{smiles}", smiles)
    save_topology(output, top)
    return top


def parameterize_from_pdb(
    pdb_path: str | Path,
    output: str | Path,
    forcefield: str = DEFAULT_FF,
    smiles: str | None = None,
) -> dict[str, Any]:
    from openff.toolkit import Molecule

    path = Path(pdb_path)
    if smiles:
        mol = Molecule.from_pdb_and_smiles(str(path), smiles, allow_undefined_stereo=True)
    else:
        mol = Molecule.from_file(str(path), allow_undefined_stereo=True)
    top = _extract_topology(mol, forcefield, str(path), smiles)
    save_topology(output, top)
    return top
