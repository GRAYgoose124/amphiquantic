"""Bootstrap parameterization without OpenFF (FOSS element data only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amphiquantic.topology.io import AQTOP_VERSION, save_topology
from amphiquantic.topology.openff_assign import LJ_BY_ELEMENT, _lj

DEFAULT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "Na": 22.99,
    "Cl": 35.45,
    "Ca": 40.078,
}


def parameterize_from_pdb_bootstrap(pdb_path: str | Path, output: str | Path) -> dict[str, Any]:
    from rustquantic import PdbFilePy

    pdb = PdbFilePy.parse(str(pdb_path))
    bonds, _, _ = pdb.determine_bonds()
    atoms = []
    for idx, ((x, y, z), elem) in enumerate(zip(pdb.coords, pdb.atom_types)):
        sigma, epsilon = _lj(elem)
        atoms.append(
            {
                "element": elem,
                "name": f"{elem}{idx + 1}",
                "mass": DEFAULT_MASSES.get(elem, 1.0),
                "charge": 0.0,
                "sigma": sigma,
                "epsilon": epsilon,
                "position": [x, y, z],
                "residue_id": 0,
                "molecule_id": 0,
            }
        )
    bond_terms = [{"i": i, "j": j, "k": 500.0, "r0": 0.15} for i, j in bonds]
    exclusions = []
    for b in bond_terms:
        exclusions.append([b["i"], b["j"]])
        exclusions.append([b["j"], b["i"]])
    top = {
        "version": AQTOP_VERSION,
        "metadata": {"openff_version": "bootstrap", "source": str(pdb_path), "smiles": None},
        "box": {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False},
        "atoms": atoms,
        "bonds": bond_terms,
        "angles": [],
        "dihedrals": [],
        "exclusions": exclusions,
    }
    save_topology(output, top)
    return top


def parameterize_from_smiles_bootstrap(smiles: str, output: str | Path) -> dict[str, Any]:
    """Bootstrap SMILES via OpenFF when available, else RDKit 3D embed + element tables."""
    src = Path(smiles)
    if src.is_file():
        smiles = src.read_text().strip()
    try:
        from amphiquantic.topology.openff_assign import parameterize_from_smiles

        return parameterize_from_smiles(smiles, output)
    except ImportError:
        pass

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "SMILES bootstrap requires openff-toolkit or rdkit (uv sync --extra docking)"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        raise RuntimeError(f"RDKit embed failed for SMILES: {smiles}")

    conf = mol.GetConformer()
    atoms = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        elem = atom.GetSymbol()
        pos = conf.GetAtomPosition(idx)
        sigma, epsilon = _lj(elem)
        atoms.append(
            {
                "element": elem,
                "name": f"{elem}{idx + 1}",
                "mass": DEFAULT_MASSES.get(elem, 1.0),
                "charge": 0.0,
                "sigma": sigma,
                "epsilon": epsilon,
                "position": [pos.x / 10.0, pos.y / 10.0, pos.z / 10.0],
                "residue_id": 0,
                "molecule_id": 0,
            }
        )
    bond_terms = []
    exclusions = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_terms.append({"i": i, "j": j, "k": 500.0, "r0": 0.15})
        exclusions.extend([[i, j], [j, i]])
    top = {
        "version": AQTOP_VERSION,
        "metadata": {"openff_version": "bootstrap-rdkit", "source": smiles, "smiles": smiles},
        "box": {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False},
        "atoms": atoms,
        "bonds": bond_terms,
        "angles": [],
        "dihedrals": [],
        "exclusions": exclusions,
    }
    save_topology(output, top)
    return top
