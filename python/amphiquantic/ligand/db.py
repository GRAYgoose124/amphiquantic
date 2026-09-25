"""Ligand lookup from public databases."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _get_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "amphiquantic/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pubchem_smiles_by_name(name: str, timeout: float = 30.0) -> str:
    """Resolve compound name to isomeric SMILES via PubChem PUG REST."""
    encoded = urllib.parse.quote(name)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IsomericSMILES,SMILES,CanonicalSMILES,ConnectivitySMILES/JSON"
    )
    try:
        data = _get_json(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise LookupError(f"PubChem name not found: {name}") from exc
    props = data["PropertyTable"]["Properties"][0]
    smiles = (
        props.get("IsomericSMILES")
        or props.get("SMILES")
        or props.get("CanonicalSMILES")
        or props.get("ConnectivitySMILES")
    )
    if not smiles:
        raise LookupError(f"No SMILES for PubChem name: {name}")
    return smiles


def pubchem_smiles_by_cid(cid: int, timeout: float = 30.0) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/property/IsomericSMILES,SMILES,CanonicalSMILES,ConnectivitySMILES/JSON"
    )
    data = _get_json(url, timeout=timeout)
    props = data["PropertyTable"]["Properties"][0]
    return (
        props.get("IsomericSMILES")
        or props.get("SMILES")
        or props.get("CanonicalSMILES")
        or props["ConnectivitySMILES"]
    )


def resolve_ligand_query(query: str, timeout: float = 30.0) -> tuple[str, str]:
    """Return (smiles, source_description) for SMILES, CID, or compound name."""
    q = query.strip()
    if re.fullmatch(r"CID:\d+", q, re.I):
        cid = int(q.split(":")[1])
        return pubchem_smiles_by_cid(cid, timeout=timeout), f"pubchem:cid:{cid}"
    if q.isdigit():
        cid = int(q)
        return pubchem_smiles_by_cid(cid, timeout=timeout), f"pubchem:cid:{cid}"
    if re.search(r"[=#()\[\]@\\/+]", q) or q.startswith("C"):
        return q, "smiles:literal"
    return pubchem_smiles_by_name(q, timeout=timeout), f"pubchem:name:{q}"


def extract_hetatm_residue(pdb_text: str, resname: str) -> str:
    """Extract HETATM records for a residue name from PDB text."""
    resname = resname.upper()
    lines = []
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip().upper() == resname:
            lines.append(line)
    if not lines:
        raise LookupError(f"No HETATM records for residue {resname}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def save_ligand_smiles(smiles: str, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(smiles.strip() + "\n")
    return out


def hetatm_pdb_to_smiles(pdb_text: str) -> str:
    """Convert HETATM PDB block to SMILES via RDKit."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit required (uv sync --extra docking)") from exc
    mol = Chem.MolFromPDBBlock(pdb_text)
    if mol is None:
        raise ValueError("Could not parse HETATM block to molecule")
    return Chem.MolToSmiles(mol)


# Co-crystal residue names → PubChem compound name (isomeric SMILES).
_HET_PUBCHEM_NAMES: dict[str, str] = {
    "FK5": "tacrolimus",
}


def smiles_has_complete_stereo(smiles: str) -> bool:
    """Return True when all RDKit-detected chiral centers are specified."""
    try:
        from rdkit import Chem
    except ImportError:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return False
    return True


def smiles_is_openff_ready(smiles: str) -> bool:
    """Return True when OpenFF can build a Molecule from SMILES without stereo errors."""
    try:
        from openff.toolkit import Molecule
        from openff.toolkit.utils.exceptions import UndefinedStereochemistryError
    except ImportError:
        return smiles_has_complete_stereo(smiles)
    try:
        Molecule.from_smiles(smiles)
        return True
    except UndefinedStereochemistryError:
        return False
    except Exception:
        return False


def ligand_smiles_from_hetatm(
    pdb_text: str,
    resname: str,
    *,
    pubchem_name: str | None = None,
) -> str:
    """Resolve ligand SMILES from a co-crystal HET residue.

    Prefer fully specified isomeric SMILES from PubChem when the HET-derived
    SMILES has undefined stereochemistry (common for macrocycles like FK506).
    """
    het = extract_hetatm_residue(pdb_text, resname)
    het_smiles: str | None = None
    try:
        het_smiles = hetatm_pdb_to_smiles(het)
        if smiles_is_openff_ready(het_smiles):
            return het_smiles
    except (ValueError, ImportError):
        pass

    name = pubchem_name or _HET_PUBCHEM_NAMES.get(resname.strip().upper())
    if name:
        return pubchem_smiles_by_name(name)
    if het_smiles is not None:
        return het_smiles
    raise ValueError(f"Could not resolve SMILES for HET residue {resname}")
