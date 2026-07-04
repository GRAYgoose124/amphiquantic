"""Ligand docking orchestration (Vina / Meeko / PDBFixer — FOSS subprocess wrappers)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)


def prepare_receptor(pdb_path: Path, out_pdb: Path) -> Path:
    """Fix receptor PDB with PDBFixer when available."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        out_pdb.write_text(pdb_path.read_text())
        return out_pdb

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(out_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return out_pdb


def prepare_ligand(smiles_or_file: str, out_pdbqt: Path) -> Path:
    """Prepare ligand PDBQT via RDKit + Meeko when available."""
    try:
        from rdkit import Chem
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError as exc:
        raise ImportError("Install docking extras: uv sync --extra docking") from exc

    src = Path(smiles_or_file)
    if src.exists() and src.suffix.lower() in {".sdf", ".mol", ".mol2"}:
        mol = Chem.MolFromMolFile(str(src), removeHs=False)
    elif src.exists():
        mol = Chem.MolFromSmiles(src.read_text().strip())
    else:
        mol = Chem.MolFromSmiles(smiles_or_file)
    if mol is None:
        raise ValueError(f"Could not parse ligand: {smiles_or_file}")
    mol = Chem.AddHs(mol)
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    pdbqt_string = PDBQTWriterLegacy.write_string(setups[0])
    out_pdbqt.write_text(pdbqt_string)
    return out_pdbqt


def dock_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    out_poses: Path,
    exhaustiveness: int = 8,
) -> Path:
    """Run AutoDock Vina or Smina."""
    vina = shutil.which("vina") or shutil.which("smina")
    if vina is None:
        raise RuntimeError("vina or smina not found on PATH")

    cmd = [
        vina,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--out",
        str(out_poses),
        "--exhaustiveness",
        str(exhaustiveness),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Docking failed: {result.stderr or result.stdout}")
    return out_poses


def dock_ligand(
    receptor_pdb: str | Path,
    ligand: str,
    output: str | Path,
    box_center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> Path:
    """End-to-end ligand docking; writes poses SDF/PDBQT to output."""
    receptor_pdb = Path(receptor_pdb)
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix="amphi_dock_") as tmp:
        tmp_path = Path(tmp)
        rec_fixed = tmp_path / "receptor.pdb"
        prepare_receptor(receptor_pdb, rec_fixed)
        lig_pdbqt = tmp_path / "ligand.pdbqt"
        prepare_ligand(ligand, lig_pdbqt)
        rec_pdbqt = tmp_path / "receptor.pdbqt"
        try:
            from meeko import Polymer, ResidueChemTemplates
            from meeko import pdbqt_writer as mw

            templates = ResidueChemTemplates.create_from_defaults()
            polymer = Polymer.from_pdb_string(rec_fixed.read_text(), templates)
            pdbqt_string, _ = mw.PDBQTWriter.write_string_from_polymer(polymer)
            rec_pdbqt.write_text(pdbqt_string)
        except Exception:
            rec_pdbqt.write_text(rec_fixed.read_text())

        center = box_center or (0.0, 0.0, 0.0)
        dock_vina(rec_pdbqt, lig_pdbqt, center, box_size, output)
    return output
