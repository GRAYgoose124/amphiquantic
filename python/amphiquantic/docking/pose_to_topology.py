"""Apply docked pose coordinates to ligand .aqtop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amphiquantic.topology.io import load_topology, save_topology


def _angstrom_to_nm(coord: tuple[float, float, float]) -> tuple[float, float, float]:
    return (coord[0] / 10.0, coord[1] / 10.0, coord[2] / 10.0)


def _parse_pdbqt_coords(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
        elif line.startswith("ROOT") or line.startswith("ENDROOT"):
            continue
        elif line.startswith("TORSDOF"):
            break
    return coords


def _parse_sdf_coords(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        return []
    n_atoms = int(lines[3].split()[0])
    coords = []
    for line in lines[4 : 4 + n_atoms]:
        parts = line.split()
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return coords


def _coords_from_pdbqt_meeko(path: Path, n_atoms: int) -> list[tuple[float, float, float]]:
    """Expand a Vina/Meeko PDBQT pose to full-atom coordinates via Meeko."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError as exc:
        raise ImportError(
            "Meeko required to map PDBQT poses onto all-atom topologies "
            "(uv sync --extra docking)"
        ) from exc

    pdbqt_mol = PDBQTMolecule.from_file(str(path))
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not mols:
        raise ValueError(f"No molecules parsed from PDBQT: {path}")
    mol = mols[0]
    if mol.GetNumAtoms() != n_atoms:
        raise ValueError(
            f"PDBQT expanded to {mol.GetNumAtoms()} atoms but topology has {n_atoms}"
        )
    if mol.GetNumConformers() == 0:
        raise ValueError(f"PDBQT molecule has no conformer: {path}")
    conf = mol.GetConformer()
    return [
        _angstrom_to_nm(
            (
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            )
        )
        for i in range(mol.GetNumAtoms())
    ]


def parse_pose_coords(
    pose_path: str | Path,
    *,
    n_atoms: int | None = None,
) -> list[tuple[float, float, float]]:
    path = Path(pose_path)
    suffix = path.suffix.lower()
    if suffix == ".pdbqt":
        coords = _parse_pdbqt_coords(path)
        if n_atoms is not None and len(coords) != n_atoms:
            return _coords_from_pdbqt_meeko(path, n_atoms)
        return [_angstrom_to_nm(c) for c in coords]
    if suffix in {".sdf", ".mol"}:
        return _parse_sdf_coords(path)
    if suffix == ".pdb":
        coords = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        if n_atoms is not None and len(coords) != n_atoms:
            raise ValueError(
                f"Pose has {len(coords)} atoms but topology has {n_atoms}"
            )
        return [_angstrom_to_nm(c) for c in coords]
    raise ValueError(f"Unsupported pose format: {path}")


def apply_pose_to_topology(
    ligand_top: dict[str, Any],
    pose_coords: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Update ligand atom positions from docked pose; preserve FF terms."""
    top = dict(ligand_top)
    atoms = list(top["atoms"])
    if len(pose_coords) != len(atoms):
        raise ValueError(
            f"Pose has {len(pose_coords)} atoms but topology has {len(atoms)}"
        )
    for i, (x, y, z) in enumerate(pose_coords):
        a = dict(atoms[i])
        a["position"] = [x, y, z]
        atoms[i] = a
    top["atoms"] = atoms
    return top


def pose_to_topology(
    ligand_aqtop: str | Path,
    pose_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    top = load_topology(ligand_aqtop)
    coords = parse_pose_coords(pose_path, n_atoms=len(top["atoms"]))
    updated = apply_pose_to_topology(top, coords)
    save_topology(output, updated)
    return updated
