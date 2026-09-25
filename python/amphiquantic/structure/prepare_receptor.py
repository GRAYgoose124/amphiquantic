"""Prepare apo receptor PDB from holo structures."""

from __future__ import annotations

from pathlib import Path

_WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"})


def prepare_apo_pdb(
    pdb_text: str,
    *,
    drop_het: bool = True,
    drop_water: bool = True,
) -> str:
    """Return PDB text with protein ATOM records only (optional HETATM removal)."""
    out: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            out.append(line)
            continue
        if not drop_het and line.startswith("HETATM"):
            if drop_water and len(line) >= 20:
                resname = line[17:20].strip().upper()
                if resname in _WATER_RESNAMES:
                    continue
            out.append(line)
    out.append("END")
    return "\n".join(out) + "\n"


def prepare_apo_file(
    input_pdb: str | Path,
    output_pdb: str | Path,
    *,
    drop_het: bool = True,
    drop_water: bool = True,
    protonate: bool = False,
    ph: float = 7.0,
) -> Path:
    """Write apo receptor PDB from holo input."""
    src = Path(input_pdb)
    dst = Path(output_pdb)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        prepare_apo_pdb(
            src.read_text(),
            drop_het=drop_het,
            drop_water=drop_water,
        )
    )
    if protonate:
        protonate_pdb_file(dst, ph=ph, output_pdb=dst)
    return dst


def protonate_pdb_file(
    input_pdb: str | Path,
    *,
    ph: float = 7.0,
    output_pdb: str | Path | None = None,
) -> Path:
    """Add missing atoms/hydrogens with PDBFixer (pH 7 default)."""
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError("PDBFixer required (uv sync --extra protein)") from exc

    src = Path(input_pdb)
    dst = Path(output_pdb) if output_pdb is not None else src
    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return dst
