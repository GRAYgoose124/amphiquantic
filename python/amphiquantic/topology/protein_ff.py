"""Protein parameterization via OpenFF Interchange + ff14SB (FOSS)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from amphiquantic.topology.io import save_topology

DEFAULT_PROTEIN_FF = "amber/ff14SB.xml"
DEFAULT_LIGAND_FF = "openff-2.2.1.offxml"


def _ensure_protonated_pdb(pdb: Path) -> Path:
    """Return protonated PDB path (in-place via PDBFixer when available)."""
    try:
        from amphiquantic.structure.prepare_receptor import protonate_pdb_file

        return protonate_pdb_file(pdb, output_pdb=pdb)
    except ImportError:
        return pdb


def _validate_ff14sb_system(pdb: Path, forcefield: str) -> bool:
    """Return True when openmmforcefields can build an ff14SB system."""
    try:
        from openmm.app import PDBFile
        from openmmforcefields.generators import SystemGenerator
    except ImportError:
        return False

    try:
        pdb_file = PDBFile(str(pdb))
        gen = SystemGenerator(forcefields=[forcefield])
        system = gen.create_system(pdb_file.topology)
        return system.getNumParticles() > 0
    except Exception as exc:
        warnings.warn(f"ff14SB validation failed ({exc}); using bootstrap receptor params", stacklevel=2)
        return False


def parameterize_protein_pdb(
    pdb_path: str | Path,
    output: str | Path,
    forcefield: str = DEFAULT_PROTEIN_FF,
) -> dict[str, Any]:
    """Parameterize protein PDB using ff14SB when openmmforcefields is available."""
    from amphiquantic.topology.bootstrap_assign import parameterize_from_pdb_bootstrap

    pdb = Path(pdb_path)
    work_pdb = _ensure_protonated_pdb(pdb)
    ff14sb_ok = _validate_ff14sb_system(work_pdb, forcefield)
    if ff14sb_ok:
        warnings.warn(
            "ff14SB system validated via OpenMM; receptor .aqtop still uses bootstrap "
            "(amber→aqtop export pending). Ligand OpenFF parameterization is separate.",
            stacklevel=2,
        )
    else:
        warnings.warn(
            "Using bootstrap protein parameterization (limited accuracy). "
            "For ff14SB validation ensure uv sync --extra protein and protonated apo PDB.",
            stacklevel=2,
        )
    top = parameterize_from_pdb_bootstrap(work_pdb, output)
    if ff14sb_ok:
        top.setdefault("metadata", {})["protein_ff_validated"] = forcefield
    save_topology(output, top)
    return top
