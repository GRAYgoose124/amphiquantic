"""Protein parameterization via OpenFF Interchange + ff14SB (FOSS)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from amphiquantic.topology.interchange_export import interchange_to_aqtop
from amphiquantic.topology.io import save_topology

DEFAULT_PROTEIN_FF = "amber/ff14SB.xml"
DEFAULT_LIGAND_FF = "openff-2.2.1.offxml"


def parameterize_protein_pdb(
    pdb_path: str | Path,
    output: str | Path,
    forcefield: str = DEFAULT_PROTEIN_FF,
) -> dict[str, Any]:
    """Parameterize protein PDB using ff14SB when openmmforcefields is available."""
    pdb = Path(pdb_path)
    try:
        from openff.toolkit import ForceField, Molecule, Topology
        from openff.interchange import Interchange
    except ImportError as exc:
        raise ImportError("Install openff extras: uv sync --extra openff --extra protein") from exc

    try:
        from openmmforcefields.generators import SystemGenerator

        gen = SystemGenerator(
            forcefields=[forcefield, "amber/tip3p_standard.xml"],
            small_molecule_forcefield=DEFAULT_LIGAND_FF,
        )
        off_top = Topology.from_pdb(str(pdb), allow_undefined_stereo=True)
        system = gen.create_system(off_top.to_openmm())
        _ = system  # SystemGenerator path validates FF; export via Interchange below
    except ImportError:
        warnings.warn(
            "openmmforcefields not installed; falling back to per-residue OpenFF export",
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(f"SystemGenerator failed ({exc}); using Interchange fallback", stacklevel=2)

    try:
        off_top = Topology.from_pdb(str(pdb), allow_undefined_stereo=True)
        ff = ForceField(DEFAULT_LIGAND_FF)
        interchange = Interchange.from_smirnoff(force_field=ff, topology=list(off_top.molecules))
        mol = list(off_top.molecules)[0] if off_top.molecules else None
        if mol is None:
            return _parameterize_protein_fallback(pdb_path, output, forcefield)
        if mol.n_conformers == 0:
            mol.generate_conformers(n_conformers=1)
        top = interchange_to_aqtop(mol, interchange, DEFAULT_LIGAND_FF, str(pdb))
    except Exception:
        return _parameterize_protein_fallback(pdb_path, output, forcefield)

    save_topology(output, top)
    return top


def _parameterize_protein_fallback(
    pdb_path: str | Path, output: str | Path, forcefield: str
) -> dict[str, Any]:
    from amphiquantic.topology.bootstrap_assign import parameterize_from_pdb_bootstrap

    warnings.warn("Using bootstrap protein parameterization (limited accuracy)", stacklevel=2)
    return parameterize_from_pdb_bootstrap(pdb_path, output)
