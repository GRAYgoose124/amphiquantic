"""Export OpenFF Interchange parameters to .aqtop dict."""

from __future__ import annotations

import math
from typing import Any

from amphiquantic.topology.io import AQTOP_VERSION

KJ_PER_MOL = 1.0
NM_TO_ANGSTROM = 10.0


def _build_exclusions(
    bonds: list[dict], angles: list[dict], dihedrals: list[dict]
) -> list[list[int]]:
    exclusions: list[list[int]] = []
    for b in bonds:
        exclusions.append([b["i"], b["j"]])
        exclusions.append([b["j"], b["i"]])
    for a in angles:
        exclusions.append([a["i"], a["k"]])
        exclusions.append([a["k"], a["i"]])
    return exclusions


def interchange_to_aqtop(
    offmol,
    interchange,
    ff_name: str,
    source: str,
    smiles: str | None = None,
    molecule_id: int = 0,
) -> dict[str, Any]:
    """Convert OpenFF Interchange object to amphiquantic .aqtop dict."""
    conformer = offmol.conformers[0]
    coords = conformer.m_as("angstrom")

    atoms: list[dict[str, Any]] = []
    for idx, atom in enumerate(offmol.atoms):
        x, y, z = coords[idx]
        charge = float(atom.partial_charge.m) if atom.partial_charge is not None else 0.0
        sigma, epsilon = 0.34, 0.36
        atoms.append(
            {
                "element": atom.symbol,
                "name": atom.name or f"{atom.symbol}{idx + 1}",
                "mass": float(atom.mass.m),
                "charge": charge,
                "sigma": sigma,
                "epsilon": epsilon,
                "position": [float(x), float(y), float(z)],
                "residue_id": int(getattr(atom, "metadata", {}).get("residue_number", 0) or 0),
                "molecule_id": molecule_id,
            }
        )

    # Nonbonded from interchange (nm -> angstrom for sigma in our engine uses nm-like scale from OpenFF)
    try:
        nb = interchange.collections["Electrostatics"]
        for idx in range(len(atoms)):
            if idx < len(nb.charges):
                atoms[idx]["charge"] = float(nb.charges[idx].m)
    except (KeyError, AttributeError, IndexError):
        pass

    try:
        vdw = interchange.collections["vdW"]
        for idx in range(len(atoms)):
            if idx < len(vdw.scale_factors):
                sf = vdw.scale_factors[idx]
                if hasattr(sf, "sigma"):
                    atoms[idx]["sigma"] = float(sf.sigma.m) * NM_TO_ANGSTROM
                if hasattr(sf, "epsilon"):
                    atoms[idx]["epsilon"] = float(sf.epsilon.m)
    except (KeyError, AttributeError, IndexError):
        pass

    bonds: list[dict[str, Any]] = []
    angles: list[dict[str, Any]] = []
    dihedrals: list[dict[str, Any]] = []
    impropers: list[dict[str, Any]] = []

    try:
        bonds_coll = interchange.collections["Bonds"]
        for idx in range(len(bonds_coll.key)):
            i, j = bonds_coll.key[idx]
            k_val = float(bonds_coll.parameters["k"][idx].m) if "k" in bonds_coll.parameters else 500.0
            r0 = float(bonds_coll.parameters["length"][idx].m) * NM_TO_ANGSTROM
            bonds.append({"i": int(i), "j": int(j), "k": k_val, "r0": r0})
    except (KeyError, AttributeError, IndexError):
        for bond in offmol.bonds:
            i, j = bond.atom1_index, bond.atom2_index
            length = offmol.get_bond_length(bond).m_as("nanometer") * NM_TO_ANGSTROM
            bonds.append({"i": i, "j": j, "k": 500.0, "r0": float(length)})

    try:
        angles_coll = interchange.collections["Angles"]
        for idx in range(len(angles_coll.key)):
            i, j, k = angles_coll.key[idx]
            k_theta = float(angles_coll.parameters["k"][idx].m)
            theta0 = float(angles_coll.parameters["angle"][idx].m)
            angles.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "k_theta": k_theta,
                    "theta0": math.radians(theta0) if theta0 > 2 * math.pi else theta0,
                }
            )
    except (KeyError, AttributeError, IndexError):
        pass

    try:
        proper = interchange.collections["ProperTorsionHandler"]
        for idx in range(len(proper.key)):
            i, j, k, l = proper.key[idx]
            k_phi = float(proper.parameters["k"][idx].m)
            period = int(proper.parameters["periodicity"][idx])
            phase = float(proper.parameters["phase"][idx].m)
            dihedrals.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "l": int(l),
                    "k_phi": k_phi,
                    "n": period,
                    "delta": math.radians(phase) if abs(phase) <= math.pi * 2 else phase,
                }
            )
    except (KeyError, AttributeError, IndexError):
        pass

    try:
        improper = interchange.collections["ImproperTorsionHandler"]
        for idx in range(len(improper.key)):
            i, j, k, l = improper.key[idx]
            k_phi = float(improper.parameters["k"][idx].m)
            period = int(improper.parameters["periodicity"][idx])
            phase = float(improper.parameters["phase"][idx].m)
            impropers.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "l": int(l),
                    "k_phi": k_phi,
                    "n": period,
                    "delta": math.radians(phase) if abs(phase) <= math.pi * 2 else phase,
                }
            )
    except (KeyError, AttributeError, IndexError):
        pass

    exclusions = _build_exclusions(bonds, angles, dihedrals)

    return {
        "version": AQTOP_VERSION,
        "metadata": {
            "openff_version": ff_name,
            "source": source,
            "smiles": smiles,
        },
        "box": {"lx": 0.0, "ly": 0.0, "lz": 0.0, "pbc": False},
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "impropers": impropers,
        "exclusions": exclusions,
    }


def create_interchange(offmol, ff_name: str):
    try:
        from openff.interchange import Interchange
    except ImportError as exc:
        raise ImportError(
            "openff-interchange not installed; install manually or use bootstrap fallback"
        ) from exc
    from openff.toolkit import ForceField

    ff = ForceField(ff_name)
    offmol.assign_partial_charges(partial_charge_method="am1bcc", normalize_partial_charges=True)
    if offmol.n_conformers == 0:
        offmol.generate_conformers(n_conformers=1)
    return Interchange.from_smirnoff(force_field=ff, topology=[offmol]), ff
