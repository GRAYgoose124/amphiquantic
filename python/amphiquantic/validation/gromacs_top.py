"""Build GROMACS .top from amphiquantic .aqtop JSON."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

RAD_TO_DEG = 180.0 / math.pi


def _sigma_nm(sigma: float) -> float:
    """Convert aqtop sigma (Å) to GROMACS nm."""
    if sigma <= 0.0:
        return 0.0
    # Values > 1 Å are already stored in Å (TIP3P OW = 3.166); small values same scale.
    return sigma / 10.0


def _atom_type_key(atom: dict[str, Any]) -> tuple[float, float, float, str]:
    return (atom["mass"], atom["sigma"], atom["epsilon"], atom["element"])


def _residue_name(atom: dict[str, Any], mol_id: int) -> str:
    name = atom.get("name", "")
    elem = atom.get("element", "X")
    if name == "OW" and elem == "O":
        return "SOL"
    if elem in {"NA", "CL", "K", "CA", "MG"} and name.startswith(elem):
        return elem[:5]
    if mol_id == 0:
        return "MOL"
    return f"M{mol_id}"[:5]


def build_gromacs_top(data: dict[str, Any], title: str = "Amphiquantic export") -> str:
    atoms: list[dict[str, Any]] = data["atoms"]
    bonds = data.get("bonds", [])
    angles = data.get("angles", [])
    dihedrals = data.get("dihedrals", [])
    impropers = data.get("impropers", [])
    exclusions = data.get("exclusions", [])

    type_map: dict[tuple[float, float, float, str], str] = {}
    type_counter: dict[str, int] = defaultdict(int)
    atom_types: list[str] = []

    for atom in atoms:
        key = _atom_type_key(atom)
        if key not in type_map:
            elem = atom["element"]
            type_counter[elem] += 1
            type_name = f"{elem}{type_counter[elem]}"[:5]
            type_map[key] = type_name
        atom_types.append(type_map[key])

    lines: list[str] = [
        f"; Generated from amphiquantic .aqtop ({title})",
        "; Units: nm, kJ/mol, degrees (angles/dihedrals)",
        "",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "1         2          yes        0.5      0.8333",
        "",
        "[ atomtypes ]",
        "; name   mass      charge  ptype  sigma      epsilon",
    ]

    seen_types: set[str] = set()
    for atom in atoms:
        tname = type_map[_atom_type_key(atom)]
        if tname in seen_types:
            continue
        seen_types.add(tname)
        lines.append(
            f"{tname:<5}  {atom['mass']:8.4f}  0.000  A  "
            f"{_sigma_nm(atom['sigma']):10.6f}  {atom['epsilon']:.6f}"
        )

    lines.extend(["", "[ moleculetype ]", "; name   nrexcl", "SYSTEM     3", "", "[ atoms ]"])
    lines.append(";  nr  type  resnr  residue  atom  cgnr  charge    mass")

    mol_ids = [a.get("molecule_id", 0) for a in atoms]
    res_counters: dict[int, int] = defaultdict(int)
    resnr_by_mol: dict[int, int] = {}

    for i, atom in enumerate(atoms, start=1):
        mol = mol_ids[i - 1]
        if mol not in resnr_by_mol:
            res_counters[mol] += 1
            resnr_by_mol[mol] = res_counters[mol]
        resnr = resnr_by_mol[mol]
        resname = _residue_name(atom, mol)
        aname = atom.get("name", atom["element"])[:5]
        lines.append(
            f"{i:5d}  {atom_types[i - 1]:<5}  {resnr:5d}  {resname:<5}  {aname:<5}  "
            f"{i:5d}  {atom['charge']:8.4f}  {atom['mass']:.4f}"
        )

    if bonds:
        lines.extend(["", "[ bonds ]", ";  ai   aj  funct  b0      kb"])
        for b in bonds:
            lines.append(
                f"{b['i'] + 1:5d}  {b['j'] + 1:5d}  1  {b['r0']:.6f}  {b['k']:.4f}"
            )

    if angles:
        lines.extend(["", "[ angles ]", ";  ai   aj   ak  funct  theta    ktheta"])
        for a in angles:
            theta_deg = a["theta0"] * RAD_TO_DEG
            lines.append(
                f"{a['i'] + 1:5d}  {a['j'] + 1:5d}  {a['k'] + 1:5d}  1  "
                f"{theta_deg:.4f}  {a['k_theta']:.4f}"
            )

    if dihedrals:
        lines.extend(["", "[ dihedrals ]", ";  ai   aj   ak   al  funct  phi0   k  mult"])
        for d in dihedrals:
            phi_deg = d["delta"] * RAD_TO_DEG
            lines.append(
                f"{d['i'] + 1:5d}  {d['j'] + 1:5d}  {d['k'] + 1:5d}  {d['l'] + 1:5d}  1  "
                f"{phi_deg:.4f}  {d['k_phi']:.4f}  {d['n']}"
            )

    if impropers:
        lines.extend(["", "[ dihedrals ]", "; impropers appended"])
        for d in impropers:
            phi_deg = d["delta"] * RAD_TO_DEG
            lines.append(
                f"{d['i'] + 1:5d}  {d['j'] + 1:5d}  {d['k'] + 1:5d}  {d['l'] + 1:5d}  2  "
                f"{phi_deg:.4f}  {d['k_phi']:.4f}  {d['n']}"
            )

    if exclusions:
        excl_by_atom: dict[int, set[int]] = defaultdict(set)
        for pair in exclusions:
            if len(pair) >= 2:
                excl_by_atom[pair[0]].add(pair[1])
        lines.extend(["", "[ exclusions ]"])
        for atom_idx in sorted(excl_by_atom):
            others = sorted(excl_by_atom[atom_idx])
            row = " ".join(str(j + 1) for j in others)
            lines.append(f"{atom_idx + 1:5d}  {row}")

    lines.extend(["", "[ system ]", title, "", "[ molecules ]", "SYSTEM  1", ""])
    return "\n".join(lines)


def gro_atom_line(
    serial: int,
    atom: dict[str, Any],
    resnr: int,
    resname: str,
) -> str:
    x, y, z = atom["position"]
    aname = atom.get("name", atom["element"])[:5]
    return f"{resnr:5d}{resname:>5}{aname:>5}{serial:5d}{x:8.3f}{y:8.3f}{z:8.3f}"


def build_gro(data: dict[str, Any], title: str = "Amphiquantic export") -> str:
    atoms = data["atoms"]
    box = data.get("box", {})
    lx = float(box.get("lx", 5.0))
    ly = float(box.get("ly", 5.0))
    lz = float(box.get("lz", 5.0))

    res_counters: dict[int, int] = defaultdict(int)
    resnr_by_mol: dict[int, int] = {}
    lines = [title, str(len(atoms))]

    for i, atom in enumerate(atoms, start=1):
        mol = atom.get("molecule_id", 0)
        if mol not in resnr_by_mol:
            res_counters[mol] += 1
            resnr_by_mol[mol] = res_counters[mol]
        resname = _residue_name(atom, mol)
        lines.append(gro_atom_line(i, atom, resnr_by_mol[mol], resname))

    lines.append(f"{lx:8.5f}{ly:8.5f}{lz:8.5f}")
    return "\n".join(lines) + "\n"
