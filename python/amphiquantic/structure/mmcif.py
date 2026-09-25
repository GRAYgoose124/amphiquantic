"""Minimal PDBx/mmCIF atom_site parser and PDB writer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MmcifAtom:
    group: str
    name: str
    element: str
    resname: str
    chain: str
    seq_id: int
    x: float
    y: float
    z: float


def _parse_loop_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    headers: list[str] = []
    rows: list[list[str]] = []
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("loop_"):
            break
        if line.startswith("_"):
            headers.append(line.split()[0])
            i += 1
            continue
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if parts:
            rows.append(parts)
        i += 1
    return headers, rows, i


def parse_atom_site(mmcif_text: str) -> list[MmcifAtom]:
    """Parse _atom_site records from mmCIF text."""
    lines = mmcif_text.splitlines()
    atoms: list[MmcifAtom] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            headers, rows, next_i = _parse_loop_table(lines, i)
            if headers and headers[0].startswith("_atom_site."):
                col = {h.split(".", 1)[1]: idx for idx, h in enumerate(headers)}

                def get(row: list[str], key: str, default: str = "") -> str:
                    idx = col.get(key)
                    if idx is None or idx >= len(row):
                        return default
                    return row[idx]

                for row in rows:
                    try:
                        atoms.append(
                            MmcifAtom(
                                group=get(row, "group_PDB", "ATOM") or "ATOM",
                                name=get(row, "auth_atom_id") or get(row, "label_atom_id", "X"),
                                element=get(row, "type_symbol", "C") or "C",
                                resname=get(row, "auth_comp_id") or get(row, "label_comp_id", "UNK"),
                                chain=get(row, "auth_asym_id") or get(row, "label_asym_id", "A"),
                                seq_id=int(float(get(row, "auth_seq_id") or get(row, "label_seq_id", "1"))),
                                x=float(get(row, "Cartn_x", "0")),
                                y=float(get(row, "Cartn_y", "0")),
                                z=float(get(row, "Cartn_z", "0")),
                            )
                        )
                    except (ValueError, IndexError):
                        continue
            i = next_i
            continue
        i += 1
    return atoms


def mmcif_to_pdb(mmcif_text: str) -> str:
    """Convert mmCIF atom_site records to legacy PDB ATOM/HETATM records."""
    atoms = parse_atom_site(mmcif_text)
    if not atoms:
        raise ValueError("No _atom_site records found in mmCIF")
    out: list[str] = []
    for idx, atom in enumerate(atoms, start=1):
        record = "HETATM" if atom.group == "HETATM" else "ATOM  "
        name = atom.name
        if len(name) < 4 and len(atom.element) == 1:
            name = f" {name:<3s}"
        else:
            name = f"{name:>4s}"
        line = (
            f"{record}"
            f"{idx:5d}"
            f"{name}"
            f" {atom.resname:>3} {atom.chain:>1}"
            f"{atom.seq_id:4d}"
            f"    {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"  1.00  0.00          {atom.element:>2}"
        )
        out.append(line)
    out.append("END")
    return "\n".join(out) + "\n"
