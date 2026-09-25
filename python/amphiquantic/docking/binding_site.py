"""Auto binding-site box estimation for docking."""

from __future__ import annotations

import json
from pathlib import Path


def _parse_pdb_coords(pdb_path: str | Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    return coords


def _ca_centroid(pdb_path: str | Path) -> tuple[float, float, float]:
    cas: list[tuple[float, float, float]] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name != "CA":
            continue
        try:
            cas.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    if not cas:
        coords = _parse_pdb_coords(pdb_path)
        if not coords:
            return (0.0, 0.0, 0.0)
        n = len(coords)
        return (
            sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n,
        )
    n = len(cas)
    return (sum(c[0] for c in cas) / n, sum(c[1] for c in cas) / n, sum(c[2] for c in cas) / n)


def auto_docking_box(
    receptor_pdb: str | Path,
    *,
    size: float = 20.0,
    reference_ligand: str | Path | None = None,
) -> dict[str, float | tuple[float, float, float]]:
    """Return center (Å) and size (Å) for Vina-style docking."""
    if reference_ligand is not None:
        ref = Path(reference_ligand)
        if ref.is_file():
            coords = _parse_pdb_coords(ref)
            if coords:
                n = len(coords)
                center = (
                    sum(c[0] for c in coords) / n,
                    sum(c[1] for c in coords) / n,
                    sum(c[2] for c in coords) / n,
                )
                return {"center": center, "size": (size, size, size)}

    center = _ca_centroid(receptor_pdb)
    return {"center": center, "size": (size, size, size)}


def write_box_json(box: dict, path: str | Path) -> None:
    out = {
        "center": list(box["center"]),
        "size": list(box["size"]),
    }
    Path(path).write_text(json.dumps(out, indent=2))
