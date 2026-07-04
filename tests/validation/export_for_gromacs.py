"""Export .aqtop coordinates for external GROMACS validation."""

from __future__ import annotations

import sys
from pathlib import Path


def export_gro(aqtop_path: Path, outdir: Path) -> None:
    from rustquantic import load_topology

    top = load_topology(str(aqtop_path))
    coords, atom_types, _ = top.to_pdb_types()
    outdir.mkdir(parents=True, exist_ok=True)
    gro = outdir / "system.gro"
    with open(gro, "w") as f:
        f.write("Amphiquantic exported structure\n")
        f.write(f"{len(coords)}\n")
        for i, ((x, y, z), elem) in enumerate(zip(coords, atom_types), start=1):
            f.write(f"{i:5d}{elem:>5} MOL {i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write("   5.00000   5.00000   5.00000\n")
    print(f"Wrote {gro}")


if __name__ == "__main__":
    export_gro(Path(sys.argv[1]), Path(sys.argv[2]))
