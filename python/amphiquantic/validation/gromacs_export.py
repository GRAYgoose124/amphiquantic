"""Export .aqtop to GROMACS .gro, .top, and run .mdp files."""

from __future__ import annotations

import json
from pathlib import Path

from amphiquantic.validation.gromacs_mdp import write_all_mdps, write_mdp
from amphiquantic.validation.gromacs_top import build_gromacs_top, build_gro


def load_aqtop(aqtop_path: Path) -> dict:
    return json.loads(aqtop_path.read_text())


def read_box_nm(aqtop_path: Path) -> tuple[float, float, float]:
    box = load_aqtop(aqtop_path).get("box", {})
    return (
        float(box.get("lx", 5.0)),
        float(box.get("ly", 5.0)),
        float(box.get("lz", 5.0)),
    )


def export_gro(aqtop_path: Path, outdir: Path) -> Path:
    data = load_aqtop(aqtop_path)
    outdir.mkdir(parents=True, exist_ok=True)
    gro = outdir / "system.gro"
    gro.write_text(build_gro(data))
    return gro


def export_top(aqtop_path: Path, outdir: Path) -> Path:
    data = load_aqtop(aqtop_path)
    outdir.mkdir(parents=True, exist_ok=True)
    top = outdir / "system.top"
    title = data.get("metadata", {}).get("source", str(aqtop_path.name))
    top.write_text(build_gromacs_top(data, title=title))
    return top


def export_em_mdp(outdir: Path) -> Path:
    return write_mdp("em", outdir)


def export_all(aqtop_path: Path, outdir: Path) -> dict[str, Path]:
    data = load_aqtop(aqtop_path)
    if not data.get("atoms"):
        raise ValueError(f"No atoms in {aqtop_path}")

    outdir.mkdir(parents=True, exist_ok=True)
    gro = outdir / "system.gro"
    top = outdir / "system.top"
    gro.write_text(build_gro(data))
    title = data.get("metadata", {}).get("source", aqtop_path.name)
    top.write_text(build_gromacs_top(data, title=title))
    mdps = write_all_mdps(outdir)
    return {"gro": gro, "top": top, **mdps}
