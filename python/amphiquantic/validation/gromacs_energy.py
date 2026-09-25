"""Parse GROMACS energies and compare with amphiquantic."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from amphiquantic.validation.gromacs_run import find_gmx, run_grompp, run_mdrun


def parse_mdrun_log(log_path: Path) -> float | None:
    text = log_path.read_text(errors="replace")
    match = re.search(r"Potential Energy\s*=\s*([-0-9.eE+]+)", text)
    if match:
        return float(match.group(1))
    return None


def parse_gmx_energy_xvg(xvg_path: Path) -> float | None:
    for line in xvg_path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def gmx_potential_energy(outdir: Path, mdp_name: str = "em", nsteps: int = 0) -> float:
    gmx = find_gmx()
    if not gmx:
        raise RuntimeError("GROMACS not installed")

    mdp = outdir / f"{mdp_name}.mdp"
    gro = outdir / "system.gro"
    top = outdir / "system.top"
    tpr = outdir / f"{mdp_name}.tpr"
    deffnm = outdir / mdp_name

    grompp = run_grompp(gmx, mdp, gro, top, tpr)
    if grompp.returncode != 0:
        raise RuntimeError(f"grompp failed: {grompp.stderr[-1500:]}")

    mdrun = run_mdrun(gmx, deffnm, steps=nsteps)
    if mdrun.returncode != 0:
        raise RuntimeError(f"mdrun failed: {mdrun.stderr[-1500:]}")

    log = outdir / f"{mdp_name}.log"
    if log.is_file():
        epot = parse_mdrun_log(log)
        if epot is not None:
            return epot

    xvg = outdir / "potential.xvg"
    subprocess.run(
        [gmx, "energy", "-f", str(deffnm.with_suffix(".edr")), "-o", str(xvg)],
        input="Potential\n",
        capture_output=True,
        text=True,
        cwd=outdir,
    )
    epot = parse_gmx_energy_xvg(xvg)
    if epot is None:
        raise RuntimeError("Could not parse GROMACS potential energy")
    return epot


def compare_em_energy(
    aqtop_path: Path,
    outdir: Path,
    rtol: float = 0.5,
) -> dict[str, float]:
    """Compare single-point / EM potential energies (kJ/mol)."""
    from rustquantic import topology_energy

    from amphiquantic.validation.gromacs_export import export_all

    export_all(aqtop_path, outdir)
    amphi_e = topology_energy(str(aqtop_path))
    gmx_e = gmx_potential_energy(outdir, mdp_name="em", nsteps=0)
    rel_diff = abs(amphi_e - gmx_e) / max(abs(amphi_e), abs(gmx_e), 1.0)
    return {
        "amphi_energy": amphi_e,
        "gmx_energy": gmx_e,
        "relative_diff": rel_diff,
        "within_rtol": rel_diff <= rtol,
    }
