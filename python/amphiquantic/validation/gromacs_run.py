"""Run GROMACS grompp/mdrun on exported validation inputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_gmx() -> str | None:
    for name in ("gmx", "gmx_mpi", "gromacs"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_grompp(gmx: str, mdp: Path, gro: Path, top: Path, tpr: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [gmx, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", "10"],
        capture_output=True,
        text=True,
    )


def run_mdrun(gmx: str, deffnm: Path, steps: int | None = None) -> subprocess.CompletedProcess:
    # Single rank avoids domain-decomposition failures on small validation boxes.
    cmd = [gmx, "mdrun", "-deffnm", str(deffnm), "-nb", "cpu", "-ntmpi", "1", "-nt", "1"]
    if steps is not None:
        cmd.extend(["-nsteps", str(steps)])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=deffnm.parent)


def validate_export(outdir: Path, mdp_name: str = "em") -> dict[str, str | int | float | bool]:
    gmx = find_gmx()
    if not gmx:
        return {"gmx_available": False}

    mdp = outdir / f"{mdp_name}.mdp"
    gro = outdir / "system.gro"
    top = outdir / "system.top"
    tpr = outdir / f"{mdp_name}.tpr"
    deffnm = outdir / mdp_name

    grompp = run_grompp(gmx, mdp, gro, top, tpr)
    result: dict[str, str | int | float | bool] = {
        "gmx_available": True,
        "grompp_ok": grompp.returncode == 0,
        "grompp_stderr": grompp.stderr[-2000:] if grompp.stderr else "",
    }
    if grompp.returncode != 0:
        return result

    mdrun = run_mdrun(gmx, deffnm, steps=0 if mdp_name == "em" else 10)
    result["mdrun_ok"] = mdrun.returncode == 0
    result["mdrun_stderr"] = mdrun.stderr[-2000:] if mdrun.stderr else ""
    edr = outdir / f"{mdp_name}.edr"
    result["edr_written"] = edr.is_file()
    return result
