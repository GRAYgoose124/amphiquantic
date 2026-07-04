"""Protein-protein assembly orchestration (LightDock or geometric fallback)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)


def dock_lightdock(
    chain_a: Path,
    chain_b: Path,
    output_pdb: Path,
    swarms: int = 50,
    glowworms: int = 50,
) -> Path:
    """Run LightDock when installed."""
    ld_setup = shutil.which("lightdock3_setup.py") or shutil.which("lightdock3_setup")
    ld_run = shutil.which("lightdock3.py") or shutil.which("lightdock3")
    if ld_setup is None or ld_run is None:
        raise RuntimeError("LightDock not found on PATH")

    with tempfile.TemporaryDirectory(prefix="amphi_assembly_") as tmp:
        tmp_path = Path(tmp)
        _run([ld_setup, str(chain_a), str(chain_b), "-s", str(swarms), "-g", str(glowworms)], cwd=tmp_path)
        _run([ld_run, "setup.json", "50", "-s", "fast"], cwd=tmp_path)
        ranked = tmp_path / "rank_by_scoring.list"
        if ranked.exists():
            first = ranked.read_text().splitlines()[0].split()[0]
            src = tmp_path / first
            output_pdb.write_text(src.read_text())
            return output_pdb
    raise RuntimeError("LightDock produced no poses")


def geometric_merge(chain_a: Path, chain_b: Path, output_pdb: Path, offset: float = 30.0) -> Path:
    """Fallback: translate chain B and merge PDBs."""
    lines_a = [ln for ln in chain_a.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
    lines_b = []
    for ln in chain_b.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        x = float(ln[30:38]) + offset
        y = float(ln[38:46])
        z = float(ln[46:54])
        new_ln = f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}"
        lines_b.append(new_ln)
    output_pdb.write_text("\n".join(lines_a + lines_b) + "\nEND\n")
    return output_pdb


def dock_assembly(
    chain_a: str | Path,
    chain_b: str | Path,
    output: str | Path,
    method: str = "auto",
) -> Path:
    chain_a = Path(chain_a)
    chain_b = Path(chain_b)
    output = Path(output)
    if method in {"auto", "lightdock"}:
        try:
            return dock_lightdock(chain_a, chain_b, output)
        except RuntimeError:
            if method == "lightdock":
                raise
    return geometric_merge(chain_a, chain_b, output)
