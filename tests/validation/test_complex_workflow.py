"""Complex workflow integration test."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.complex
@pytest.mark.tier_b
@pytest.mark.rust
def test_complex_build_script(tmp_path, built_extension, force_backend_cpu):
    out = tmp_path / "complex"
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "steps" / "complex" / "build.sh"),
            str(ROOT / "tests" / "fixtures" / "complex" / "ala_dipeptide.pdb"),
            "",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out / "equil.aqtop").is_file()

    from rustquantic import load_topology

    n_complex = load_topology(str(out / "complex.aqtop")).num_atoms()
    n_solv = load_topology(str(out / "solvated.aqtop")).num_atoms()
    assert n_solv > n_complex
    assert abs(load_topology(str(out / "ionized.aqtop")).net_charge()) < 0.5
