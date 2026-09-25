"""GROMACS export format tests (Tier A); live GROMACS is Tier B."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.validation.gromacs_export import export_all, load_aqtop, read_box_nm
from amphiquantic.validation.gromacs_top import build_gromacs_top

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.tier_a
def test_read_box_from_fixture():
    aqtop = FIXTURES / "complex" / "dipeptide_solv.aqtop"
    lx, ly, lz = read_box_nm(aqtop)
    assert lx == pytest.approx(3.3)
    assert ly == pytest.approx(3.15)
    assert lz == pytest.approx(3.0)


@pytest.mark.tier_a
def test_top_contains_bonds_and_atomtypes():
    data = load_aqtop(FIXTURES / "complex" / "dipeptide_solv.aqtop")
    top = build_gromacs_top(data)
    assert "[ atomtypes ]" in top
    assert "[ bonds ]" in top
    assert "[ angles ]" in top
    assert "[ exclusions ]" in top
    assert "SYSTEM  1" in top


@pytest.mark.tier_a
def test_top_angle_uses_degrees():
    data = load_aqtop(FIXTURES / "complex" / "dipeptide_solv.aqtop")
    top = build_gromacs_top(data)
    assert "104.5" in top  # TIP3P HOH ~104.52°


@pytest.mark.tier_a
@pytest.mark.rust
def test_export_writes_gro_top_mdps(tmp_path, built_extension):
    aqtop = FIXTURES / "complex" / "dipeptide_solv.aqtop"
    outdir = tmp_path / "gmx"
    paths = export_all(aqtop, outdir)
    gro_lines = paths["gro"].read_text().splitlines()
    lx, ly, lz = read_box_nm(aqtop)
    box_parts = gro_lines[-1].split()
    assert float(box_parts[0]) == pytest.approx(lx)
    assert paths["top"].read_text().startswith("; Generated")
    assert paths["em"].is_file()
    assert paths["nvt"].is_file()
    assert "integrator" in paths["em"].read_text()


@pytest.mark.tier_b
def test_grompp_on_fixture(tmp_path):
    pytest.importorskip("subprocess")
    from amphiquantic.validation.gromacs_run import find_gmx, validate_export

    if not find_gmx():
        pytest.skip("GROMACS not installed")

    aqtop = FIXTURES / "complex" / "dipeptide_solv.aqtop"
    outdir = tmp_path / "gmx"
    export_all(aqtop, outdir)
    result = validate_export(outdir, mdp_name="em")
    assert result["grompp_ok"], result.get("grompp_stderr", "")
