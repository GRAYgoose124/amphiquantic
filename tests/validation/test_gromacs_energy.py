"""GROMACS energy comparison tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from amphiquantic.validation.gromacs_energy import compare_em_energy
from amphiquantic.validation.gromacs_run import find_gmx

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.tier_b
def test_gromacs_em_energy_vs_amphiquantic(tmp_path, built_extension, force_backend_cpu):
    if not find_gmx():
        pytest.skip("GROMACS not installed")

    result = compare_em_energy(
        FIXTURES / "methane.aqtop",
        tmp_path / "gmx",
        rtol=0.9,
    )
    assert result["within_rtol"], (
        f"amphi={result['amphi_energy']:.1f} gmx={result['gmx_energy']:.1f} "
        f"rel={result['relative_diff']:.3f}"
    )
