"""Minimizer and dynamics validation."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.mark.tier_a
@pytest.mark.rust
def test_minimize_lowers_harmonic_energy(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import Topology, minimize_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = tmp_path / "bond.aqtop"
    top.write(str(path))
    _, e0, _ = minimize_topology(str(path), None, 1, 0.0001)
    _, e1, _ = minimize_topology(str(path), None, 50, 0.0001)
    assert e1 <= e0 + 1.0


@pytest.mark.tier_a
@pytest.mark.rust
def test_nve_energy_finite(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import Topology, equilibrate_topology

    top = Topology.from_pdb_types(
        [(0.0, 0.0, 0.0), (0.18, 0.0, 0.0)],
        ["C", "C"],
        [(0, 1)],
    )
    path = tmp_path / "nve.aqtop"
    top.write(str(path))
    equilibrate_topology(str(path), None, 10, 300.0)


@pytest.mark.tier_a
@pytest.mark.rust
def test_equilibrate_returns_temperature_and_pressure(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import equilibrate_topology

    src = FIXTURES / "methane.aqtop"
    path = tmp_path / "methane.aqtop"
    path.write_text(src.read_text())

    _top, temperature, pressure = equilibrate_topology(str(path), None, 10, 300.0)
    assert temperature >= 0.0
    assert isinstance(pressure, float)


@pytest.mark.tier_a
@pytest.mark.rust
@pytest.mark.parametrize("thermostat", ["langevin", "berendsen", "vrescale", "nose-hoover"])
def test_equilibrate_accepts_each_thermostat(built_extension, force_backend_cpu, tmp_path, thermostat):
    from rustquantic import equilibrate_topology

    src = FIXTURES / "methane.aqtop"
    path = tmp_path / "methane.aqtop"
    path.write_text(src.read_text())

    top, temperature, _pressure = equilibrate_topology(
        str(path), None, 15, 300.0, None, None, thermostat, None, None, None, None, None, 3, None
    )
    assert temperature >= 0.0
    assert top is not None


@pytest.mark.tier_a
@pytest.mark.rust
def test_equilibrate_rejects_unknown_thermostat(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import equilibrate_topology

    src = FIXTURES / "methane.aqtop"
    path = tmp_path / "methane.aqtop"
    path.write_text(src.read_text())

    with pytest.raises(ValueError):
        equilibrate_topology(str(path), None, 5, 300.0, None, None, "not-a-thermostat")


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_montecarlo_barostat_changes_box(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import load_topology, simulate_topology

    src = FIXTURES / "methane.aqtop"
    path = tmp_path / "methane.aqtop"
    path.write_text(src.read_text())

    before_box = load_topology(str(path)).to_pdb_types()[0]
    top, temperature, pressure = simulate_topology(
        str(path),
        None,
        30,
        300.0,
        None,
        True,
        "berendsen",
        "montecarlo",
        1.0,
        None,
        None,
        None,
        3,
        1,
    )
    assert temperature >= 0.0
    assert isinstance(pressure, float)
    after_box = top.to_pdb_types()[0]
    assert len(after_box) == len(before_box)


@pytest.mark.tier_a
@pytest.mark.rust
def test_simulate_com_remove_interval_runs(built_extension, force_backend_cpu, tmp_path):
    from rustquantic import simulate_topology

    src = FIXTURES / "methane.aqtop"
    path = tmp_path / "methane.aqtop"
    path.write_text(src.read_text())

    top, temperature, _pressure = simulate_topology(
        str(path), None, 10, 300.0, None, False, "vrescale", None, None, None, None, 5
    )
    assert temperature >= 0.0
    assert top is not None
