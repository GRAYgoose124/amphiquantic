"""Implicit solvation and GB energy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
METHANE = ROOT / "tests" / "fixtures" / "methane.aqtop"


@pytest.mark.tier_a
@pytest.mark.rust
def test_prepare_implicit_metadata(tmp_path, built_extension):
    from amphiquantic.topology.implicit import prepare_implicit_file
    from amphiquantic.validation.gromacs_export import load_aqtop

    out = tmp_path / "implicit.aqtop"
    prepare_implicit_file(str(METHANE), str(out))
    top = load_aqtop(out)
    assert top["metadata"]["solvation_model"] == "gb_obc2"
    assert top["box"]["pbc"] is False
    assert all("born_r" in a for a in top["atoms"])


@pytest.mark.tier_a
@pytest.mark.rust
def test_gb_energy_finite(tmp_path, built_extension):
    from amphiquantic.topology.implicit import prepare_implicit_file
    from rustquantic import topology_energy

    out = tmp_path / "implicit.aqtop"
    prepare_implicit_file(str(METHANE), str(out))
    e = topology_energy(str(out))
    assert e == e  # not NaN
    assert abs(e) < 1e12


@pytest.mark.tier_b
@pytest.mark.rust
def test_implicit_md_methane_no_nan(tmp_path, built_extension, force_backend_cpu):
    """Short implicit GB MD stays finite on methane."""
    from amphiquantic.topology.implicit import prepare_implicit_file
    from amphiquantic.validation.trajectory_analysis import parse_aqtrj
    from rustquantic import simulate_topology

    implicit = tmp_path / "implicit.aqtop"
    traj = tmp_path / "md.aqtrj"
    prepare_implicit_file(str(METHANE), str(implicit))
    simulate_topology(str(implicit), str(traj), 10, 300.0, None, False)
    frames = parse_aqtrj(traj)
    assert len(frames) >= 1
    for frame in frames:
        assert frame.epot == frame.epot
        for pos in frame.positions:
            assert all(v == v for v in pos)


@pytest.mark.tier_a
def test_render_trajectory_frames(tmp_path):
    from amphiquantic.visualization.traj_render import render_trajectory_frames

    traj = tmp_path / "t.aqtrj"
    traj.write_text(
        "# aqtrj v1\n"
        "FRAME 0 2 1.0 2.0 3.0 3.0 3.0 0\n"
        "0 0 0\n"
        "1 0 0\n"
        "FRAME 1 2 1.5 2.5 3.0 3.0 3.0 0\n"
        "0.1 0 0\n"
        "1.1 0 0\n"
    )
    frames_dir = tmp_path / "frames"
    paths = render_trajectory_frames(traj, frames_dir)
    assert len(paths) == 2
    assert all(p.is_file() for p in paths)


@pytest.mark.tier_a
def test_atom_colors_by_molecule():
    from amphiquantic.visualization.traj_render import atom_colors

    top = {
        "atoms": [
            {"element": "C", "molecule_id": 0},
            {"element": "H", "molecule_id": 0},
            {"element": "C", "molecule_id": 1},
            {"element": "O", "molecule_id": 1},
        ]
    }
    colors = atom_colors(top)
    assert len(colors) == 4
    assert colors[0] == colors[1]
    assert colors[2] == colors[3]
    assert colors[0] != colors[2]


@pytest.mark.tier_a
def test_render_topology_spin_frames(tmp_path):
    from amphiquantic.visualization.traj_render import render_topology_spin

    out = tmp_path / "dock_preview.mp4"
    result = render_topology_spin(METHANE, out, n_frames=8, fps=8)
    assert result is not None
    dock_frames = tmp_path / "dock_frames"
    if out.is_file():
        assert out.stat().st_size > 0
    else:
        assert result == dock_frames
        assert len(list(dock_frames.glob("frame_*.png"))) == 8


@pytest.mark.tier_a
def test_render_trajectory_with_topology(tmp_path):
    import json

    from amphiquantic.visualization.traj_render import render_trajectory_frames

    top_path = tmp_path / "complex.aqtop"
    top = {
        "version": 1,
        "metadata": {},
        "box": {"lx": 3.0, "ly": 3.0, "lz": 3.0, "pbc": True},
        "atoms": [
            {
                "element": "C",
                "position": [0.0, 0.0, 0.0],
                "molecule_id": 0,
            },
            {
                "element": "O",
                "position": [1.0, 0.0, 0.0],
                "molecule_id": 1,
            },
        ],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "exclusions": [],
    }
    top_path.write_text(json.dumps(top))
    traj = tmp_path / "t.aqtrj"
    traj.write_text(
        "# aqtrj v1\n"
        "FRAME 0 2 1.0 2.0 3.0 3.0 3.0 0\n"
        "0 0 0\n"
        "1 0 0\n"
    )

    plain_dir = tmp_path / "plain"
    colored_dir = tmp_path / "colored"
    plain_paths = render_trajectory_frames(traj, plain_dir)
    colored_paths = render_trajectory_frames(traj, colored_dir, topology_path=top_path)
    assert len(plain_paths) == len(colored_paths) == 1
    assert plain_paths[0].stat().st_size != colored_paths[0].stat().st_size


@pytest.mark.tier_a
def test_solute_atom_indices():
    from amphiquantic.topology.io import load_topology
    from amphiquantic.visualization.traj_render import solute_atom_indices

    methane = load_topology(ROOT / "tests" / "fixtures" / "methane.aqtop")
    assert solute_atom_indices(methane) == list(range(len(methane["atoms"])))

    mixed = {
        "atoms": [
            {"molecule_id": 0, "name": "CA"},
            {"molecule_id": 1, "name": "C1"},
            {"molecule_id": 2, "name": "OW"},
            {"molecule_id": 2, "name": "HW1"},
            {"molecule_id": 3, "name": "NA"},
        ]
    }
    assert solute_atom_indices(mixed) == [0, 1]
