"""Render .aqtrj trajectories to PNG frames and optional MP4."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from amphiquantic.topology.io import load_topology
from amphiquantic.validation.trajectory_analysis import TrajFrame, parse_aqtrj

_MOLECULE_COLORS = ("steelblue", "coral", "mediumseagreen", "goldenrod", "orchid")
_ELEMENT_COLORS = {
    "H": "#d0d0d0",
    "C": "#606060",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "S": "#ffff30",
    "P": "#ff8000",
}


def solute_atom_indices(topology: dict[str, Any]) -> list[int]:
    """Indices of receptor (mol 0) and ligand (mol 1); exclude solvent/ions."""
    atoms = topology["atoms"]
    return [i for i, atom in enumerate(atoms) if atom.get("molecule_id", 0) in (0, 1)]


def _filter_positions(
    positions: list[tuple[float, float, float]],
    indices: list[int] | None,
) -> list[tuple[float, float, float]]:
    if indices is None:
        return positions
    return [positions[i] for i in indices]


def _filter_colors(colors: list[str] | None, indices: list[int] | None) -> list[str] | None:
    if colors is None or indices is None:
        return colors
    return [colors[i] for i in indices]


def atom_colors(topology: dict[str, Any], *, atom_indices: list[int] | None = None) -> list[str]:
    """Color atoms by molecule_id; fall back to element tint when ids are uniform."""
    atoms = topology["atoms"]
    mol_ids = [a.get("molecule_id", 0) for a in atoms]
    unique_ids = sorted(set(mol_ids))
    if len(unique_ids) > 1:
        colors = [_MOLECULE_COLORS[mid % len(_MOLECULE_COLORS)] for mid in mol_ids]
    else:
        colors = [_ELEMENT_COLORS.get(str(a.get("element", "C")).upper(), "#808080") for a in atoms]
    if atom_indices is not None:
        return [colors[i] for i in atom_indices]
    return colors


def _positions_from_topology(topology: dict[str, Any]) -> list[tuple[float, float, float]]:
    return [tuple(a["position"]) for a in topology["atoms"]]


def _finite_positions(
    positions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for p in positions:
        if all(np.isfinite(v) for v in p):
            out.append(p)
    return out


def _centered_limits(positions: list[tuple[float, float, float]]) -> tuple[float, float, float, float, float, float]:
    arr = np.array(positions)
    center = arr.mean(axis=0)
    span = max(float(np.ptp(arr, axis=0).max()), 0.5)
    half = span / 2 + 0.2
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        center[2] - half,
        center[2] + half,
    )


def render_frame(
    frame: TrajFrame,
    *,
    ax=None,
    title: str | None = None,
    colors: list[str] | None = None,
    atom_indices: list[int] | None = None,
) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    positions = _finite_positions(_filter_positions(frame.positions, atom_indices))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    ax.clear()
    scatter_color = _filter_colors(colors, atom_indices) if colors else "steelblue"
    if positions:
        ax.scatter(xs, ys, zs, s=20, c=scatter_color, alpha=0.8)
    if atom_indices is not None and positions:
        xmin, xmax, ymin, ymax, zmin, zmax = _centered_limits(positions)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    elif positions and (frame.box_lx <= 0 or frame.box_ly <= 0 or frame.box_lz <= 0):
        xmin, xmax, ymin, ymax, zmin, zmax = _centered_limits(positions)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    elif frame.box_lx > 0 and frame.box_ly > 0 and frame.box_lz > 0:
        ax.set_xlim(0, frame.box_lx)
        ax.set_ylim(0, frame.box_ly)
        ax.set_zlim(0, frame.box_lz)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    if title:
        ax.set_title(title)
    return ax


def render_structure_frame(
    positions: list[tuple[float, float, float]],
    *,
    ax=None,
    title: str | None = None,
    colors: list[str] | None = None,
    elev: float = 20.0,
    azim: float = 0.0,
) -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    ax.clear()
    scatter_color = colors if colors else "steelblue"
    ax.scatter(xs, ys, zs, s=20, c=scatter_color, alpha=0.8)
    if positions:
        arr = np.array(positions)
        center = arr.mean(axis=0)
        span = max(float(np.ptp(arr, axis=0).max()), 0.5)
        half = span / 2 + 0.2
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    if title:
        ax.set_title(title)
    return ax


def render_trajectory_frames(
    traj_path: str | Path,
    out_dir: str | Path,
    *,
    stride: int = 1,
    topology_path: str | Path | None = None,
    solute_only: bool = False,
) -> list[Path]:
    frames = parse_aqtrj(traj_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors: list[str] | None = None
    atom_indices: list[int] | None = None
    if topology_path is not None:
        topology = load_topology(topology_path)
        if len(topology["atoms"]) != len(frames[0].positions):
            raise ValueError(
                f"Topology has {len(topology['atoms'])} atoms but trajectory has "
                f"{len(frames[0].positions)}"
            )
        if solute_only:
            atom_indices = solute_atom_indices(topology)
        colors = atom_colors(topology, atom_indices=atom_indices)
    paths: list[Path] = []
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for idx, frame in enumerate(frames[::stride]):
        render_frame(
            frame,
            ax=ax,
            title=f"frame {frame.step}",
            colors=colors,
            atom_indices=atom_indices,
        )
        path = out_dir / f"frame_{idx:04d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def render_topology_spin_frames(
    topology_path: str | Path,
    out_dir: str | Path,
    *,
    n_frames: int = 36,
) -> list[Path]:
    topology = load_topology(topology_path)
    positions = _positions_from_topology(topology)
    colors = atom_colors(topology)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx in range(n_frames):
        azim = 360.0 * idx / n_frames
        render_structure_frame(
            positions,
            ax=ax,
            title="docked complex",
            colors=colors,
            elev=20.0,
            azim=azim,
        )
        path = out_dir / f"frame_{idx:04d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def frames_to_mp4(frame_paths: list[Path], mp4_path: Path, fps: int = 24) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not frame_paths:
        return False
    list_file = mp4_path.parent / "frames.txt"
    with open(list_file, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.resolve()}'\n")
            f.write(f"duration {1.0 / fps}\n")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not mp4_path.is_file() or mp4_path.stat().st_size == 0:
        if mp4_path.is_file() and mp4_path.stat().st_size == 0:
            mp4_path.unlink()
        return False
    return True


def render_topology_spin(
    topology_path: str | Path,
    output: str | Path | None = None,
    *,
    n_frames: int = 36,
    fps: int = 12,
) -> Path | None:
    topology_path = Path(topology_path)
    out = Path(output) if output else topology_path.with_suffix(".mp4")
    frames_dir = out.parent / "dock_frames"
    frame_paths = render_topology_spin_frames(topology_path, frames_dir, n_frames=n_frames)
    if not frame_paths:
        return None
    if out.suffix.lower() == ".mp4":
        if frames_to_mp4(frame_paths, out, fps=fps):
            return out
        return frames_dir
    return frames_dir


def render_trajectory(
    traj_path: str | Path,
    topology_path: str | Path | None = None,
    output: str | Path | None = None,
    *,
    stride: int = 1,
    fps: int = 24,
    solute_only: bool = False,
) -> Path | None:
    traj_path = Path(traj_path)
    out = Path(output) if output else traj_path.with_suffix(".mp4")
    frames_dir = out.parent / "frames"
    frame_paths = render_trajectory_frames(
        traj_path,
        frames_dir,
        stride=stride,
        topology_path=topology_path,
        solute_only=solute_only,
    )
    if not frame_paths:
        return None
    if out.suffix.lower() == ".mp4":
        if frames_to_mp4(frame_paths, out, fps=fps):
            return out
        return frames_dir
    return frames_dir
