"""Loaders for amphiquantic's native trajectory (.aqtrj) and topology (.aqtop) files.

The .aqtrj format (see ``rust/src/trajectory/mod.rs``) is a plain-text,
append-only format::

    # aqtrj v1 step natoms epot ekin box_lx box_ly box_lz pbc
    FRAME <step> <natoms> <epot> <ekin> <lx> <ly> <lz> <pbc>
    x y z
    x y z
    ...
    FRAME <step> ...
    ...

Coordinates and box lengths are in nanometers, energies in kJ/mol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from amphiquantic.topology.io import load_topology


@dataclass
class Frame:
    """A single trajectory frame."""

    step: int
    positions: np.ndarray  # (n_atoms, 3)
    epot: float
    ekin: float
    box: np.ndarray  # (3,) lx, ly, lz
    pbc: bool


@dataclass
class Trajectory:
    """An in-memory trajectory: stacked frames plus optional atom metadata.

    Attributes
    ----------
    positions:
        Array of shape ``(n_frames, n_atoms, 3)`` in nanometers.
    box:
        Array of shape ``(n_frames, 3)`` of orthorhombic box lengths (nm).
    steps, epot, ekin:
        Per-frame scalars, shape ``(n_frames,)``.
    topology:
        Optional dict loaded from a .aqtop file (see
        ``amphiquantic.topology.io.load_topology``).
    """

    positions: np.ndarray
    box: np.ndarray
    steps: np.ndarray
    epot: np.ndarray
    ekin: np.ndarray
    pbc: np.ndarray
    topology: dict[str, Any] | None = field(default=None)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[1]

    def atom_field(self, key: str) -> np.ndarray:
        """Return a per-atom field (e.g. 'name', 'element', 'residue_id') from topology."""
        if self.topology is None:
            raise ValueError("Trajectory has no attached topology; pass topology=... to load_aqtrj")
        atoms = self.topology["atoms"]
        return np.array([a.get(key) for a in atoms])

    def masses(self) -> np.ndarray:
        return self.atom_field("mass").astype(float)


def _parse_aqtrj(path: str | Path) -> list[Frame]:
    frames: list[Frame] = []
    step = natoms = 0
    epot = ekin = 0.0
    box = np.zeros(3)
    pbc = True
    positions: list[list[float]] = []

    def _flush():
        if positions:
            frames.append(
                Frame(
                    step=step,
                    positions=np.asarray(positions, dtype=float),
                    epot=epot,
                    ekin=ekin,
                    box=box.copy(),
                    pbc=pbc,
                )
            )

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("FRAME"):
                _flush()
                positions = []
                parts = line.split()
                step = int(parts[1])
                natoms = int(parts[2])
                epot = float(parts[3])
                ekin = float(parts[4])
                box = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                pbc = parts[8] != "0"
                continue
            parts = line.split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
    _flush()

    for f in frames:
        if natoms and f.positions.shape[0] != natoms and f is frames[-1]:
            # Tolerate a truncated final frame written mid-flush; still expose it.
            pass
    return frames


def load_aqtrj(path: str | Path, topology: str | Path | dict | None = None) -> Trajectory:
    """Load a .aqtrj trajectory into a :class:`Trajectory` of stacked numpy arrays.

    Parameters
    ----------
    path:
        Path to the .aqtrj file.
    topology:
        Optional .aqtop path (or already-loaded dict) providing per-atom
        metadata (name/element/residue) for selections.
    """
    frames = _parse_aqtrj(path)
    if not frames:
        raise ValueError(f"No frames found in trajectory: {path}")

    n_atoms = frames[0].positions.shape[0]
    # Frames should all have the same atom count; if a trailing frame is
    # short (e.g. writer interrupted), drop it rather than crash.
    frames = [f for f in frames if f.positions.shape[0] == n_atoms]

    positions = np.stack([f.positions for f in frames])
    box = np.stack([f.box for f in frames])
    steps = np.array([f.step for f in frames], dtype=np.int64)
    epot = np.array([f.epot for f in frames], dtype=float)
    ekin = np.array([f.ekin for f in frames], dtype=float)
    pbc = np.array([f.pbc for f in frames], dtype=bool)

    top: dict[str, Any] | None
    if topology is None:
        top = None
    elif isinstance(topology, dict):
        top = topology
    else:
        top = load_topology(topology)

    return Trajectory(
        positions=positions,
        box=box,
        steps=steps,
        epot=epot,
        ekin=ekin,
        pbc=pbc,
        topology=top,
    )
