"""Python-side helpers for reading trajectory files produced by rustquantic.

The heavy lifting (binary parsing) happens in the Rust extension; these
helpers just reshape the results into numpy arrays for downstream analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dcd(path: str | Path) -> dict:
    """Load a DCD trajectory into numpy arrays.

    Returns a dict with:
        natoms: int
        positions: ndarray, shape (n_frames, natoms, 3), float64
        box: ndarray, shape (n_frames, 3), float64 (a, b, c lengths)
    """
    from rustquantic import read_dcd_trajectory

    natoms, frames, boxes = read_dcd_trajectory(str(path))
    positions = np.asarray(frames, dtype=np.float64).reshape(len(frames), natoms, 3)
    box = np.asarray(boxes, dtype=np.float64).reshape(len(boxes), 3)
    return {"natoms": natoms, "positions": positions, "box": box}


def load_checkpoint(path: str | Path) -> dict:
    """Load a binary checkpoint into numpy arrays.

    Returns a dict with:
        positions: ndarray (natoms, 3)
        velocities: ndarray (natoms, 3)
        box: tuple (lx, ly, lz)
        pbc: bool
        step: int
        seed: int
    """
    from rustquantic import read_checkpoint_file

    positions, velocities, box, pbc, step, seed = read_checkpoint_file(str(path))
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "velocities": np.asarray(velocities, dtype=np.float64),
        "box": box,
        "pbc": pbc,
        "step": step,
        "seed": seed,
    }


def save_checkpoint(
    path: str | Path,
    positions,
    velocities,
    box,
    pbc: bool,
    step: int,
    seed: int = 0,
) -> None:
    """Write a binary checkpoint from numpy-array-like positions/velocities."""
    from rustquantic import write_checkpoint_file

    pos = [tuple(map(float, p)) for p in np.asarray(positions)]
    vel = [tuple(map(float, v)) for v in np.asarray(velocities)]
    write_checkpoint_file(str(path), pos, vel, tuple(map(float, box)), bool(pbc), int(step), int(seed))


__all__ = ["load_dcd", "load_checkpoint", "save_checkpoint"]
