"""Structural analysis: Kabsch superposition, RMSD/RMSF, Rg, geometric series."""

from __future__ import annotations

import numpy as np


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the optimal rotation + translation superposing ``mobile`` onto ``target``.

    Parameters
    ----------
    mobile, target:
        ``(n, 3)`` coordinate arrays (same atom correspondence/order).

    Returns
    -------
    (R, t_mobile, t_target):
        ``R`` is a ``(3, 3)`` rotation matrix and ``t_mobile``/``t_target``
        are the centroids that were subtracted, such that::

            fitted = (mobile - t_mobile) @ R + t_target
    """
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    c_mobile = mobile.mean(axis=0)
    c_target = target.mean(axis=0)
    mm = mobile - c_mobile
    tt = target - c_target

    h = mm.T @ tt
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    r = u @ correction @ vt
    # r satisfies mm @ r ~ tt
    return r, c_mobile, c_target


def superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``mobile`` optimally rotated/translated onto ``target`` (Kabsch)."""
    r, c_mobile, c_target = kabsch(mobile, target)
    return (np.asarray(mobile, dtype=float) - c_mobile) @ r + c_target


def rmsd(a: np.ndarray, b: np.ndarray, fit: bool = False) -> float:
    """RMSD between two ``(n, 3)`` coordinate sets, optionally after Kabsch fitting."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if fit:
        a = superpose(a, b)
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def rmsd_series(
    positions: np.ndarray,
    reference: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    fit: bool = True,
) -> np.ndarray:
    """RMSD of each frame against a reference frame (default: frame 0).

    Parameters
    ----------
    positions:
        ``(n_frames, n_atoms, 3)``.
    reference:
        ``(n_atoms_selected, 3)`` reference coordinates; defaults to the
        (selected) first frame of ``positions``.
    indices:
        Optional atom subset (indices into the atom axis) used for both
        fitting and the RMSD calculation.
    fit:
        If True, each frame is superposed (Kabsch) onto the reference
        before computing RMSD ("fitted" RMSD, as in ``gmx rms``). If
        False, RMSD is computed on the coordinates as-is.
    """
    positions = np.asarray(positions, dtype=float)
    if indices is not None:
        sel = positions[:, indices, :]
    else:
        sel = positions
    ref = sel[0] if reference is None else np.asarray(reference, dtype=float)

    out = np.empty(sel.shape[0], dtype=float)
    for i in range(sel.shape[0]):
        out[i] = rmsd(sel[i], ref, fit=fit)
    return out


def rmsf(
    positions: np.ndarray,
    indices: np.ndarray | None = None,
    fit: bool = True,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the (fitted) mean structure.

    Returns an array of shape ``(n_selected_atoms,)``.
    """
    positions = np.asarray(positions, dtype=float)
    if indices is not None:
        sel = positions[:, indices, :]
    else:
        sel = positions

    n_frames = sel.shape[0]
    if fit:
        ref = sel[0] if reference is None else np.asarray(reference, dtype=float)
        fitted = np.empty_like(sel)
        for i in range(n_frames):
            fitted[i] = superpose(sel[i], ref)
        sel = fitted

    mean_struct = sel.mean(axis=0)
    disp2 = np.sum((sel - mean_struct) ** 2, axis=-1)  # (n_frames, n_atoms)
    return np.sqrt(disp2.mean(axis=0))


def rmsf_per_residue(rmsf_per_atom: np.ndarray, residue_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a per-atom RMSF array by residue id.

    Returns ``(unique_residue_ids, mean_rmsf_per_residue)``.
    """
    residue_ids = np.asarray(residue_ids)
    unique = np.unique(residue_ids)
    out = np.array([rmsf_per_atom[residue_ids == r].mean() for r in unique])
    return unique, out


def radius_of_gyration(positions: np.ndarray, masses: np.ndarray | None = None) -> float:
    """Mass-weighted (or unweighted) radius of gyration of a single ``(n, 3)`` frame."""
    positions = np.asarray(positions, dtype=float)
    if masses is None:
        masses = np.ones(positions.shape[0])
    masses = np.asarray(masses, dtype=float)
    total_mass = masses.sum()
    com = (masses[:, None] * positions).sum(axis=0) / total_mass
    diff = positions - com
    rg2 = (masses * np.sum(diff * diff, axis=-1)).sum() / total_mass
    return float(np.sqrt(rg2))


def radius_of_gyration_series(
    positions: np.ndarray, masses: np.ndarray | None = None, indices: np.ndarray | None = None
) -> np.ndarray:
    """Rg time series for a trajectory of shape ``(n_frames, n_atoms, 3)``."""
    positions = np.asarray(positions, dtype=float)
    if indices is not None:
        positions = positions[:, indices, :]
        if masses is not None:
            masses = np.asarray(masses)[indices]
    return np.array([radius_of_gyration(frame, masses) for frame in positions])


def distance_series(positions: np.ndarray, i: int, j: int) -> np.ndarray:
    """Distance between atoms ``i`` and ``j`` across all frames, shape ``(n_frames,)``."""
    positions = np.asarray(positions, dtype=float)
    diff = positions[:, i, :] - positions[:, j, :]
    return np.linalg.norm(diff, axis=-1)


def angle_series(positions: np.ndarray, i: int, j: int, k: int, degrees: bool = True) -> np.ndarray:
    """Angle i-j-k (vertex at j) across all frames."""
    positions = np.asarray(positions, dtype=float)
    b1 = positions[:, i, :] - positions[:, j, :]
    b2 = positions[:, k, :] - positions[:, j, :]
    cos_theta = np.sum(b1 * b2, axis=-1) / (np.linalg.norm(b1, axis=-1) * np.linalg.norm(b2, axis=-1))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta) if degrees else theta


def dihedral_series(positions: np.ndarray, i: int, j: int, k: int, l: int, degrees: bool = True) -> np.ndarray:
    """Dihedral (torsion) angle i-j-k-l across all frames."""
    positions = np.asarray(positions, dtype=float)
    p0 = positions[:, i, :]
    p1 = positions[:, j, :]
    p2 = positions[:, k, :]
    p3 = positions[:, l, :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    theta = np.arctan2(y, x)
    return np.degrees(theta) if degrees else theta
