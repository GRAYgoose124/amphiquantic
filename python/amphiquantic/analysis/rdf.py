"""Radial distribution function g(r) with orthorhombic minimum-image PBC."""

from __future__ import annotations

import numpy as np


def _min_image(diff: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply orthorhombic minimum-image convention to displacement vectors."""
    return diff - box * np.round(diff / box)


def radial_distribution_function(
    positions: np.ndarray,
    box: np.ndarray,
    indices_a: np.ndarray | None = None,
    indices_b: np.ndarray | None = None,
    r_max: float | None = None,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute g(r) between two (possibly identical) atom groups, averaged over frames.

    Parameters
    ----------
    positions:
        ``(n_frames, n_atoms, 3)``.
    box:
        Either ``(3,)`` (constant box) or ``(n_frames, 3)`` orthorhombic box
        lengths, same units as ``positions`` (nm).
    indices_a, indices_b:
        Index arrays selecting the two groups. If ``indices_b`` is omitted,
        it defaults to ``indices_a`` (self-RDF, i<j pairs only). If both are
        omitted, all atoms are used.
    r_max:
        Histogram cutoff; defaults to half of the smallest box dimension
        across frames (safe for minimum-image convention).
    n_bins:
        Number of histogram bins.

    Returns
    -------
    (r, g_r):
        Bin-center distances and g(r) values (g(r) -> 1 for an
        uncorrelated / ideal gas at large ``r``, given a large enough box
        and enough frames).
    """
    positions = np.asarray(positions, dtype=float)
    n_frames, n_atoms, _ = positions.shape
    box = np.asarray(box, dtype=float)
    if box.ndim == 1:
        box = np.tile(box, (n_frames, 1))

    if indices_a is None:
        indices_a = np.arange(n_atoms)
    indices_a = np.asarray(indices_a)
    self_rdf = indices_b is None
    indices_b = indices_a if self_rdf else np.asarray(indices_b)

    if r_max is None:
        r_max = float(np.min(box)) / 2.0

    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)

    n_a = len(indices_a)
    n_b = len(indices_b)
    total_pairs = 0

    for f in range(n_frames):
        pa = positions[f, indices_a, :]
        pb = positions[f, indices_b, :]
        b = box[f]

        diff = pa[:, None, :] - pb[None, :, :]
        diff = _min_image(diff, b)
        dist = np.linalg.norm(diff, axis=-1)

        if self_rdf:
            iu = np.triu_indices(n_a, k=1)
            dist_flat = dist[iu]
            total_pairs += len(dist_flat)
        else:
            dist_flat = dist.ravel()
            total_pairs += n_a * n_b

        hist, _ = np.histogram(dist_flat, bins=edges)
        counts += hist

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    mean_box_volume = np.prod(box, axis=1).mean()
    if self_rdf:
        n_pairs_possible = n_a * (n_a - 1) / 2.0
    else:
        n_pairs_possible = float(n_a * n_b)
    density_pairs = n_pairs_possible / mean_box_volume  # pairs per unit volume, per frame

    ideal_counts = density_pairs * shell_volumes * n_frames
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(ideal_counts > 0, counts / ideal_counts, 0.0)

    return r_centers, g_r


def coordination_number(r: np.ndarray, g_r: np.ndarray, density: float, r_cut: float) -> float:
    """Integrate 4*pi*r^2*rho*g(r) dr up to ``r_cut`` (trapezoidal rule).

    ``density`` is the number density (atoms / volume) of the *target*
    group used to compute ``g_r``.
    """
    mask = r <= r_cut
    integrand = 4.0 * np.pi * r[mask] ** 2 * density * g_r[mask]
    return float(np.trapezoid(integrand, r[mask]))
