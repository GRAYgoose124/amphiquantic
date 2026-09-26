"""Mean-squared displacement, PBC unwrapping and Einstein diffusion coefficients."""

from __future__ import annotations

import numpy as np


def unwrap_pbc(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Unwrap coordinates that were periodically wrapped into an orthorhombic box.

    Parameters
    ----------
    positions:
        ``(n_frames, n_atoms, 3)`` wrapped coordinates.
    box:
        ``(3,)`` constant box lengths, or ``(n_frames, 3)`` per-frame box.

    Returns
    -------
    Unwrapped coordinates of the same shape, continuous in time (no jumps
    from periodic images).
    """
    positions = np.asarray(positions, dtype=float)
    n_frames = positions.shape[0]
    box = np.asarray(box, dtype=float)
    if box.ndim == 1:
        box = np.tile(box, (n_frames, 1))

    unwrapped = positions.copy()
    for f in range(1, n_frames):
        b = box[f]
        delta = unwrapped[f] - unwrapped[f - 1]
        shift = np.round(delta / b) * b
        unwrapped[f] -= shift
    return unwrapped


def mean_squared_displacement(
    positions: np.ndarray,
    indices: np.ndarray | None = None,
    box: np.ndarray | None = None,
) -> np.ndarray:
    """MSD(t) averaged over atoms, using every frame as a time origin.

    If ``box`` is given, positions are first unwrapped with
    :func:`unwrap_pbc`. Returns an array of shape ``(n_frames,)`` where
    entry ``k`` is the MSD at lag ``k`` frames, averaged over all valid
    origins and the selected atoms.
    """
    positions = np.asarray(positions, dtype=float)
    if indices is not None:
        positions = positions[:, indices, :]
    if box is not None:
        positions = unwrap_pbc(positions, box)

    n_frames = positions.shape[0]
    msd = np.zeros(n_frames, dtype=float)
    for lag in range(n_frames):
        if lag == 0:
            msd[0] = 0.0
            continue
        disp = positions[lag:] - positions[:-lag]
        sq = np.sum(disp * disp, axis=-1)  # (n_origins, n_atoms)
        msd[lag] = sq.mean()
    return msd


def diffusion_coefficient(
    msd: np.ndarray,
    dt: float,
    dimensions: int = 3,
    fit_start: float = 0.1,
    fit_end: float = 0.9,
) -> tuple[float, float, float]:
    """Fit the Einstein relation MSD(t) = 2*d*D*t + b over a linear window.

    Parameters
    ----------
    msd:
        MSD time series, e.g. from :func:`mean_squared_displacement`.
    dt:
        Time between frames.
    dimensions:
        Number of spatial dimensions the MSD was computed over (3 for a
        bulk 3D diffusion coefficient).
    fit_start, fit_end:
        Fractional window (of the full time series) used for the linear
        fit, to avoid the noisy short-time ballistic regime and the poorly
        sampled tail.

    Returns
    -------
    (D, slope, intercept):
        Diffusion coefficient and the raw linear-fit slope/intercept, in
        units of (length_unit)^2 / (time_unit) for ``slope``, and
        ``D = slope / (2 * dimensions)``.
    """
    n = len(msd)
    lo = max(1, int(fit_start * n))
    hi = max(lo + 2, int(fit_end * n))
    hi = min(hi, n)
    t = np.arange(lo, hi) * dt
    y = msd[lo:hi]
    slope, intercept = np.polyfit(t, y, 1)
    d = slope / (2.0 * dimensions)
    return float(d), float(slope), float(intercept)
