"""Energy-log statistics: block averaging, statistical inefficiency,
autocorrelation and simple equilibration detection.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation function C(k) = <(x_t - mean)(x_{t+k} - mean)> / var.

    Returns an array of length ``max_lag + 1`` (default: ``len(series) - 1``),
    with ``C(0) == 1``.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_lag is None:
        max_lag = n - 1
    x = series - series.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    c = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        c[k] = np.dot(x[: n - k], x[k:]) / (n - k) / var
    return c


def statistical_inefficiency(series: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate the statistical inefficiency g = 1 + 2*sum_{k=1}^{K} (1 - k/N) C(k).

    Uses the standard truncation: summation stops at the first negative
    autocorrelation (or ``max_lag``), following common MD analysis
    practice (e.g. pymbar's approach) to avoid noise-dominated long-lag
    terms inflating the estimate. ``g`` is always >= 1; the number of
    effectively independent samples is ``N / g``.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 2:
        return 1.0
    c = autocorrelation(series, max_lag=max_lag if max_lag is not None else n - 1)
    g = 1.0
    for k in range(1, len(c)):
        if c[k] <= 0:
            break
        g += 2.0 * (1.0 - k / n) * c[k]
    return max(g, 1.0)


def block_average(series: np.ndarray, block_size: int) -> np.ndarray:
    """Split ``series`` into non-overlapping blocks and return their means."""
    series = np.asarray(series, dtype=float)
    n_blocks = len(series) // block_size
    if n_blocks == 0:
        raise ValueError("block_size larger than series length")
    trimmed = series[: n_blocks * block_size].reshape(n_blocks, block_size)
    return trimmed.mean(axis=1)


def block_average_error(series: np.ndarray, block_sizes: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Standard error of the mean estimated over a range of block sizes.

    Returns ``(block_sizes, std_errors)``; the standard error should
    plateau once ``block_size`` exceeds the correlation time, giving a
    reliable error estimate on the overall mean (block-averaging method).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if block_sizes is None:
        max_blocks = max(1, n // 4)
        block_sizes = sorted(set(np.geomspace(1, max_blocks, num=min(20, max_blocks)).astype(int)))
    errors = []
    used_sizes = []
    for bs in block_sizes:
        n_blocks = n // bs
        if n_blocks < 2:
            continue
        means = block_average(series, bs)
        errors.append(means.std(ddof=1) / np.sqrt(n_blocks))
        used_sizes.append(bs)
    return np.array(used_sizes), np.array(errors)


def equilibration_time(series: np.ndarray, n_candidates: int | None = None) -> int:
    """Estimate the equilibration cutoff index using the pymbar-style heuristic.

    Tries discarding the first ``t`` samples for a range of candidate
    ``t`` values and picks the one maximizing the number of effectively
    uncorrelated samples remaining, ``N_eff(t) = (N - t) / g(t)``. Returns
    the index ``t`` after which the series is considered equilibrated.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 10:
        return 0
    if n_candidates is None:
        n_candidates = min(n - 5, 100)
    candidates = np.unique(np.linspace(0, n - 5, n_candidates).astype(int))

    best_t = 0
    best_neff = -np.inf
    for t in candidates:
        tail = series[t:]
        if len(tail) < 5:
            continue
        g = statistical_inefficiency(tail)
        neff = len(tail) / g
        if neff > best_neff:
            best_neff = neff
            best_t = int(t)
    return best_t
