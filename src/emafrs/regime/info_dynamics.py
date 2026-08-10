"""Hurst exponent (R/S + DFA cross-check) and Shannon entropy of returns.

PDF Eq. (2) gives a single-window R/S formula H = log(R(n)/S(n)) / log(n), which as a
single-n estimate is extremely noisy in practice. This module implements the standard
generalization -- R/S computed across multiple sub-window sizes with a log-log linear
regression for the slope -- and cross-checks it against Detrended Fluctuation Analysis
(DFA), per docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md's documented correction #3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hurst_rs(x: np.ndarray, min_chunk: int = 8, n_chunk_sizes: int = 10) -> float:
    """Rescaled-range Hurst estimate via log-log regression of R/S across chunk sizes.
    Returns NaN if the window is too short to support at least 2 chunk sizes."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_chunk = n // 2
    if max_chunk < min_chunk:
        return np.nan

    chunk_sizes = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=n_chunk_sizes).astype(int))
    log_sizes, log_rs = [], []
    for size in chunk_sizes:
        if size < 2:
            continue
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = x[i * size : (i + 1) * size]
            deviations = np.cumsum(chunk - chunk.mean())
            r = deviations.max() - deviations.min()
            s = chunk.std(ddof=0)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            log_sizes.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_list)))

    if len(log_sizes) < 2:
        return np.nan
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def dfa_alpha(x: np.ndarray, min_box: int = 4, n_box_sizes: int = 10) -> float:
    """Detrended Fluctuation Analysis scaling exponent alpha, used as an independent
    cross-check of hurst_rs -- alpha ~ H for stationary long-memory processes."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.cumsum(x - x.mean())
    max_box = n // 4
    if max_box < min_box:
        return np.nan

    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), num=n_box_sizes).astype(int))
    log_sizes, log_fluct = [], []
    for size in box_sizes:
        if size < 4:
            continue
        n_boxes = n // size
        if n_boxes < 1:
            continue
        rms = []
        t = np.arange(size)
        for i in range(n_boxes):
            segment = y[i * size : (i + 1) * size]
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms:
            log_sizes.append(np.log(size))
            log_fluct.append(np.log(np.mean(rms)))

    if len(log_sizes) < 2:
        return np.nan
    slope, _ = np.polyfit(log_sizes, log_fluct, 1)
    return float(slope)


def shannon_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """PDF Eq. (3): Shannon entropy (bits) of the empirical return distribution over a window."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    hist, _ = np.histogram(x, bins=n_bins)
    total = hist.sum()
    if total == 0:
        return np.nan
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def rolling_hurst(returns: pd.Series, window: int = 126, min_chunk: int = 8) -> pd.Series:
    return returns.rolling(window, min_periods=window).apply(
        lambda w: hurst_rs(w, min_chunk=min_chunk), raw=True
    )


def rolling_dfa(returns: pd.Series, window: int = 126, min_box: int = 4) -> pd.Series:
    return returns.rolling(window, min_periods=window).apply(
        lambda w: dfa_alpha(w, min_box=min_box), raw=True
    )


def rolling_entropy(returns: pd.Series, window: int = 21, n_bins: int = 10) -> pd.Series:
    return returns.rolling(window, min_periods=window).apply(
        lambda w: shannon_entropy(w, n_bins=n_bins), raw=True
    )
