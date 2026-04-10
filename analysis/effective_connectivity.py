"""Effective connectivity — directed causal influences between neurons.

EC goes beyond functional connectivity by estimating directed causal
influences. Implements simplified DCM using time-lagged cross-correlation
and partial correlation to distinguish direct vs indirect connections.

Scientific basis:
- Friston (2011), NeuroImage — DCM for electrophysiology
- Seth et al. (2015), J Neurosci — Granger causality in neuroscience
- Bressler & Seth (2011), NeuroImage — Wiener-Granger causality
"""

import numpy as np
from scipy import linalg
from .loader import SpikeData


def _build_binned_trains(data: SpikeData, bin_size_ms: float) -> tuple[np.ndarray, list[int]]:
    """Bin spike trains into count matrix (n_electrodes x n_bins)."""
    bin_sec = bin_size_ms / 1000.0
    t0, t1 = data.time_range
    if t1 <= t0:
        return np.empty((0, 0)), []
    bins = np.arange(t0, t1 + bin_sec, bin_sec)
    eids = data.electrode_ids
    binned = np.zeros((len(eids), len(bins) - 1))
    for i, e in enumerate(eids):
        ts = data.times[data.electrodes == e]
        if len(ts) > 0:
            binned[i], _ = np.histogram(ts, bins=bins)
    return binned, eids


def _time_lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-lagged correlation. Positive lag = x leads y."""
    n = len(x)
    if n < max_lag + 1:
        return np.array([0.0]), np.array([0])
    xn = x - np.mean(x)
    yn = y - np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.zeros(2 * max_lag + 1), np.arange(-max_lag, max_lag + 1)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))
    for idx, lag in enumerate(lags):
        if lag >= 0:
            corrs[idx] = np.sum(xn[:n - lag] * yn[lag:]) / ((n - abs(lag)) * sx * sy)
        else:
            corrs[idx] = np.sum(xn[-lag:] * yn[:n + lag]) / ((n - abs(lag)) * sx * sy)
    return corrs, lags


def _partial_correlation_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """Partial correlation matrix — controls for all other variables."""
    nv = data_matrix.shape[0]
    if nv < 2:
        return np.zeros((nv, nv))
    cov = np.cov(data_matrix) + 1e-8 * np.eye(nv)
    try:
        prec = linalg.inv(cov)
    except linalg.LinAlgError:
        prec = linalg.pinv(cov)
    diag = np.sqrt(np.abs(np.diag(prec)))
    diag[diag == 0] = 1.0
    D = np.diag(1.0 / diag)
    pcorr = -D @ prec @ D
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def estimate_effective_connectivity(
    data: SpikeData, bin_size_ms: float = 10.0,
    max_lag_bins: int = 10, significance_threshold: float = 0.05,
) -> dict:
    """Build directed connectivity matrix using time-lagged cross-correlation
    + partial correlation (controlling for confounds).

    Distinguish direct vs indirect connections.
    """
    binned, eids = _build_binned_trains(data, bin_size_ms)
    n = len(eids)
    if n < 2:
        return {"directed_matrix": [], "lag_matrix": [],
                "direct_connections": [], "indirect_connections": [],
                "electrode_ids": eids, "n_direct": 0, "n_indirect": 0,
                "asymmetry_index": 0.0}

    # Step 1: directed strength via time-lagged cross-correlation
    directed_raw = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            corrs, lags = _time_lagged_correlation(binned[i], binned[j], max_lag_bins)
            pos = lags > 0
            if np.any(pos):
                pc = np.abs(corrs[pos])
                directed_raw[i, j] = float(np.max(pc))
                lag_matrix[i, j] = float(lags[pos][np.argmax(pc)]) * bin_size_ms

    # Step 2: partial correlation for direct vs indirect
    pcorr = _partial_correlation_matrix(binned)

    # Step 3: combine
    directed = np.zeros((n, n))
    direct_conn, indirect_conn = [], []
    sig = significance_threshold
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            raw = directed_raw[i, j]
            pc = abs(pcorr[i, j])
            if raw > sig:
                info = {"source": int(eids[i]), "target": int(eids[j]),
                        "raw_correlation": float(raw),
                        "partial_correlation": float(pc),
                        "lag_ms": float(lag_matrix[i, j])}
                if pc > sig:
                    directed[i, j] = raw * pc
                    info["strength"] = float(directed[i, j])
                    info["type"] = "direct"
                    direct_conn.append(info)
                else:
                    info["type"] = "indirect"
                    indirect_conn.append(info)

    # Asymmetry index
    if np.sum(np.abs(directed)) > 0:
        sym = (directed + directed.T) / 2.0
        asym = (directed - directed.T) / 2.0
        asymmetry = float(np.sum(np.abs(asym)) / (np.sum(np.abs(sym) + np.abs(asym))))
    else:
        asymmetry = 0.0

    return {"directed_matrix": directed.tolist(), "lag_matrix": lag_matrix.tolist(),
            "direct_connections": direct_conn, "indirect_connections": indirect_conn,
            "electrode_ids": eids, "n_direct": len(direct_conn),
            "n_indirect": len(indirect_conn), "asymmetry_index": asymmetry}


def compute_causal_hierarchy(
    data: SpikeData, bin_size_ms: float = 10.0,
    max_lag_bins: int = 10, significance_threshold: float = 0.05,
) -> dict:
    """Order electrodes by causal influence (ratio of outgoing to incoming).

    Return hierarchy levels and flow direction.
    """
    ec = estimate_effective_connectivity(data, bin_size_ms, max_lag_bins,
                                         significance_threshold)
    eids = ec["electrode_ids"]
    n = len(eids)
    if n < 2:
        return {"hierarchy_levels": [], "flow_direction": "balanced",
                "hierarchy_score": 0.0, "electrode_ids": eids,
                "driver_nodes": [], "receiver_nodes": []}

    dm = np.array(ec["directed_matrix"])
    outgoing = np.sum(dm, axis=1)
    incoming = np.sum(dm, axis=0)
    total = outgoing + incoming
    safe_total = np.where(total > 0, total, 1.0)
    ratio = np.where(total > 0, outgoing / safe_total, 0.5)

    order = np.argsort(-ratio)
    levels = []
    for rank, idx in enumerate(order):
        levels.append({"electrode_id": int(eids[idx]), "level": rank,
                        "outgoing_strength": float(outgoing[idx]),
                        "incoming_strength": float(incoming[idx]),
                        "influence_ratio": float(ratio[idx])})

    nq = max(1, n // 4)
    drivers = [int(eids[order[i]]) for i in range(nq)]
    receivers = [int(eids[order[-(i + 1)]]) for i in range(nq)]

    mr = float(np.mean(ratio))
    flow = "feedforward" if mr > 0.6 else ("feedback" if mr < 0.4 else "balanced")
    hs = min(1.0, max(0.0, float(np.std(ratio) * 2.0)))

    return {"hierarchy_levels": levels, "flow_direction": flow,
            "hierarchy_score": hs, "electrode_ids": eids,
            "driver_nodes": drivers, "receiver_nodes": receivers}
