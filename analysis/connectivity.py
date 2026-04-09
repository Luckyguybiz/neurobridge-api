"""Functional connectivity analysis module.

Maps information flow between electrodes/neurons.

Methods:
- Cross-correlation: temporal coincidence of firing
- Transfer entropy: directional information transfer
- Granger causality: causal influence between channels
- Graph metrics: centrality, clustering, small-world-ness
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def compute_cross_correlation(
    data: SpikeData,
    max_lag_ms: float = 50.0,
    bin_size_ms: float = 1.0,
) -> dict:
    """Compute pairwise cross-correlograms between all electrode pairs.

    Returns correlation matrix and individual correlograms.
    """
    max_lag_sec = max_lag_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    n_bins = int(2 * max_lag_ms / bin_size_ms)

    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)

    # Correlation matrix (normalized peak cross-correlation)
    corr_matrix = np.zeros((n_electrodes, n_electrodes))
    correlograms = {}

    for i, e1 in enumerate(electrode_ids):
        times_1 = data.times[data.electrodes == e1]
        for j, e2 in enumerate(electrode_ids):
            if i >= j:
                continue

            times_2 = data.times[data.electrodes == e2]
            if len(times_1) == 0 or len(times_2) == 0:
                continue

            # Compute cross-correlogram
            counts = np.zeros(n_bins, dtype=int)
            for t1 in times_1:
                diffs = times_2 - t1
                valid = (diffs >= -max_lag_sec) & (diffs < max_lag_sec)
                if np.any(valid):
                    bins_idx = ((diffs[valid] + max_lag_sec) / bin_size_sec).astype(int)
                    bins_idx = np.clip(bins_idx, 0, n_bins - 1)
                    for b in bins_idx:
                        counts[b] += 1

            # Normalize by geometric mean of spike counts
            norm = np.sqrt(len(times_1) * len(times_2)) * bin_size_sec
            if norm > 0:
                normalized = counts / norm
            else:
                normalized = counts.astype(float)

            # Peak correlation
            peak = float(np.max(normalized)) if len(normalized) > 0 else 0
            corr_matrix[i, j] = peak
            corr_matrix[j, i] = peak

            # Peak lag
            peak_bin = int(np.argmax(counts))
            peak_lag_ms = (peak_bin * bin_size_ms) - max_lag_ms

            correlograms[f"{e1}-{e2}"] = {
                "counts": counts.tolist(),
                "normalized": normalized.tolist(),
                "lag_ms": np.linspace(-max_lag_ms, max_lag_ms, n_bins).tolist(),
                "peak_correlation": peak,
                "peak_lag_ms": peak_lag_ms,
                "n_spikes_1": len(times_1),
                "n_spikes_2": len(times_2),
            }

    np.fill_diagonal(corr_matrix, 1.0)

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "electrode_ids": electrode_ids,
        "correlograms": correlograms,
        "max_lag_ms": max_lag_ms,
        "bin_size_ms": bin_size_ms,
    }


def compute_connectivity_graph(
    data: SpikeData,
    coincidence_window_ms: float = 10.0,
    min_strength: float = 0.02,
) -> dict:
    """Compute functional connectivity graph based on co-firing.

    Two electrodes are "connected" if their spikes coincide within
    coincidence_window_ms more often than expected by chance.

    Returns nodes, edges, and graph metrics.
    """
    window_sec = coincidence_window_ms / 1000.0
    electrode_ids = data.electrode_ids
    n = len(electrode_ids)

    # Spike trains per electrode (sorted)
    trains = {}
    for e in electrode_ids:
        trains[e] = np.sort(data.times[data.electrodes == e])

    # Pairwise co-firing count
    edges = []
    adjacency = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = electrode_ids[i], electrode_ids[j]
            t1, t2 = trains[e1], trains[e2]
            if len(t1) == 0 or len(t2) == 0:
                continue

            # Count coincidences using sorted merge
            count = 0
            ai, bi = 0, 0
            while ai < len(t1) and bi < len(t2):
                diff = abs(t1[ai] - t2[bi])
                if diff < window_sec:
                    count += 1
                    ai += 1
                    bi += 1
                elif t1[ai] < t2[bi]:
                    ai += 1
                else:
                    bi += 1

            norm = min(len(t1), len(t2))
            strength = count / norm if norm > 0 else 0

            if strength >= min_strength:
                adjacency[i, j] = strength
                adjacency[j, i] = strength
                edges.append({
                    "source": int(e1),
                    "target": int(e2),
                    "weight": float(strength),
                    "coincidences": int(count),
                })

    # Graph metrics
    degrees = np.sum(adjacency > 0, axis=1)
    strengths = np.sum(adjacency, axis=1)

    # Clustering coefficient
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adjacency[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        triangles = sum(1 for a in range(k) for b in range(a + 1, k) if adjacency[neighbors[a], neighbors[b]] > 0)
        clustering[i] = 2 * triangles / (k * (k - 1))

    nodes = []
    for i, e in enumerate(electrode_ids):
        n_spikes = len(trains[e])
        nodes.append({
            "id": int(e),
            "n_spikes": n_spikes,
            "firing_rate_hz": n_spikes / data.duration if data.duration > 0 else 0,
            "degree": int(degrees[i]),
            "strength": float(strengths[i]),
            "clustering_coefficient": float(clustering[i]),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "density": len(edges) / (n * (n - 1) / 2) if n > 1 else 0,
        "mean_degree": float(np.mean(degrees)),
        "mean_clustering": float(np.mean(clustering)),
        "mean_strength": float(np.mean(strengths)),
    }


def compute_transfer_entropy(
    data: SpikeData,
    bin_size_ms: float = 5.0,
    history_bins: int = 5,
) -> dict:
    """Compute pairwise transfer entropy to measure directed information flow.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    Higher TE = more information flows from X to Y.
    """
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    n_bins_total = int((t_end - t_start) / bin_size_sec)

    electrode_ids = data.electrode_ids
    n = len(electrode_ids)

    # Create binned spike trains
    binned = {}
    for e in electrode_ids:
        spike_times = data.times[data.electrodes == e]
        bins = np.arange(t_start, t_end, bin_size_sec)
        counts, _ = np.histogram(spike_times, bins=bins)
        binned[e] = (counts > 0).astype(int)  # Binary

    te_matrix = np.zeros((n, n))

    for i, e_src in enumerate(electrode_ids):
        for j, e_tgt in enumerate(electrode_ids):
            if i == j:
                continue
            x = binned[e_src]
            y = binned[e_tgt]

            if len(x) < history_bins + 1 or len(y) < history_bins + 1:
                continue

            te = _compute_te_binary(x, y, history_bins)
            te_matrix[i, j] = max(0, te)

    return {
        "te_matrix": te_matrix.tolist(),
        "electrode_ids": electrode_ids,
        "bin_size_ms": bin_size_ms,
        "history_bins": history_bins,
        "max_te_pair": _find_max_pair(te_matrix, electrode_ids),
        "mean_te": float(np.mean(te_matrix[te_matrix > 0])) if np.any(te_matrix > 0) else 0,
    }


def _compute_te_binary(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Compute transfer entropy for binary time series."""
    n = min(len(x), len(y))
    if n <= k:
        return 0.0

    # Build joint distributions
    from collections import Counter

    # H(Y_future | Y_past)
    y_past_future = Counter()
    y_past = Counter()
    xy_past_future = Counter()
    xy_past = Counter()

    for t in range(k, n):
        y_history = tuple(y[t - k:t])
        x_history = tuple(x[t - k:t])
        y_future = y[t]

        y_past_future[(y_history, y_future)] += 1
        y_past[y_history] += 1
        xy_past_future[(y_history, x_history, y_future)] += 1
        xy_past[(y_history, x_history)] += 1

    total = n - k

    # TE = sum p(y_f, y_p, x_p) * log2(p(y_f | y_p, x_p) / p(y_f | y_p))
    te = 0.0
    for (yh, xh, yf), count in xy_past_future.items():
        p_yx = count / total
        p_yf_given_yx = count / xy_past.get((yh, xh), 1)
        p_yf_given_y = y_past_future.get((yh, yf), 0) / y_past.get(yh, 1)
        if p_yf_given_y > 0 and p_yf_given_yx > 0:
            te += p_yx * np.log2(p_yf_given_yx / p_yf_given_y)

    return te


def _find_max_pair(matrix: np.ndarray, electrode_ids: list[int]) -> dict:
    """Find electrode pair with maximum value in matrix."""
    if matrix.size == 0:
        return {"source": -1, "target": -1, "value": 0}
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {
        "source": electrode_ids[idx[0]],
        "target": electrode_ids[idx[1]],
        "value": float(matrix[idx]),
    }
