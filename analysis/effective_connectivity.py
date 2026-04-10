"""Effective connectivity analysis — directed causal influences between neurons.

Effective connectivity (EC) goes beyond functional connectivity by estimating
directed causal influences between neural populations. This module implements
a simplified Dynamic Causal Modeling (DCM) approach using time-lagged
cross-correlation and partial correlation to distinguish direct from
indirect connections.

Scientific basis:
- Friston (2011), NeuroImage — DCM for electrophysiology
- Seth et al. (2015), J Neurosci — Granger causality in neuroscience
- Bressler & Seth (2011), NeuroImage — Wiener-Granger causality

Methods:
- Time-lagged cross-correlation for directed connectivity
- Partial correlation to control for confounding paths
- Causal hierarchy ordering via outgoing/incoming influence ratio
"""

import numpy as np
from scipy import linalg
from .loader import SpikeData


def _build_binned_trains(data: SpikeData, bin_size_ms: float) -> tuple[np.ndarray, list[int]]:
    """Bin spike trains into count matrix (n_electrodes x n_bins)."""
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    if t_end <= t_start:
        return np.empty((0, 0)), []

    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)
    n_bins = len(bins) - 1

    binned = np.zeros((n_electrodes, n_bins), dtype=np.float64)
    for i, e in enumerate(electrode_ids):
        spike_times = data.times[data.electrodes == e]
        if len(spike_times) > 0:
            counts, _ = np.histogram(spike_times, bins=bins)
            binned[i] = counts.astype(np.float64)

    return binned, electrode_ids


def _time_lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-lagged correlation between x and y.

    Returns correlation values and corresponding lags.
    Positive lag means x leads y.
    """
    n = len(x)
    if n < max_lag + 1:
        return np.array([0.0]), np.array([0])

    x_normed = (x - np.mean(x))
    y_normed = (y - np.mean(y))
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return np.zeros(2 * max_lag + 1), np.arange(-max_lag, max_lag + 1)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))

    for idx, lag in enumerate(lags):
        if lag >= 0:
            c = np.sum(x_normed[:n - lag] * y_normed[lag:]) / ((n - abs(lag)) * sx * sy)
        else:
            c = np.sum(x_normed[-lag:] * y_normed[:n + lag]) / ((n - abs(lag)) * sx * sy)
        correlations[idx] = c

    return correlations, lags


def _partial_correlation_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix from data (variables x observations).

    Partial correlation between i and j controls for all other variables,
    revealing direct connections only.
    """
    n_vars = data_matrix.shape[0]
    if n_vars < 2:
        return np.zeros((n_vars, n_vars))

    # Covariance matrix
    cov = np.cov(data_matrix)
    if cov.ndim == 0:
        return np.zeros((n_vars, n_vars))

    # Regularize for numerical stability
    cov += 1e-8 * np.eye(n_vars)

    try:
        precision = linalg.inv(cov)
    except linalg.LinAlgError:
        # Fallback: pseudoinverse
        precision = linalg.pinv(cov)

    # Partial correlation from precision matrix
    diag = np.sqrt(np.abs(np.diag(precision)))
    diag[diag == 0] = 1.0
    D = np.diag(1.0 / diag)
    pcorr = -D @ precision @ D
    np.fill_diagonal(pcorr, 1.0)

    return pcorr


def _compute_directed_strength(
    binned: np.ndarray, i: int, j: int, max_lag: int,
) -> tuple[float, float, float]:
    """Compute directed connection strength from i to j.

    Returns:
        strength_i_to_j: Positive-lag peak (i leads j).
        strength_j_to_i: Negative-lag peak (j leads i).
        optimal_lag_ms: Lag at peak in bins.
    """
    corr, lags = _time_lagged_correlation(binned[i], binned[j], max_lag)

    # i -> j: positive lags (i leads)
    pos_mask = lags > 0
    neg_mask = lags < 0

    if np.any(pos_mask):
        pos_corr = np.abs(corr[pos_mask])
        i_to_j = float(np.max(pos_corr))
        peak_lag = int(lags[pos_mask][np.argmax(pos_corr)])
    else:
        i_to_j = 0.0
        peak_lag = 0

    if np.any(neg_mask):
        neg_corr = np.abs(corr[neg_mask])
        j_to_i = float(np.max(neg_corr))
    else:
        j_to_i = 0.0

    return i_to_j, j_to_i, float(peak_lag)


def estimate_effective_connectivity(
    data: SpikeData,
    bin_size_ms: float = 10.0,
    max_lag_bins: int = 10,
    significance_threshold: float = 0.05,
) -> dict:
    """Build a directed connectivity matrix using time-lagged cross-correlation
    combined with partial correlation to control for confounds.

    Distinguishes direct vs indirect connections: time-lagged correlation
    reveals directionality, partial correlation removes spurious edges
    caused by shared inputs or indirect paths.

    Args:
        data: SpikeData with spike times and electrode IDs.
        bin_size_ms: Bin width for spike train discretization.
        max_lag_bins: Maximum lag in bins to search for directed influence.
        significance_threshold: Minimum partial correlation to retain edge.

    Returns:
        Dict with keys:
        - directed_matrix: list[list[float]], directed adjacency (i->j at [i][j]).
        - lag_matrix: list[list[float]], optimal lag in ms for each connection.
        - direct_connections: list[dict], direct edges after partial corr filtering.
        - indirect_connections: list[dict], edges removed by partial corr.
        - electrode_ids: list[int].
        - n_direct, n_indirect, asymmetry_index.
    """
    binned, electrode_ids = _build_binned_trains(data, bin_size_ms)
    n = len(electrode_ids)

    if n < 2:
        return {
            "directed_matrix": [],
            "lag_matrix": [],
            "direct_connections": [],
            "indirect_connections": [],
            "electrode_ids": electrode_ids,
            "n_direct": 0,
            "n_indirect": 0,
            "asymmetry_index": 0.0,
        }

    # Step 1: Time-lagged cross-correlation for directed connectivity
    directed_raw = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            strength_ij, _, opt_lag = _compute_directed_strength(
                binned, i, j, max_lag_bins
            )
            directed_raw[i, j] = strength_ij
            lag_matrix[i, j] = opt_lag * bin_size_ms

    # Step 2: Partial correlation to identify direct connections
    # Build time-lagged data for partial correlation
    # For each pair, create lagged version and compute partial corr
    pcorr = _partial_correlation_matrix(binned)

    # Step 3: Combine — edge is "direct" if both directed and partial corr are significant
    directed_matrix = np.zeros((n, n))
    direct_connections = []
    indirect_connections = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            raw_strength = directed_raw[i, j]
            partial_strength = abs(pcorr[i, j])

            if raw_strength > significance_threshold:
                if partial_strength > significance_threshold:
                    # Direct connection
                    directed_matrix[i, j] = raw_strength * partial_strength
                    direct_connections.append({
                        "source": int(electrode_ids[i]),
                        "target": int(electrode_ids[j]),
                        "strength": float(directed_matrix[i, j]),
                        "raw_correlation": float(raw_strength),
                        "partial_correlation": float(partial_strength),
                        "lag_ms": float(lag_matrix[i, j]),
                        "type": "direct",
                    })
                else:
                    # Indirect connection (confound-mediated)
                    indirect_connections.append({
                        "source": int(electrode_ids[i]),
                        "target": int(electrode_ids[j]),
                        "raw_correlation": float(raw_strength),
                        "partial_correlation": float(partial_strength),
                        "lag_ms": float(lag_matrix[i, j]),
                        "type": "indirect",
                    })

    # Asymmetry index: how asymmetric is the connectivity?
    # 0 = fully symmetric, 1 = fully asymmetric
    if np.sum(np.abs(directed_matrix)) > 0:
        sym = (directed_matrix + directed_matrix.T) / 2.0
        asym = (directed_matrix - directed_matrix.T) / 2.0
        asymmetry = float(
            np.sum(np.abs(asym)) / np.sum(np.abs(sym) + np.abs(asym))
        )
    else:
        asymmetry = 0.0

    return {
        "directed_matrix": directed_matrix.tolist(),
        "lag_matrix": lag_matrix.tolist(),
        "direct_connections": direct_connections,
        "indirect_connections": indirect_connections,
        "electrode_ids": electrode_ids,
        "n_direct": len(direct_connections),
        "n_indirect": len(indirect_connections),
        "asymmetry_index": asymmetry,
    }


def compute_causal_hierarchy(
    data: SpikeData,
    bin_size_ms: float = 10.0,
    max_lag_bins: int = 10,
    significance_threshold: float = 0.05,
) -> dict:
    """Order electrodes by causal influence in the network.

    Computes a hierarchy based on the ratio of outgoing to incoming
    effective connections. Nodes at the top of the hierarchy predominantly
    drive activity; nodes at the bottom predominantly receive.

    The flow direction indicates overall information routing: feedforward
    (top-down) vs feedback (bottom-up) balance.

    Args:
        data: SpikeData with spike times and electrode IDs.
        bin_size_ms: Bin width for spike train discretization.
        max_lag_bins: Maximum lag in bins.
        significance_threshold: Minimum strength to count as connection.

    Returns:
        Dict with keys:
        - hierarchy_levels: list[dict], sorted by causal influence (descending).
            Each dict has electrode_id, level, outgoing, incoming, influence_ratio.
        - flow_direction: 'feedforward' | 'feedback' | 'balanced'.
        - hierarchy_score: float, 0-1 how hierarchical the network is.
        - electrode_ids: list[int].
        - driver_nodes: list[int], top-hierarchy electrodes.
        - receiver_nodes: list[int], bottom-hierarchy electrodes.
    """
    ec_result = estimate_effective_connectivity(
        data, bin_size_ms, max_lag_bins, significance_threshold
    )

    electrode_ids = ec_result["electrode_ids"]
    n = len(electrode_ids)

    if n < 2:
        return {
            "hierarchy_levels": [],
            "flow_direction": "balanced",
            "hierarchy_score": 0.0,
            "electrode_ids": electrode_ids,
            "driver_nodes": [],
            "receiver_nodes": [],
        }

    directed = np.array(ec_result["directed_matrix"])

    # Outgoing and incoming strength per node
    outgoing = np.sum(directed, axis=1)  # row sums: i -> all
    incoming = np.sum(directed, axis=0)  # col sums: all -> j

    # Influence ratio: out / (out + in), range [0, 1]
    # > 0.5 = driver, < 0.5 = receiver
    total = outgoing + incoming
    influence_ratio = np.where(total > 0, outgoing / total, 0.5)

    # Sort by influence ratio descending
    order = np.argsort(-influence_ratio)

    hierarchy_levels = []
    for rank, idx in enumerate(order):
        hierarchy_levels.append({
            "electrode_id": int(electrode_ids[idx]),
            "level": rank,
            "outgoing_strength": float(outgoing[idx]),
            "incoming_strength": float(incoming[idx]),
            "influence_ratio": float(influence_ratio[idx]),
        })

    # Classify drivers (top 25%) and receivers (bottom 25%)
    n_quarter = max(1, n // 4)
    driver_nodes = [int(electrode_ids[order[i]]) for i in range(n_quarter)]
    receiver_nodes = [int(electrode_ids[order[-(i + 1)]]) for i in range(n_quarter)]

    # Flow direction based on mean influence ratio of top vs bottom
    mean_ratio = float(np.mean(influence_ratio))
    if mean_ratio > 0.6:
        flow_direction = "feedforward"
    elif mean_ratio < 0.4:
        flow_direction = "feedback"
    else:
        flow_direction = "balanced"

    # Hierarchy score: variance of influence ratios
    # High variance = clear hierarchy, low variance = egalitarian
    hierarchy_score = float(np.std(influence_ratio) * 2.0)
    hierarchy_score = min(1.0, max(0.0, hierarchy_score))

    return {
        "hierarchy_levels": hierarchy_levels,
        "flow_direction": flow_direction,
        "hierarchy_score": hierarchy_score,
        "electrode_ids": electrode_ids,
        "driver_nodes": driver_nodes,
        "receiver_nodes": receiver_nodes,
    }
