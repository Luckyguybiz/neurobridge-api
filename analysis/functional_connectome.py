"""Full functional connectome analysis with graph theory metrics.

Builds weighted connectivity graphs from spike correlations and applies
graph-theoretic analysis: community detection (spectral clustering),
rich-club coefficients, small-world index, characteristic path length,
global efficiency, and betweenness centrality.

Scientific basis:
- Bullmore & Sporns (2009), Nat Rev Neurosci — complex brain networks
- Rubinov & Sporns (2010), NeuroImage — graph metrics for brain networks
- Newman (2006), PNAS — modularity and community structure

Methods:
- Cross-correlation / coherence based adjacency matrices
- Spectral clustering for community detection
- Rich-club, small-world, efficiency, centrality metrics
"""

import numpy as np
from typing import Optional
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import SpectralClustering
from .loader import SpikeData


def _build_binned_trains(data: SpikeData, bin_size_ms: float) -> tuple[np.ndarray, list[int]]:
    """Bin spike trains into binary matrix (n_electrodes x n_bins)."""
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


def _cross_correlation_matrix(binned: np.ndarray) -> np.ndarray:
    """Compute normalized cross-correlation matrix from binned trains."""
    n = binned.shape[0]
    if n == 0:
        return np.empty((0, 0))

    # Zero-mean each row
    means = binned.mean(axis=1, keepdims=True)
    stds = binned.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    normed = (binned - means) / stds

    # Pearson correlation
    corr = (normed @ normed.T) / binned.shape[1]
    np.fill_diagonal(corr, 0.0)
    return corr


def _coherence_matrix(binned: np.ndarray) -> np.ndarray:
    """Compute spectral coherence matrix using FFT."""
    n_electrodes, n_bins = binned.shape
    if n_electrodes == 0 or n_bins < 4:
        return np.empty((0, 0))

    # FFT of each binned train
    spectra = np.fft.rfft(binned, axis=1)
    power = np.abs(spectra) ** 2

    coh_matrix = np.zeros((n_electrodes, n_electrodes))
    for i in range(n_electrodes):
        for j in range(i + 1, n_electrodes):
            cross = np.abs(np.sum(spectra[i] * np.conj(spectra[j]))) ** 2
            auto_i = np.sum(power[i])
            auto_j = np.sum(power[j])
            denom = auto_i * auto_j
            coh = cross / denom if denom > 0 else 0.0
            coh_matrix[i, j] = coh
            coh_matrix[j, i] = coh

    return coh_matrix


def build_full_connectome(
    data: SpikeData,
    method: str = "cross_correlation",
    threshold: float = 0.05,
    bin_size_ms: float = 5.0,
) -> dict:
    """Build weighted adjacency matrix from spike correlations.

    Constructs a functional connectome by computing pairwise
    correlation/coherence between all electrode pairs and thresholding.

    Args:
        data: SpikeData with spike times and electrode IDs.
        method: 'cross_correlation' or 'coherence'.
        threshold: Minimum absolute weight to retain an edge.
        bin_size_ms: Bin width for discretizing spike trains.

    Returns:
        Dict with keys:
        - adjacency_matrix: list[list[float]], weighted adjacency (n x n).
        - edge_list: list[dict], each with source, target, weight.
        - electrode_ids: list[int].
        - n_nodes, n_edges, density, mean_weight, max_weight.
        - method, threshold, bin_size_ms.
    """
    binned, electrode_ids = _build_binned_trains(data, bin_size_ms)
    n = len(electrode_ids)

    if n < 2:
        return {
            "adjacency_matrix": [],
            "edge_list": [],
            "electrode_ids": electrode_ids,
            "n_nodes": n, "n_edges": 0, "density": 0.0,
            "mean_weight": 0.0, "max_weight": 0.0,
            "method": method, "threshold": threshold, "bin_size_ms": bin_size_ms,
        }

    if method == "coherence":
        raw_matrix = _coherence_matrix(binned)
    else:
        raw_matrix = _cross_correlation_matrix(binned)

    # Apply threshold (use absolute value for negative correlations)
    adj = np.where(np.abs(raw_matrix) >= threshold, raw_matrix, 0.0)
    np.fill_diagonal(adj, 0.0)

    # Build edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w != 0.0:
                edge_list.append({
                    "source": int(electrode_ids[i]),
                    "target": int(electrode_ids[j]),
                    "weight": float(w),
                })

    max_edges = n * (n - 1) / 2
    weights = [e["weight"] for e in edge_list]

    return {
        "adjacency_matrix": adj.tolist(),
        "edge_list": edge_list,
        "electrode_ids": electrode_ids,
        "n_nodes": n,
        "n_edges": len(edge_list),
        "density": len(edge_list) / max_edges if max_edges > 0 else 0.0,
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
        "max_weight": float(np.max(np.abs(weights))) if weights else 0.0,
        "method": method,
        "threshold": threshold,
        "bin_size_ms": bin_size_ms,
    }


def detect_communities(
    data: SpikeData,
    method: str = "spectral",
    n_communities: Optional[int] = None,
    bin_size_ms: float = 5.0,
    threshold: float = 0.05,
) -> dict:
    """Community detection using spectral clustering on the connectivity matrix.

    Reveals modular organization in the neural network — groups of neurons
    that fire together more densely than expected by chance.

    Args:
        data: SpikeData with spike times and electrode IDs.
        method: 'spectral' (spectral clustering on affinity matrix).
        n_communities: Number of communities. If None, estimated from
            eigenvalue gap of the graph Laplacian (max 10).
        bin_size_ms: Bin width for spike train discretization.
        threshold: Minimum correlation to form an edge.

    Returns:
        Dict with keys:
        - community_assignments: dict[int, int], electrode_id -> community.
        - n_communities: int.
        - modularity: float, Newman modularity Q.
        - community_sizes: list[int].
        - electrode_ids: list[int].
    """
    binned, electrode_ids = _build_binned_trains(data, bin_size_ms)
    n = len(electrode_ids)

    if n < 3:
        assignments = {int(e): 0 for e in electrode_ids}
        return {
            "community_assignments": assignments,
            "n_communities": 1,
            "modularity": 0.0,
            "community_sizes": [n],
            "electrode_ids": electrode_ids,
        }

    # Build affinity matrix (absolute correlation, thresholded)
    corr = _cross_correlation_matrix(binned)
    affinity = np.abs(corr)
    affinity[affinity < threshold] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Estimate n_communities from eigenvalue gap of graph Laplacian
    if n_communities is None:
        degrees = np.sum(affinity, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
        L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
        eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
        # Largest gap in first few eigenvalues
        max_k = min(10, n - 1)
        gaps = np.diff(eigvals[:max_k + 1])
        n_communities = int(np.argmax(gaps) + 1)
        n_communities = max(2, min(n_communities, n // 2))

    # Spectral clustering
    sc = SpectralClustering(
        n_clusters=n_communities,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(affinity + 1e-10 * np.eye(n))

    # Compute Newman modularity Q
    W = affinity.copy()
    total_weight = np.sum(W) / 2.0
    if total_weight > 0:
        ki = np.sum(W, axis=1)
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += W[i, j] - (ki[i] * ki[j]) / (2.0 * total_weight)
        Q /= (2.0 * total_weight)
    else:
        Q = 0.0

    assignments = {int(electrode_ids[i]): int(labels[i]) for i in range(n)}
    community_sizes = [int(np.sum(labels == c)) for c in range(n_communities)]

    return {
        "community_assignments": assignments,
        "n_communities": n_communities,
        "modularity": float(Q),
        "community_sizes": community_sizes,
        "electrode_ids": electrode_ids,
    }


def compute_graph_theory_metrics(
    data: SpikeData,
    bin_size_ms: float = 5.0,
    threshold: float = 0.05,
) -> dict:
    """Compute comprehensive graph theory metrics on the functional connectome.

    Metrics:
    - Rich-club coefficient: tendency of high-degree nodes to connect
    - Small-world index (sigma): clustering / path_length vs random graph
    - Characteristic path length: average shortest path
    - Global efficiency: inverse of average shortest path
    - Betweenness centrality: fraction of shortest paths through each node

    Args:
        data: SpikeData with spike times and electrode IDs.
        bin_size_ms: Bin width for spike train discretization.
        threshold: Minimum correlation to form an edge.

    Returns:
        Dict with keys:
        - rich_club_coefficients: dict[int, float], degree k -> phi(k).
        - small_world_index: float (sigma).
        - characteristic_path_length: float.
        - global_efficiency: float.
        - betweenness_centrality: dict[int, float], electrode_id -> centrality.
        - clustering_coefficients: dict[int, float], electrode_id -> C.
        - degree_distribution: dict[int, int], degree -> count.
        - electrode_ids: list[int].
    """
    binned, electrode_ids = _build_binned_trains(data, bin_size_ms)
    n = len(electrode_ids)

    if n < 3:
        empty = {int(e): 0.0 for e in electrode_ids}
        return {
            "rich_club_coefficients": {},
            "small_world_index": 0.0,
            "characteristic_path_length": 0.0,
            "global_efficiency": 0.0,
            "betweenness_centrality": empty,
            "clustering_coefficients": empty,
            "degree_distribution": {},
            "electrode_ids": electrode_ids,
        }

    # Build binary adjacency
    corr = _cross_correlation_matrix(binned)
    adj = (np.abs(corr) >= threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)

    # Weighted adjacency for efficiency calculations
    W = np.abs(corr)
    W[W < threshold] = 0.0
    np.fill_diagonal(W, 0.0)

    degrees = np.sum(adj, axis=1).astype(int)

    # --- Clustering coefficients ---
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        triangles = 0
        for a_idx in range(k):
            for b_idx in range(a_idx + 1, k):
                if adj[neighbors[a_idx], neighbors[b_idx]] > 0:
                    triangles += 1
        clustering[i] = 2.0 * triangles / (k * (k - 1))

    # --- Shortest paths (unweighted) ---
    # Convert to distance: connected = 1, disconnected = inf
    dist_matrix = np.full((n, n), np.inf)
    dist_matrix[adj > 0] = 1.0
    np.fill_diagonal(dist_matrix, 0.0)

    sp = shortest_path(sparse.csr_matrix(dist_matrix), method="D", directed=False)

    # Characteristic path length (exclude infinities and self)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    finite_paths = sp[mask & np.isfinite(sp)]
    char_path_length = float(np.mean(finite_paths)) if len(finite_paths) > 0 else float("inf")

    # --- Global efficiency ---
    inv_sp = np.zeros_like(sp)
    finite_mask = (sp > 0) & np.isfinite(sp)
    inv_sp[finite_mask] = 1.0 / sp[finite_mask]
    np.fill_diagonal(inv_sp, 0.0)
    global_eff = float(np.sum(inv_sp) / (n * (n - 1))) if n > 1 else 0.0

    # --- Betweenness centrality (Brandes algorithm simplified) ---
    betweenness = np.zeros(n)
    for s in range(n):
        # BFS from source s
        visited = np.full(n, False)
        dist_from_s = np.full(n, -1)
        sigma = np.zeros(n)  # number of shortest paths
        pred = [[] for _ in range(n)]
        queue = [s]
        dist_from_s[s] = 0
        sigma[s] = 1.0
        visited[s] = True
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)
            neighbors = np.where(adj[v] > 0)[0]
            for w in neighbors:
                if not visited[w]:
                    visited[w] = True
                    dist_from_s[w] = dist_from_s[v] + 1
                    queue.append(w)
                if dist_from_s[w] == dist_from_s[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Back-propagation
        delta = np.zeros(n)
        for w in reversed(order):
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalize
    norm_factor = (n - 1) * (n - 2)
    if norm_factor > 0:
        betweenness /= norm_factor

    # --- Rich-club coefficients ---
    rich_club = {}
    unique_degrees = sorted(set(degrees))
    for k in unique_degrees:
        if k < 1:
            continue
        # Nodes with degree > k
        rich_nodes = np.where(degrees > k)[0]
        n_rich = len(rich_nodes)
        if n_rich < 2:
            continue
        # Edges among rich nodes
        sub_adj = adj[np.ix_(rich_nodes, rich_nodes)]
        e_rich = np.sum(sub_adj) / 2.0
        max_e = n_rich * (n_rich - 1) / 2.0
        rich_club[int(k)] = float(e_rich / max_e) if max_e > 0 else 0.0

    # --- Small-world index (sigma = C/C_rand / L/L_rand) ---
    # Random graph expectations: C_rand ~ p, L_rand ~ ln(n)/ln(k_mean)
    total_edges = np.sum(adj) / 2.0
    max_edges = n * (n - 1) / 2.0
    p = total_edges / max_edges if max_edges > 0 else 0.0
    mean_degree = float(np.mean(degrees))
    C_actual = float(np.mean(clustering))
    C_rand = p if p > 0 else 1e-10

    if mean_degree > 1 and n > 1:
        L_rand = np.log(n) / np.log(mean_degree) if mean_degree > 1 else float("inf")
    else:
        L_rand = float("inf")

    gamma = C_actual / C_rand if C_rand > 0 else 0.0
    lam = char_path_length / L_rand if L_rand > 0 and np.isfinite(L_rand) else 0.0
    sigma = gamma / lam if lam > 0 else 0.0

    # --- Degree distribution ---
    degree_dist = {}
    for d in degrees:
        d_int = int(d)
        degree_dist[d_int] = degree_dist.get(d_int, 0) + 1

    return {
        "rich_club_coefficients": rich_club,
        "small_world_index": float(sigma),
        "characteristic_path_length": float(char_path_length),
        "global_efficiency": float(global_eff),
        "betweenness_centrality": {
            int(electrode_ids[i]): float(betweenness[i]) for i in range(n)
        },
        "clustering_coefficients": {
            int(electrode_ids[i]): float(clustering[i]) for i in range(n)
        },
        "degree_distribution": degree_dist,
        "electrode_ids": electrode_ids,
    }
