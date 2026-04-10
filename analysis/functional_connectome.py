"""Full functional connectome analysis with graph theory metrics.

Scientific basis:
- Bullmore & Sporns (2009), Nat Rev Neurosci — complex brain networks
- Rubinov & Sporns (2010), NeuroImage — graph metrics for brain networks
- Newman (2006), PNAS — modularity and community structure

Methods: cross-correlation/coherence adjacency, spectral clustering for
community detection, rich-club, small-world, efficiency, centrality.
"""

import numpy as np
from typing import Optional
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import SpectralClustering
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


def _correlation_matrix(binned: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix from binned trains."""
    if binned.shape[0] == 0:
        return np.empty((0, 0))
    mu = binned.mean(axis=1, keepdims=True)
    std = binned.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    normed = (binned - mu) / std
    corr = (normed @ normed.T) / binned.shape[1]
    np.fill_diagonal(corr, 0.0)
    return corr


def build_full_connectome(
    data: SpikeData, method: str = "cross_correlation",
    threshold: float = 0.05, bin_size_ms: float = 5.0,
) -> dict:
    """Build weighted adjacency matrix from spike correlations.

    Returns matrix, edge list, and basic graph stats.
    """
    binned, eids = _build_binned_trains(data, bin_size_ms)
    n = len(eids)
    if n < 2:
        return {"adjacency_matrix": [], "edge_list": [], "electrode_ids": eids,
                "n_nodes": n, "n_edges": 0, "density": 0.0,
                "mean_weight": 0.0, "max_weight": 0.0, "method": method,
                "threshold": threshold, "bin_size_ms": bin_size_ms}

    raw = _correlation_matrix(binned)
    if method == "coherence":
        spectra = np.fft.rfft(binned, axis=1)
        power = np.abs(spectra) ** 2
        raw = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                cross = np.abs(np.sum(spectra[i] * np.conj(spectra[j]))) ** 2
                denom = np.sum(power[i]) * np.sum(power[j])
                c = cross / denom if denom > 0 else 0.0
                raw[i, j] = raw[j, i] = c

    adj = np.where(np.abs(raw) >= threshold, raw, 0.0)
    np.fill_diagonal(adj, 0.0)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0.0:
                edges.append({"source": int(eids[i]), "target": int(eids[j]),
                              "weight": float(adj[i, j])})
    max_e = n * (n - 1) / 2
    ws = [e["weight"] for e in edges]
    return {"adjacency_matrix": adj.tolist(), "edge_list": edges,
            "electrode_ids": eids, "n_nodes": n, "n_edges": len(edges),
            "density": len(edges) / max_e if max_e > 0 else 0.0,
            "mean_weight": float(np.mean(ws)) if ws else 0.0,
            "max_weight": float(np.max(np.abs(ws))) if ws else 0.0,
            "method": method, "threshold": threshold, "bin_size_ms": bin_size_ms}


def detect_communities(
    data: SpikeData, method: str = "spectral",
    n_communities: Optional[int] = None,
    bin_size_ms: float = 5.0, threshold: float = 0.05,
) -> dict:
    """Community detection using spectral clustering on the connectivity matrix.

    Return community assignments, modularity score, n_communities.
    """
    binned, eids = _build_binned_trains(data, bin_size_ms)
    n = len(eids)
    if n < 3:
        return {"community_assignments": {int(e): 0 for e in eids},
                "n_communities": 1, "modularity": 0.0,
                "community_sizes": [n], "electrode_ids": eids}

    corr = _correlation_matrix(binned)
    aff = np.abs(corr)
    aff[aff < threshold] = 0.0
    np.fill_diagonal(aff, 0.0)

    if n_communities is None:
        deg = np.sum(aff, axis=1)
        D_inv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-10)))
        L = np.eye(n) - D_inv @ aff @ D_inv
        eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))
        gaps = np.diff(eigvals[:min(10, n)])
        n_communities = max(2, min(int(np.argmax(gaps) + 1), n // 2))

    sc = SpectralClustering(n_clusters=n_communities, affinity="precomputed",
                            assign_labels="kmeans", random_state=42)
    labels = sc.fit_predict(aff + 1e-10 * np.eye(n))

    W = aff.copy()
    m2 = np.sum(W)
    Q = 0.0
    if m2 > 0:
        ki = np.sum(W, axis=1)
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += W[i, j] - ki[i] * ki[j] / m2
        Q /= m2

    return {"community_assignments": {int(eids[i]): int(labels[i]) for i in range(n)},
            "n_communities": n_communities, "modularity": float(Q),
            "community_sizes": [int(np.sum(labels == c)) for c in range(n_communities)],
            "electrode_ids": eids}


def compute_graph_theory_metrics(
    data: SpikeData, bin_size_ms: float = 5.0, threshold: float = 0.05,
) -> dict:
    """Rich-club coefficient, small-world index (sigma), characteristic path
    length, global efficiency, betweenness centrality per node.
    """
    binned, eids = _build_binned_trains(data, bin_size_ms)
    n = len(eids)
    empty = {int(e): 0.0 for e in eids}
    if n < 3:
        return {"rich_club_coefficients": {}, "small_world_index": 0.0,
                "characteristic_path_length": 0.0, "global_efficiency": 0.0,
                "betweenness_centrality": empty, "clustering_coefficients": empty,
                "degree_distribution": {}, "electrode_ids": eids}

    corr = _correlation_matrix(binned)
    adj = (np.abs(corr) >= threshold).astype(float)
    np.fill_diagonal(adj, 0.0)
    degrees = np.sum(adj, axis=1).astype(int)

    # Clustering coefficients
    clust = np.zeros(n)
    for i in range(n):
        nb = np.where(adj[i] > 0)[0]
        k = len(nb)
        if k < 2:
            continue
        tri = sum(1 for a in range(k) for b in range(a + 1, k) if adj[nb[a], nb[b]] > 0)
        clust[i] = 2.0 * tri / (k * (k - 1))

    # Shortest paths
    dm = np.full((n, n), np.inf)
    dm[adj > 0] = 1.0
    np.fill_diagonal(dm, 0.0)
    sp = shortest_path(sparse.csr_matrix(dm), method="D", directed=False)
    mask = np.ones((n, n), bool)
    np.fill_diagonal(mask, False)
    fp = sp[mask & np.isfinite(sp)]
    cpl = float(np.mean(fp)) if len(fp) > 0 else float("inf")

    inv_sp = np.zeros_like(sp)
    ok = (sp > 0) & np.isfinite(sp)
    inv_sp[ok] = 1.0 / sp[ok]
    ge = float(np.sum(inv_sp) / (n * (n - 1))) if n > 1 else 0.0

    # Betweenness centrality (Brandes)
    bc = np.zeros(n)
    for s in range(n):
        visited = np.full(n, False)
        dist_s = np.full(n, -1)
        sigma = np.zeros(n)
        pred = [[] for _ in range(n)]
        q, dist_s[s], sigma[s], visited[s] = [s], 0, 1.0, True
        order = []
        while q:
            v = q.pop(0)
            order.append(v)
            for w in np.where(adj[v] > 0)[0]:
                if not visited[w]:
                    visited[w] = True
                    dist_s[w] = dist_s[v] + 1
                    q.append(w)
                if dist_s[w] == dist_s[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        for w in reversed(order):
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    nf = (n - 1) * (n - 2)
    if nf > 0:
        bc /= nf

    # Rich-club
    rc = {}
    for k in sorted(set(degrees)):
        if k < 1:
            continue
        rich = np.where(degrees > k)[0]
        nr = len(rich)
        if nr < 2:
            continue
        er = np.sum(adj[np.ix_(rich, rich)]) / 2.0
        rc[int(k)] = float(er / (nr * (nr - 1) / 2.0))

    # Small-world index
    te = np.sum(adj) / 2.0
    me = n * (n - 1) / 2.0
    p = te / me if me > 0 else 0.0
    md = float(np.mean(degrees))
    C = float(np.mean(clust))
    Cr = max(p, 1e-10)
    Lr = np.log(n) / np.log(md) if md > 1 else float("inf")
    gamma = C / Cr
    lam = cpl / Lr if np.isfinite(Lr) and Lr > 0 else 0.0
    sigma = gamma / lam if lam > 0 else 0.0

    dd = {}
    for d in degrees:
        dd[int(d)] = dd.get(int(d), 0) + 1

    return {"rich_club_coefficients": rc, "small_world_index": float(sigma),
            "characteristic_path_length": float(cpl), "global_efficiency": ge,
            "betweenness_centrality": {int(eids[i]): float(bc[i]) for i in range(n)},
            "clustering_coefficients": {int(eids[i]): float(clust[i]) for i in range(n)},
            "degree_distribution": dd, "electrode_ids": eids}
