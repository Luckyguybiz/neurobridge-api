"""Topological data analysis (TDA) of neural spike train activity.

Betti numbers count "holes" in the activity manifold — a signature of
complex computation (Giusti et al., 2016; Curto, 2017).
Simplified implementation without external TDA libraries.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy import sparse
from .loader import SpikeData


def _build_pop_vectors(data: SpikeData, n_windows: int) -> tuple[np.ndarray, list[int]]:
    """Bin spikes into time windows, return (n_windows x n_electrodes) matrix."""
    eids = data.electrode_ids
    ne = len(eids)
    t0, t1 = data.time_range
    if t1 <= t0 or ne == 0 or n_windows < 2:
        return np.empty((0, 0)), eids
    ws = (t1 - t0) / n_windows
    pv = np.zeros((n_windows, ne))
    for w in range(n_windows):
        for i, e in enumerate(eids):
            mask = (data.times >= t0 + w * ws) & (data.times < t0 + (w + 1) * ws) & (data.electrodes == e)
            pv[w, i] = float(np.sum(mask))
    norms = np.linalg.norm(pv, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pv / norms, eids


def _n_components(adj: np.ndarray) -> int:
    """Count connected components (beta_0)."""
    if adj.shape[0] == 0:
        return 0
    nc, _ = connected_components(sparse.csr_matrix(adj > 0), directed=False)
    return int(nc)


def _cycle_rank(adj: np.ndarray) -> int:
    """Count independent cycles (beta_1) = E - V + C."""
    n = adj.shape[0]
    if n == 0:
        return 0
    e = int(np.sum(adj > 0)) // 2
    return max(0, e - n + _n_components(adj))


def _estimate_beta2(adj: np.ndarray, dist: np.ndarray, eps: float) -> int:
    """Estimate beta_2 via hollow tetrahedra (K4 with distance variance)."""
    n = adj.shape[0]
    if n < 4:
        return 0
    cav = 0
    for i in range(n):
        ni = set(np.where(adj[i] > 0)[0])
        for j in (x for x in ni if x > i):
            nij = ni & set(np.where(adj[j] > 0)[0])
            for k in (x for x in nij if x > j):
                for l in (x for x in nij & set(np.where(adj[k] > 0)[0]) if x > k):
                    ds = [dist[a, b] for a, b in [(i,j),(i,k),(i,l),(j,k),(j,l),(k,l)]]
                    if np.std(ds) > eps * 0.3:
                        cav += 1
    return cav


def compute_betti_numbers(
    data: SpikeData, max_dimension: int = 2,
    n_windows: int = 20, n_thresholds: int = 20,
) -> dict:
    """Bin spikes into windows, compute distance matrix between population
    vectors, build Vietoris-Rips filtration, extract Betti numbers.

    beta_0=components, beta_1=loops, beta_2=cavities.
    """
    pv, eids = _build_pop_vectors(data, n_windows)
    np_ = pv.shape[0]
    empty = [0] * n_thresholds
    if np_ < 3:
        return {"betti_curves": {"beta_0": empty, "beta_1": empty, "beta_2": empty},
                "thresholds": list(np.linspace(0, 1, n_thresholds)),
                "mean_betti": {"beta_0": 0.0, "beta_1": 0.0, "beta_2": 0.0},
                "max_betti": {"beta_0": 0, "beta_1": 0, "beta_2": 0},
                "n_windows": n_windows, "n_points": np_}

    dist = squareform(pdist(pv, "euclidean"))
    md = float(np.max(dist)) or 1.0
    thresholds = np.linspace(0, md, n_thresholds)
    b0, b1, b2 = [], [], []

    for eps in thresholds:
        adj = (dist <= eps).astype(float)
        np.fill_diagonal(adj, 0.0)
        b0.append(_n_components(adj))
        b1.append(_cycle_rank(adj) if max_dimension >= 1 else 0)
        b2.append(_estimate_beta2(adj, dist, eps) if max_dimension >= 2 else 0)

    return {"betti_curves": {"beta_0": b0, "beta_1": b1, "beta_2": b2},
            "thresholds": thresholds.tolist(),
            "mean_betti": {"beta_0": float(np.mean(b0)), "beta_1": float(np.mean(b1)),
                           "beta_2": float(np.mean(b2))},
            "max_betti": {"beta_0": int(np.max(b0)), "beta_1": int(np.max(b1)),
                          "beta_2": int(np.max(b2))},
            "n_windows": n_windows, "n_points": np_}


def compute_persistence_diagram(
    data: SpikeData, n_windows: int = 20,
    n_thresholds: int = 50, max_dimension: int = 1,
) -> dict:
    """Persistent homology — track when topological features appear/die.

    Return birth/death pairs per dimension.
    """
    pv, _ = _build_pop_vectors(data, n_windows)
    np_ = pv.shape[0]
    if np_ < 3:
        return {"persistence_pairs": {"dim_0": [], "dim_1": []},
                "total_persistence": {"dim_0": 0.0, "dim_1": 0.0},
                "max_persistence": {"dim_0": 0.0, "dim_1": 0.0},
                "n_features": {"dim_0": 0, "dim_1": 0}, "thresholds": []}

    dist = squareform(pdist(pv, "euclidean"))
    md = float(np.max(dist)) or 1.0
    thresholds = np.linspace(0, md, n_thresholds)

    # Dim 0: track component merging
    prev_labels = np.arange(np_)
    comp_birth = {i: 0.0 for i in range(np_)}
    d0_pairs = []

    prev_cycles = 0
    cycle_births = []
    d1_pairs = []

    for step, eps in enumerate(thresholds):
        adj = (dist <= eps).astype(float)
        np.fill_diagonal(adj, 0.0)
        _, labels = connected_components(sparse.csr_matrix(adj > 0), directed=False)

        if step > 0:
            # Detect component merges
            for i in range(np_):
                for j in range(i + 1, np_):
                    if prev_labels[i] != prev_labels[j] and labels[i] == labels[j]:
                        ci, cj = prev_labels[i], prev_labels[j]
                        if ci in comp_birth and cj in comp_birth:
                            dying = ci if comp_birth[ci] >= comp_birth[cj] else cj
                            if dying in comp_birth:
                                d0_pairs.append([comp_birth[dying], float(eps)])
                                del comp_birth[dying]
            new_b = {}
            for i in range(np_):
                nl, ol = labels[i], prev_labels[i]
                b = comp_birth.get(ol, 0.0)
                new_b[nl] = min(new_b.get(nl, b), b)
            comp_birth = new_b
        prev_labels = labels.copy()

        if max_dimension >= 1:
            cc = _cycle_rank(adj)
            if cc > prev_cycles:
                cycle_births.extend([float(eps)] * (cc - prev_cycles))
            elif cc < prev_cycles:
                for _ in range(min(prev_cycles - cc, len(cycle_births))):
                    cycle_births.sort()
                    d1_pairs.append([cycle_births.pop(0), float(eps)])
            prev_cycles = cc

    for b in comp_birth.values():
        if b < md:
            d0_pairs.append([b, md])
    for b in cycle_births:
        d1_pairs.append([b, md])

    def _stats(pairs):
        if not pairs:
            return 0.0, 0.0, 0
        lt = [d - b for b, d in pairs]
        return sum(lt), max(lt), len(pairs)

    t0, m0, n0 = _stats(d0_pairs)
    t1, m1, n1 = _stats(d1_pairs)

    return {"persistence_pairs": {"dim_0": d0_pairs, "dim_1": d1_pairs},
            "total_persistence": {"dim_0": t0, "dim_1": t1},
            "max_persistence": {"dim_0": m0, "dim_1": m1},
            "n_features": {"dim_0": n0, "dim_1": n1},
            "thresholds": thresholds.tolist()}


def compute_topological_complexity(data: SpikeData, n_windows: int = 20) -> dict:
    """Composite score from Betti numbers — higher = more structured computation.

    Combines beta_0 entropy, beta_1 presence, beta_2 presence,
    persistence stability, and effective dimensionality.
    """
    br = compute_betti_numbers(data, max_dimension=2, n_windows=n_windows)
    pr = compute_persistence_diagram(data, n_windows=n_windows)

    # Component scores
    b0c = np.array(br["betti_curves"]["beta_0"], dtype=float)
    if b0c.max() > 0:
        p = (b0c / b0c.max()); p = p / (p.sum() + 1e-10); p = p[p > 0]
        s_b0 = float(-np.sum(p * np.log2(p + 1e-10))) / max(np.log2(len(b0c)), 1e-10)
    else:
        s_b0 = 0.0
    mb1 = br["max_betti"]["beta_1"]
    s_b1 = min(1.0, float(np.mean(br["betti_curves"]["beta_1"])) / max(1, n_windows * 0.1)) if mb1 > 0 else 0.0
    s_b2 = min(1.0, br["max_betti"]["beta_2"] / 5.0) if br["max_betti"]["beta_2"] > 0 else 0.0
    tp = pr["total_persistence"]["dim_0"] + pr["total_persistence"]["dim_1"]
    nf = pr["n_features"]["dim_0"] + pr["n_features"]["dim_1"]
    mt = max(br["thresholds"]) if br["thresholds"] else 1.0
    s_pers = min(1.0, (tp / max(1, nf)) / max(mt, 1e-10))
    pv, _ = _build_pop_vectors(data, n_windows)
    eff_dim, s_dim = 0.0, 0.0
    if pv.shape[0] >= 2 and pv.shape[1] >= 2:
        ev = np.sort(np.real(np.linalg.eigvalsh((pv - pv.mean(0)).T @ (pv - pv.mean(0)) / (pv.shape[0] - 1))))[::-1]
        ev = ev[ev > 0]
        if len(ev) > 0:
            pr_ = ev / ev.sum()
            eff_dim = float(1.0 / np.sum(pr_ ** 2))
            s_dim = eff_dim / min(pv.shape)
    scores = {"beta_0_structure": s_b0, "beta_1_loops": s_b1, "beta_2_cavities": s_b2,
              "persistence": s_pers, "dimensionality": s_dim}
    w = {"beta_0_structure": 0.15, "beta_1_loops": 0.25, "beta_2_cavities": 0.15,
         "persistence": 0.25, "dimensionality": 0.20}
    cx = min(1.0, max(0.0, sum(w[k] * scores[k] for k in w)))

    if cx > 0.7:
        interp = "High topological complexity: rich multi-scale structure."
    elif cx > 0.4:
        interp = "Moderate topological complexity: meaningful cyclic dynamics."
    elif cx > 0.15:
        interp = "Low topological complexity: limited higher-order features."
    else:
        interp = "Minimal topological complexity: sparse activity patterns."

    return {"complexity_score": float(cx), "components": scores,
            "interpretation": interp,
            "betti_summary": {f"mean_beta_{d}": br["mean_betti"][f"beta_{d}"]
                              for d in range(3)} | {f"max_beta_{d}": br["max_betti"][f"beta_{d}"]
                              for d in range(3)},
            "effective_dimension": eff_dim}
