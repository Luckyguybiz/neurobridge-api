"""Topological data analysis (TDA) of neural spike train activity.

Reveals high-dimensional structure in neural activity that is invisible to
standard correlation-based methods. Betti numbers count topological features
("holes") in the activity manifold — a signature of complex computation.

Scientific basis:
- Giusti, Ghrist & Bassett (2016), J Comp Neurosci — clique topology
  reveals structure in neural correlations
- Curto (2017), PNAS — geometric combinatorics of neural codes
- Rybakken et al. (2019), Front Comp Neurosci — persistent homology
  for neural decoding

Methods:
- Simplified Betti number computation from distance matrices
  (without external TDA libraries):
  - beta_0: connected components (graph connectivity)
  - beta_1: independent cycles (loop detection via cycle rank)
  - beta_2: cavities (estimated from clique analysis)
- Persistent homology via threshold filtration
- Composite topological complexity score
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy import sparse
from .loader import SpikeData


def _build_population_vectors(
    data: SpikeData, n_windows: int,
) -> tuple[np.ndarray, list[int]]:
    """Bin spikes into time windows and build population activity vectors.

    Each window becomes a point in n_electrode-dimensional space.
    Returns (n_windows x n_electrodes) matrix and electrode IDs.
    """
    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)
    t_start, t_end = data.time_range

    if t_end <= t_start or n_electrodes == 0 or n_windows < 2:
        return np.empty((0, 0)), electrode_ids

    window_size = (t_end - t_start) / n_windows
    pop_vectors = np.zeros((n_windows, n_electrodes), dtype=np.float64)

    for w in range(n_windows):
        w_start = t_start + w * window_size
        w_end = w_start + window_size
        for i, e in enumerate(electrode_ids):
            mask = (data.times >= w_start) & (data.times < w_end) & (data.electrodes == e)
            pop_vectors[w, i] = float(np.sum(mask))

    # Normalize each window to unit vector (if nonzero)
    norms = np.linalg.norm(pop_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pop_vectors = pop_vectors / norms

    return pop_vectors, electrode_ids


def _distance_matrix(pop_vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between population vectors."""
    if pop_vectors.shape[0] < 2:
        return np.zeros((pop_vectors.shape[0], pop_vectors.shape[0]))
    dists = squareform(pdist(pop_vectors, metric="euclidean"))
    return dists


def _count_connected_components(adj: np.ndarray) -> int:
    """Count connected components in an adjacency matrix (beta_0)."""
    n = adj.shape[0]
    if n == 0:
        return 0
    graph = sparse.csr_matrix(adj > 0)
    n_components, _ = connected_components(graph, directed=False)
    return int(n_components)


def _count_independent_cycles(adj: np.ndarray) -> int:
    """Count independent cycles in a graph (beta_1).

    Uses the cycle rank formula: beta_1 = E - V + C
    where E = edges, V = vertices, C = connected components.
    """
    n = adj.shape[0]
    if n == 0:
        return 0

    # Count edges (undirected)
    n_edges = int(np.sum(adj > 0)) // 2
    n_components = _count_connected_components(adj)
    cycle_rank = n_edges - n + n_components

    return max(0, cycle_rank)


def _estimate_cavities(adj: np.ndarray, dist_matrix: np.ndarray, threshold: float) -> int:
    """Estimate beta_2 (2D cavities) from clique analysis.

    A cavity (void) exists when the boundary of a set of triangles
    forms a closed surface without being filled by a tetrahedron.
    We approximate this by counting "hollow tetrahedra" — sets of
    4 nodes where all 6 edges exist (complete subgraph K4) but
    at least one face has high internal distance.
    """
    n = adj.shape[0]
    if n < 4:
        return 0

    # Find all triangles first
    triangles = []
    for i in range(n):
        neighbors_i = set(np.where(adj[i] > 0)[0])
        for j in neighbors_i:
            if j <= i:
                continue
            neighbors_j = set(np.where(adj[j] > 0)[0])
            common = neighbors_i & neighbors_j
            for k in common:
                if k <= j:
                    continue
                triangles.append((i, j, k))

    if len(triangles) < 4:
        return 0

    # Check for hollow tetrahedra (K4 minus solid interior)
    cavity_count = 0
    triangle_set = set(triangles)

    # For each set of 4 mutually connected nodes
    for i in range(n):
        ni = set(np.where(adj[i] > 0)[0])
        for j in ni:
            if j <= i:
                continue
            nj = set(np.where(adj[j] > 0)[0])
            common_ij = ni & nj
            for k in common_ij:
                if k <= j:
                    continue
                nk = set(np.where(adj[k] > 0)[0])
                common_ijk = common_ij & nk
                for l_node in common_ijk:
                    if l_node <= k:
                        continue
                    # All 4 nodes are mutually connected (K4)
                    # Check if "hollow": mean internal distance > threshold
                    nodes = [i, j, k, l_node]
                    internal_dists = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            internal_dists.append(dist_matrix[nodes[a], nodes[b]])
                    # A cavity if distances vary a lot (not a tight cluster)
                    dist_std = np.std(internal_dists)
                    if dist_std > threshold * 0.3:
                        cavity_count += 1

    return cavity_count


def compute_betti_numbers(
    data: SpikeData,
    max_dimension: int = 2,
    n_windows: int = 20,
    n_thresholds: int = 20,
) -> dict:
    """Compute Betti numbers from spike train topology.

    Bins spikes into time windows to create population activity vectors,
    computes distance matrix, then builds Vietoris-Rips filtration at
    multiple thresholds to extract Betti numbers.

    beta_0 = connected components (how fragmented the activity space is)
    beta_1 = independent loops (cyclic patterns in neural dynamics)
    beta_2 = cavities (higher-order structure, voids)

    Args:
        data: SpikeData with spike times and electrode IDs.
        max_dimension: Maximum Betti number to compute (0, 1, or 2).
        n_windows: Number of time windows to divide the recording into.
        n_thresholds: Number of distance thresholds for filtration.

    Returns:
        Dict with keys:
        - betti_curves: dict with 'beta_0', 'beta_1', 'beta_2' as
            lists of values across thresholds.
        - thresholds: list[float], distance thresholds used.
        - mean_betti: dict with mean beta_0, beta_1, beta_2.
        - max_betti: dict with max beta_0, beta_1, beta_2.
        - n_windows, n_points.
    """
    pop_vectors, electrode_ids = _build_population_vectors(data, n_windows)
    n_points = pop_vectors.shape[0]

    if n_points < 3:
        empty_curve = [0] * n_thresholds
        return {
            "betti_curves": {"beta_0": empty_curve, "beta_1": empty_curve, "beta_2": empty_curve},
            "thresholds": list(np.linspace(0, 1, n_thresholds)),
            "mean_betti": {"beta_0": 0.0, "beta_1": 0.0, "beta_2": 0.0},
            "max_betti": {"beta_0": 0, "beta_1": 0, "beta_2": 0},
            "n_windows": n_windows,
            "n_points": n_points,
        }

    dist = _distance_matrix(pop_vectors)

    # Filtration thresholds from 0 to max distance
    max_dist = float(np.max(dist))
    if max_dist == 0:
        max_dist = 1.0
    thresholds = np.linspace(0, max_dist, n_thresholds)

    betti_0 = []
    betti_1 = []
    betti_2 = []

    for eps in thresholds:
        # Build adjacency at this threshold
        adj = (dist <= eps).astype(np.float64)
        np.fill_diagonal(adj, 0.0)

        # beta_0: connected components
        b0 = _count_connected_components(adj)
        betti_0.append(b0)

        # beta_1: independent cycles
        if max_dimension >= 1:
            b1 = _count_independent_cycles(adj)
            betti_1.append(b1)
        else:
            betti_1.append(0)

        # beta_2: cavities (expensive, only if requested)
        if max_dimension >= 2:
            b2 = _estimate_cavities(adj, dist, eps)
            betti_2.append(b2)
        else:
            betti_2.append(0)

    return {
        "betti_curves": {
            "beta_0": [int(x) for x in betti_0],
            "beta_1": [int(x) for x in betti_1],
            "beta_2": [int(x) for x in betti_2],
        },
        "thresholds": thresholds.tolist(),
        "mean_betti": {
            "beta_0": float(np.mean(betti_0)),
            "beta_1": float(np.mean(betti_1)),
            "beta_2": float(np.mean(betti_2)),
        },
        "max_betti": {
            "beta_0": int(np.max(betti_0)),
            "beta_1": int(np.max(betti_1)),
            "beta_2": int(np.max(betti_2)),
        },
        "n_windows": n_windows,
        "n_points": n_points,
    }


def compute_persistence_diagram(
    data: SpikeData,
    n_windows: int = 20,
    n_thresholds: int = 50,
    max_dimension: int = 1,
) -> dict:
    """Compute persistent homology — track topological features across scales.

    For each topological feature (component, loop, cavity), records when it
    is "born" (appears) and "dies" (disappears) as the distance threshold
    increases. Long-lived features (large death - birth) represent robust
    topological structure.

    Args:
        data: SpikeData with spike times and electrode IDs.
        n_windows: Number of time windows.
        n_thresholds: Number of filtration steps (higher = finer resolution).
        max_dimension: Maximum dimension for persistence (0 or 1).

    Returns:
        Dict with keys:
        - persistence_pairs: dict with 'dim_0', 'dim_1' as lists of
            [birth, death] pairs.
        - total_persistence: dict with summed lifetimes per dimension.
        - max_persistence: dict with longest-lived feature per dimension.
        - n_features: dict with count of features per dimension.
        - thresholds: list[float].
    """
    pop_vectors, electrode_ids = _build_population_vectors(data, n_windows)
    n_points = pop_vectors.shape[0]

    if n_points < 3:
        return {
            "persistence_pairs": {"dim_0": [], "dim_1": []},
            "total_persistence": {"dim_0": 0.0, "dim_1": 0.0},
            "max_persistence": {"dim_0": 0.0, "dim_1": 0.0},
            "n_features": {"dim_0": 0, "dim_1": 0},
            "thresholds": [],
        }

    dist = _distance_matrix(pop_vectors)
    max_dist = float(np.max(dist))
    if max_dist == 0:
        max_dist = 1.0
    thresholds = np.linspace(0, max_dist, n_thresholds)

    # Track features across filtration
    # Dimension 0: track connected components (birth at 0, die when merged)
    prev_labels = np.arange(n_points)
    component_birth = {i: 0.0 for i in range(n_points)}
    dim0_pairs = []

    # Dimension 1: track cycle birth/death
    prev_cycle_count = 0
    cycle_births = []
    dim1_pairs = []

    for step, eps in enumerate(thresholds):
        adj = (dist <= eps).astype(np.float64)
        np.fill_diagonal(adj, 0.0)

        # --- Dimension 0: component merging ---
        graph = sparse.csr_matrix(adj > 0)
        n_comp, labels = connected_components(graph, directed=False)

        # Detect merges: when two previously separate components join
        if step > 0:
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if prev_labels[i] != prev_labels[j] and labels[i] == labels[j]:
                        # Components merged — younger one dies
                        comp_i = prev_labels[i]
                        comp_j = prev_labels[j]
                        if comp_i in component_birth and comp_j in component_birth:
                            birth_i = component_birth[comp_i]
                            birth_j = component_birth[comp_j]
                            # Younger component dies
                            if birth_i >= birth_j:
                                dying = comp_i
                            else:
                                dying = comp_j
                            if dying in component_birth:
                                dim0_pairs.append([
                                    float(component_birth[dying]),
                                    float(eps),
                                ])
                                del component_birth[dying]

            # Update birth records for new label mapping
            new_births = {}
            for i in range(n_points):
                new_label = labels[i]
                old_label = prev_labels[i]
                if new_label not in new_births:
                    if old_label in component_birth:
                        new_births[new_label] = component_birth[old_label]
                    else:
                        new_births[new_label] = 0.0
                else:
                    if old_label in component_birth:
                        new_births[new_label] = min(
                            new_births[new_label],
                            component_birth[old_label],
                        )
            component_birth = new_births

        prev_labels = labels.copy()

        # --- Dimension 1: cycle detection ---
        if max_dimension >= 1:
            cycle_count = _count_independent_cycles(adj)
            if cycle_count > prev_cycle_count:
                # New cycles born
                for _ in range(cycle_count - prev_cycle_count):
                    cycle_births.append(float(eps))
            elif cycle_count < prev_cycle_count:
                # Cycles died (filled in)
                n_died = prev_cycle_count - cycle_count
                for _ in range(min(n_died, len(cycle_births))):
                    birth = cycle_births.pop(0)
                    dim1_pairs.append([birth, float(eps)])
            prev_cycle_count = cycle_count

    # Remaining components get infinite death (use max_dist)
    for comp, birth in component_birth.items():
        if birth < max_dist:
            dim0_pairs.append([float(birth), float(max_dist)])

    # Remaining cycles get infinite death
    for birth in cycle_births:
        dim1_pairs.append([float(birth), float(max_dist)])

    # Summary statistics
    def _persistence_stats(pairs):
        if not pairs:
            return 0.0, 0.0, 0
        lifetimes = [d - b for b, d in pairs]
        return float(sum(lifetimes)), float(max(lifetimes)), len(pairs)

    total_0, max_0, n_0 = _persistence_stats(dim0_pairs)
    total_1, max_1, n_1 = _persistence_stats(dim1_pairs)

    return {
        "persistence_pairs": {
            "dim_0": dim0_pairs,
            "dim_1": dim1_pairs,
        },
        "total_persistence": {"dim_0": total_0, "dim_1": total_1},
        "max_persistence": {"dim_0": max_0, "dim_1": max_1},
        "n_features": {"dim_0": n_0, "dim_1": n_1},
        "thresholds": thresholds.tolist(),
    }


def compute_topological_complexity(
    data: SpikeData,
    n_windows: int = 20,
) -> dict:
    """Compute composite topological complexity score from Betti numbers.

    Higher complexity indicates more structured, multi-scale computation
    in the neural network. Organoids with higher topological complexity
    tend to show richer information processing capabilities.

    The score combines:
    - beta_0 entropy: diversity of component sizes
    - beta_1 presence: cyclic structure in activity space
    - Persistence: stability of topological features
    - Dimensionality: effective dimension of the activity manifold

    Args:
        data: SpikeData with spike times and electrode IDs.
        n_windows: Number of time windows.

    Returns:
        Dict with keys:
        - complexity_score: float, composite score (0-1).
        - components: dict with individual score components.
        - interpretation: str, qualitative description.
        - betti_summary: dict, summary of Betti numbers.
        - effective_dimension: float.
    """
    # Compute Betti numbers
    betti_result = compute_betti_numbers(data, max_dimension=2, n_windows=n_windows)
    persistence_result = compute_persistence_diagram(data, n_windows=n_windows)

    # Component 1: beta_0 structure
    # Optimal: not all connected (1 component) and not all disconnected
    beta_0_curve = betti_result["betti_curves"]["beta_0"]
    if len(beta_0_curve) > 0 and max(beta_0_curve) > 0:
        beta_0_normed = np.array(beta_0_curve) / max(beta_0_curve)
        # Entropy of the curve (more variation = more structure)
        probs = beta_0_normed / (np.sum(beta_0_normed) + 1e-10)
        probs = probs[probs > 0]
        beta_0_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(len(beta_0_curve))
        beta_0_score = beta_0_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        beta_0_score = 0.0

    # Component 2: beta_1 presence (loops)
    beta_1_curve = betti_result["betti_curves"]["beta_1"]
    max_beta_1 = betti_result["max_betti"]["beta_1"]
    # Score based on presence and count of loops
    if max_beta_1 > 0:
        beta_1_score = min(1.0, float(np.mean(beta_1_curve)) / max(1, n_windows * 0.1))
    else:
        beta_1_score = 0.0

    # Component 3: beta_2 presence (cavities)
    max_beta_2 = betti_result["max_betti"]["beta_2"]
    beta_2_score = min(1.0, max_beta_2 / 5.0) if max_beta_2 > 0 else 0.0

    # Component 4: Persistence stability
    total_pers = (
        persistence_result["total_persistence"]["dim_0"]
        + persistence_result["total_persistence"]["dim_1"]
    )
    n_features = (
        persistence_result["n_features"]["dim_0"]
        + persistence_result["n_features"]["dim_1"]
    )
    mean_pers = total_pers / max(1, n_features)
    # Normalize by max threshold
    thresholds = betti_result["thresholds"]
    max_thresh = max(thresholds) if thresholds else 1.0
    persistence_score = min(1.0, mean_pers / max_thresh) if max_thresh > 0 else 0.0

    # Component 5: Effective dimensionality (from PCA of population vectors)
    pop_vectors, _ = _build_population_vectors(data, n_windows)
    if pop_vectors.shape[0] >= 2 and pop_vectors.shape[1] >= 2:
        centered = pop_vectors - pop_vectors.mean(axis=0)
        cov = centered.T @ centered / (pop_vectors.shape[0] - 1)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) > 0:
            probs = eigvals / np.sum(eigvals)
            participation_ratio = 1.0 / np.sum(probs ** 2)
            eff_dim = float(participation_ratio)
            max_dim = float(min(pop_vectors.shape))
            dim_score = eff_dim / max_dim if max_dim > 0 else 0.0
        else:
            eff_dim = 0.0
            dim_score = 0.0
    else:
        eff_dim = 0.0
        dim_score = 0.0

    # Composite score (weighted combination)
    weights = {
        "beta_0_structure": 0.15,
        "beta_1_loops": 0.25,
        "beta_2_cavities": 0.15,
        "persistence": 0.25,
        "dimensionality": 0.20,
    }
    scores = {
        "beta_0_structure": beta_0_score,
        "beta_1_loops": beta_1_score,
        "beta_2_cavities": beta_2_score,
        "persistence": persistence_score,
        "dimensionality": dim_score,
    }
    complexity = sum(weights[k] * scores[k] for k in weights)
    complexity = min(1.0, max(0.0, complexity))

    # Interpretation
    if complexity > 0.7:
        interpretation = "High topological complexity: rich multi-scale structure indicating sophisticated information processing."
    elif complexity > 0.4:
        interpretation = "Moderate topological complexity: meaningful structure with some cyclic dynamics present."
    elif complexity > 0.15:
        interpretation = "Low topological complexity: basic connectivity structure with limited higher-order features."
    else:
        interpretation = "Minimal topological complexity: sparse or trivial activity patterns."

    return {
        "complexity_score": float(complexity),
        "components": scores,
        "interpretation": interpretation,
        "betti_summary": {
            "mean_beta_0": betti_result["mean_betti"]["beta_0"],
            "mean_beta_1": betti_result["mean_betti"]["beta_1"],
            "mean_beta_2": betti_result["mean_betti"]["beta_2"],
            "max_beta_0": betti_result["max_betti"]["beta_0"],
            "max_beta_1": betti_result["max_betti"]["beta_1"],
            "max_beta_2": betti_result["max_betti"]["beta_2"],
        },
        "effective_dimension": eff_dim,
    }
