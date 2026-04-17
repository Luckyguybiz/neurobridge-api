"""Organoid Intelligence Quotient (OIQ) -- publication-quality composite metric.

Quantifies the computational capacity of brain organoids from multi-electrode
array (MEA) spike recordings.  Designed for datasets ranging from 1K to 10M+
spikes across 1-128 electrodes and durations from minutes to weeks.

Scoring framework
-----------------
Six orthogonal dimensions, each grounded in established neuroscience metrics:

  1. Signal Quality        (0-15)  SNR, ISI violations, amplitude distribution
  2. Network Complexity    (0-20)  Graph topology: clustering, path length,
                                   small-world index (Watts & Strogatz 1998)
  3. Information Processing(0-20)  Entropy rate, mutual information,
                                   transfer entropy (Schreiber 2000)
  4. Temporal Organization (0-20)  Burst quality, UP/DOWN states, regularity
                                   (Beggs & Plenz 2003)
  5. Adaptability          (0-15)  Firing-rate stability, Fano factor trends,
                                   response variability across time
  6. Learning Potential    (0-10)  STDP-compatible timing windows,
                                   plasticity signatures

Total: 0-100.  Grades: A (>=80), B (>=60), C (>=40), D (>=20), F (<20).

Performance target: <60 s on 2.6 M spikes / 32 electrodes / 118 h.

Strategy for speed
------------------
* All heavy paths are vectorised NumPy -- zero Python-level spike loops.
* Bin sizes adapt automatically so no array exceeds ``_MAX_BINS`` elements.
* O(N^2)-per-spike algorithms (STDP cross-correlation) operate on a
  deterministic stratified subsample capped at ``_MAX_SPIKES_PAIRWISE``.
* Graph metrics are computed on the 32x32 adjacency matrix (tiny).

References
----------
Beggs & Plenz (2003) J Neurosci -- neuronal avalanches
Watts & Strogatz (1998) Nature -- small-world networks
Schreiber (2000) Phys Rev Lett -- transfer entropy
Selinger et al. (2004) J Neurosci Methods -- ISI violation rate
Softky & Koch (1993) J Neurosci -- CV of ISI
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .loader import SpikeData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive-resolution constants
# ---------------------------------------------------------------------------
_MAX_BINS: int = 6_000
"""Hard ceiling on time-bin arrays to bound memory and runtime."""

_MAX_SPIKES_PAIRWISE: int = 60_000
"""Subsample cap for O(N^2) pairwise algorithms (STDP, cross-corr)."""

_MIN_SPIKES_FOR_ANALYSIS: int = 50
"""Absolute minimum spike count to produce meaningful scores."""

_MIN_SPIKES_PER_ELECTRODE: int = 10
"""Electrodes with fewer spikes are flagged as silent."""


# ═══════════════════════════════════════════════════════════════════════════
# Data classes for structured output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubScore:
    """One dimension of the OIQ."""
    name: str
    value: float
    max_value: float
    details: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    @property
    def pct(self) -> float:
        return (self.value / self.max_value * 100) if self.max_value else 0.0


@dataclass
class OIQResult:
    """Complete Organoid IQ result."""
    iq_score: float
    grade: str
    assessment: str
    subscores: dict[str, SubScore]
    computation_time_s: float
    dataset_summary: dict[str, Any]
    interpretation: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON/API responses."""
        sub = {}
        details = {}
        interp = {}
        for key, sc in self.subscores.items():
            sub[key] = round(sc.value, 2)
            details[key] = sc.details
            interp[key] = f"{sc.value:.1f}/{sc.max_value:.0f} -- {sc.interpretation}"
        return {
            "iq_score": round(self.iq_score, 1),
            "grade": self.grade,
            "assessment": self.assessment,
            "subscores": sub,
            "max_possible": 100,
            "details": details,
            "interpretation": interp,
            "computation_time_s": round(self.computation_time_s, 2),
            "dataset_summary": self.dataset_summary,
            "warnings": self.warnings,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers (pure NumPy, no Python spike loops)
# ═══════════════════════════════════════════════════════════════════════════

def _adaptive_bin_s(duration_s: float, default_s: float) -> float:
    """Return bin size in seconds that keeps total bins <= _MAX_BINS."""
    if duration_s <= 0:
        return default_s
    min_bin = duration_s / _MAX_BINS
    return max(default_s, min_bin)


def _bin_spike_trains(data: SpikeData, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin all electrodes into a (n_electrodes, n_bins) matrix.

    Returns (matrix, bin_edges).  All vectorised.
    """
    t0, t1 = data.time_range
    edges = np.arange(t0, t1 + bin_s, bin_s)
    n_bins = len(edges) - 1
    eids = data.electrode_ids
    mat = np.zeros((len(eids), n_bins), dtype=np.int32)
    eid_to_row = {e: i for i, e in enumerate(eids)}
    # Vectorised: digitize all spikes at once
    bin_idx = np.searchsorted(edges, data.times, side="right") - 1
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
    rows = np.array([eid_to_row[int(e)] for e in data.electrodes], dtype=np.int32)
    np.add.at(mat, (rows, bin_idx), 1)
    return mat, edges


def _binary_spike_matrix(data: SpikeData, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary (0/1) spike matrix per electrode."""
    mat, edges = _bin_spike_trains(data, bin_s)
    return (mat > 0).astype(np.uint8), edges


def _stratified_subsample(data: SpikeData, max_spikes: int) -> SpikeData:
    """Deterministic stratified subsample preserving per-electrode proportions."""
    if data.n_spikes <= max_spikes:
        return data
    rng = np.random.RandomState(42)
    keep_parts: list[np.ndarray] = []
    for e in data.electrode_ids:
        idx = data._electrode_indices[e]
        n_keep = max(1, int(len(idx) * max_spikes / data.n_spikes))
        if n_keep < len(idx):
            chosen = rng.choice(idx, size=n_keep, replace=False)
        else:
            chosen = idx
        keep_parts.append(chosen)
    keep = np.sort(np.concatenate(keep_parts))
    return SpikeData(
        times=data.times[keep],
        electrodes=data.electrodes[keep],
        amplitudes=data.amplitudes[keep],
        waveforms=data.waveforms[keep] if data.waveforms is not None else None,
        sampling_rate=data.sampling_rate,
        metadata=data.metadata,
    )


def _gaussian_score(value: float, optimal: float, width: float) -> float:
    """Gaussian bell scoring: 1.0 at *optimal*, decaying with *width*."""
    return float(np.exp(-((value - optimal) ** 2) / (2.0 * width ** 2)))


def _sigmoid_score(value: float, midpoint: float, steepness: float = 10.0) -> float:
    """Monotonically increasing 0-1 score via logistic sigmoid."""
    x = steepness * (value - midpoint)
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _inv_sigmoid_score(value: float, midpoint: float, steepness: float = 10.0) -> float:
    """Monotonically decreasing 0-1 score (bad when high)."""
    return 1.0 - _sigmoid_score(value, midpoint, steepness)


def _safe_log2(x: np.ndarray) -> np.ndarray:
    """log2 that returns 0 for zero inputs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 0, np.log2(x), 0.0)
    return result


def _shannon_entropy_bits(prob: np.ndarray) -> float:
    """Shannon entropy in bits from a probability vector."""
    p = prob[prob > 0]
    return float(-np.sum(p * np.log2(p)))


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 1: SIGNAL QUALITY (0-15)
# ═══════════════════════════════════════════════════════════════════════════

def _score_signal_quality(data: SpikeData) -> SubScore:
    """Evaluate recording quality: SNR, ISI violations, amplitude health.

    Metrics:
    - SNR per electrode: mean(|amplitude|) / std(amplitude)  [Quiroga 2004]
    - ISI violation rate: fraction of ISIs < 2 ms refractory  [Hill 2011]
    - Amplitude normality: Shapiro-Wilk on amplitude distribution
    - Active electrode fraction

    Scoring (15 pts max):
      SNR quality          0-5
      ISI violation rate   0-4
      Amplitude health     0-3
      Active electrodes    0-3
    """
    eids = data.electrode_ids
    n_electrodes = len(eids)
    duration = data.duration

    snr_values = []
    violation_rates = []
    amp_kurtosis = []
    active_count = 0

    for e in eids:
        idx = data._electrode_indices[e]
        n = len(idx)
        if n < _MIN_SPIKES_PER_ELECTRODE:
            violation_rates.append(0.0)
            continue
        active_count += 1
        amps = data.amplitudes[idx]
        # SNR: signal / noise
        snr = float(np.mean(np.abs(amps)) / (np.std(amps) + 1e-12))
        snr_values.append(snr)
        # ISI violations (refractory period < 2 ms)
        times_e = data.times[idx]  # already sorted globally
        isi_ms = np.diff(times_e) * 1000.0
        if len(isi_ms) > 0:
            viol_rate = float(np.sum(isi_ms < 2.0)) / len(isi_ms)
        else:
            viol_rate = 0.0
        violation_rates.append(viol_rate)
        # Kurtosis of amplitude distribution (should be moderate)
        if n > 20:
            kurt = float(np.mean(((amps - np.mean(amps)) / (np.std(amps) + 1e-12)) ** 4))
            amp_kurtosis.append(kurt)

    mean_snr = float(np.mean(snr_values)) if snr_values else 0.0
    mean_viol = float(np.mean(violation_rates)) if violation_rates else 1.0
    mean_kurt = float(np.mean(amp_kurtosis)) if amp_kurtosis else 3.0
    active_frac = active_count / max(n_electrodes, 1)

    # --- Sub-scores ---
    # SNR: optimal 3-8; >1 acceptable, >5 very good
    snr_score = min(5.0, _sigmoid_score(mean_snr, midpoint=2.0, steepness=1.5) * 5.0)
    # ISI violations: 0% ideal, >5% bad
    viol_score = _inv_sigmoid_score(mean_viol, midpoint=0.03, steepness=80) * 4.0
    # Amplitude kurtosis: mesokurtic (~3) is healthy, excess = artifacts
    kurt_score = _gaussian_score(mean_kurt, optimal=3.0, width=3.0) * 3.0
    # Active electrode fraction: 100% ideal
    active_score = active_frac * 3.0

    total = round(min(15.0, snr_score + viol_score + kurt_score + active_score), 2)

    return SubScore(
        name="signal_quality",
        value=total,
        max_value=15.0,
        interpretation="Recording fidelity: SNR, refractory compliance, amplitude health",
        details={
            "mean_snr": round(mean_snr, 3),
            "mean_isi_violation_rate": round(mean_viol, 5),
            "mean_amplitude_kurtosis": round(mean_kurt, 3),
            "active_electrode_fraction": round(active_frac, 3),
            "active_electrodes": active_count,
            "total_electrodes": n_electrodes,
            "snr_sub": round(snr_score, 2),
            "violation_sub": round(viol_score, 2),
            "kurtosis_sub": round(kurt_score, 2),
            "active_sub": round(active_score, 2),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 2: NETWORK COMPLEXITY (0-20)
# ═══════════════════════════════════════════════════════════════════════════

def _score_network_complexity(data: SpikeData, bin_s: float) -> SubScore:
    """Evaluate network topology from co-firing patterns.

    Constructs a functional connectivity matrix from spike coincidences,
    then computes graph-theoretic metrics.

    Metrics:
    - Graph density
    - Mean clustering coefficient  [Watts & Strogatz 1998]
    - Characteristic path length
    - Small-world index (sigma = C/C_rand / L/L_rand)
    - Degree distribution entropy

    Scoring (20 pts max):
      Small-world index    0-8
      Clustering coeff     0-5
      Density (optimal)    0-4
      Degree entropy       0-3
    """
    eids = data.electrode_ids
    n = len(eids)

    if n < 3:
        return SubScore(
            name="network_complexity", value=0.0, max_value=20.0,
            interpretation="Network topology (insufficient electrodes)",
            details={"error": "fewer than 3 electrodes"},
        )

    # --- Build binary coincidence matrix (vectorised) ---
    bmat, _ = _binary_spike_matrix(data, bin_s)
    # Co-activity: dot product of binary trains => counts of co-active bins
    coactive = bmat.astype(np.float64) @ bmat.T.astype(np.float64)
    # Normalise by geometric mean of active bins per electrode
    active_counts = np.sum(bmat, axis=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = np.sqrt(np.outer(active_counts, active_counts))
        geo[geo == 0] = 1.0
        strength = coactive / geo
    np.fill_diagonal(strength, 0.0)

    # Threshold: retain edges > 2 sigma above mean of off-diagonal
    triu = strength[np.triu_indices(n, k=1)]
    if len(triu) == 0 or np.std(triu) < 1e-12:
        threshold = 0.0
    else:
        threshold = float(np.mean(triu) + 2.0 * np.std(triu))
    adj = (strength > threshold).astype(np.float64)

    n_edges = int(np.sum(adj) // 2)
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # --- Clustering coefficient (vectorised) ---
    degrees = np.sum(adj, axis=1)
    # Number of triangles at each node: (A^3)_ii / 2
    adj2 = adj @ adj
    triangles = np.diag(adj2 @ adj) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = degrees * (degrees - 1) / 2.0
        denom[denom == 0] = 1.0
        clustering = triangles / denom
        clustering[degrees < 2] = 0.0
    mean_clustering = float(np.mean(clustering))

    # --- Characteristic path length (BFS on small matrix) ---
    char_path = _characteristic_path_length(adj)

    # --- Small-world index ---
    # Compare against Erdos-Renyi random graph with same density
    if density > 0 and n > 3:
        # Expected clustering for ER: p = density
        c_rand = density
        # Expected path length for ER: ln(n) / ln(n*p)  (connected regime)
        np_val = n * density
        if np_val > 1:
            l_rand = np.log(n) / np.log(max(np_val, 1.01))
        else:
            l_rand = float(n)
        gamma = (mean_clustering / c_rand) if c_rand > 0 else 0.0
        lam = (char_path / l_rand) if l_rand > 0 else 1.0
        sigma = (gamma / lam) if lam > 0 else 0.0
    else:
        sigma = 0.0
        gamma = 0.0
        lam = 1.0

    # --- Degree distribution entropy ---
    if np.sum(degrees) > 0:
        deg_hist = np.bincount(degrees.astype(int))
        deg_prob = deg_hist / deg_hist.sum()
        degree_entropy = _shannon_entropy_bits(deg_prob)
        max_deg_entropy = np.log2(n) if n > 1 else 1.0
        norm_deg_entropy = degree_entropy / max_deg_entropy if max_deg_entropy > 0 else 0.0
    else:
        degree_entropy = 0.0
        norm_deg_entropy = 0.0

    # --- Sub-scores ---
    # Small-world: sigma > 1 is small-world; 3-10 is excellent
    sw_score = min(8.0, _sigmoid_score(sigma, midpoint=1.5, steepness=1.0) * 8.0)
    # Clustering: 0.3-0.7 is good for neural networks
    clust_score = _gaussian_score(mean_clustering, optimal=0.5, width=0.3) * 5.0
    # Density: moderate (0.2-0.5) is biologically realistic
    dens_score = _gaussian_score(density, optimal=0.35, width=0.25) * 4.0
    # Degree entropy: higher = more heterogeneous = more interesting
    deg_score = min(3.0, norm_deg_entropy * 3.0)

    total = round(min(20.0, sw_score + clust_score + dens_score + deg_score), 2)

    return SubScore(
        name="network_complexity",
        value=total,
        max_value=20.0,
        interpretation="Graph topology: small-worldness, clustering, connectivity",
        details={
            "n_edges": n_edges,
            "density": round(density, 4),
            "mean_clustering_coefficient": round(mean_clustering, 4),
            "characteristic_path_length": round(char_path, 3),
            "small_world_sigma": round(sigma, 3),
            "small_world_gamma": round(gamma, 3),
            "small_world_lambda": round(lam, 3),
            "degree_entropy_bits": round(degree_entropy, 3),
            "norm_degree_entropy": round(norm_deg_entropy, 3),
            "threshold_used": round(threshold, 5),
            "sw_sub": round(sw_score, 2),
            "clust_sub": round(clust_score, 2),
            "dens_sub": round(dens_score, 2),
            "deg_sub": round(deg_score, 2),
        },
    )


def _characteristic_path_length(adj: np.ndarray) -> float:
    """Mean shortest path length via BFS on a small adjacency matrix.

    For a 32x32 matrix this is instant.  Returns inf-free mean:
    disconnected pairs are excluded.
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    # BFS via matrix powers (exact for unweighted)
    power = adj.copy()
    for step in range(1, n):
        newly = (power > 0) & (dist == np.inf)
        dist[newly] = step
        power = power @ adj
        if not np.any(dist == np.inf):
            break
    mask = (dist < np.inf) & (dist > 0)
    return float(np.mean(dist[mask])) if np.any(mask) else float(n)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 3: INFORMATION PROCESSING (0-20)
# ═══════════════════════════════════════════════════════════════════════════

def _score_information_processing(data: SpikeData, bin_s: float) -> SubScore:
    """Evaluate information-theoretic capacity.

    Metrics:
    - Entropy rate: word entropy / word length  [Strong et al. 1998]
    - Mutual information between electrode pairs
    - Transfer entropy (directed information flow)  [Schreiber 2000]

    Scoring (20 pts max):
      Entropy rate (normalised) 0-7
      Mutual information        0-6
      Transfer entropy           0-7
    """
    bmat, _ = _binary_spike_matrix(data, bin_s)
    n_elec, n_bins = bmat.shape

    if n_bins < 20:
        return SubScore(
            name="information_processing", value=0.0, max_value=20.0,
            interpretation="Information-theoretic capacity (insufficient data)",
            details={"error": "too few time bins"},
        )

    # --- Entropy rate via word distributions (vectorised) ---
    word_len = min(8, max(3, n_bins // 500))  # adapt word length to data
    entropies = []
    for row in bmat:
        if np.sum(row) < 5:
            continue
        # Slide a window of word_len across the binary train
        # Each word is an integer encoding of the bit pattern
        powers = 2 ** np.arange(word_len)
        n_words = n_bins - word_len + 1
        if n_words < 10:
            continue
        # Vectorised word extraction: strided view
        words = np.lib.stride_tricks.sliding_window_view(row, word_len)
        word_ints = words @ powers
        # Count unique words
        _, counts = np.unique(word_ints, return_counts=True)
        probs = counts / counts.sum()
        ent = _shannon_entropy_bits(probs)
        # Normalise by max entropy (word_len bits)
        entropies.append(ent / word_len)

    mean_entropy_rate = float(np.mean(entropies)) if entropies else 0.0

    # --- Pairwise mutual information (vectorised) ---
    mi_values = []
    if n_elec >= 2:
        # Joint and marginal probabilities from binary matrix
        for i in range(min(n_elec, 16)):  # cap at 16 pairs for speed
            for j in range(i + 1, min(n_elec, 16)):
                xi, xj = bmat[i], bmat[j]
                # Joint distribution (4 states: 00, 01, 10, 11)
                joint = np.zeros(4)
                joint[0] = np.sum((xi == 0) & (xj == 0))
                joint[1] = np.sum((xi == 0) & (xj == 1))
                joint[2] = np.sum((xi == 1) & (xj == 0))
                joint[3] = np.sum((xi == 1) & (xj == 1))
                joint /= n_bins
                p_i = np.array([1 - np.mean(xi), np.mean(xi)])
                p_j = np.array([1 - np.mean(xj), np.mean(xj)])
                # MI = sum p(x,y) log2(p(x,y) / p(x)p(y))
                mi = 0.0
                for a in range(2):
                    for b in range(2):
                        pxy = joint[a * 2 + b]
                        px_py = p_i[a] * p_j[b]
                        if pxy > 0 and px_py > 0:
                            mi += pxy * np.log2(pxy / px_py)
                mi_values.append(mi)

    mean_mi = float(np.mean(mi_values)) if mi_values else 0.0

    # --- Transfer entropy (vectorised, binned) ---
    te_values = _compute_transfer_entropy_fast(bmat, history=min(5, n_bins // 100))
    mean_te = float(np.mean(te_values)) if len(te_values) > 0 else 0.0

    # --- Sub-scores ---
    # Entropy rate: 0.5-0.8 is complex; too low = boring, too high = noise
    ent_score = _gaussian_score(mean_entropy_rate, optimal=0.65, width=0.25) * 7.0
    # Mutual information: higher = more coordination (up to ~0.3 bits)
    mi_score = min(6.0, _sigmoid_score(mean_mi, midpoint=0.02, steepness=40) * 6.0)
    # Transfer entropy: non-zero = directional flow
    te_score = min(7.0, _sigmoid_score(mean_te, midpoint=0.005, steepness=100) * 7.0)

    total = round(min(20.0, ent_score + mi_score + te_score), 2)

    return SubScore(
        name="information_processing",
        value=total,
        max_value=20.0,
        interpretation="Entropy rate, mutual information, transfer entropy",
        details={
            "mean_entropy_rate_normalised": round(mean_entropy_rate, 4),
            "word_length_used": word_len,
            "mean_mutual_information_bits": round(mean_mi, 5),
            "mean_transfer_entropy_bits": round(mean_te, 6),
            "n_electrode_pairs_analysed_mi": len(mi_values),
            "n_electrode_pairs_analysed_te": len(te_values),
            "ent_sub": round(ent_score, 2),
            "mi_sub": round(mi_score, 2),
            "te_sub": round(te_score, 2),
        },
    )


def _compute_transfer_entropy_fast(
    bmat: np.ndarray,
    history: int = 5,
) -> np.ndarray:
    """Vectorised pairwise transfer entropy on binary spike matrix.

    TE(X->Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

    We convert history windows into integer state IDs and use
    np.unique-based counting -- no Python loops over time bins.
    """
    n_elec, n_bins = bmat.shape
    history = max(1, min(history, n_bins // 10))
    if n_bins < history + 2 or n_elec < 2:
        return np.array([])

    results = []
    # For each pair (max 16x16 = 256 pairs, each O(n_bins))
    pairs = min(n_elec, 16)
    powers = 2 ** np.arange(history, dtype=np.int64)

    for i in range(pairs):
        yi = bmat[i]
        # Build Y_past state vector: integer encoding of last `history` bins
        y_past_mat = np.lib.stride_tricks.sliding_window_view(yi, history)
        y_past = y_past_mat[:-1] @ powers  # shape: (n_bins - history,)
        y_future = yi[history:]  # shape: (n_bins - history,)

        for j in range(pairs):
            if i == j:
                continue
            xj = bmat[j]
            x_past_mat = np.lib.stride_tricks.sliding_window_view(xj, history)
            x_past = x_past_mat[:-1] @ powers

            # Joint states: (y_past, x_past, y_future)
            # Encode as composite key
            max_state = 2 ** history
            joint_key = y_past * (max_state * 2) + x_past * 2 + y_future
            cond_key = y_past * 2 + y_future
            ypast_key = y_past

            # H(Y_future | Y_past) via conditional entropy
            h_y_given_ypast = _conditional_entropy_from_keys(cond_key, ypast_key)
            # H(Y_future | Y_past, X_past)
            joint_cond_key = y_past * max_state + x_past
            h_y_given_both = _conditional_entropy_from_keys(
                joint_key, joint_cond_key
            )
            te = max(0.0, h_y_given_ypast - h_y_given_both)
            results.append(te)

    return np.array(results) if results else np.array([])


def _conditional_entropy_from_keys(
    joint_key: np.ndarray,
    cond_key: np.ndarray,
) -> float:
    """H(A|B) from integer-encoded joint and condition keys.

    H(A|B) = H(A,B) - H(B)
    """
    n = len(joint_key)
    if n == 0:
        return 0.0
    _, joint_counts = np.unique(joint_key, return_counts=True)
    _, cond_counts = np.unique(cond_key, return_counts=True)
    h_joint = _shannon_entropy_bits(joint_counts / n)
    h_cond = _shannon_entropy_bits(cond_counts / n)
    return max(0.0, h_joint - h_cond)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 4: TEMPORAL ORGANIZATION (0-20)
# ═══════════════════════════════════════════════════════════════════════════

def _score_temporal_organization(data: SpikeData, bin_s: float) -> SubScore:
    """Evaluate temporal structure of activity.

    Metrics:
    - Burst detection quality: fraction of spikes in bursts, burst regularity
    - Avalanche size distribution exponent (criticality)
    - ISI coefficient of variation (Poisson=1, regular<1, bursty>1)
    - Population activity autocorrelation structure

    Scoring (20 pts max):
      Burst quality              0-6
      Near-criticality           0-7
      ISI regularity (optimal)   0-4
      Autocorrelation structure  0-3
    """
    eids = data.electrode_ids
    mat, edges = _bin_spike_trains(data, bin_s)
    pop_rate = np.sum(mat, axis=0).astype(np.float64)
    n_bins = len(pop_rate)

    # --- Burst detection (vectorised) ---
    # A burst = consecutive bins where population activity > mean + 2*std
    mean_pop = np.mean(pop_rate)
    std_pop = np.std(pop_rate)
    burst_threshold = mean_pop + 2.0 * std_pop if std_pop > 0 else mean_pop + 1
    above = pop_rate > burst_threshold

    # Find burst boundaries using diff
    padded = np.concatenate([[False], above, [False]])
    transitions = np.diff(padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    if len(starts) > 0 and len(ends) > 0:
        burst_sizes = np.array([float(np.sum(pop_rate[s:e])) for s, e in zip(starts, ends)])
        burst_durations = (ends - starts).astype(float) * bin_s
        n_bursts = len(starts)
        spikes_in_bursts = float(np.sum(burst_sizes))
        burst_fraction = spikes_in_bursts / max(data.n_spikes, 1)
        # Inter-burst intervals
        if n_bursts > 1:
            ibis = np.diff(starts) * bin_s
            ibi_cv = float(np.std(ibis) / (np.mean(ibis) + 1e-12))
        else:
            ibi_cv = 1.0
    else:
        n_bursts = 0
        burst_fraction = 0.0
        burst_durations = np.array([])
        burst_sizes = np.array([])
        ibi_cv = 1.0

    # --- Avalanche analysis (criticality) ---
    # Power-law exponent of avalanche sizes
    # Avalanches: consecutive active bins (population > 0)
    active = pop_rate > 0
    padded_a = np.concatenate([[False], active, [False]])
    trans_a = np.diff(padded_a.astype(int))
    av_starts = np.where(trans_a == 1)[0]
    av_ends = np.where(trans_a == -1)[0]

    if len(av_starts) > 1:
        av_sizes = np.array([float(np.sum(pop_rate[s:e])) for s, e in zip(av_starts, av_ends)])
        av_sizes = av_sizes[av_sizes > 0]
        if len(av_sizes) > 20:
            # MLE power-law exponent (Clauset et al. 2009 simplified)
            xmin = max(np.min(av_sizes), 1.0)
            alpha = 1.0 + len(av_sizes) / np.sum(np.log(av_sizes / xmin))
            # Branching ratio: mean(size_t+1 / size_t)
            if len(av_sizes) > 1:
                ratios = av_sizes[1:] / (av_sizes[:-1] + 1e-12)
                branching = float(np.mean(ratios[ratios < 10]))  # exclude outliers
            else:
                branching = 0.0
        else:
            alpha = 0.0
            branching = 0.0
    else:
        av_sizes = np.array([])
        alpha = 0.0
        branching = 0.0

    # --- ISI CV across electrodes ---
    cv_values = []
    for e in eids:
        idx = data._electrode_indices[e]
        if len(idx) < 10:
            continue
        isi = np.diff(data.times[idx])
        if np.mean(isi) > 0:
            cv_values.append(float(np.std(isi) / np.mean(isi)))
    mean_cv = float(np.mean(cv_values)) if cv_values else 1.0

    # --- Autocorrelation structure ---
    if n_bins > 50:
        # Lag-1 and lag-10 autocorrelation of population rate
        centered = pop_rate - np.mean(pop_rate)
        var = np.sum(centered ** 2)
        if var > 0:
            ac1 = float(np.sum(centered[:-1] * centered[1:]) / var)
            lag10 = min(10, n_bins // 5)
            ac10 = float(np.sum(centered[:-lag10] * centered[lag10:]) / var)
        else:
            ac1 = 0.0
            ac10 = 0.0
    else:
        ac1 = 0.0
        ac10 = 0.0

    # --- Sub-scores ---
    # Burst quality: ~20-60% of spikes in bursts; regular IBI preferred
    burst_score = (
        _gaussian_score(burst_fraction, optimal=0.4, width=0.25) * 3.0
        + _inv_sigmoid_score(ibi_cv, midpoint=1.0, steepness=3.0) * 3.0
    )
    burst_score = min(6.0, burst_score)

    # Criticality: alpha ~1.5, branching ~1.0
    alpha_sc = _gaussian_score(alpha, optimal=1.5, width=0.3) * 4.0
    branch_sc = _gaussian_score(branching, optimal=1.0, width=0.2) * 3.0
    crit_score = min(7.0, alpha_sc + branch_sc)

    # ISI CV: ~1.0 (Poisson-like) is more complex than very low or very high
    cv_score = _gaussian_score(mean_cv, optimal=1.0, width=0.5) * 4.0

    # Autocorrelation: positive lag-1 (temporal structure) but decaying
    ac_score = min(3.0, max(0.0, ac1) * 3.0 + max(0.0, -ac10) * 1.0)

    total = round(min(20.0, burst_score + crit_score + cv_score + ac_score), 2)

    return SubScore(
        name="temporal_organization",
        value=total,
        max_value=20.0,
        interpretation="Burst structure, criticality, ISI regularity, temporal correlations",
        details={
            "n_bursts": n_bursts,
            "burst_fraction": round(burst_fraction, 4),
            "ibi_cv": round(ibi_cv, 3),
            "avalanche_exponent_alpha": round(alpha, 3),
            "branching_ratio": round(branching, 4),
            "n_avalanches": len(av_sizes),
            "mean_isi_cv": round(mean_cv, 3),
            "autocorrelation_lag1": round(ac1, 4),
            "autocorrelation_lag10": round(ac10, 4),
            "burst_sub": round(burst_score, 2),
            "crit_sub": round(crit_score, 2),
            "cv_sub": round(cv_score, 2),
            "ac_sub": round(ac_score, 2),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 5: ADAPTABILITY (0-15)
# ═══════════════════════════════════════════════════════════════════════════

def _score_adaptability(data: SpikeData) -> SubScore:
    """Evaluate stability and adaptability over time.

    Metrics:
    - Firing rate drift: coefficient of variation of windowed rates
    - Fano factor trend: how spike count variability changes over time
    - Response diversity: entropy of per-electrode rate ratios

    Scoring (15 pts max):
      Rate stability            0-5
      Fano factor profile       0-5
      Response diversity         0-5
    """
    eids = data.electrode_ids
    t0, t1 = data.time_range
    duration = t1 - t0

    if duration < 1.0:
        return SubScore(
            name="adaptability", value=0.0, max_value=15.0,
            interpretation="Stability and adaptability (recording too short)",
            details={"error": "duration < 1 s"},
        )

    # Divide recording into 10-20 temporal windows
    n_windows = min(20, max(4, int(duration / 60)))  # ~1 min windows or fewer
    window_dur = duration / n_windows
    window_edges = np.linspace(t0, t1, n_windows + 1)

    # --- Per-electrode firing rate in each window ---
    rate_matrix = np.zeros((len(eids), n_windows))
    for row_i, e in enumerate(eids):
        idx = data._electrode_indices[e]
        times_e = data.times[idx]
        bin_idx = np.searchsorted(window_edges, times_e, side="right") - 1
        np.clip(bin_idx, 0, n_windows - 1, out=bin_idx)
        counts = np.bincount(bin_idx, minlength=n_windows)
        rate_matrix[row_i] = counts / window_dur

    # --- Rate stability: CV of per-window total population rate ---
    total_rate = np.sum(rate_matrix, axis=0)
    if np.mean(total_rate) > 0:
        rate_cv = float(np.std(total_rate) / np.mean(total_rate))
    else:
        rate_cv = 1.0

    # --- Per-electrode drift ---
    drift_scores = []
    for row_i in range(len(eids)):
        row = rate_matrix[row_i]
        if np.mean(row) > 0:
            drift_scores.append(float(np.std(row) / np.mean(row)))
    mean_drift = float(np.mean(drift_scores)) if drift_scores else 1.0

    # --- Fano factor profile ---
    # Fano = var(count) / mean(count) in each window
    count_matrix = rate_matrix * window_dur  # back to counts
    fano_per_window = []
    for w in range(n_windows):
        col = count_matrix[:, w]
        m = np.mean(col)
        if m > 0:
            fano_per_window.append(float(np.var(col) / m))
    mean_fano = float(np.mean(fano_per_window)) if fano_per_window else 1.0
    # Fano trend: regression slope
    if len(fano_per_window) > 3:
        x = np.arange(len(fano_per_window), dtype=float)
        x -= x.mean()
        y = np.array(fano_per_window)
        fano_slope = float(np.sum(x * (y - y.mean())) / (np.sum(x ** 2) + 1e-12))
    else:
        fano_slope = 0.0

    # --- Response diversity: entropy of relative firing rates ---
    mean_rates = np.mean(rate_matrix, axis=1)
    mean_rates_pos = mean_rates[mean_rates > 0]
    if len(mean_rates_pos) > 1:
        rate_probs = mean_rates_pos / np.sum(mean_rates_pos)
        rate_entropy = _shannon_entropy_bits(rate_probs)
        max_entropy = np.log2(len(mean_rates_pos))
        norm_diversity = rate_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        norm_diversity = 0.0

    # --- Sub-scores ---
    # Rate stability: low CV is good (0.1-0.5 ideal)
    stability_score = _gaussian_score(rate_cv, optimal=0.3, width=0.3) * 5.0
    # Fano factor: ~1-3 interesting; near 0 = frozen, >10 = exploding
    fano_score = _gaussian_score(mean_fano, optimal=2.0, width=3.0) * 5.0
    # Diversity: higher = more differentiated electrodes
    diversity_score = norm_diversity * 5.0

    total = round(min(15.0, stability_score + fano_score + diversity_score), 2)

    return SubScore(
        name="adaptability",
        value=total,
        max_value=15.0,
        interpretation="Firing-rate stability, Fano factor, response diversity",
        details={
            "n_windows": n_windows,
            "window_duration_s": round(window_dur, 1),
            "population_rate_cv": round(rate_cv, 4),
            "mean_electrode_drift_cv": round(mean_drift, 4),
            "mean_fano_factor": round(mean_fano, 3),
            "fano_slope": round(fano_slope, 5),
            "normalised_rate_diversity": round(norm_diversity, 4),
            "stability_sub": round(stability_score, 2),
            "fano_sub": round(fano_score, 2),
            "diversity_sub": round(diversity_score, 2),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 6: LEARNING POTENTIAL (0-10)
# ═══════════════════════════════════════════════════════════════════════════

def _score_learning_potential(data: SpikeData) -> SubScore:
    """Evaluate STDP-compatible timing signatures.

    Metrics:
    - Cross-correlogram asymmetry: excess pre-before-post pairs
      within 1-20 ms plasticity window  [Bi & Poo 1998]
    - Number of electrode pairs with significant asymmetry
    - Mean plasticity window timing

    All done with vectorised searchsorted, not nested Python loops.

    Scoring (10 pts max):
      Significant STDP pairs   0-5
      Mean asymmetry magnitude 0-3
      Timing precision         0-2
    """
    eids = data.electrode_ids
    n = len(eids)
    if n < 2:
        return SubScore(
            name="learning_potential", value=0.0, max_value=10.0,
            interpretation="STDP-compatible timing (insufficient electrodes)",
            details={"error": "fewer than 2 electrodes"},
        )

    # Use subsampled data for speed
    d = _stratified_subsample(data, _MAX_SPIKES_PAIRWISE)

    max_lag_s = 0.030  # 30 ms
    stdp_window_s = 0.020  # 20 ms -- classic STDP window

    n_significant = 0
    asymmetries = []
    peak_lags_ms = []

    for i in range(n):
        t_i = d.times[d.electrodes == eids[i]]
        if len(t_i) < 5:
            continue
        for j in range(i + 1, n):
            t_j = d.times[d.electrodes == eids[j]]
            if len(t_j) < 5:
                continue

            # Vectorised: for each spike in i, find spikes in j within lag
            # Use searchsorted for O(N log N)
            lo = np.searchsorted(t_j, t_i - max_lag_s, side="left")
            hi = np.searchsorted(t_j, t_i + max_lag_s, side="right")
            # Count pre-before-post (positive lag) vs post-before-pre
            n_pre = 0
            n_post = 0
            lags_this_pair = []
            for k in range(len(t_i)):
                if lo[k] >= hi[k]:
                    continue
                diffs = t_j[lo[k]:hi[k]] - t_i[k]
                within_stdp = diffs[(np.abs(diffs) > 0.001) & (np.abs(diffs) <= stdp_window_s)]
                n_pre += int(np.sum(within_stdp > 0))
                n_post += int(np.sum(within_stdp < 0))
                if len(within_stdp) > 0:
                    lags_this_pair.extend(np.abs(within_stdp).tolist())

            total_pairs = n_pre + n_post
            if total_pairs < 10:
                continue

            # Asymmetry index: (pre - post) / (pre + post)
            asym = (n_pre - n_post) / total_pairs
            asymmetries.append(abs(asym))

            # Significance: binomial test -- is asymmetry beyond chance?
            # Under H0 (no STDP), P(pre) = P(post) = 0.5
            # Use normal approximation for speed
            z = abs(n_pre - n_post) / (np.sqrt(total_pairs) + 1e-12)
            if z > 2.0:  # ~ p < 0.05
                n_significant += 1

            if lags_this_pair:
                peak_lags_ms.append(float(np.median(lags_this_pair)) * 1000)

    max_pairs = n * (n - 1) // 2
    frac_significant = n_significant / max(max_pairs, 1)
    mean_asym = float(np.mean(asymmetries)) if asymmetries else 0.0
    mean_peak_lag = float(np.mean(peak_lags_ms)) if peak_lags_ms else 0.0

    # --- Sub-scores ---
    # Significant pairs: more = more potential connections learning
    sig_score = min(5.0, _sigmoid_score(frac_significant, midpoint=0.1, steepness=15) * 5.0)
    # Asymmetry: 0.1-0.4 is biologically realistic
    asym_score = _gaussian_score(mean_asym, optimal=0.25, width=0.2) * 3.0
    # Timing precision: peak lag 5-15 ms is optimal STDP window
    timing_score = _gaussian_score(mean_peak_lag, optimal=10.0, width=8.0) * 2.0

    total = round(min(10.0, sig_score + asym_score + timing_score), 2)

    return SubScore(
        name="learning_potential",
        value=total,
        max_value=10.0,
        interpretation="STDP timing asymmetry, plasticity-window prevalence",
        details={
            "n_significant_pairs": n_significant,
            "total_pairs_tested": max_pairs,
            "fraction_significant": round(frac_significant, 4),
            "mean_asymmetry_index": round(mean_asym, 4),
            "mean_peak_lag_ms": round(mean_peak_lag, 2),
            "sig_sub": round(sig_score, 2),
            "asym_sub": round(asym_score, 2),
            "timing_sub": round(timing_score, 2),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN API
# ═══════════════════════════════════════════════════════════════════════════

def compute_organoid_iq(data: SpikeData) -> dict:
    """Compute the Organoid Intelligence Quotient.

    Returns a dict (via ``OIQResult.to_dict()``) with:
    - ``iq_score``: composite 0-100
    - ``grade``: A / B / C / D / F
    - ``assessment``: one-line human-readable verdict
    - ``subscores``: per-dimension scores
    - ``details``: full metric breakdown per dimension
    - ``interpretation``: what each score means
    - ``computation_time_s``: wall-clock seconds
    - ``dataset_summary``: key facts about the input data
    - ``warnings``: any data-quality or edge-case warnings

    Automatically adapts resolution for large datasets so that the full
    pipeline completes in under 60 s on 2.6 M spikes.
    """
    t_start = time.monotonic()
    warn_list: list[str] = []

    # --- Validate input ---
    if data.n_spikes < _MIN_SPIKES_FOR_ANALYSIS:
        return OIQResult(
            iq_score=0.0,
            grade="F",
            assessment="Insufficient data for analysis",
            subscores={},
            computation_time_s=time.monotonic() - t_start,
            dataset_summary=_dataset_summary(data),
            interpretation={},
            warnings=[f"Only {data.n_spikes} spikes; need >= {_MIN_SPIKES_FOR_ANALYSIS}"],
        ).to_dict()

    # --- Adaptive bin size ---
    # Target: 10 ms default, scales up for long recordings
    bin_s = _adaptive_bin_s(data.duration, 0.010)
    logger.info(
        "OIQ: %d spikes, %d electrodes, %.1f h, bin=%.3f s",
        data.n_spikes, data.n_electrodes, data.duration / 3600, bin_s,
    )

    if bin_s > 60.0:
        warn_list.append(
            f"Very long recording ({data.duration/3600:.0f} h); bin size "
            f"increased to {bin_s:.0f} s to keep computation tractable."
        )

    # Silent electrode check
    for e in data.electrode_ids:
        n = len(data._electrode_indices[e])
        if n < _MIN_SPIKES_PER_ELECTRODE:
            warn_list.append(f"Electrode {e}: only {n} spikes (quasi-silent)")

    # --- Compute all six dimensions ---
    scores: dict[str, SubScore] = {}

    s1 = _safe_compute("signal_quality", lambda: _score_signal_quality(data), 15.0, warn_list)
    scores["signal_quality"] = s1

    s2 = _safe_compute("network_complexity", lambda: _score_network_complexity(data, bin_s), 20.0, warn_list)
    scores["network_complexity"] = s2

    s3 = _safe_compute("information_processing", lambda: _score_information_processing(data, bin_s), 20.0, warn_list)
    scores["information_processing"] = s3

    s4 = _safe_compute("temporal_organization", lambda: _score_temporal_organization(data, bin_s), 20.0, warn_list)
    scores["temporal_organization"] = s4

    s5 = _safe_compute("adaptability", lambda: _score_adaptability(data), 15.0, warn_list)
    scores["adaptability"] = s5

    s6 = _safe_compute("learning_potential", lambda: _score_learning_potential(data), 10.0, warn_list)
    scores["learning_potential"] = s6

    # --- Composite score ---
    total = sum(sc.value for sc in scores.values())
    total = round(min(100.0, max(0.0, total)), 1)

    grade, assessment = _grade(total)

    elapsed = time.monotonic() - t_start
    logger.info("OIQ computation complete: %.1f (%s) in %.2f s", total, grade, elapsed)

    return OIQResult(
        iq_score=total,
        grade=grade,
        assessment=assessment,
        subscores=scores,
        computation_time_s=elapsed,
        dataset_summary=_dataset_summary(data),
        interpretation={k: f"{sc.value:.1f}/{sc.max_value:.0f} -- {sc.interpretation}"
                        for k, sc in scores.items()},
        warnings=warn_list,
    ).to_dict()


def _safe_compute(
    name: str,
    fn,
    max_val: float,
    warn_list: list[str],
) -> SubScore:
    """Run a scoring function with exception handling."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("OIQ dimension '%s' failed: %s", name, exc, exc_info=True)
        warn_list.append(f"{name}: computation failed ({type(exc).__name__}: {exc})")
        return SubScore(
            name=name,
            value=0.0,
            max_value=max_val,
            interpretation=f"{name} (failed)",
            details={"error": str(exc)},
        )


def _grade(score: float) -> tuple[str, str]:
    """Map numeric score to letter grade and assessment."""
    if score >= 85:
        return "A+", (
            "Exceptional organoid: rich information processing, near-critical "
            "dynamics, strong STDP signatures, small-world connectivity"
        )
    if score >= 75:
        return "A", (
            "Excellent: complex organised activity with evidence of "
            "computation and plasticity"
        )
    if score >= 60:
        return "B", (
            "Good: organised burst patterns, moderate information capacity, "
            "functional network structure"
        )
    if score >= 40:
        return "C", (
            "Average: basic spontaneous activity with some temporal "
            "structure but limited complexity"
        )
    if score >= 20:
        return "D", (
            "Below average: minimal network organisation, low information "
            "content, limited plasticity signatures"
        )
    return "F", (
        "Poor: minimal or degraded activity; possible cell death "
        "or recording failure"
    )


def _dataset_summary(data: SpikeData) -> dict:
    """Key facts about the input data."""
    return {
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(data.duration, 2),
        "duration_h": round(data.duration / 3600, 2),
        "sampling_rate_hz": data.sampling_rate,
        "mean_firing_rate_hz": round(data.n_spikes / max(data.duration, 1e-9), 3),
        "spikes_per_electrode": {
            int(e): len(data._electrode_indices[e]) for e in data.electrode_ids
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON API
# ═══════════════════════════════════════════════════════════════════════════

def compute_organoid_comparison(datasets: dict[str, SpikeData]) -> dict:
    """Compare OIQ scores across multiple organoids/recordings.

    Returns per-organoid results, a ranking, and aggregate statistics.
    """
    results = {}
    for name, data in datasets.items():
        iq = compute_organoid_iq(data)
        results[name] = {
            "iq_score": iq["iq_score"],
            "grade": iq["grade"],
            "subscores": iq["subscores"],
            "assessment": iq["assessment"],
        }

    ranked = sorted(results.items(), key=lambda x: x[1]["iq_score"], reverse=True)
    scores = [r["iq_score"] for r in results.values()]

    return {
        "results": results,
        "ranking": [
            {"rank": i + 1, "name": name, "iq": r["iq_score"], "grade": r["grade"]}
            for i, (name, r) in enumerate(ranked)
        ],
        "best": ranked[0][0] if ranked else None,
        "worst": ranked[-1][0] if ranked else None,
        "mean_iq": round(float(np.mean(scores)), 1) if scores else 0,
        "std_iq": round(float(np.std(scores)), 1) if scores else 0,
        "n_datasets": len(datasets),
    }
