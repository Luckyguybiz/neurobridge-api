"""Metastability analysis — brain-like dynamic switching between states.

Measures Kuramoto order parameter (global synchronization),
metastability index, and state transition dynamics.
"""
import numpy as np
from .loader import SpikeData

def compute_kuramoto_parameter(data: SpikeData, bin_size_ms: float = 20.0) -> dict:
    """Compute Kuramoto order parameter — measure of global synchronization."""
    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)
    n_bins = len(bins) - 1

    if n_bins < 2:
        return {"kuramoto_mean": 0.0, "metastability_index": 0.0}

    # Compute per-electrode rates
    rates = np.zeros((data.n_electrodes, n_bins))
    for e_idx, e_id in enumerate(data.electrode_ids):
        mask = data.electrodes == e_id
        e_times = data.times[mask]
        counts, _ = np.histogram(e_times, bins=bins)
        rates[e_idx] = counts

    # Hilbert transform to get instantaneous phase
    from scipy.signal import hilbert
    phases = np.zeros_like(rates)
    for i in range(data.n_electrodes):
        if np.std(rates[i]) > 0:
            analytic = hilbert(rates[i] - np.mean(rates[i]))
            phases[i] = np.angle(analytic)

    # Kuramoto order parameter R(t) = |1/N * sum(exp(j*phi_i))|
    R = np.abs(np.mean(np.exp(1j * phases), axis=0))

    return {
        "kuramoto_mean": float(np.mean(R)),
        "kuramoto_std": float(np.std(R)),
        "metastability_index": float(np.std(R)),  # variance of R = metastability
        "kuramoto_max": float(np.max(R)),
        "kuramoto_min": float(np.min(R)),
        "synchronization_level": "high" if np.mean(R) > 0.7 else "moderate" if np.mean(R) > 0.4 else "low",
        "kuramoto_timeseries": R.tolist()[:500],
        "time_bins": bins[:501].tolist(),
    }

def compute_state_transitions(data: SpikeData, n_states: int = 4, bin_size_ms: float = 50.0) -> dict:
    """Compute state transition matrix using K-means clustering."""
    from sklearn.cluster import KMeans

    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)
    n_bins = len(bins) - 1

    if n_bins < n_states * 2:
        return {"n_states": 0, "reason": "insufficient data"}

    rates = np.zeros((n_bins, data.n_electrodes))
    for e_idx, e_id in enumerate(data.electrode_ids):
        mask = data.electrodes == e_id
        counts, _ = np.histogram(data.times[mask], bins=bins)
        rates[:, e_idx] = counts

    # Cluster into states
    n_states = min(n_states, n_bins // 2)
    km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    labels = km.fit_predict(rates)

    # Transition matrix
    trans = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i+1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans, row_sums, where=row_sums > 0, out=np.zeros_like(trans))

    # Dwell times
    dwell_times = {s: [] for s in range(n_states)}
    current_state = labels[0]
    current_dwell = 1
    for i in range(1, len(labels)):
        if labels[i] == current_state:
            current_dwell += 1
        else:
            dwell_times[current_state].append(current_dwell * bin_size_ms)
            current_state = labels[i]
            current_dwell = 1
    dwell_times[current_state].append(current_dwell * bin_size_ms)

    return {
        "n_states": n_states,
        "transition_matrix": trans_prob.tolist(),
        "state_labels": labels.tolist()[:500],
        "dwell_times": {str(k): v for k, v in dwell_times.items()},
        "mean_dwell_ms": {str(k): float(np.mean(v)) if v else 0.0 for k, v in dwell_times.items()},
        "n_transitions": int(np.sum(np.diff(labels) != 0)),
    }

def analyze_metastability(data: SpikeData) -> dict:
    """Full metastability analysis."""
    kuramoto = compute_kuramoto_parameter(data)
    transitions = compute_state_transitions(data)

    return {
        "kuramoto": kuramoto,
        "state_transitions": transitions,
        "is_metastable": kuramoto["metastability_index"] > 0.15,
    }
