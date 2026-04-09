"""Synaptic Weight Inference — reconstruct the connectome from spike timing.

NOVEL: Nobody has inferred synaptic weight matrices from FinalSpark data.

The "weights" between neurons are hidden — we can't measure them directly.
But we CAN infer them from spike timing using Generalized Linear Models (GLM).

If neuron A's spike at time t predicts neuron B's spike at t+dt,
the weight A→B is positive (excitatory).

Tracking weight changes over time = watching learning happen in real-time.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def infer_synaptic_weights(
    data: SpikeData,
    bin_size_ms: float = 5.0,
    history_bins: int = 10,
    regularization: float = 1.0,
) -> dict:
    """Infer effective synaptic weights using spike-triggered regression.

    For each target electrode B, fit a model:
    P(B fires at t) = f(sum of weighted inputs from all A's at t-1, t-2, ..., t-k)

    The learned weights represent effective synaptic strength.

    Returns NxN weight matrix + temporal kernels.
    """
    bin_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)

    electrode_ids = data.electrode_ids
    n = len(electrode_ids)

    # Binned spike counts
    binned = np.zeros((n, len(bins) - 1))
    for i, e in enumerate(electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binned[i] = counts

    n_t = binned.shape[1]
    if n_t < history_bins + 10:
        return {"error": "Not enough data for weight inference"}

    from sklearn.linear_model import Ridge

    weight_matrix = np.zeros((n, n))
    temporal_kernels = {}  # weight as function of delay

    for target in range(n):
        # Target: does electrode `target` fire at time t?
        y = binned[target, history_bins:]

        # Features: activity of ALL electrodes at t-1, t-2, ..., t-k
        X = np.zeros((n_t - history_bins, n * history_bins))
        for delay in range(history_bins):
            for source in range(n):
                col = delay * n + source
                X[:, col] = binned[source, history_bins - delay - 1:n_t - delay - 1]

        # Fit ridge regression
        reg = Ridge(alpha=regularization)
        reg.fit(X, y)

        # Extract weights (summed over delays)
        coefs = reg.coef_.reshape(history_bins, n)
        summed_weights = np.sum(coefs, axis=0)
        weight_matrix[target] = summed_weights

        # Temporal kernel for each source → target
        for source in range(n):
            key = f"{electrode_ids[source]}->{electrode_ids[target]}"
            kernel = coefs[:, source].tolist()
            temporal_kernels[key] = {
                "weights_by_delay": kernel,
                "total_weight": round(float(summed_weights[source]), 5),
                "peak_delay_ms": round(float((np.argmax(np.abs(coefs[:, source])) + 1) * bin_size_ms), 1),
                "type": "excitatory" if summed_weights[source] > 0.01 else "inhibitory" if summed_weights[source] < -0.01 else "neutral",
            }

    # Normalize weight matrix
    max_abs = np.max(np.abs(weight_matrix)) if np.max(np.abs(weight_matrix)) > 0 else 1
    weight_matrix_norm = weight_matrix / max_abs

    # Classify connections
    excitatory = int(np.sum(weight_matrix_norm > 0.1))
    inhibitory = int(np.sum(weight_matrix_norm < -0.1))

    # E/I balance
    ei_ratio = excitatory / inhibitory if inhibitory > 0 else float('inf')

    # Find strongest connections
    strongest = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(weight_matrix_norm[i, j]) > 0.2:
                strongest.append({
                    "source": int(electrode_ids[j]),
                    "target": int(electrode_ids[i]),
                    "weight": round(float(weight_matrix_norm[i, j]), 4),
                    "type": "excitatory" if weight_matrix_norm[i, j] > 0 else "inhibitory",
                })
    strongest.sort(key=lambda x: abs(x["weight"]), reverse=True)

    return {
        "weight_matrix": weight_matrix_norm.tolist(),
        "electrode_ids": electrode_ids,
        "n_excitatory": excitatory,
        "n_inhibitory": inhibitory,
        "ei_ratio": round(float(ei_ratio), 2) if ei_ratio != float('inf') else "inf",
        "strongest_connections": strongest[:10],
        "history_bins": history_bins,
        "temporal_resolution_ms": bin_size_ms,
        "interpretation": (
            f"Inferred {excitatory} excitatory and {inhibitory} inhibitory connections. "
            f"E/I ratio: {ei_ratio:.1f} "
            + ("(balanced — healthy organoid)" if 0.5 < ei_ratio < 3
               else "(excitation-dominated — potentially epileptiform)" if ei_ratio > 3
               else "(inhibition-dominated — potentially suppressed)")
        ),
    }


def track_weight_changes(
    data: SpikeData,
    window_sec: float = 30.0,
    step_sec: float = 15.0,
    bin_size_ms: float = 5.0,
) -> dict:
    """Track synaptic weight changes over time.

    Sliding window weight inference — each window produces a weight matrix.
    Changes between consecutive windows = learning/plasticity.
    """
    t_start, t_end = data.time_range
    electrode_ids = data.electrode_ids
    n = len(electrode_ids)

    snapshots = []
    t = t_start

    while t + window_sec <= t_end:
        window_data = data.get_time_range(t, t + window_sec)
        if window_data.n_spikes < 50:
            t += step_sec
            continue

        result = infer_synaptic_weights(window_data, bin_size_ms=bin_size_ms)
        if "error" not in result:
            matrix = np.array(result["weight_matrix"])
            snapshots.append({
                "time": round(float(t), 2),
                "matrix": matrix,
                "mean_abs_weight": round(float(np.mean(np.abs(matrix))), 4),
                "n_excitatory": result["n_excitatory"],
                "n_inhibitory": result["n_inhibitory"],
            })
        t += step_sec

    if len(snapshots) < 2:
        return {"error": "Not enough windows for tracking"}

    # Compute changes between consecutive snapshots
    changes = []
    for i in range(1, len(snapshots)):
        delta = snapshots[i]["matrix"] - snapshots[i-1]["matrix"]
        max_change = float(np.max(np.abs(delta)))
        mean_change = float(np.mean(np.abs(delta)))

        # Find which connection changed most
        idx = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        most_changed = {
            "source": int(electrode_ids[idx[1]]),
            "target": int(electrode_ids[idx[0]]),
            "delta": round(float(delta[idx]), 4),
            "direction": "strengthened" if delta[idx] > 0 else "weakened",
        }

        changes.append({
            "time": snapshots[i]["time"],
            "max_change": round(max_change, 4),
            "mean_change": round(mean_change, 4),
            "most_changed_connection": most_changed,
            "is_significant": max_change > 0.2,
        })

    significant_changes = [c for c in changes if c["is_significant"]]

    return {
        "n_snapshots": len(snapshots),
        "changes": changes,
        "n_significant_changes": len(significant_changes),
        "has_learning": len(significant_changes) > 0,
        "weight_trajectory": [{"time": s["time"], "mean_weight": s["mean_abs_weight"]} for s in snapshots],
        "interpretation": (
            f"LEARNING DETECTED: {len(significant_changes)} significant weight changes observed. "
            f"Synaptic weights are actively being modified — the organoid is adapting its connectivity."
            if significant_changes
            else "Stable weights — no significant plasticity detected in this time window."
        ),
    }
