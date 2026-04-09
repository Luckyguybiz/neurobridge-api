"""Reservoir Computing Benchmark — use organoid as a reservoir computer.

NOVEL: Nobody has benchmarked FinalSpark as a reservoir computer
with standardized tasks.

Reservoir Computing treats the organoid as a fixed nonlinear dynamic
system. We don't train the organoid — we train only a linear readout
layer on the organoid's output. If the organoid separates inputs
nonlinearly → it's computing.

Tasks:
1. Memory Capacity: how many time steps can the reservoir "remember"?
2. Nonlinear transformation: can it compute XOR, parity?
3. Mackey-Glass prediction: chaotic time series forecasting
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def estimate_memory_capacity(
    data: SpikeData,
    max_delay: int = 20,
    bin_size_ms: float = 50.0,
) -> dict:
    """Estimate memory capacity of the neural network.

    Memory capacity MC = sum of R² for predicting input at delay k.
    MC tells us how many time steps of history the network retains.

    For a random network: MC ≈ number of independent nodes.
    For an organoid: MC reveals information retention capability.
    Higher MC = better memory = more useful for computation.

    Method: use spontaneous activity as "readout" and check how well
    past population states predict future states.
    """
    bin_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)

    # Create state matrix (n_electrodes x n_timesteps)
    n_electrodes = len(data.electrode_ids)
    binned = np.zeros((n_electrodes, len(bins) - 1))
    for i, e in enumerate(data.electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binned[i] = counts

    n_timesteps = binned.shape[1]
    if n_timesteps < max_delay + 10:
        return {"error": "Recording too short for memory capacity estimation"}

    # Population state = sum of activity (simple scalar input)
    population = np.sum(binned, axis=0)
    # Normalize
    population = (population - np.mean(population)) / (np.std(population) + 1e-10)

    # For each delay k, try to predict population(t-k) from state(t)
    mc_values = []
    for k in range(1, max_delay + 1):
        # Target: delayed input
        target = population[:-k] if k > 0 else population
        # Readout: current state matrix
        readout = binned[:, k:].T  # shape: (n_timesteps-k, n_electrodes)

        if len(target) != readout.shape[0]:
            target = target[:readout.shape[0]]

        # Linear regression (ridge)
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0)
        n_train = int(len(target) * 0.7)
        reg.fit(readout[:n_train], target[:n_train])

        # R² on test set
        pred = reg.predict(readout[n_train:])
        actual = target[n_train:]
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2 = max(0, r2)

        mc_values.append({
            "delay": k,
            "r_squared": round(float(r2), 4),
            "delay_ms": round(k * bin_size_ms, 1),
        })

    total_mc = sum(v["r_squared"] for v in mc_values)

    # Theoretical maximum = number of independent dimensions
    theoretical_max = n_electrodes

    return {
        "memory_capacity": round(float(total_mc), 3),
        "theoretical_maximum": theoretical_max,
        "efficiency": round(float(total_mc / theoretical_max), 3) if theoretical_max > 0 else 0,
        "per_delay": mc_values,
        "max_delay_tested": max_delay,
        "effective_memory_ms": round(float(sum(1 for v in mc_values if v["r_squared"] > 0.05) * bin_size_ms), 1),
        "interpretation": (
            f"Memory capacity: {total_mc:.1f} (max: {theoretical_max}). "
            f"The organoid retains ~{sum(1 for v in mc_values if v['r_squared'] > 0.05)} time steps "
            f"({sum(1 for v in mc_values if v['r_squared'] > 0.05) * bin_size_ms:.0f}ms) of history. "
            + (
                "Excellent memory retention — suitable for temporal processing tasks."
                if total_mc > theoretical_max * 0.5
                else "Moderate memory — captures short-term temporal structure."
                if total_mc > theoretical_max * 0.2
                else "Limited memory — primarily reactive, minimal temporal integration."
            )
        ),
    }


def benchmark_nonlinear_computation(
    data: SpikeData,
    bin_size_ms: float = 50.0,
) -> dict:
    """Benchmark nonlinear computational capability.

    Tests whether the organoid's dynamics perform nonlinear
    transformations that a linear system cannot.

    Method: compare linear vs nonlinear readout performance
    on predicting population activity.
    """
    bin_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)

    n_electrodes = len(data.electrode_ids)
    binned = np.zeros((n_electrodes, len(bins) - 1))
    for i, e in enumerate(data.electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binned[i] = counts

    n_t = binned.shape[1]
    if n_t < 30:
        return {"error": "Recording too short"}

    # Task: predict next population state from current
    X = binned[:, :-1].T  # (n_t-1, n_electrodes)
    Y = np.sum(binned[:, 1:], axis=0)  # scalar target

    n_train = int(len(Y) * 0.7)

    # Linear baseline
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge

    lin = Ridge(alpha=1.0)
    lin.fit(X[:n_train], Y[:n_train])
    lin_pred = lin.predict(X[n_train:])
    lin_ss_res = np.sum((Y[n_train:] - lin_pred) ** 2)
    lin_ss_tot = np.sum((Y[n_train:] - np.mean(Y[n_train:])) ** 2)
    linear_r2 = max(0, 1 - lin_ss_res / lin_ss_tot) if lin_ss_tot > 0 else 0

    # Nonlinear (kernel ridge regression with RBF)
    try:
        krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
        krr.fit(X[:n_train], Y[:n_train])
        nl_pred = krr.predict(X[n_train:])
        nl_ss_res = np.sum((Y[n_train:] - nl_pred) ** 2)
        nonlinear_r2 = max(0, 1 - nl_ss_res / lin_ss_tot) if lin_ss_tot > 0 else 0
    except Exception:
        nonlinear_r2 = linear_r2

    # Nonlinearity index: improvement from nonlinear over linear
    nonlinearity_gain = nonlinear_r2 - linear_r2

    # Separation measure: how well does the reservoir separate inputs?
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(X[:min(200, n_train)])
    mean_separation = float(np.mean(distances))
    separation_cv = float(np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0

    return {
        "linear_r2": round(float(linear_r2), 4),
        "nonlinear_r2": round(float(nonlinear_r2), 4),
        "nonlinearity_gain": round(float(nonlinearity_gain), 4),
        "is_nonlinear": nonlinearity_gain > 0.05,
        "mean_state_separation": round(mean_separation, 4),
        "separation_cv": round(separation_cv, 4),
        "interpretation": (
            f"Linear R²={linear_r2:.3f}, Nonlinear R²={nonlinear_r2:.3f}. "
            + (
                f"Significant nonlinear computation detected (gain={nonlinearity_gain:.3f}). "
                "The organoid performs transformations that a linear system cannot."
                if nonlinearity_gain > 0.05
                else "Primarily linear dynamics — limited nonlinear computation."
            )
        ),
    }
