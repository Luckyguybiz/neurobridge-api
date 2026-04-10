"""Homeostatic plasticity monitoring.

Tracks whether the organoid maintains stable firing rates through
synaptic scaling. Biological networks regulate their own excitability
to prevent runaway excitation or silence.
"""
import numpy as np
from .loader import SpikeData


def monitor_homeostasis(data: SpikeData, window_sec: float = 5.0) -> dict:
    """Monitor homeostatic regulation of firing rates over time."""
    bins = np.arange(0, data.duration, window_sec)
    n_windows = len(bins) - 1
    if n_windows < 3:
        return {"homeostasis_active": False, "reason": "recording too short"}

    rates = np.zeros((data.n_electrodes, n_windows))
    for idx, eid in enumerate(data.electrode_ids):
        mask = data.electrodes == eid
        for w in range(n_windows):
            t_mask = (data.times[mask] >= bins[w]) & (data.times[mask] < bins[w + 1])
            rates[idx, w] = float(np.sum(t_mask)) / window_sec

    # Homeostasis = firing rates return to baseline after perturbations
    mean_rates = np.mean(rates, axis=1)
    rate_cv_over_time = np.std(rates, axis=1) / np.maximum(mean_rates, 0.01)
    mean_cv = float(np.mean(rate_cv_over_time))

    # Trend: is the network drifting or stable?
    population_rate = np.mean(rates, axis=0)
    if len(population_rate) > 2:
        trend_coeff = float(np.polyfit(np.arange(n_windows), population_rate, 1)[0])
    else:
        trend_coeff = 0.0

    # Stability score
    stability = 1.0 / (1.0 + mean_cv)
    homeostasis_active = mean_cv < 0.5 and abs(trend_coeff) < 1.0

    # Detect compensation events (rate drops then recovers)
    compensations = 0
    for i in range(1, n_windows - 1):
        if population_rate[i] < population_rate[i-1] * 0.5 and population_rate[i+1] > population_rate[i] * 1.3:
            compensations += 1

    return {
        "homeostasis_active": bool(homeostasis_active),
        "stability_score": float(stability),
        "mean_cv_over_time": mean_cv,
        "trend_slope": trend_coeff,
        "trend_direction": "increasing" if trend_coeff > 0.5 else "decreasing" if trend_coeff < -0.5 else "stable",
        "n_compensation_events": compensations,
        "mean_rates_per_electrode": {str(eid): float(r) for eid, r in zip(data.electrode_ids, mean_rates)},
        "population_rate_timeline": population_rate.tolist()[:100],
        "window_sec": window_sec,
        "n_windows": n_windows,
    }
