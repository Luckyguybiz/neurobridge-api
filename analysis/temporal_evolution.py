"""Temporal evolution tracking module.

Tracks how organoid properties change over long recordings (hours/days).
Uses sliding-window analysis to compute timeseries of key neural metrics,
detect linear trends (improving/degrading/stable), and find critical
moments where metrics change dramatically.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def _compute_window_metrics(window_data: SpikeData, window_duration: float) -> dict:
    """Compute key metrics for a single time window."""
    n = window_data.n_spikes
    if n == 0:
        return {
            "firing_rate": 0.0,
            "burst_rate": 0.0,
            "complexity": 0.0,
            "synchrony": 0.0,
        }

    firing_rate = n / window_duration if window_duration > 0 else 0.0

    # Burst rate: count ISI < 10ms transitions
    burst_count = 0
    if n >= 3:
        isi = np.diff(window_data.times) * 1000  # ms
        in_burst = isi < 10.0
        transitions = np.diff(in_burst.astype(int))
        burst_count = int(np.sum(transitions == 1))
    burst_rate = burst_count / window_duration if window_duration > 0 else 0.0

    # Complexity: ISI coefficient of variation (higher = more complex/irregular)
    complexity = 0.0
    if n >= 2:
        isi = np.diff(window_data.times) * 1000
        if np.mean(isi) > 0:
            complexity = float(np.std(isi) / np.mean(isi))

    # Synchrony: fraction of spikes within 5ms of another electrode's spike
    synchrony = 0.0
    if n >= 2 and window_data.n_electrodes >= 2:
        sync_count = 0
        for i, t in enumerate(window_data.times):
            nearby = np.abs(window_data.times - t) < 0.005  # 5ms
            diff_electrode = window_data.electrodes != window_data.electrodes[i]
            if np.any(nearby & diff_electrode):
                sync_count += 1
        synchrony = sync_count / n if n > 0 else 0.0

    return {
        "firing_rate": round(firing_rate, 4),
        "burst_rate": round(burst_rate, 6),
        "complexity": round(complexity, 4),
        "synchrony": round(synchrony, 4),
    }


def track_evolution(
    data: SpikeData,
    window_sec: float = 60.0,
    step_sec: Optional[float] = None,
) -> dict:
    """Compute metrics in sliding windows over the full recording.

    Returns timeseries of firing rate, burst rate, complexity, and synchrony.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    if step_sec is None:
        step_sec = window_sec / 2  # 50% overlap by default

    t_start, t_end = data.time_range
    windows: list[dict] = []
    timestamps: list[float] = []
    metrics_series = {"firing_rate": [], "burst_rate": [], "complexity": [], "synchrony": []}

    t = t_start
    while t + window_sec <= t_end:
        window_data = data.get_time_range(t, t + window_sec)
        metrics = _compute_window_metrics(window_data, window_sec)

        center = t + window_sec / 2
        timestamps.append(round(center, 2))
        for key in metrics_series:
            metrics_series[key].append(metrics[key])

        windows.append({
            "center_s": round(center, 2),
            "n_spikes": window_data.n_spikes,
            **metrics,
        })
        t += step_sec

    return {
        "n_windows": len(windows),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "timestamps": timestamps,
        "timeseries": metrics_series,
        "windows": windows,
        "duration_covered_s": round(t_end - t_start, 2),
    }


def detect_trends(
    data: SpikeData,
    window_sec: float = 60.0,
    step_sec: Optional[float] = None,
) -> dict:
    """Linear regression on each metric timeseries.

    Determines if each metric is improving, degrading, or stable.
    Uses p < 0.05 for significance.
    """
    evo = track_evolution(data, window_sec, step_sec)
    if "error" in evo:
        return evo

    timestamps = np.array(evo["timestamps"])
    if len(timestamps) < 3:
        return {"error": "Not enough windows for trend analysis", "n_windows": len(timestamps)}

    trends: dict[str, dict] = {}
    for metric_name, values in evo["timeseries"].items():
        y = np.array(values)

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(timestamps, y)

        # Rate of change per hour
        slope_per_hour = slope * 3600

        # Percent change over recording
        initial = intercept + slope * timestamps[0]
        final_pred = intercept + slope * timestamps[-1]
        pct_change = ((final_pred - initial) / abs(initial) * 100) if abs(initial) > 1e-10 else 0.0

        significant = p_value < 0.05

        if not significant:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Biological interpretation
        interpretations = {
            "firing_rate": {
                "increasing": "Organoid becoming more active — possible excitotoxicity risk if extreme",
                "decreasing": "Organoid activity declining — possible degradation or adaptation",
                "stable": "Stable firing rate — healthy homeostasis",
            },
            "burst_rate": {
                "increasing": "More bursty — developing more complex dynamics",
                "decreasing": "Less bursty — possible loss of network structure",
                "stable": "Stable burst patterns",
            },
            "complexity": {
                "increasing": "Increasing ISI variability — richer temporal coding",
                "decreasing": "Becoming more regular — possible pathological state",
                "stable": "Stable complexity",
            },
            "synchrony": {
                "increasing": "Increasing synchronization — stronger network coupling",
                "decreasing": "Desynchronizing — possible network decoupling",
                "stable": "Stable synchrony levels",
            },
        }

        trends[metric_name] = {
            "slope": round(float(slope), 8),
            "slope_per_hour": round(float(slope_per_hour), 6),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": float(p_value),
            "significant": significant,
            "direction": direction,
            "pct_change": round(float(pct_change), 2),
            "interpretation": interpretations.get(metric_name, {}).get(direction, ""),
        }

    # Overall organoid health assessment
    directions = [t["direction"] for t in trends.values()]
    if all(d == "stable" for d in directions):
        overall = "Stable — organoid in steady-state"
    elif trends["firing_rate"]["direction"] == "decreasing":
        overall = "Degrading — firing rate declining"
    elif trends["complexity"]["direction"] == "increasing" and trends["firing_rate"]["direction"] != "decreasing":
        overall = "Improving — developing more complex dynamics"
    else:
        overall = "Mixed — some metrics changing, monitor closely"

    return {
        "trends": trends,
        "overall_assessment": overall,
        "window_sec": window_sec,
        "n_windows": evo["n_windows"],
    }


def find_critical_moments(
    data: SpikeData,
    window_sec: float = 60.0,
    step_sec: Optional[float] = None,
    threshold_std: float = 2.0,
) -> dict:
    """Find time points where metrics change dramatically.

    A critical moment is when a metric deviates >threshold_std standard
    deviations from the running mean. These may indicate state transitions,
    stimulation effects, or pathological events.
    """
    evo = track_evolution(data, window_sec, step_sec)
    if "error" in evo:
        return evo

    timestamps = evo["timestamps"]
    if len(timestamps) < 5:
        return {"error": "Not enough windows for critical moment detection", "n_windows": len(timestamps)}

    critical_moments: list[dict] = []

    for metric_name, values in evo["timeseries"].items():
        arr = np.array(values)
        running_mean = np.convolve(arr, np.ones(5) / 5, mode="same")
        running_std = np.array([
            np.std(arr[max(0, i - 2):min(len(arr), i + 3)])
            for i in range(len(arr))
        ])

        # Avoid division by zero
        running_std = np.maximum(running_std, 1e-10)

        z_scores = (arr - running_mean) / running_std

        for i, z in enumerate(z_scores):
            if abs(z) > threshold_std:
                direction = "spike" if z > 0 else "drop"
                critical_moments.append({
                    "time_s": timestamps[i],
                    "metric": metric_name,
                    "value": float(arr[i]),
                    "running_mean": round(float(running_mean[i]), 4),
                    "z_score": round(float(z), 2),
                    "direction": direction,
                    "severity": "extreme" if abs(z) > 3.0 else "significant",
                })

    # Sort by absolute z-score
    critical_moments.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    # Cluster nearby moments (within 2 windows)
    clusters: list[dict] = []
    used = set()
    for i, cm in enumerate(critical_moments):
        if i in used:
            continue
        cluster = [cm]
        for j, other in enumerate(critical_moments):
            if j != i and j not in used and abs(cm["time_s"] - other["time_s"]) < window_sec * 2:
                cluster.append(other)
                used.add(j)
        used.add(i)
        metrics_involved = list(set(c["metric"] for c in cluster))
        clusters.append({
            "time_s": cm["time_s"],
            "n_metrics_affected": len(metrics_involved),
            "metrics": metrics_involved,
            "max_z_score": max(abs(c["z_score"]) for c in cluster),
            "is_global_event": len(metrics_involved) >= 3,
        })

    return {
        "critical_moments": critical_moments[:50],  # Top 50
        "n_critical_moments": len(critical_moments),
        "clusters": clusters[:20],
        "n_clusters": len(clusters),
        "threshold_std": threshold_std,
        "window_sec": window_sec,
        "has_state_transitions": any(c["is_global_event"] for c in clusters),
    }
