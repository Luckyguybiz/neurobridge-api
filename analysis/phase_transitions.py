"""Phase Transition Detection — finding moments when organoid reorganizes.

NOVEL: Nobody detects phase transitions in organoid activity in real-time.

Before learning occurs, neural networks undergo phase transitions —
sudden reorganization of activity patterns. Like water freezing to ice,
the network "snaps" into a new configuration.

Detecting these moments tells us WHEN the organoid is learning,
allowing us to time stimulation for maximum effect.

Methods:
- Order parameter tracking (magnetization analog for neural networks)
- Susceptibility peaks (variance maxima = critical points)
- Change point detection (statistical shifts in activity regime)
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def detect_phase_transitions(
    data: SpikeData,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> dict:
    """Detect phase transitions in neural activity.

    Tracks an "order parameter" (synchrony) over time.
    Phase transitions = moments when order parameter changes abruptly.

    The order parameter: mean pairwise correlation of electrode activity.
    Low correlation = disordered phase (independent neurons)
    High correlation = ordered phase (synchronized firing)
    Transition between them = phase transition = potential learning moment
    """
    t_start, t_end = data.time_range
    bin_sec = 0.01  # 10ms bins within each window

    windows = []
    t = t_start

    while t + window_sec <= t_end:
        window_data = data.get_time_range(t, t + window_sec)

        # Bin spike trains within window
        bins = np.arange(t, t + window_sec + bin_sec, bin_sec)
        n_bins = len(bins) - 1
        n_el = len(data.electrode_ids)

        if n_bins < 5 or window_data.n_spikes < 5:
            t += step_sec
            continue

        binned = np.zeros((n_el, n_bins))
        for i, e in enumerate(data.electrode_ids):
            counts, _ = np.histogram(window_data.times[window_data.electrodes == e], bins=bins)
            binned[i] = counts

        # Order parameter: mean pairwise correlation
        if n_el >= 2:
            corr_matrix = np.corrcoef(binned)
            # Remove diagonal and NaN
            mask = ~np.eye(n_el, dtype=bool)
            valid_corrs = corr_matrix[mask]
            valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
            order_param = float(np.mean(np.abs(valid_corrs))) if len(valid_corrs) > 0 else 0
        else:
            order_param = 0

        # Susceptibility: variance of activity across electrodes
        electrode_rates = np.sum(binned, axis=1)
        susceptibility = float(np.var(electrode_rates)) if len(electrode_rates) > 0 else 0

        # Population firing rate
        pop_rate = float(np.sum(binned)) / window_sec

        windows.append({
            "time": round(float(t), 2),
            "order_parameter": round(order_param, 4),
            "susceptibility": round(susceptibility, 4),
            "population_rate": round(pop_rate, 2),
            "n_spikes": window_data.n_spikes,
        })

        t += step_sec

    if len(windows) < 5:
        return {"error": "Not enough data for phase transition detection"}

    # Detect change points (abrupt changes in order parameter)
    order_params = np.array([w["order_parameter"] for w in windows])
    transitions = _detect_change_points(order_params, threshold=2.0)

    # Susceptibility peaks (variance maxima → critical points)
    susceptibilities = np.array([w["susceptibility"] for w in windows])
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(susceptibilities, height=np.percentile(susceptibilities, 80), distance=3)

    # Classify phases
    median_op = float(np.median(order_params))
    for w in windows:
        w["phase"] = "ordered" if w["order_parameter"] > median_op * 1.3 else "disordered" if w["order_parameter"] < median_op * 0.7 else "transitional"

    # Build transition events
    transition_events = []
    for cp_idx in transitions:
        if cp_idx < len(windows):
            before = order_params[max(0, cp_idx - 2):cp_idx]
            after = order_params[cp_idx:min(len(order_params), cp_idx + 3)]

            before_mean = float(np.mean(before)) if len(before) > 0 else 0
            after_mean = float(np.mean(after)) if len(after) > 0 else 0
            direction = "ordering" if after_mean > before_mean else "disordering"

            transition_events.append({
                "time": windows[cp_idx]["time"],
                "window_idx": int(cp_idx),
                "direction": direction,
                "order_change": round(float(after_mean - before_mean), 4),
                "before_order": round(before_mean, 4),
                "after_order": round(after_mean, 4),
            })

    critical_points = [{"time": windows[p]["time"], "susceptibility": windows[p]["susceptibility"]} for p in peaks]

    return {
        "windows": windows,
        "transitions": transition_events,
        "critical_points": critical_points[:10],
        "n_transitions": len(transition_events),
        "n_critical_points": len(critical_points),
        "mean_order_parameter": round(float(np.mean(order_params)), 4),
        "order_parameter_cv": round(float(np.std(order_params) / np.mean(order_params)), 4) if np.mean(order_params) > 0 else 0,
        "interpretation": (
            f"PHASE TRANSITIONS DETECTED: {len(transition_events)} reorganization events. "
            f"These are moments when the organoid's computational state shifts — "
            f"potential learning windows for stimulation. "
            f"Critical points (susceptibility peaks): {len(critical_points)}."
            if transition_events
            else "Stable dynamics — no significant phase transitions detected. "
            "The organoid maintains consistent activity patterns."
        ),
        "stimulation_advice": (
            f"Optimal stimulation timing: near phase transitions at "
            f"{[t['time'] for t in transition_events[:3]]} seconds. "
            f"The system is most plastic during transitions between ordered and disordered states."
            if transition_events
            else "No clear optimal timing detected — stimulate during periods of moderate activity."
        ),
    }


def _detect_change_points(signal: np.ndarray, threshold: float = 2.0) -> list[int]:
    """Detect change points using CUSUM-like method."""
    if len(signal) < 5:
        return []

    diff = np.abs(np.diff(signal))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    if std_diff == 0:
        return []

    # Points where change exceeds threshold * std
    change_points = np.where(diff > mean_diff + threshold * std_diff)[0]

    # Remove close neighbors
    if len(change_points) > 0:
        filtered = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered[-1] > 3:
                filtered.append(cp)
        return [int(cp) for cp in filtered]

    return []
