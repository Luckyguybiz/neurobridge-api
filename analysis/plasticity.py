"""Plasticity & STDP analysis module.

Spike-Timing-Dependent Plasticity — the fundamental learning rule of biological neurons.
If neuron A fires before B within 20ms → connection strengthens (LTP).
If neuron A fires after B → connection weakens (LTD).

This module detects STDP-like patterns in organoid data — evidence of learning.
Nobody has systematically mapped STDP on FinalSpark yet.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def compute_stdp_matrix(
    data: SpikeData,
    max_lag_ms: float = 30.0,
    bin_size_ms: float = 1.0,
) -> dict:
    """Compute STDP curves for all electrode pairs.

    For each pair (A, B): count spike timing differences.
    Positive lag = A fires before B (pre-before-post = LTP window).
    Negative lag = A fires after B (post-before-pre = LTD window).

    The asymmetry of the histogram reveals the direction of plasticity.
    """
    max_lag_sec = max_lag_ms / 1000.0
    n_bins = int(2 * max_lag_ms / bin_size_ms)
    electrode_ids = data.electrode_ids
    n = len(electrode_ids)

    pair_results = {}
    plasticity_matrix = np.zeros((n, n))  # Net plasticity direction

    for i in range(n):
        t_i = data.times[data.electrodes == electrode_ids[i]]
        for j in range(n):
            if i == j:
                continue
            t_j = data.times[data.electrodes == electrode_ids[j]]

            if len(t_i) < 5 or len(t_j) < 5:
                continue

            # Compute all pairwise timing differences
            diffs_ms = []
            for ti in t_i:
                valid = t_j[(t_j > ti - max_lag_sec) & (t_j < ti + max_lag_sec)]
                for tj in valid:
                    diffs_ms.append((tj - ti) * 1000)  # ms, positive = j fires after i

            if len(diffs_ms) < 3:
                continue

            diffs = np.array(diffs_ms)

            # Histogram
            bins_edges = np.linspace(-max_lag_ms, max_lag_ms, n_bins + 1)
            hist, _ = np.histogram(diffs, bins=bins_edges)

            # STDP metrics
            # LTP window: pre fires before post (positive lag for j, negative for i→j)
            ltp_mask = diffs > 0  # j fires AFTER i
            ltd_mask = diffs < 0  # j fires BEFORE i

            ltp_count = int(np.sum(ltp_mask))
            ltd_count = int(np.sum(ltd_mask))

            # Asymmetry index: (LTP - LTD) / (LTP + LTD)
            total = ltp_count + ltd_count
            asymmetry = (ltp_count - ltd_count) / total if total > 0 else 0

            # Mean timing in LTP/LTD windows
            mean_ltp_ms = float(np.mean(diffs[ltp_mask])) if ltp_count > 0 else 0
            mean_ltd_ms = float(np.mean(np.abs(diffs[ltd_mask]))) if ltd_count > 0 else 0

            plasticity_matrix[i, j] = asymmetry

            key = f"{electrode_ids[i]}-{electrode_ids[j]}"
            pair_results[key] = {
                "source": int(electrode_ids[i]),
                "target": int(electrode_ids[j]),
                "n_pairs": len(diffs),
                "ltp_count": ltp_count,
                "ltd_count": ltd_count,
                "asymmetry_index": round(float(asymmetry), 4),
                "mean_ltp_delay_ms": round(mean_ltp_ms, 2),
                "mean_ltd_delay_ms": round(mean_ltd_ms, 2),
                "histogram": hist.tolist(),
                "interpretation": (
                    "Strong LTP (potentiation)" if asymmetry > 0.3
                    else "Weak LTP" if asymmetry > 0.1
                    else "Balanced" if abs(asymmetry) <= 0.1
                    else "Weak LTD" if asymmetry > -0.3
                    else "Strong LTD (depression)"
                ),
            }

    # Find strongest plasticity pairs
    top_ltp = sorted(
        [p for p in pair_results.values() if p["asymmetry_index"] > 0.1],
        key=lambda x: x["asymmetry_index"], reverse=True
    )[:5]

    top_ltd = sorted(
        [p for p in pair_results.values() if p["asymmetry_index"] < -0.1],
        key=lambda x: x["asymmetry_index"]
    )[:5]

    return {
        "pairs": pair_results,
        "plasticity_matrix": plasticity_matrix.tolist(),
        "electrode_ids": electrode_ids,
        "top_ltp_pairs": top_ltp,
        "top_ltd_pairs": top_ltd,
        "mean_asymmetry": round(float(np.mean(np.abs(plasticity_matrix[plasticity_matrix != 0]))), 4) if np.any(plasticity_matrix != 0) else 0,
        "n_significant_pairs": int(np.sum(np.abs(plasticity_matrix) > 0.1)),
        "has_learning_signatures": bool(np.any(np.abs(plasticity_matrix) > 0.2)),
        "max_lag_ms": max_lag_ms,
    }


def detect_learning_episodes(
    data: SpikeData,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
) -> dict:
    """Detect episodes where STDP patterns change over time.

    Sliding window analysis — compares plasticity patterns between
    consecutive time windows. Changes in STDP = evidence of learning/adaptation.
    """
    t_start, t_end = data.time_range
    windows = []
    t = t_start

    while t + window_sec <= t_end:
        window_data = data.get_time_range(t, t + window_sec)
        if window_data.n_spikes > 20:
            stdp = compute_stdp_matrix(window_data, max_lag_ms=20.0)
            matrix = np.array(stdp["plasticity_matrix"])
            windows.append({
                "start": round(float(t), 2),
                "end": round(float(t + window_sec), 2),
                "mean_asymmetry": round(float(np.mean(np.abs(matrix[matrix != 0]))), 4) if np.any(matrix != 0) else 0,
                "n_significant": int(np.sum(np.abs(matrix) > 0.1)),
                "matrix_norm": round(float(np.linalg.norm(matrix)), 4),
            })
        t += step_sec

    # Detect changes between consecutive windows
    changes = []
    for i in range(1, len(windows)):
        delta_asymmetry = windows[i]["mean_asymmetry"] - windows[i-1]["mean_asymmetry"]
        delta_significant = windows[i]["n_significant"] - windows[i-1]["n_significant"]

        if abs(delta_asymmetry) > 0.05 or abs(delta_significant) > 2:
            changes.append({
                "time": windows[i]["start"],
                "delta_asymmetry": round(float(delta_asymmetry), 4),
                "delta_significant_pairs": int(delta_significant),
                "type": "plasticity_increase" if delta_asymmetry > 0 else "plasticity_decrease",
            })

    return {
        "windows": windows,
        "n_windows": len(windows),
        "changes": changes,
        "n_learning_episodes": len(changes),
        "has_learning": len(changes) > 0,
        "window_sec": window_sec,
    }
