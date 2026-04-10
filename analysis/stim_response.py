"""Stimulation response analysis module.

Analyzes how the organoid responds to electrical stimulation events.
Measures response latency, jitter, rate changes, and dose-response
curves. Can also estimate stimulation times from spike data if
explicit timestamps are not available.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def detect_response(
    data: SpikeData,
    stim_times: list[float],
    window_ms: float = 200.0,
    baseline_ms: float = 200.0,
) -> dict:
    """Measure organoid response in post-stimulus windows.

    For each stimulation event, computes:
    - Rate change vs baseline
    - First-spike latency
    - Response jitter across trials
    - Response reliability (fraction of trials with response)
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    if not stim_times:
        return {"error": "No stimulation times provided"}

    stim_times = sorted(stim_times)
    window_sec = window_ms / 1000.0
    baseline_sec = baseline_ms / 1000.0

    trial_responses: list[dict] = []
    latencies: list[float] = []
    rate_changes: list[float] = []

    for stim_t in stim_times:
        # Baseline window: [stim_t - baseline_sec, stim_t]
        baseline_data = data.get_time_range(stim_t - baseline_sec, stim_t)
        baseline_rate = baseline_data.n_spikes / baseline_sec if baseline_sec > 0 else 0

        # Response window: [stim_t, stim_t + window_sec]
        response_data = data.get_time_range(stim_t, stim_t + window_sec)
        response_rate = response_data.n_spikes / window_sec if window_sec > 0 else 0

        rate_change = response_rate - baseline_rate
        rate_changes.append(rate_change)

        # First-spike latency
        if response_data.n_spikes > 0:
            first_spike = float(response_data.times[0])
            latency_ms = (first_spike - stim_t) * 1000
            latencies.append(latency_ms)
            has_response = True
        else:
            latency_ms = None
            has_response = False

        trial_responses.append({
            "stim_time_s": round(stim_t, 4),
            "baseline_rate_hz": round(baseline_rate, 3),
            "response_rate_hz": round(response_rate, 3),
            "rate_change_hz": round(rate_change, 3),
            "first_spike_latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "n_response_spikes": response_data.n_spikes,
            "has_response": has_response,
        })

    # Aggregate statistics
    reliability = sum(1 for t in trial_responses if t["has_response"]) / len(trial_responses)

    latency_stats = {}
    if latencies:
        latency_stats = {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "std_ms": round(float(np.std(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "jitter_ms": round(float(np.std(latencies)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
        }

    mean_rate_change = float(np.mean(rate_changes))

    # Per-electrode response
    electrode_responses: dict[int, dict] = {}
    for e in data.electrode_ids:
        e_latencies = []
        e_responded = 0
        for stim_t in stim_times:
            e_data = data.get_time_range(stim_t, stim_t + window_sec)
            e_mask = e_data.electrodes == e
            if np.sum(e_mask) > 0:
                e_responded += 1
                first = float(e_data.times[e_mask][0])
                e_latencies.append((first - stim_t) * 1000)

        electrode_responses[e] = {
            "reliability": round(e_responded / len(stim_times), 3),
            "mean_latency_ms": round(float(np.mean(e_latencies)), 2) if e_latencies else None,
            "n_responses": e_responded,
        }

    return {
        "n_stimulations": len(stim_times),
        "window_ms": window_ms,
        "baseline_ms": baseline_ms,
        "trials": trial_responses,
        "reliability": round(reliability, 3),
        "mean_rate_change_hz": round(mean_rate_change, 3),
        "latency": latency_stats,
        "electrode_responses": electrode_responses,
        "response_type": (
            "Excitatory" if mean_rate_change > 1.0
            else "Inhibitory" if mean_rate_change < -1.0
            else "Weak/No response"
        ),
    }


def compute_dose_response(
    data: SpikeData,
    stim_times: list[float],
    stim_amplitudes: list[float],
    window_ms: float = 200.0,
    baseline_ms: float = 200.0,
) -> dict:
    """How does response scale with stimulation strength?

    Groups stimulation events by amplitude and computes mean response
    for each dose level. Fits a sigmoid (Hill curve) to the dose-response data.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    if len(stim_times) != len(stim_amplitudes):
        return {"error": "stim_times and stim_amplitudes must have same length"}

    if not stim_times:
        return {"error": "No stimulation events provided"}

    window_sec = window_ms / 1000.0
    baseline_sec = baseline_ms / 1000.0

    # Group by unique amplitudes
    unique_amps = sorted(set(stim_amplitudes))
    dose_groups: dict[float, list[float]] = {a: [] for a in unique_amps}

    for stim_t, amp in zip(stim_times, stim_amplitudes):
        baseline = data.get_time_range(stim_t - baseline_sec, stim_t)
        response = data.get_time_range(stim_t, stim_t + window_sec)
        baseline_rate = baseline.n_spikes / baseline_sec if baseline_sec > 0 else 0
        response_rate = response.n_spikes / window_sec if window_sec > 0 else 0
        dose_groups[amp].append(response_rate - baseline_rate)

    # Compute mean response per dose
    dose_response_curve: list[dict] = []
    x_doses = []
    y_responses = []

    for amp in unique_amps:
        responses = dose_groups[amp]
        mean_resp = float(np.mean(responses))
        std_resp = float(np.std(responses))
        dose_response_curve.append({
            "amplitude": amp,
            "n_trials": len(responses),
            "mean_response_hz": round(mean_resp, 3),
            "std_response_hz": round(std_resp, 3),
            "sem_response_hz": round(std_resp / np.sqrt(len(responses)), 3) if len(responses) > 1 else 0.0,
        })
        x_doses.append(amp)
        y_responses.append(mean_resp)

    x_arr = np.array(x_doses)
    y_arr = np.array(y_responses)

    # Fit linear trend as baseline
    fit_result = {}
    if len(x_arr) >= 3:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_arr, y_arr)
        fit_result = {
            "linear_slope": round(float(slope), 4),
            "linear_r_squared": round(float(r_value ** 2), 4),
            "linear_p_value": float(p_value),
            "is_dose_dependent": p_value < 0.05 and slope > 0,
        }

    # Monotonicity check
    diffs = np.diff(y_arr)
    monotonic_increasing = bool(np.all(diffs >= 0))
    monotonic_score = float(np.sum(diffs > 0)) / max(len(diffs), 1)

    # EC50 estimate (amplitude at half-max response)
    ec50 = None
    if len(y_arr) >= 3 and np.max(y_arr) > 0:
        half_max = np.max(y_arr) / 2
        above = np.where(y_arr >= half_max)[0]
        if len(above) > 0 and above[0] > 0:
            idx = above[0]
            # Linear interpolation between points
            x0, x1 = x_arr[idx - 1], x_arr[idx]
            y0, y1 = y_arr[idx - 1], y_arr[idx]
            if y1 != y0:
                ec50 = round(float(x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)), 4)

    return {
        "dose_response_curve": dose_response_curve,
        "n_dose_levels": len(unique_amps),
        "n_total_trials": len(stim_times),
        "fit": fit_result,
        "monotonicity_score": round(monotonic_score, 3),
        "is_monotonic": monotonic_increasing,
        "ec50_estimate": ec50,
        "max_response_hz": round(float(np.max(y_arr)), 3) if len(y_arr) > 0 else 0.0,
        "dose_range": [float(np.min(x_arr)), float(np.max(x_arr))] if len(x_arr) > 0 else [],
    }


def estimate_stim_times(
    data: SpikeData,
    min_gap_ms: float = 500.0,
    min_burst_spikes: int = 10,
    regularity_threshold: float = 0.3,
) -> dict:
    """Estimate stimulation times from spike data patterns.

    Looks for regular high-activity bursts that likely correspond
    to stimulation events. Stimulation typically produces stereotyped
    responses with regular inter-stim intervals.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    min_gap_sec = min_gap_ms / 1000.0

    # Find population bursts (dense clusters of spikes)
    times = data.times
    burst_starts: list[float] = []
    burst_sizes: list[int] = []

    i = 0
    while i < len(times):
        # Count spikes within 20ms window
        window_end = times[i] + 0.020  # 20ms
        j = i
        while j < len(times) and times[j] <= window_end:
            j += 1
        cluster_size = j - i

        if cluster_size >= min_burst_spikes:
            burst_starts.append(float(times[i]))
            burst_sizes.append(cluster_size)
            # Skip past this burst + min gap
            while i < len(times) and times[i] < times[j - 1] + min_gap_sec:
                i += 1
        else:
            i += 1

    if len(burst_starts) < 2:
        return {
            "estimated_stim_times": [],
            "n_detected": 0,
            "confidence": "low",
            "method": "burst_detection",
            "note": "Too few bursts detected to estimate stimulation",
        }

    # Check regularity of inter-burst intervals
    ibis = np.diff(burst_starts) * 1000  # ms
    ibi_cv = float(np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else float("inf")

    is_regular = ibi_cv < regularity_threshold

    if is_regular:
        confidence = "high"
        interpretation = f"Regular pattern detected (CV={ibi_cv:.3f}), likely stimulation-evoked"
    elif ibi_cv < 0.6:
        confidence = "medium"
        interpretation = f"Semi-regular pattern (CV={ibi_cv:.3f}), possibly stimulation"
    else:
        confidence = "low"
        interpretation = f"Irregular pattern (CV={ibi_cv:.3f}), may be spontaneous bursts"

    return {
        "estimated_stim_times": [round(t, 4) for t in burst_starts],
        "n_detected": len(burst_starts),
        "inter_stim_interval_ms": {
            "mean": round(float(np.mean(ibis)), 2),
            "std": round(float(np.std(ibis)), 2),
            "cv": round(ibi_cv, 4),
        },
        "burst_sizes": burst_sizes,
        "mean_burst_size": round(float(np.mean(burst_sizes)), 1),
        "confidence": confidence,
        "is_regular": is_regular,
        "interpretation": interpretation,
        "method": "burst_detection",
        "parameters": {
            "min_gap_ms": min_gap_ms,
            "min_burst_spikes": min_burst_spikes,
            "regularity_threshold": regularity_threshold,
        },
    }
