"""Burst detection and analysis module.

Network bursts = synchronized firing across multiple electrodes.
Critical for understanding organoid computation and plasticity.

Methods:
- Threshold-based: N spikes on M+ electrodes within T ms window
- Rank surprise: statistical surprise of spike coincidence
- CMA (Cumulative Moving Average): adaptive burst detection
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def detect_bursts(
    data: SpikeData,
    min_electrodes: int = 3,
    window_ms: float = 50.0,
    min_spikes_per_electrode: int = 2,
) -> dict:
    """Detect network bursts (synchronized multi-electrode firing).

    Algorithm:
    1. Slide window across time
    2. At each position, count electrodes with >= min_spikes spikes
    3. If >= min_electrodes are active, mark as burst
    4. Merge overlapping burst windows
    """
    if data.n_spikes == 0:
        return {"bursts": [], "n_bursts": 0}

    window_sec = window_ms / 1000.0
    t_start, t_end = data.time_range
    step_sec = window_sec / 4  # 75% overlap

    burst_windows = []
    t = t_start

    while t < t_end:
        mask = (data.times >= t) & (data.times < t + window_sec)
        if np.sum(mask) == 0:
            t += step_sec
            continue

        window_spikes = data.electrodes[mask]
        unique_electrodes, counts = np.unique(window_spikes, return_counts=True)
        active = unique_electrodes[counts >= min_spikes_per_electrode]

        if len(active) >= min_electrodes:
            burst_windows.append({
                "start": float(t),
                "end": float(t + window_sec),
                "n_electrodes": int(len(active)),
                "electrodes": active.tolist(),
                "n_spikes": int(np.sum(mask)),
            })

        t += step_sec

    # Merge overlapping bursts
    bursts = _merge_bursts(burst_windows)

    # Compute burst metrics
    for burst in bursts:
        burst_mask = (data.times >= burst["start"]) & (data.times <= burst["end"])
        burst_spikes = data.times[burst_mask]
        burst["duration_ms"] = (burst["end"] - burst["start"]) * 1000
        burst["peak_firing_rate"] = float(len(burst_spikes) / (burst["end"] - burst["start"])) if burst["end"] > burst["start"] else 0

    # Inter-burst intervals
    ibis = []
    for i in range(1, len(bursts)):
        ibi = bursts[i]["start"] - bursts[i - 1]["end"]
        ibis.append(ibi * 1000)  # ms

    return {
        "bursts": bursts,
        "n_bursts": len(bursts),
        "burst_rate_per_min": len(bursts) / (data.duration / 60) if data.duration > 0 else 0,
        "mean_duration_ms": float(np.mean([b["duration_ms"] for b in bursts])) if bursts else 0,
        "mean_n_electrodes": float(np.mean([b["n_electrodes"] for b in bursts])) if bursts else 0,
        "mean_ibi_ms": float(np.mean(ibis)) if ibis else 0,
        "cv_ibi": float(np.std(ibis) / np.mean(ibis)) if ibis and np.mean(ibis) > 0 else 0,
        "total_burst_time_pct": sum(b["duration_ms"] for b in bursts) / (data.duration * 1000) * 100 if data.duration > 0 else 0,
    }


def _merge_bursts(windows: list[dict], gap_ms: float = 10.0) -> list[dict]:
    """Merge overlapping or close burst windows."""
    if not windows:
        return []

    gap_sec = gap_ms / 1000.0
    sorted_w = sorted(windows, key=lambda w: w["start"])
    merged = [sorted_w[0].copy()]

    for w in sorted_w[1:]:
        if w["start"] - merged[-1]["end"] <= gap_sec:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["n_electrodes"] = max(merged[-1]["n_electrodes"], w["n_electrodes"])
            all_el = set(merged[-1]["electrodes"]) | set(w["electrodes"])
            merged[-1]["electrodes"] = sorted(all_el)
            merged[-1]["n_spikes"] = merged[-1].get("n_spikes", 0) + w.get("n_spikes", 0)
        else:
            merged.append(w.copy())

    return merged


def compute_burst_profiles(data: SpikeData, bursts: list[dict]) -> dict:
    """Compute detailed profiles for each burst.

    For each burst:
    - Per-electrode spike count
    - Temporal profile (firing rate curve within burst)
    - Recruitment order (which electrodes fire first)
    """
    profiles = []

    for burst in bursts:
        mask = (data.times >= burst["start"]) & (data.times <= burst["end"])
        b_times = data.times[mask]
        b_electrodes = data.electrodes[mask]

        # Per-electrode counts
        electrode_counts = {}
        electrode_first_spike = {}
        for e in np.unique(b_electrodes):
            e_mask = b_electrodes == e
            electrode_counts[int(e)] = int(np.sum(e_mask))
            electrode_first_spike[int(e)] = float(b_times[e_mask][0])

        # Recruitment order (sorted by first spike time)
        recruitment = sorted(electrode_first_spike.items(), key=lambda x: x[1])

        # Temporal profile (1ms bins within burst)
        duration = burst["end"] - burst["start"]
        n_bins = max(1, int(duration * 1000))
        bins = np.linspace(burst["start"], burst["end"], n_bins + 1)
        profile, _ = np.histogram(b_times, bins=bins)

        profiles.append({
            "start": burst["start"],
            "end": burst["end"],
            "electrode_counts": electrode_counts,
            "recruitment_order": [e for e, _ in recruitment],
            "recruitment_latencies_ms": [(t - burst["start"]) * 1000 for _, t in recruitment],
            "temporal_profile": profile.tolist(),
            "peak_bin_idx": int(np.argmax(profile)),
        })

    return {"profiles": profiles, "n_bursts": len(profiles)}


def detect_single_channel_bursts(
    spike_times: np.ndarray,
    max_isi_ms: float = 100.0,
    min_spikes: int = 3,
) -> list[dict]:
    """Detect bursts on a single electrode using ISI-based method.

    A burst = consecutive spikes with ISI < max_isi_ms, at least min_spikes.
    """
    if len(spike_times) < min_spikes:
        return []

    max_isi_sec = max_isi_ms / 1000.0
    isi = np.diff(spike_times)
    in_burst = isi < max_isi_sec

    bursts = []
    burst_start = None
    burst_count = 0

    for i, is_burst in enumerate(in_burst):
        if is_burst:
            if burst_start is None:
                burst_start = i
                burst_count = 2
            else:
                burst_count += 1
        else:
            if burst_start is not None and burst_count >= min_spikes:
                bursts.append({
                    "start": float(spike_times[burst_start]),
                    "end": float(spike_times[i]),
                    "n_spikes": burst_count,
                    "duration_ms": float((spike_times[i] - spike_times[burst_start]) * 1000),
                    "mean_isi_ms": float(np.mean(isi[burst_start:i]) * 1000),
                    "intra_burst_rate_hz": burst_count / (spike_times[i] - spike_times[burst_start]) if spike_times[i] > spike_times[burst_start] else 0,
                })
            burst_start = None
            burst_count = 0

    # Handle burst at end of recording
    if burst_start is not None and burst_count >= min_spikes:
        end_idx = len(spike_times) - 1
        bursts.append({
            "start": float(spike_times[burst_start]),
            "end": float(spike_times[end_idx]),
            "n_spikes": burst_count,
            "duration_ms": float((spike_times[end_idx] - spike_times[burst_start]) * 1000),
            "mean_isi_ms": float(np.mean(isi[burst_start:end_idx]) * 1000),
            "intra_burst_rate_hz": burst_count / (spike_times[end_idx] - spike_times[burst_start]) if spike_times[end_idx] > spike_times[burst_start] else 0,
        })

    return bursts
