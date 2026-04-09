"""Comprehensive statistical analysis module.

Computes everything you'd want to know about a spike recording:
- Per-electrode summary statistics
- Temporal dynamics (stationarity, trends)
- Population-level metrics
- Quality metrics (SNR, artifact detection)
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def compute_full_summary(data: SpikeData) -> dict:
    """Compute comprehensive summary of entire dataset."""
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    t_start, t_end = data.time_range

    # Per-electrode stats
    electrode_stats = []
    for e in data.electrode_ids:
        mask = data.electrodes == e
        n = int(np.sum(mask))
        times_e = data.times[mask]
        amps_e = data.amplitudes[mask]

        firing_rate = n / data.duration if data.duration > 0 else 0

        # ISI stats
        if n >= 2:
            isi = np.diff(times_e) * 1000  # ms
            isi_mean = float(np.mean(isi))
            isi_cv = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0
            refractory_violations = int(np.sum(isi < 2.0))
        else:
            isi_mean = 0
            isi_cv = 0
            refractory_violations = 0

        electrode_stats.append({
            "electrode": int(e),
            "n_spikes": n,
            "firing_rate_hz": round(firing_rate, 3),
            "mean_amplitude_uv": round(float(np.mean(amps_e)), 2) if n > 0 else 0,
            "std_amplitude_uv": round(float(np.std(amps_e)), 2) if n > 0 else 0,
            "mean_isi_ms": round(isi_mean, 2),
            "cv_isi": round(isi_cv, 3),
            "refractory_violations": refractory_violations,
            "first_spike_s": round(float(times_e[0]), 4) if n > 0 else None,
            "last_spike_s": round(float(times_e[-1]), 4) if n > 0 else None,
        })

    # Population metrics
    all_rates = [s["firing_rate_hz"] for s in electrode_stats]
    all_counts = [s["n_spikes"] for s in electrode_stats]

    # Temporal stability (compare first half vs second half firing rates)
    midpoint = (t_start + t_end) / 2
    first_half = data.get_time_range(t_start, midpoint)
    second_half = data.get_time_range(midpoint, t_end)
    half_duration = (t_end - t_start) / 2

    stability = {}
    for e in data.electrode_ids:
        r1 = np.sum(first_half.electrodes == e) / half_duration if half_duration > 0 else 0
        r2 = np.sum(second_half.electrodes == e) / half_duration if half_duration > 0 else 0
        change_pct = ((r2 - r1) / r1 * 100) if r1 > 0 else 0
        stability[int(e)] = {
            "first_half_hz": round(float(r1), 3),
            "second_half_hz": round(float(r2), 3),
            "change_pct": round(float(change_pct), 1),
        }

    return {
        "dataset": {
            "n_spikes": data.n_spikes,
            "n_electrodes": data.n_electrodes,
            "duration_s": round(data.duration, 3),
            "time_range": [round(t_start, 4), round(t_end, 4)],
            "sampling_rate": data.sampling_rate,
            "metadata": data.metadata,
        },
        "population": {
            "total_spikes": data.n_spikes,
            "mean_firing_rate_hz": round(float(np.mean(all_rates)), 3),
            "std_firing_rate_hz": round(float(np.std(all_rates)), 3),
            "min_firing_rate_hz": round(float(np.min(all_rates)), 3),
            "max_firing_rate_hz": round(float(np.max(all_rates)), 3),
            "most_active_electrode": int(data.electrode_ids[np.argmax(all_counts)]),
            "least_active_electrode": int(data.electrode_ids[np.argmin(all_counts)]),
            "mean_amplitude_uv": round(float(np.mean(data.amplitudes)), 2),
            "spikes_per_electrode": {int(e): int(c) for e, c in zip(data.electrode_ids, all_counts)},
        },
        "electrodes": electrode_stats,
        "temporal_stability": stability,
    }


def compute_temporal_dynamics(
    data: SpikeData,
    bin_size_sec: float = 60.0,
) -> dict:
    """Analyze how activity changes over time.

    Detects trends, periodicities, and state transitions.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes"}

    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    n_bins = len(bins) - 1

    # Population firing rate over time
    pop_counts, _ = np.histogram(data.times, bins=bins)
    pop_rate = pop_counts / bin_size_sec

    # Per-electrode rates over time
    electrode_rates = {}
    for e in data.electrode_ids:
        e_times = data.times[data.electrodes == e]
        counts, _ = np.histogram(e_times, bins=bins)
        electrode_rates[int(e)] = (counts / bin_size_sec).tolist()

    # Trend analysis (linear regression on population rate)
    if n_bins >= 3:
        bin_centers = (bins[:-1] + bins[1:]) / 2
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(bin_centers, pop_rate)
        trend = {
            "slope_hz_per_sec": round(float(slope), 6),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "direction": "increasing" if slope > 0 else "decreasing",
        }
    else:
        trend = {"slope_hz_per_sec": 0, "r_squared": 0, "significant": False}

    # Coefficient of variation of population rate (measure of stationarity)
    cv_rate = float(np.std(pop_rate) / np.mean(pop_rate)) if np.mean(pop_rate) > 0 else 0

    # Fano factor per electrode (variance/mean of spike counts)
    fano_factors = {}
    for e in data.electrode_ids:
        e_times = data.times[data.electrodes == e]
        counts, _ = np.histogram(e_times, bins=bins)
        mean_c = np.mean(counts)
        fano = float(np.var(counts) / mean_c) if mean_c > 0 else 0
        fano_factors[int(e)] = round(fano, 3)

    return {
        "bin_size_sec": bin_size_sec,
        "n_bins": n_bins,
        "bin_centers": ((bins[:-1] + bins[1:]) / 2).tolist(),
        "population_rate": pop_rate.tolist(),
        "electrode_rates": electrode_rates,
        "trend": trend,
        "cv_population_rate": round(cv_rate, 3),
        "is_stationary": cv_rate < 0.5 and not trend.get("significant", False),
        "fano_factors": fano_factors,
        "mean_fano_factor": round(float(np.mean(list(fano_factors.values()))), 3),
    }


def compute_quality_metrics(data: SpikeData) -> dict:
    """Compute data quality metrics.

    Checks for:
    - Refractory period violations (suggests poor spike detection)
    - Amplitude distribution abnormalities
    - Missing electrodes
    - Recording gaps
    """
    issues = []

    # Check refractory violations per electrode
    total_violations = 0
    for e in data.electrode_ids:
        e_times = np.sort(data.times[data.electrodes == e])
        if len(e_times) >= 2:
            isi = np.diff(e_times) * 1000  # ms
            violations = int(np.sum(isi < 1.0))
            total_violations += violations
            if violations > len(isi) * 0.01:
                issues.append(f"Electrode {e}: {violations} refractory violations ({violations/len(isi)*100:.1f}%)")

    # Check for silent electrodes
    expected_electrodes = set(range(max(data.electrode_ids) + 1))
    active_electrodes = set(data.electrode_ids)
    missing = expected_electrodes - active_electrodes
    if missing:
        issues.append(f"Silent electrodes: {sorted(missing)}")

    # Check for recording gaps (>5 seconds without any spikes)
    all_times = np.sort(data.times)
    if len(all_times) >= 2:
        gaps = np.diff(all_times)
        large_gaps = np.where(gaps > 5.0)[0]
        if len(large_gaps) > 0:
            issues.append(f"{len(large_gaps)} recording gaps > 5s detected")

    # Amplitude distribution check
    if len(data.amplitudes) > 0:
        kurt = float(scipy_stats.kurtosis(data.amplitudes))
        skew = float(scipy_stats.skew(data.amplitudes))
    else:
        kurt, skew = 0, 0

    # SNR estimate (peak amplitude / noise floor)
    snr_per_electrode = {}
    for e in data.electrode_ids:
        amps = np.abs(data.amplitudes[data.electrodes == e])
        if len(amps) > 10:
            signal = np.percentile(amps, 90)
            noise = np.percentile(amps, 10)
            snr = signal / noise if noise > 0 else 0
            snr_per_electrode[int(e)] = round(float(snr), 2)

    quality_score = max(0, 100 - len(issues) * 15 - total_violations * 0.1)

    return {
        "quality_score": round(min(100, quality_score), 1),
        "total_refractory_violations": total_violations,
        "violation_rate_pct": round(total_violations / data.n_spikes * 100, 2) if data.n_spikes > 0 else 0,
        "n_active_electrodes": len(active_electrodes),
        "missing_electrodes": sorted(missing),
        "amplitude_kurtosis": round(kurt, 3),
        "amplitude_skewness": round(skew, 3),
        "snr_per_electrode": snr_per_electrode,
        "mean_snr": round(float(np.mean(list(snr_per_electrode.values()))), 2) if snr_per_electrode else 0,
        "issues": issues,
        "n_issues": len(issues),
    }
