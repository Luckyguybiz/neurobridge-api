"""Multi-organoid comparative analysis module.

FinalSpark provides 4 MEAs x 8 electrodes = 32 channels total.
This module splits a 32-electrode dataset into virtual organoids
and compares their activity patterns to identify the best-performing
organoid, correlate behaviour across MEAs, and rank them.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def split_into_organoids(
    data: SpikeData,
    electrodes_per_organoid: int = 8,
) -> dict:
    """Split dataset into N sub-SpikeData objects by electrode ranges.

    Default: 4 organoids of 8 electrodes each (matching FinalSpark layout).
    Electrodes 0-7 -> organoid 0, 8-15 -> organoid 1, etc.

    Returns dict with organoid sub-datasets and metadata.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    all_electrodes = sorted(data.electrode_ids)
    n_total = max(all_electrodes) + 1 if all_electrodes else 0
    n_organoids = max(1, int(np.ceil(n_total / electrodes_per_organoid)))

    organoids: dict[int, SpikeData] = {}
    organoid_info: list[dict] = []

    for org_idx in range(n_organoids):
        e_start = org_idx * electrodes_per_organoid
        e_end = e_start + electrodes_per_organoid
        electrode_range = list(range(e_start, e_end))

        # Filter spikes for this organoid
        mask = np.isin(data.electrodes, electrode_range)
        if np.sum(mask) == 0:
            continue

        sub_data = SpikeData(
            times=data.times[mask],
            electrodes=data.electrodes[mask],
            amplitudes=data.amplitudes[mask],
            waveforms=data.waveforms[mask] if data.waveforms is not None else None,
            sampling_rate=data.sampling_rate,
            metadata={**data.metadata, "organoid_index": org_idx},
        )
        organoids[org_idx] = sub_data

        active_in_range = [e for e in electrode_range if e in all_electrodes]
        organoid_info.append({
            "organoid_index": org_idx,
            "electrode_range": [e_start, e_end - 1],
            "active_electrodes": active_in_range,
            "n_active": len(active_in_range),
            "n_spikes": sub_data.n_spikes,
            "duration_s": round(sub_data.duration, 3),
        })

    return {
        "n_organoids": len(organoids),
        "electrodes_per_organoid": electrodes_per_organoid,
        "organoids": organoid_info,
        "_sub_datasets": organoids,
    }


def compare_organoids(
    data: SpikeData,
    electrodes_per_organoid: int = 8,
    bin_size_sec: float = 1.0,
) -> dict:
    """Compute per-organoid stats and cross-organoid correlation.

    For each organoid computes: firing rate, burst count, ISI CV.
    Then ranks organoids, finds best/worst, and computes pairwise
    cross-correlation of population firing rates.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    split = split_into_organoids(data, electrodes_per_organoid)
    sub_datasets: dict[int, SpikeData] = split.get("_sub_datasets", {})

    if len(sub_datasets) < 2:
        return {"error": "Need at least 2 organoids to compare", "n_found": len(sub_datasets)}

    # Compute per-organoid metrics
    t_start, t_end = data.time_range
    duration = t_end - t_start
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    organoid_metrics: list[dict] = []
    rate_timeseries: dict[int, np.ndarray] = {}

    for org_idx, sub_data in sorted(sub_datasets.items()):
        n_spikes = sub_data.n_spikes
        firing_rate = n_spikes / duration if duration > 0 else 0.0

        # ISI coefficient of variation
        if n_spikes >= 2:
            isi = np.diff(sub_data.times) * 1000  # ms
            isi_cv = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0.0
            isi_mean_ms = float(np.mean(isi))
        else:
            isi_cv = 0.0
            isi_mean_ms = 0.0

        # Burst detection (simplified: ISI < 10ms clusters)
        burst_count = 0
        if n_spikes >= 3:
            isi_arr = np.diff(sub_data.times) * 1000
            in_burst = isi_arr < 10.0
            transitions = np.diff(in_burst.astype(int))
            burst_count = int(np.sum(transitions == 1))

        # Binned rate for correlation
        counts, _ = np.histogram(sub_data.times, bins=bins)
        rate = counts / bin_size_sec
        rate_timeseries[org_idx] = rate

        # Amplitude stats
        mean_amp = float(np.mean(np.abs(sub_data.amplitudes))) if n_spikes > 0 else 0.0

        organoid_metrics.append({
            "organoid_index": org_idx,
            "n_spikes": n_spikes,
            "firing_rate_hz": round(firing_rate, 3),
            "burst_count": burst_count,
            "isi_cv": round(isi_cv, 4),
            "isi_mean_ms": round(isi_mean_ms, 2),
            "mean_amplitude_uv": round(mean_amp, 2),
            "n_active_electrodes": sub_data.n_electrodes,
        })

    # Rank by composite score (rate + burst_count + complexity via ISI_CV)
    for m in organoid_metrics:
        rates = [x["firing_rate_hz"] for x in organoid_metrics]
        max_rate = max(rates) if max(rates) > 0 else 1.0
        m["score"] = round(
            0.4 * (m["firing_rate_hz"] / max_rate)
            + 0.3 * min(m["burst_count"] / 10.0, 1.0)
            + 0.3 * min(m["isi_cv"], 2.0) / 2.0,
            4,
        )

    ranked = sorted(organoid_metrics, key=lambda x: x["score"], reverse=True)
    for rank, m in enumerate(ranked):
        m["rank"] = rank + 1

    best = ranked[0]
    worst = ranked[-1]

    # Cross-organoid correlation matrix
    org_indices = sorted(rate_timeseries.keys())
    n_org = len(org_indices)
    corr_matrix = np.zeros((n_org, n_org))

    for i, idx_i in enumerate(org_indices):
        for j, idx_j in enumerate(org_indices):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif len(rate_timeseries[idx_i]) > 2 and len(rate_timeseries[idx_j]) > 2:
                min_len = min(len(rate_timeseries[idx_i]), len(rate_timeseries[idx_j]))
                r, p = scipy_stats.pearsonr(
                    rate_timeseries[idx_i][:min_len],
                    rate_timeseries[idx_j][:min_len],
                )
                corr_matrix[i, j] = float(r) if not np.isnan(r) else 0.0

    mean_cross_corr = float(np.mean(corr_matrix[np.triu_indices(n_org, k=1)]))

    return {
        "n_organoids": n_org,
        "electrodes_per_organoid": electrodes_per_organoid,
        "organoid_metrics": ranked,
        "best_organoid": {
            "index": best["organoid_index"],
            "score": best["score"],
            "firing_rate_hz": best["firing_rate_hz"],
        },
        "worst_organoid": {
            "index": worst["organoid_index"],
            "score": worst["score"],
            "firing_rate_hz": worst["firing_rate_hz"],
        },
        "cross_organoid_correlation": corr_matrix.tolist(),
        "mean_cross_correlation": round(mean_cross_corr, 4),
        "organoid_indices": org_indices,
        "synchrony_interpretation": (
            "Highly synchronized" if mean_cross_corr > 0.7
            else "Moderately synchronized" if mean_cross_corr > 0.3
            else "Weakly synchronized" if mean_cross_corr > 0.0
            else "Anti-correlated"
        ),
    }
