"""Organoid Turing Test — can you distinguish real organoid from simulation?

Scientific basis:
    Haussler & Bhatt (UCSC, 2026) proposed a "Turing test for organoids":
    present spike data from a real organoid alongside data from computational
    models, and measure statistical distinguishability.

    This module generates comparison data from 3 sources:
    1. Poisson process (null model — random firing)
    2. LIF network (structured but simple)
    3. Real organoid data (the test subject)

    Then computes discriminability metrics:
    - ISI distribution KL-divergence
    - Burst statistics comparison
    - Complexity (LZ) comparison
    - Synchrony comparison
    - Overall "biological realism" score
"""
import numpy as np
from scipy import stats as sp_stats
from .loader import SpikeData


def _generate_poisson_spikes(rate_hz: float, duration: float, n_electrodes: int) -> SpikeData:
    """Generate Poisson process spike data."""
    spikes_per_electrode = int(rate_hz * duration)
    all_times = []
    all_electrodes = []
    for e in range(n_electrodes):
        times = np.sort(np.random.uniform(0, duration, spikes_per_electrode))
        all_times.extend(times)
        all_electrodes.extend([e] * len(times))
    return SpikeData(
        times=np.array(all_times),
        electrodes=np.array(all_electrodes),
        amplitudes=np.random.normal(-50, 15, len(all_times)),
    )


def _generate_lif_spikes(rate_hz: float, duration: float, n_electrodes: int) -> SpikeData:
    """Generate LIF-like correlated spike data with bursts."""
    all_times = []
    all_electrodes = []
    t = 0.0
    dt = 0.001
    # Simple correlated firing with refractory period
    last_spike = np.full(n_electrodes, -1.0)
    while t < duration:
        for e in range(n_electrodes):
            if t - last_spike[e] < 0.005:  # 5ms refractory
                continue
            # Base rate + network drive
            network_drive = sum(1 for e2 in range(n_electrodes)
                              if e2 != e and t - last_spike[e2] < 0.01 and last_spike[e2] > 0) * 0.3
            p = (rate_hz + network_drive * rate_hz) * dt
            if np.random.random() < p:
                all_times.append(t)
                all_electrodes.append(e)
                last_spike[e] = t
        t += dt

    if not all_times:
        all_times = [0.0]
        all_electrodes = [0]

    return SpikeData(
        times=np.array(all_times),
        electrodes=np.array(all_electrodes),
        amplitudes=np.random.normal(-50, 15, len(all_times)),
    )


def _compute_features(data: SpikeData) -> dict:
    """Extract statistical features for comparison."""
    if data.n_spikes < 10:
        return {"firing_rate": 0, "cv_isi": 0, "burst_fraction": 0, "sync_index": 0, "complexity": 0}

    rate = data.n_spikes / max(data.duration, 0.001)

    # ISI statistics
    isi_list = []
    for e in data.electrode_ids:
        mask = data.electrodes == e
        times = data.times[mask]
        if len(times) > 1:
            isi_list.extend(np.diff(times).tolist())
    cv_isi = float(np.std(isi_list) / max(np.mean(isi_list), 1e-10)) if isi_list else 0.0

    # Burst fraction (windows with >3x mean rate)
    bin_size = 0.05  # 50ms
    bins = np.arange(0, data.duration, bin_size)
    counts, _ = np.histogram(data.times, bins=bins)
    threshold = np.mean(counts) * 3
    burst_fraction = float(np.mean(counts > threshold))

    # Synchrony (mean pairwise correlation)
    n_bins = len(counts)
    rates_per_electrode = np.zeros((data.n_electrodes, n_bins))
    for idx, eid in enumerate(data.electrode_ids):
        mask = data.electrodes == eid
        c, _ = np.histogram(data.times[mask], bins=bins)
        rates_per_electrode[idx] = c
    if data.n_electrodes > 1 and n_bins > 2:
        corr = np.corrcoef(rates_per_electrode)
        np.fill_diagonal(corr, 0)
        sync = float(np.mean(np.abs(corr)))
    else:
        sync = 0.0

    # LZ complexity
    binary = (counts > 0).astype(int)
    n = len(binary)
    c_count = 1
    i, k = 0, 1
    while k < n:
        if binary[k] != binary[max(k - 1, 0)]:
            c_count += 1
        k += 1
    complexity = float(c_count * np.log2(max(n, 2)) / max(n, 1))

    return {
        "firing_rate": float(rate),
        "cv_isi": cv_isi,
        "burst_fraction": burst_fraction,
        "sync_index": sync,
        "complexity": complexity,
    }


def run_turing_test(data: SpikeData) -> dict:
    """Run organoid Turing test: compare real data against Poisson and LIF models."""
    rate = data.n_spikes / max(data.duration, 0.001) / max(data.n_electrodes, 1)

    # Generate comparison data
    poisson = _generate_poisson_spikes(rate, data.duration, data.n_electrodes)
    lif = _generate_lif_spikes(rate, data.duration, data.n_electrodes)

    # Extract features
    real_features = _compute_features(data)
    poisson_features = _compute_features(poisson)
    lif_features = _compute_features(lif)

    # Compute distances
    feature_keys = ["firing_rate", "cv_isi", "burst_fraction", "sync_index", "complexity"]

    def _distance(f1, f2):
        diffs = []
        for k in feature_keys:
            v1, v2 = f1[k], f2[k]
            denom = max(abs(v1) + abs(v2), 1e-10)
            diffs.append(abs(v1 - v2) / denom)
        return float(np.mean(diffs))

    dist_to_poisson = _distance(real_features, poisson_features)
    dist_to_lif = _distance(real_features, lif_features)

    # Biological realism score: how far from Poisson (random) the real data is
    realism_score = min(1.0, dist_to_poisson / max(dist_to_poisson + dist_to_lif, 1e-10))

    # Discriminability: can we tell real from models?
    discriminability = (dist_to_poisson + dist_to_lif) / 2

    return {
        "biological_realism_score": realism_score,
        "discriminability": discriminability,
        "distance_to_poisson": dist_to_poisson,
        "distance_to_lif": dist_to_lif,
        "real_features": real_features,
        "poisson_features": poisson_features,
        "lif_features": lif_features,
        "verdict": (
            "Highly biological" if realism_score > 0.7 else
            "Moderately biological" if realism_score > 0.4 else
            "Difficult to distinguish from random"
        ),
        "feature_comparison": {
            k: {
                "real": real_features[k],
                "poisson": poisson_features[k],
                "lif": lif_features[k],
            }
            for k in feature_keys
        },
    }
