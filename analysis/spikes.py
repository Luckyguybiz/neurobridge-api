"""Spike analysis module — detection, sorting, waveform analysis.

Implements:
- Spike detection from raw recordings (threshold-based, FinalSpark method)
- Spike sorting via PCA + K-means or HDBSCAN
- Waveform template matching
- Inter-spike interval analysis
- Refractory period violations
"""

import numpy as np
from scipy import signal as sig
from scipy.stats import median_abs_deviation
from typing import Optional
from .loader import SpikeData


def detect_spikes(
    raw_signal: np.ndarray,
    sampling_rate: float = 30000.0,
    threshold_factor: float = 6.0,
    window_ms: float = 30.0,
    refractory_ms: float = 1.0,
    waveform_pre_ms: float = 1.0,
    waveform_post_ms: float = 2.0,
) -> dict:
    """Detect spikes from single-channel raw signal.

    FinalSpark method: T = threshold_factor × Median({σi})
    where σi is std dev over window_ms windows.

    Returns dict with times, amplitudes, waveforms arrays.
    """
    n_samples = len(raw_signal)
    window_samples = int(window_ms * sampling_rate / 1000)
    refractory_samples = int(refractory_ms * sampling_rate / 1000)
    pre_samples = int(waveform_pre_ms * sampling_rate / 1000)
    post_samples = int(waveform_post_ms * sampling_rate / 1000)

    # Bandpass filter 300-3000 Hz (spike frequency band)
    sos = sig.butter(4, [300, 3000], btype="band", fs=sampling_rate, output="sos")
    filtered = sig.sosfiltfilt(sos, raw_signal)

    # Compute threshold using windowed std dev
    n_windows = max(1, n_samples // window_samples)
    stds = np.array([
        np.std(filtered[i * window_samples:(i + 1) * window_samples])
        for i in range(n_windows)
    ])
    threshold = threshold_factor * np.median(stds)

    # Find negative peaks below threshold
    below_threshold = filtered < -threshold
    crossings = np.where(np.diff(below_threshold.astype(int)) == 1)[0]

    times = []
    amplitudes = []
    waveforms = []
    last_spike = -refractory_samples

    for cx in crossings:
        # Enforce refractory period
        if cx - last_spike < refractory_samples:
            continue

        # Find negative peak within 1.5ms
        search_end = min(cx + int(1.5 * sampling_rate / 1000), n_samples)
        peak_idx = cx + np.argmin(filtered[cx:search_end])

        # Extract waveform
        wf_start = peak_idx - pre_samples
        wf_end = peak_idx + post_samples
        if wf_start >= 0 and wf_end < n_samples:
            waveform = filtered[wf_start:wf_end].copy()
            waveforms.append(waveform)
            times.append(peak_idx / sampling_rate)
            amplitudes.append(float(filtered[peak_idx]))
            last_spike = peak_idx

    return {
        "times": np.array(times),
        "amplitudes": np.array(amplitudes),
        "waveforms": np.array(waveforms) if waveforms else np.array([]).reshape(0, pre_samples + post_samples),
        "threshold": threshold,
        "n_spikes": len(times),
    }


def sort_spikes(
    waveforms: np.ndarray,
    method: str = "pca_kmeans",
    n_components: int = 3,
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 20,
) -> dict:
    """Sort spikes by waveform shape using dimensionality reduction + clustering.

    Methods:
    - pca_kmeans: PCA + K-means (fast, deterministic)
    - pca_hdbscan: PCA + HDBSCAN (better for unknown cluster count)

    Returns dict with labels, centroids, explained_variance.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if len(waveforms) < min_cluster_size:
        return {"labels": np.zeros(len(waveforms), dtype=int), "n_clusters": 1, "method": method}

    # Normalize waveforms
    scaler = StandardScaler()
    wf_scaled = scaler.fit_transform(waveforms)

    # PCA
    n_comp = min(n_components, wf_scaled.shape[1], len(wf_scaled) - 1)
    pca = PCA(n_components=n_comp)
    features = pca.fit_transform(wf_scaled)

    if method == "pca_kmeans":
        from sklearn.cluster import KMeans
        if n_clusters is None:
            # Auto-determine cluster count using silhouette score
            from sklearn.metrics import silhouette_score
            best_k, best_score = 2, -1
            max_k = min(8, len(waveforms) // min_cluster_size)
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(features)
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_k, best_score = k, score
            n_clusters = best_k

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(features)
        centroids = km.cluster_centers_

    elif method == "pca_hdbscan":
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(features)
        centroids = np.array([features[labels == l].mean(axis=0) for l in set(labels) if l >= 0])

    else:
        raise ValueError(f"Unknown sorting method: {method}")

    # Compute mean waveform per cluster
    unique_labels = sorted(set(labels))
    mean_waveforms = {int(l): waveforms[labels == l].mean(axis=0).tolist() for l in unique_labels if l >= 0}

    return {
        "labels": labels.tolist(),
        "n_clusters": len([l for l in unique_labels if l >= 0]),
        "pca_features": features.tolist(),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "mean_waveforms": mean_waveforms,
        "method": method,
    }


def compute_isi(data: SpikeData, electrode: Optional[int] = None) -> dict:
    """Compute inter-spike intervals for one or all electrodes.

    Returns distribution stats and histogram data.
    """
    results = {}

    electrodes = [electrode] if electrode is not None else data.electrode_ids

    for e in electrodes:
        spike_times = data.times[data.electrodes == e]
        if len(spike_times) < 2:
            results[e] = {"n_intervals": 0}
            continue

        isi = np.diff(spike_times) * 1000  # Convert to ms
        isi = isi[isi > 0]

        # Histogram (log-spaced bins from 0.5ms to 10s)
        bins = np.logspace(np.log10(0.5), np.log10(10000), 80)
        hist, bin_edges = np.histogram(isi, bins=bins)

        # Refractory period violations (ISI < 2ms)
        violations = int(np.sum(isi < 2.0))

        results[e] = {
            "n_intervals": len(isi),
            "mean_ms": float(np.mean(isi)),
            "median_ms": float(np.median(isi)),
            "std_ms": float(np.std(isi)),
            "cv": float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0,
            "min_ms": float(np.min(isi)),
            "max_ms": float(np.max(isi)),
            "percentile_5": float(np.percentile(isi, 5)),
            "percentile_95": float(np.percentile(isi, 95)),
            "refractory_violations": violations,
            "violation_rate": violations / len(isi) if len(isi) > 0 else 0,
            "histogram": {"counts": hist.tolist(), "bin_edges": bin_edges.tolist()},
        }

    return results


def compute_firing_rates(
    data: SpikeData,
    bin_size_sec: float = 1.0,
) -> dict:
    """Compute firing rates over time for each electrode.

    Returns time-binned firing rates (spikes/second).
    """
    if data.n_spikes == 0:
        return {"bins": [], "rates": {}}

    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    rates = {}
    for e in data.electrode_ids:
        spike_times = data.times[data.electrodes == e]
        counts, _ = np.histogram(spike_times, bins=bins)
        rates[e] = (counts / bin_size_sec).tolist()

    return {
        "bins": bins[:-1].tolist(),
        "bin_size_sec": bin_size_sec,
        "rates": rates,
        "mean_rates": {e: float(np.mean(r)) for e, r in rates.items()},
    }


def compute_amplitude_stats(data: SpikeData) -> dict:
    """Compute amplitude distribution statistics per electrode."""
    results = {}
    for e in data.electrode_ids:
        amps = data.amplitudes[data.electrodes == e]
        if len(amps) == 0:
            continue
        results[e] = {
            "n_spikes": len(amps),
            "mean_uv": float(np.mean(amps)),
            "std_uv": float(np.std(amps)),
            "median_uv": float(np.median(amps)),
            "mad_uv": float(median_abs_deviation(amps)),
            "min_uv": float(np.min(amps)),
            "max_uv": float(np.max(amps)),
            "percentile_10": float(np.percentile(amps, 10)),
            "percentile_90": float(np.percentile(amps, 90)),
            "histogram": {
                "counts": np.histogram(amps, bins=50)[0].tolist(),
                "bin_edges": np.histogram(amps, bins=50)[1].tolist(),
            },
        }
    return results
