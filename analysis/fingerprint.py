"""Organoid Fingerprinting — unique identity signature for each organoid.

NOVEL: Nobody creates reproducible "fingerprints" of organoids.

Each organoid has unique:
- Firing rate distribution across electrodes
- Burst patterns
- Connectivity topology
- ISI statistics
- Spectral profile

The fingerprint is a compact vector that uniquely identifies an organoid.
Useful for:
- Tracking changes over time (aging, learning, degradation)
- Comparing organoids (which is best for a task?)
- Quality control (is this the same organoid as yesterday?)
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def compute_fingerprint(data: SpikeData) -> dict:
    """Compute a unique fingerprint vector for this organoid/recording.

    Returns a 64-dimensional feature vector that characterizes the
    organoid's computational identity.
    """
    if data.n_spikes < 50:
        return {"error": "Not enough spikes for fingerprinting"}

    features = {}
    feature_vector = []
    feature_names = []

    # 1. Firing rate profile (8 dims — rate per electrode, sorted)
    rates = []
    for e in data.electrode_ids:
        n = int(np.sum(data.electrodes == e))
        rates.append(n / data.duration if data.duration > 0 else 0)

    rates_sorted = sorted(rates, reverse=True)
    # Normalize to sum=1
    total_rate = sum(rates_sorted)
    rates_norm = [r / total_rate if total_rate > 0 else 0 for r in rates_sorted]

    for i, r in enumerate(rates_norm[:8]):
        feature_vector.append(r)
        feature_names.append(f"rate_rank_{i}")
    # Pad to 8 if fewer electrodes
    while len(feature_vector) < 8:
        feature_vector.append(0)
        feature_names.append(f"rate_rank_{len(feature_vector) - 1}")

    features["firing_rate_profile"] = rates_norm[:8]
    features["rate_entropy"] = round(float(-sum(r * np.log2(r + 1e-10) for r in rates_norm if r > 0)), 4)

    # 2. ISI statistics (8 dims)
    all_isi = []
    for e in data.electrode_ids:
        e_times = np.sort(data.times[data.electrodes == e])
        if len(e_times) >= 2:
            all_isi.extend(np.diff(e_times).tolist())

    if all_isi:
        all_isi = np.array(all_isi) * 1000  # ms
        isi_features = [
            float(np.mean(all_isi)),
            float(np.std(all_isi)),
            float(np.median(all_isi)),
            float(np.std(all_isi) / np.mean(all_isi)) if np.mean(all_isi) > 0 else 0,
            float(np.percentile(all_isi, 5)),
            float(np.percentile(all_isi, 25)),
            float(np.percentile(all_isi, 75)),
            float(np.percentile(all_isi, 95)),
        ]
    else:
        isi_features = [0] * 8

    for i, f in enumerate(isi_features):
        feature_vector.append(f)
    feature_names.extend(["isi_mean", "isi_std", "isi_median", "isi_cv", "isi_p5", "isi_p25", "isi_p75", "isi_p95"])
    features["isi_stats"] = dict(zip(feature_names[-8:], isi_features))

    # 3. Amplitude distribution (8 dims)
    amps = np.abs(data.amplitudes)
    if len(amps) > 0:
        amp_features = [
            float(np.mean(amps)),
            float(np.std(amps)),
            float(np.median(amps)),
            float(np.percentile(amps, 10)),
            float(np.percentile(amps, 90)),
            float(np.max(amps)),
            float(np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else 0,
            float(len(amps[amps > np.mean(amps) + 2 * np.std(amps)]) / len(amps)),  # outlier fraction
        ]
    else:
        amp_features = [0] * 8

    for f in amp_features:
        feature_vector.append(f)
    feature_names.extend(["amp_mean", "amp_std", "amp_median", "amp_p10", "amp_p90", "amp_max", "amp_cv", "amp_outlier_frac"])

    # 4. Burst characteristics (8 dims)
    from .bursts import analyze_bursts
    burst_result = analyze_bursts(data)
    bursts = burst_result.get("network", {}).get("bursts", burst_result.get("bursts", []))

    burst_features = [
        float(burst_result.get("burst_rate_per_min", 0)),
        float(burst_result.get("mean_duration_ms", 0)),
        float(burst_result.get("mean_n_electrodes", 0)),
        float(burst_result.get("total_burst_time_pct", 0)),
        float(burst_result.get("cv_ibi", 0)),
        float(len(bursts)),
        float(burst_result.get("mean_ibi_ms", 0)),
        float(max([b.get("peak_firing_rate", 0) for b in bursts], default=0)),
    ]

    for f in burst_features:
        feature_vector.append(f)
    feature_names.extend(["burst_rate", "burst_dur", "burst_electrodes", "burst_time_pct", "ibi_cv", "n_bursts", "mean_ibi", "peak_burst_rate"])

    # 5. Connectivity fingerprint (8 dims)
    from .connectivity import compute_connectivity_graph, connectivity_to_dict
    conn = connectivity_to_dict(compute_connectivity_graph(data))

    gm = conn.get("graph_metrics", {})
    conn_features = [
        float(gm.get("density", 0)),
        float(gm.get("mean_clustering", 0)),
        float(gm.get("mean_degree", 0)),
        float(conn.get("n_edges", 0)),  # mean_strength not in dict, use n_edges
        float(conn.get("n_edges", 0)),
        0, 0, 0,  # placeholder for future graph metrics
    ]

    # Hub detection: which electrode has most connections?
    nodes = conn.get("nodes", [])
    if nodes:
        degrees = [n.get("degree", 0) for n in nodes]
        conn_features[5] = float(np.max(degrees))  # max degree
        conn_features[6] = float(np.std(degrees))  # degree heterogeneity
        conn_features[7] = float(np.max(degrees) / (np.mean(degrees) + 1e-10))  # hub ratio

    for f in conn_features:
        feature_vector.append(f)
    feature_names.extend(["conn_density", "clustering", "mean_degree", "mean_strength", "n_edges", "max_degree", "degree_std", "hub_ratio"])

    # Normalize feature vector
    fv = np.array(feature_vector)
    fv_norm = fv / (np.linalg.norm(fv) + 1e-10)

    # Hash for quick comparison
    fingerprint_hash = hash(tuple(np.round(fv_norm, 4).tolist())) & 0xFFFFFFFF

    return {
        "fingerprint_vector": fv_norm.tolist(),
        "fingerprint_hash": f"{fingerprint_hash:08x}",
        "feature_names": feature_names,
        "n_features": len(feature_vector),
        "features": features,
        "summary": {
            "dominant_characteristic": (
                "bursty" if burst_features[0] > 5
                else "highly_connected" if conn_features[0] > 0.5
                else "regular" if isi_features[3] < 0.5
                else "sparse" if sum(rates) < 5
                else "balanced"
            ),
            "rate_entropy": features["rate_entropy"],
            "n_electrodes": data.n_electrodes,
        },
    }


def compare_fingerprints(fp1: dict, fp2: dict) -> dict:
    """Compare two organoid fingerprints — how similar are they?"""
    v1 = np.array(fp1["fingerprint_vector"])
    v2 = np.array(fp2["fingerprint_vector"])

    # Cosine similarity
    cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
    # Euclidean distance
    euclidean = float(np.linalg.norm(v1 - v2))
    # Correlation
    correlation = float(np.corrcoef(v1, v2)[0, 1]) if len(v1) > 1 else 0

    return {
        "cosine_similarity": round(cosine, 4),
        "euclidean_distance": round(euclidean, 4),
        "correlation": round(correlation, 4),
        "same_organoid_likely": cosine > 0.9,
        "interpretation": (
            "Very similar — likely same organoid or very similar conditions"
            if cosine > 0.9
            else "Similar — comparable activity patterns"
            if cosine > 0.7
            else "Different — distinct computational profiles"
            if cosine > 0.4
            else "Very different — fundamentally different activity"
        ),
    }
