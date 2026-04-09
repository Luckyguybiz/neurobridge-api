"""ML pipeline — anomaly detection, state classification, dimensionality reduction.

Advanced machine learning for organoid neural data:
- Anomaly detection (Isolation Forest)
- State classification (active, resting, bursting, transitional)
- PCA/UMAP dimensionality reduction of neural state space
- Feature extraction for downstream analysis
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def extract_features(
    data: SpikeData,
    window_sec: float = 1.0,
) -> dict:
    """Extract multi-scale features from spike data in sliding windows.

    Features per window per electrode:
    - Spike count, firing rate
    - Mean/std amplitude
    - ISI mean, CV
    - Burst indicator
    """
    t_start, t_end = data.time_range
    windows = np.arange(t_start, t_end, window_sec)
    n_windows = len(windows)
    n_electrodes = data.n_electrodes

    # Feature matrix: (n_windows, n_features)
    feature_names = []
    for e in data.electrode_ids:
        feature_names.extend([
            f"e{e}_rate", f"e{e}_mean_amp", f"e{e}_std_amp", f"e{e}_cv_isi",
        ])
    feature_names.extend(["total_rate", "n_active_electrodes", "synchrony_index"])

    n_features = len(feature_names)
    features = np.zeros((n_windows, n_features))

    for w_idx, w_start in enumerate(windows):
        w_end = w_start + window_sec
        mask = (data.times >= w_start) & (data.times < w_end)
        w_times = data.times[mask]
        w_electrodes = data.electrodes[mask]
        w_amplitudes = data.amplitudes[mask]

        f_idx = 0
        n_active = 0
        total_spikes = len(w_times)

        for e in data.electrode_ids:
            e_mask = w_electrodes == e
            n_spikes = int(np.sum(e_mask))

            features[w_idx, f_idx] = n_spikes / window_sec  # rate
            f_idx += 1

            if n_spikes > 0:
                features[w_idx, f_idx] = np.mean(w_amplitudes[e_mask])  # mean amp
                features[w_idx, f_idx + 1] = np.std(w_amplitudes[e_mask]) if n_spikes > 1 else 0  # std amp
                n_active += 1
            f_idx += 2

            # CV ISI
            if n_spikes >= 2:
                e_times = np.sort(w_times[e_mask])
                isi = np.diff(e_times)
                cv = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0
                features[w_idx, f_idx] = cv
            f_idx += 1

        # Population features
        features[w_idx, f_idx] = total_spikes / window_sec
        features[w_idx, f_idx + 1] = n_active
        # Synchrony index (Fano factor of spike counts across electrodes)
        if n_active > 0:
            counts = np.array([np.sum(w_electrodes == e) for e in data.electrode_ids])
            mean_c = np.mean(counts)
            features[w_idx, f_idx + 2] = np.var(counts) / mean_c if mean_c > 0 else 0

    return {
        "features": features.tolist(),
        "feature_names": feature_names,
        "n_windows": n_windows,
        "n_features": n_features,
        "window_sec": window_sec,
        "window_times": windows.tolist(),
    }


def detect_anomalies(
    data: SpikeData,
    window_sec: float = 1.0,
    contamination: float = 0.05,
) -> dict:
    """Detect anomalous time windows using Isolation Forest.

    Anomalies = unusual neural activity patterns (potential artifacts,
    seizure-like events, or interesting biological phenomena).
    """
    from sklearn.ensemble import IsolationForest

    feat_result = extract_features(data, window_sec)
    features = np.array(feat_result["features"])

    if len(features) < 10:
        return {"error": "Not enough data windows for anomaly detection"}

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = clf.fit_predict(features)
    scores = clf.score_samples(features)

    anomaly_windows = []
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == -1:
            anomaly_windows.append({
                "window_idx": int(i),
                "time_start": round(feat_result["window_times"][i], 3),
                "time_end": round(feat_result["window_times"][i] + window_sec, 3),
                "anomaly_score": round(float(-score), 4),
            })

    anomaly_windows.sort(key=lambda x: x["anomaly_score"], reverse=True)

    return {
        "anomalies": anomaly_windows,
        "n_anomalies": len(anomaly_windows),
        "n_windows_total": len(features),
        "anomaly_rate": round(len(anomaly_windows) / len(features), 4) if len(features) > 0 else 0,
        "anomaly_scores": (-scores).tolist(),
        "threshold": round(float(-clf.offset_), 4),
    }


def classify_states(
    data: SpikeData,
    window_sec: float = 2.0,
) -> dict:
    """Classify neural activity states using unsupervised clustering.

    States: resting, active, bursting, transitional.
    Uses K-means on extracted features with automatic state labeling.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feat_result = extract_features(data, window_sec)
    features = np.array(feat_result["features"])

    if len(features) < 10:
        return {"error": "Not enough data"}

    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Find optimal k (2-5 states)
    from sklearn.metrics import silhouette_score
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(features) // 3)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(scaled)

    # Label states based on mean firing rate
    total_rate_idx = feat_result["feature_names"].index("total_rate")
    state_rates = {}
    for c in range(best_k):
        mask = labels == c
        mean_rate = float(np.mean(features[mask, total_rate_idx]))
        state_rates[c] = mean_rate

    # Sort by rate and assign names
    sorted_states = sorted(state_rates.items(), key=lambda x: x[1])
    state_names = {}
    names = ["resting", "low_activity", "active", "bursting", "high_bursting"]
    for i, (state_id, _) in enumerate(sorted_states):
        state_names[state_id] = names[min(i, len(names) - 1)]

    # Build timeline
    timeline = []
    for i, label in enumerate(labels):
        timeline.append({
            "window_idx": int(i),
            "time_start": round(feat_result["window_times"][i], 3),
            "state": state_names[int(label)],
            "state_id": int(label),
        })

    # State statistics
    state_stats = {}
    for state_id, name in state_names.items():
        mask = labels == state_id
        state_stats[name] = {
            "n_windows": int(np.sum(mask)),
            "fraction": round(float(np.sum(mask) / len(labels)), 3),
            "mean_firing_rate": round(float(np.mean(features[mask, total_rate_idx])), 2),
        }

    return {
        "n_states": best_k,
        "silhouette_score": round(float(best_score), 4),
        "states": state_stats,
        "timeline": timeline,
        "state_names": {int(k): v for k, v in state_names.items()},
    }


def compute_pca_embedding(
    data: SpikeData,
    window_sec: float = 1.0,
    n_components: int = 3,
) -> dict:
    """PCA dimensionality reduction of neural state space.

    Projects high-dimensional feature space into 2D/3D for visualization.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    feat_result = extract_features(data, window_sec)
    features = np.array(feat_result["features"])
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    if len(features) < n_components + 1:
        return {"error": "Not enough data"}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    n_comp = min(n_components, scaled.shape[1], len(scaled) - 1)
    pca = PCA(n_components=n_comp)
    embedding = pca.fit_transform(scaled)

    return {
        "embedding": embedding.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "n_components": n_comp,
        "window_times": feat_result["window_times"],
        "top_features": [
            {"feature": feat_result["feature_names"][i], "loading": round(float(pca.components_[0, i]), 4)}
            for i in np.argsort(np.abs(pca.components_[0]))[-5:]
        ],
    }
