"""Transfer learning analysis for biological neural networks.

Does learning one pattern help (or hinder) learning another?
Positive transfer = shared representations accelerate new learning.
Negative transfer = interference slows or disrupts new learning.

In organoids, we measure this by asking: does neural activity in
segment N predict activity in segment N+1 better than chance?
If yes -- the network has built transferable representations.

Representational Similarity Analysis (RSA, Kriegeskorte et al. 2008)
provides a framework for comparing neural representations across
time, conditions, and even across different networks.

This is critical for NeuroBridge: if organoid networks show positive
transfer, they can be pre-trained on simple tasks before deployment
on complex ones -- dramatically reducing setup time.

References:
- Kriegeskorte et al. (2008) "Representational similarity analysis"
  Frontiers in Systems Neuroscience 2:4
- Kornblith et al. (2019) "Similarity of neural network representations
  revisited" ICML 2019
- Pan & Yang (2010) "A survey on transfer learning"
  IEEE TKDE 22(10):1345-1359
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Optional
from .loader import SpikeData


def _bin_spike_trains(
    data: SpikeData,
    bin_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin spike trains into population activity vectors.

    Returns:
        binned: Array of shape (n_electrodes, n_bins).
        bin_centers: Array of bin center times in seconds.
    """
    t_start, t_end = data.time_range
    bin_sec = bin_ms / 1000.0
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins = len(bins) - 1
    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)

    binned = np.zeros((n_electrodes, n_bins))
    for i, eid in enumerate(electrode_ids):
        mask = data.electrodes == eid
        counts, _ = np.histogram(data.times[mask], bins=bins)
        binned[i] = counts

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return binned, bin_centers


def _ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Simple Ridge regression prediction without sklearn dependency.

    Solves: w = (X^T X + alpha I)^{-1} X^T y

    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training targets (n_samples, n_targets).
        X_test: Test features (m_samples, n_features).
        alpha: Regularization strength.

    Returns:
        Predictions of shape (m_samples, n_targets).
    """
    n_features = X_train.shape[1]
    XtX = X_train.T @ X_train + alpha * np.eye(n_features)
    try:
        w = np.linalg.solve(XtX, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, X_train.T @ y_train, rcond=None)[0]
    return X_test @ w


def measure_transfer(
    data: SpikeData,
    n_segments: int = 4,
    bin_ms: float = 50.0,
    alpha: float = 1.0,
) -> dict:
    """Measure transfer learning between consecutive time segments.

    Splits the recording into n_segments. For each consecutive pair (N, N+1):
    1. Train Ridge regression on segment N to predict neural activity
       from time-lagged activity.
    2. Test on segment N+1. If R^2 > random baseline, then patterns
       from N generalize to N+1 (positive transfer).
    3. Random baseline: train on shuffled segment N, test on N+1.

    Args:
        data: Spike data to analyze.
        n_segments: Number of equal-length segments.
        bin_ms: Bin size for spike train binning.
        alpha: Ridge regression regularization parameter.

    Returns:
        Dict with keys:
            - transfers: Per-pair results with R^2 scores and transfer type.
            - mean_transfer_score: Average R^2 gain over baseline.
            - has_positive_transfer: True if mean gain > 0.
            - has_negative_transfer: True if any pair shows interference.
    """
    if data.n_spikes < 100:
        return {
            "error": "Not enough spikes for transfer learning analysis",
            "transfers": [],
            "mean_transfer_score": 0.0,
        }

    binned, bin_centers = _bin_spike_trains(data, bin_ms=bin_ms)
    n_electrodes, n_bins = binned.shape
    seg_len = n_bins // n_segments

    if seg_len < 10 or n_electrodes < 2:
        return {
            "error": "Segments too short or too few electrodes",
            "transfers": [],
            "mean_transfer_score": 0.0,
        }

    transfers = []

    for seg_idx in range(n_segments - 1):
        # Segment N (train) and N+1 (test)
        s_train = binned[:, seg_idx * seg_len:(seg_idx + 1) * seg_len]
        s_test = binned[:, (seg_idx + 1) * seg_len:(seg_idx + 2) * seg_len]

        # Build time-lagged features: predict t from t-1
        # X = activity at time t, Y = activity at time t+1
        X_train = s_train[:, :-1].T  # (seg_len-1, n_electrodes)
        Y_train = s_train[:, 1:].T
        X_test = s_test[:, :-1].T
        Y_test = s_test[:, 1:].T

        if X_train.shape[0] < 5:
            continue

        # Train model on segment N, test on segment N+1
        Y_pred = _ridge_predict(X_train, Y_train, X_test, alpha=alpha)

        # R^2 score
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
        r2_transfer = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        r2_transfer = max(-1.0, r2_transfer)  # Clip extreme negatives

        # Baseline: shuffle training data
        n_shuffles = 5
        r2_baselines = []
        for _ in range(n_shuffles):
            perm = np.random.permutation(X_train.shape[0])
            Y_train_shuffled = Y_train[perm]
            Y_pred_shuf = _ridge_predict(X_train, Y_train_shuffled, X_test, alpha=alpha)
            ss_res_shuf = np.sum((Y_test - Y_pred_shuf) ** 2)
            r2_shuf = 1.0 - (ss_res_shuf / ss_tot) if ss_tot > 1e-12 else 0.0
            r2_baselines.append(max(-1.0, r2_shuf))

        r2_baseline = float(np.mean(r2_baselines))
        transfer_gain = r2_transfer - r2_baseline

        t_train_center = bin_centers[seg_idx * seg_len + seg_len // 2]
        t_test_center = bin_centers[(seg_idx + 1) * seg_len + seg_len // 2]

        if transfer_gain > 0.05:
            transfer_type = "positive"
        elif transfer_gain < -0.05:
            transfer_type = "negative"
        else:
            transfer_type = "neutral"

        transfers.append({
            "segment_train": seg_idx,
            "segment_test": seg_idx + 1,
            "time_train_center": round(float(t_train_center), 3),
            "time_test_center": round(float(t_test_center), 3),
            "r2_transfer": round(float(r2_transfer), 4),
            "r2_baseline": round(float(r2_baseline), 4),
            "transfer_gain": round(float(transfer_gain), 4),
            "transfer_type": transfer_type,
        })

    if not transfers:
        return {
            "transfers": [],
            "mean_transfer_score": 0.0,
            "has_positive_transfer": False,
            "has_negative_transfer": False,
            "interpretation": "Could not compute transfer -- insufficient data.",
        }

    gains = [t["transfer_gain"] for t in transfers]
    mean_gain = float(np.mean(gains))
    has_positive = any(t["transfer_type"] == "positive" for t in transfers)
    has_negative = any(t["transfer_type"] == "negative" for t in transfers)

    n_positive = sum(1 for t in transfers if t["transfer_type"] == "positive")
    n_negative = sum(1 for t in transfers if t["transfer_type"] == "negative")
    n_neutral = sum(1 for t in transfers if t["transfer_type"] == "neutral")

    return {
        "transfers": transfers,
        "mean_transfer_score": round(mean_gain, 4),
        "max_transfer_gain": round(float(np.max(gains)), 4),
        "min_transfer_gain": round(float(np.min(gains)), 4),
        "has_positive_transfer": has_positive,
        "has_negative_transfer": has_negative,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": n_neutral,
        "n_segments": n_segments,
        "interpretation": (
            f"POSITIVE TRANSFER -- mean gain {mean_gain:.3f}. "
            f"{n_positive}/{len(transfers)} segment pairs show positive transfer. "
            f"The organoid builds representations that generalize across time -- "
            f"evidence of abstract learning."
            if mean_gain > 0.05
            else f"NEGATIVE TRANSFER (interference) -- mean gain {mean_gain:.3f}. "
            f"{n_negative}/{len(transfers)} pairs show interference. "
            f"New activity patterns disrupt the network's ability to maintain "
            f"predictable dynamics."
            if mean_gain < -0.05
            else f"NEUTRAL -- mean gain {mean_gain:.3f}. "
            f"No significant transfer between segments. Each segment's "
            f"dynamics are largely independent."
        ),
    }


def compute_representational_similarity(
    data: SpikeData,
    window_sec: float = 5.0,
    bin_ms: float = 50.0,
    step_sec: Optional[float] = None,
) -> dict:
    """Representational Similarity Analysis (RSA) across time windows.

    For each time window, compute the Representational Dissimilarity Matrix
    (RDM): a matrix of pairwise distances between neural population states
    (binned activity vectors within the window).

    Then compute second-order similarity: correlate RDMs across windows.
    High RDM correlation = similar representational geometry = the network
    encodes information the same way across time.

    Args:
        data: Spike data to analyze.
        window_sec: Duration of each analysis window.
        bin_ms: Bin size for population activity vectors.
        step_sec: Step between windows (defaults to window_sec / 2).

    Returns:
        Dict with keys:
            - rdm_similarities: Matrix of RDM correlations between windows.
            - window_centers: Time centers of each window.
            - mean_similarity: Average RDM correlation (higher = more stable
              representations).
            - representational_stability: Summary score.
    """
    if data.n_spikes < 50:
        return {
            "error": "Not enough spikes for RSA",
            "rdm_similarities": [],
            "window_centers": [],
            "mean_similarity": 0.0,
        }

    if step_sec is None:
        step_sec = window_sec / 2.0

    t_start, t_end = data.time_range
    bin_sec = bin_ms / 1000.0
    n_bins_per_window = max(3, int(window_sec / bin_sec))

    # Collect RDMs per window
    rdms = []
    window_centers = []
    t = t_start

    while t + window_sec <= t_end:
        w_data = data.get_time_range(t, t + window_sec)
        if w_data.n_spikes < 10:
            t += step_sec
            continue

        # Bin activity within this window
        binned, _ = _bin_spike_trains(w_data, bin_ms=bin_ms)
        n_elec, n_bins = binned.shape

        if n_bins < 3 or n_elec < 2:
            t += step_sec
            continue

        # Each time bin is a "condition" -- compute pairwise distances
        # between population vectors at different time bins
        # RDM shape: (n_bins, n_bins)
        states = binned.T  # (n_bins, n_electrodes)

        # Euclidean distance matrix
        rdm = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            for j in range(i + 1, n_bins):
                d = np.linalg.norm(states[i] - states[j])
                rdm[i, j] = d
                rdm[j, i] = d

        rdms.append(rdm)
        window_centers.append(round(t + window_sec / 2.0, 3))
        t += step_sec

    n_windows = len(rdms)
    if n_windows < 2:
        return {
            "rdm_similarities": [],
            "window_centers": window_centers,
            "mean_similarity": 0.0,
            "n_windows": n_windows,
            "interpretation": "Not enough valid windows for RSA comparison.",
        }

    # Compare RDMs across windows using Spearman correlation
    # of upper-triangle vectors (standard RSA approach)
    rdm_sim_matrix = np.zeros((n_windows, n_windows))

    for i in range(n_windows):
        for j in range(i, n_windows):
            # Get upper triangle of each RDM
            min_size = min(rdms[i].shape[0], rdms[j].shape[0])
            if min_size < 3:
                rdm_sim_matrix[i, j] = 0.0
                rdm_sim_matrix[j, i] = 0.0
                continue

            idx = np.triu_indices(min_size, k=1)
            vec_i = rdms[i][:min_size, :min_size][idx]
            vec_j = rdms[j][:min_size, :min_size][idx]

            if len(vec_i) < 3 or np.std(vec_i) < 1e-12 or np.std(vec_j) < 1e-12:
                rdm_sim_matrix[i, j] = 0.0
                rdm_sim_matrix[j, i] = 0.0
                continue

            rho, _ = spearmanr(vec_i, vec_j)
            if np.isnan(rho):
                rho = 0.0
            rdm_sim_matrix[i, j] = rho
            rdm_sim_matrix[j, i] = rho

    # Extract off-diagonal values for summary
    offdiag_idx = np.triu_indices(n_windows, k=1)
    offdiag_vals = rdm_sim_matrix[offdiag_idx]
    mean_sim = float(np.mean(offdiag_vals)) if len(offdiag_vals) > 0 else 0.0

    # Adjacent vs distant window similarities
    adjacent_sims = []
    distant_sims = []
    for i in range(n_windows - 1):
        adjacent_sims.append(float(rdm_sim_matrix[i, i + 1]))
    for i in range(n_windows):
        for j in range(i + 2, n_windows):
            distant_sims.append(float(rdm_sim_matrix[i, j]))

    mean_adjacent = float(np.mean(adjacent_sims)) if adjacent_sims else 0.0
    mean_distant = float(np.mean(distant_sims)) if distant_sims else 0.0

    return {
        "rdm_similarities": rdm_sim_matrix.tolist(),
        "window_centers": window_centers,
        "n_windows": n_windows,
        "window_sec": window_sec,
        "mean_similarity": round(mean_sim, 4),
        "mean_adjacent_similarity": round(mean_adjacent, 4),
        "mean_distant_similarity": round(mean_distant, 4),
        "representational_stability": round(mean_sim, 4),
        "drift_index": round(mean_adjacent - mean_distant, 4) if distant_sims else 0.0,
        "interpretation": (
            f"STABLE REPRESENTATIONS -- mean RDM correlation {mean_sim:.3f}. "
            f"The network maintains similar representational geometry across time. "
            f"Adjacent windows: {mean_adjacent:.3f}, distant: {mean_distant:.3f}."
            if mean_sim > 0.5
            else f"DRIFTING REPRESENTATIONS -- mean RDM correlation {mean_sim:.3f}. "
            f"Representational geometry changes over time. "
            f"Adjacent: {mean_adjacent:.3f}, distant: {mean_distant:.3f}. "
            f"The network reorganizes its internal representations."
            if mean_sim > 0.2
            else f"UNSTABLE REPRESENTATIONS -- mean RDM correlation {mean_sim:.3f}. "
            f"Little consistency in how the network encodes information across time."
        ),
    }
