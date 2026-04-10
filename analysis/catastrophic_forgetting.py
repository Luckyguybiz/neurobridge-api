"""Catastrophic forgetting analysis for biological neural networks.

When an organoid learns task B after task A, does it forget task A?
This is "catastrophic forgetting" -- a major unsolved problem in both
biological and artificial neural networks.

Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017, PNAS) showed
that protecting important synaptic weights prevents forgetting in ANNs.
The biological analogue: synaptic consolidation via protein synthesis.

In organoids, we can measure forgetting by tracking how connectivity
patterns (cross-correlation matrices) change over time. If early patterns
are destroyed when new ones form -- that's catastrophic forgetting.
If they coexist -- the organoid has multi-task memory.

This directly addresses the 45-minute memory barrier in FinalSpark organoids.

References:
- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural
  networks" PNAS 114(13):3521-3526
- French (1999) "Catastrophic forgetting in connectionist networks"
  Trends in Cognitive Sciences 3(4):128-135
- McCloskey & Cohen (1989) "Catastrophic interference in connectionist
  networks" Psychology of Learning and Motivation 24:109-165
"""

import numpy as np
from scipy.spatial.distance import correlation as dist_correlation
from scipy.stats import pearsonr
from typing import Optional
from .loader import SpikeData


def _build_xcorr_matrix(
    data: SpikeData,
    max_lag_ms: float = 20.0,
    bin_ms: float = 5.0,
) -> np.ndarray:
    """Build cross-correlation matrix for all electrode pairs.

    Each entry (i, j) is the peak normalized cross-correlation
    between spike trains of electrodes i and j within max_lag.

    Returns:
        Symmetric matrix of shape (n_electrodes, n_electrodes).
    """
    electrode_ids = data.electrode_ids
    n = len(electrode_ids)
    t_start, t_end = data.time_range

    if t_end <= t_start or data.n_spikes < 10:
        return np.zeros((n, n))

    bin_sec = bin_ms / 1000.0
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins = len(bins) - 1

    if n_bins < 2:
        return np.zeros((n, n))

    # Bin spike trains
    binned = np.zeros((n, n_bins))
    for i, eid in enumerate(electrode_ids):
        mask = data.electrodes == eid
        counts, _ = np.histogram(data.times[mask], bins=bins)
        binned[i] = counts

    max_lag_bins = max(1, int(max_lag_ms / bin_ms))
    xcorr = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if np.sum(binned[i]) < 3 or np.sum(binned[j]) < 3:
                continue
            # Normalized cross-correlation at multiple lags
            norm_i = np.linalg.norm(binned[i])
            norm_j = np.linalg.norm(binned[j])
            if norm_i == 0 or norm_j == 0:
                continue

            best_corr = 0.0
            for lag in range(-max_lag_bins, max_lag_bins + 1):
                if lag >= 0:
                    seg_i = binned[i, :n_bins - lag]
                    seg_j = binned[j, lag:]
                else:
                    seg_i = binned[i, -lag:]
                    seg_j = binned[j, :n_bins + lag]
                if len(seg_i) < 2:
                    continue
                ni = np.linalg.norm(seg_i)
                nj = np.linalg.norm(seg_j)
                if ni > 0 and nj > 0:
                    c = float(np.dot(seg_i, seg_j) / (ni * nj))
                    if abs(c) > abs(best_corr):
                        best_corr = c

            xcorr[i, j] = best_corr
            xcorr[j, i] = best_corr

    return xcorr


def _matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized distance between two matrices.

    Uses 1 - cosine_similarity of flattened upper triangles.
    Returns value in [0, 1]: 0 = identical, 1 = orthogonal.
    """
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices_from(a, k=1)
    vec_a = a[idx]
    vec_b = b[idx]

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0

    cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, 1.0 - cosine_sim))


def measure_forgetting(
    data: SpikeData,
    window_pairs: Optional[list[tuple[tuple[float, float], tuple[float, float]]]] = None,
    max_lag_ms: float = 20.0,
) -> dict:
    """Measure catastrophic forgetting between time windows.

    Splits data into time segments and measures how connectivity patterns
    (cross-correlation matrices) from earlier windows are preserved or
    destroyed in later windows.

    If window_pairs is None, automatically creates sequential pairs from
    equal-length windows spanning the full recording.

    Args:
        data: Spike data to analyze.
        window_pairs: List of ((start_a, end_a), (start_b, end_b)) tuples.
            Each pair compares the connectivity pattern from window A
            (earlier) to window B (later).
        max_lag_ms: Maximum lag for cross-correlation computation.

    Returns:
        Dict with keys:
            - pairs: List of per-pair results (distance, retention, times).
            - mean_forgetting: Average forgetting score (0 = no forgetting, 1 = total).
            - mean_retention: Average retention score (1 - forgetting).
            - has_catastrophic_forgetting: True if mean forgetting > 0.5.
            - interpretation: Human-readable summary.
    """
    if data.n_spikes < 50:
        return {
            "error": "Not enough spikes for forgetting analysis",
            "pairs": [],
            "mean_forgetting": 0.0,
            "mean_retention": 0.0,
            "has_catastrophic_forgetting": False,
        }

    t_start, t_end = data.time_range
    duration = t_end - t_start

    # Auto-generate window pairs if not provided
    if window_pairs is None:
        n_auto_windows = 4
        win_len = duration / n_auto_windows
        if win_len < 0.5:
            return {
                "error": "Recording too short for automatic window pairing",
                "pairs": [],
                "mean_forgetting": 0.0,
                "mean_retention": 0.0,
                "has_catastrophic_forgetting": False,
            }
        window_pairs = []
        for i in range(n_auto_windows - 1):
            w_a = (t_start + i * win_len, t_start + (i + 1) * win_len)
            for j in range(i + 1, n_auto_windows):
                w_b = (t_start + j * win_len, t_start + (j + 1) * win_len)
                window_pairs.append((w_a, w_b))

    pair_results = []
    for (start_a, end_a), (start_b, end_b) in window_pairs:
        data_a = data.get_time_range(start_a, end_a)
        data_b = data.get_time_range(start_b, end_b)

        xcorr_a = _build_xcorr_matrix(data_a, max_lag_ms=max_lag_ms)
        xcorr_b = _build_xcorr_matrix(data_b, max_lag_ms=max_lag_ms)

        # Ensure same shape
        min_n = min(xcorr_a.shape[0], xcorr_b.shape[0])
        if min_n < 2:
            continue
        xcorr_a = xcorr_a[:min_n, :min_n]
        xcorr_b = xcorr_b[:min_n, :min_n]

        dist = _matrix_distance(xcorr_a, xcorr_b)
        retention = 1.0 - dist
        time_gap = start_b - end_a

        pair_results.append({
            "window_a": (round(start_a, 3), round(end_a, 3)),
            "window_b": (round(start_b, 3), round(end_b, 3)),
            "time_gap_sec": round(time_gap, 3),
            "distance": round(dist, 4),
            "retention": round(retention, 4),
            "forgetting": round(dist, 4),
            "spikes_a": data_a.n_spikes,
            "spikes_b": data_b.n_spikes,
        })

    if not pair_results:
        return {
            "pairs": [],
            "mean_forgetting": 0.0,
            "mean_retention": 0.0,
            "has_catastrophic_forgetting": False,
            "interpretation": "Could not compute forgetting -- insufficient data in windows.",
        }

    forgetting_scores = [p["forgetting"] for p in pair_results]
    mean_forgetting = float(np.mean(forgetting_scores))
    mean_retention = 1.0 - mean_forgetting

    return {
        "pairs": pair_results,
        "mean_forgetting": round(mean_forgetting, 4),
        "mean_retention": round(mean_retention, 4),
        "max_forgetting": round(float(np.max(forgetting_scores)), 4),
        "min_forgetting": round(float(np.min(forgetting_scores)), 4),
        "has_catastrophic_forgetting": mean_forgetting > 0.5,
        "n_pairs": len(pair_results),
        "interpretation": (
            f"CATASTROPHIC FORGETTING DETECTED -- mean forgetting score {mean_forgetting:.2f}. "
            f"Connectivity patterns from earlier time windows are largely destroyed in later windows. "
            f"The organoid does not preserve its earlier connectivity structure."
            if mean_forgetting > 0.5
            else f"MODERATE FORGETTING -- score {mean_forgetting:.2f}. "
            f"Some connectivity patterns are preserved across time windows, "
            f"but partial degradation is visible."
            if mean_forgetting > 0.25
            else f"LOW FORGETTING -- score {mean_forgetting:.2f}. "
            f"Connectivity patterns are well preserved across time windows. "
            f"The organoid maintains stable network structure -- evidence of memory retention."
        ),
    }


def compute_retention_curve(
    data: SpikeData,
    n_windows: int = 10,
    max_lag_ms: float = 20.0,
) -> dict:
    """Track how much of early-window patterns survive across subsequent windows.

    Divides the recording into n_windows equal segments. Uses the first
    window as the reference pattern. Measures how similar each subsequent
    window's connectivity is to the reference.

    This produces a retention curve: if it decays to 0 quickly, the organoid
    forgets. If it stays high, it remembers. The shape of the curve reveals
    the forgetting dynamics (exponential, gradual, sudden).

    Args:
        data: Spike data to analyze.
        n_windows: Number of time windows to divide the recording into.
        max_lag_ms: Maximum lag for cross-correlation computation.

    Returns:
        Dict with keys:
            - retention_scores: List of retention values (0-1) per window.
            - window_centers: Time center of each window in seconds.
            - reference_window: (start, end) of the reference window.
            - half_life_sec: Estimated time for retention to drop to 0.5.
            - decay_type: "exponential", "gradual", "sudden", or "stable".
            - final_retention: Retention score of the last window.
    """
    t_start, t_end = data.time_range
    duration = t_end - t_start
    win_len = duration / n_windows

    if win_len < 0.2 or data.n_spikes < 50:
        return {
            "error": "Recording too short or too few spikes for retention curve",
            "retention_scores": [],
            "window_centers": [],
        }

    # Build reference matrix from first window
    ref_data = data.get_time_range(t_start, t_start + win_len)
    ref_xcorr = _build_xcorr_matrix(ref_data, max_lag_ms=max_lag_ms)

    if np.linalg.norm(ref_xcorr) < 1e-12:
        return {
            "error": "Reference window has no significant correlations",
            "retention_scores": [],
            "window_centers": [],
        }

    retention_scores = [1.0]  # First window is the reference (perfect match)
    window_centers = [round(t_start + win_len / 2, 3)]

    for i in range(1, n_windows):
        w_start = t_start + i * win_len
        w_end = w_start + win_len
        w_data = data.get_time_range(w_start, w_end)

        xcorr = _build_xcorr_matrix(w_data, max_lag_ms=max_lag_ms)

        min_n = min(ref_xcorr.shape[0], xcorr.shape[0])
        if min_n < 2:
            retention_scores.append(0.0)
        else:
            dist = _matrix_distance(
                ref_xcorr[:min_n, :min_n],
                xcorr[:min_n, :min_n],
            )
            retention_scores.append(round(1.0 - dist, 4))

        window_centers.append(round(w_start + win_len / 2, 3))

    # Estimate half-life (when retention drops below 0.5)
    half_life_sec = None
    for i, score in enumerate(retention_scores):
        if score < 0.5:
            half_life_sec = round(window_centers[i] - window_centers[0], 3)
            break

    # Classify decay type
    final_retention = retention_scores[-1]
    mid_retention = retention_scores[len(retention_scores) // 2] if len(retention_scores) > 2 else final_retention

    if final_retention > 0.7:
        decay_type = "stable"
    elif mid_retention > 0.6 and final_retention < 0.3:
        decay_type = "sudden"
    else:
        # Check if decay looks exponential: log(retention) should be linear
        valid_scores = [s for s in retention_scores if s > 0.01]
        if len(valid_scores) >= 3:
            log_scores = np.log(np.array(valid_scores) + 1e-10)
            xs = np.arange(len(log_scores))
            if len(xs) >= 3:
                r, _ = pearsonr(xs, log_scores)
                decay_type = "exponential" if abs(r) > 0.85 else "gradual"
            else:
                decay_type = "gradual"
        else:
            decay_type = "gradual"

    return {
        "retention_scores": retention_scores,
        "window_centers": window_centers,
        "reference_window": (round(t_start, 3), round(t_start + win_len, 3)),
        "window_duration_sec": round(win_len, 3),
        "half_life_sec": half_life_sec,
        "decay_type": decay_type,
        "final_retention": round(final_retention, 4),
        "mean_retention": round(float(np.mean(retention_scores)), 4),
        "n_windows": n_windows,
        "interpretation": (
            f"STABLE MEMORY -- retention remains at {final_retention:.2f} after full recording. "
            f"The organoid preserves its initial connectivity pattern. "
            f"No catastrophic forgetting observed."
            if decay_type == "stable"
            else f"SUDDEN FORGETTING at ~{half_life_sec:.1f}s. "
            f"Pattern retained until midpoint ({mid_retention:.2f}) then collapsed ({final_retention:.2f}). "
            f"Suggests a phase transition or catastrophic event."
            if decay_type == "sudden"
            else f"EXPONENTIAL DECAY -- half-life {half_life_sec:.1f}s. "
            f"Pattern fades exponentially, consistent with passive synaptic decay. "
            f"Final retention: {final_retention:.2f}."
            if decay_type == "exponential" and half_life_sec is not None
            else f"GRADUAL FORGETTING -- final retention {final_retention:.2f}. "
            f"Connectivity pattern slowly degrades over time."
        ),
    }
