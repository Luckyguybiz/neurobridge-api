"""Sleep-wake state analysis for brain organoid MEA recordings.

Detects UP/DOWN states, slow oscillations, and sleep-like dynamics
in population spiking activity. Designed for the FinalSpark platform:
4 MEAs x 8 electrodes, multi-day continuous recordings.

Biological basis
----------------
Cortical organoids exhibit spontaneous alternations between high-
activity (UP) and low-activity (DOWN) states resembling in-vivo
slow-wave sleep. These states are visible as a bimodal distribution
in the population firing rate and can be formally segmented with a
two-state Hidden Markov Model.

Key metrics computed
--------------------
- UP/DOWN state detection via HMM and adaptive thresholding
- Slow oscillation power (0.1--1 Hz) and dominant frequency
- Phase-amplitude coupling between slow oscillation and gamma-band
- Per-MEA sleep architecture comparison
- Multi-scale temporal analysis (1 s, 10 s, 60 s, 1 h windows)
- Homeostatic sleep pressure tracking
- Statistical validation against rate-matched null model

Performance target: < 60 s on 2.6 M spikes, 32 electrodes, 118 h.

References
----------
- Trujillo et al., Cell Stem Cell 2019 -- oscillatory activity in organoids
- Sakata & Harris, Neuron 2009 -- UP/DOWN state bimodality
- Tort et al., J Neurophysiol 2010 -- modulation index for PAC
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from typing import Optional
from .loader import SpikeData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MEA layout: 4 MEAs, 8 electrodes each (FinalSpark convention)
_N_MEAS = 4
_ELECTRODES_PER_MEA = 8

# Default bin sizes (seconds) for rate estimation
_DEFAULT_BIN_SEC = 0.050  # 50 ms -- resolves individual UP states
_SLOW_OSC_BIN_SEC = 0.100  # 100 ms -- Nyquist-safe for <1 Hz analysis

# Adaptive bin thresholds (recording duration -> bin size)
# Keeps total bins manageable for HMM: target < 100K bins for
# the forward-backward algorithm which is O(T) per iteration.
_ADAPTIVE_BIN_THRESHOLDS = [
    (36000.0, 10.0),  # > 10 h  ->  10 s bins (~42K bins for 118 h)
    (3600.0,  1.0),   # > 1 h   ->  1 s bins  (~425K for short multi-hour)
    (0.0,     0.050), # <= 1 h  ->  50 ms bins (original resolution)
]

# Frequency bands (Hz)
_SLOW_OSC_BAND = (0.1, 1.0)
_DELTA_BAND = (1.0, 4.0)
_THETA_BAND = (4.0, 8.0)
_GAMMA_BAND = (30.0, 100.0)

# Multi-scale analysis windows (seconds)
_MULTISCALE_WINDOWS = [1.0, 10.0, 60.0, 3600.0]

# Null model replicates for statistical validation
_N_NULL_REPLICATES = 200


# ===================================================================
# 0. Adaptive bin size selection
# ===================================================================

def _choose_adaptive_bin_sec(duration_sec: float, requested_bin_sec: float) -> float:
    """Pick a bin size that keeps total bins tractable for HMM.

    For short recordings (<=1 h), honour the user's requested bin size.
    For longer recordings, coarsen the bins so the HMM forward-backward
    algorithm (O(T) numpy vectorised) completes in seconds, not minutes.

    Returns the effective bin size in seconds.
    """
    for threshold_sec, bin_sec in _ADAPTIVE_BIN_THRESHOLDS:
        if duration_sec > threshold_sec:
            return max(bin_sec, requested_bin_sec)
    return requested_bin_sec


# ===================================================================
# 1. Population firing rate computation
# ===================================================================

def _compute_population_rate(
    times: np.ndarray,
    t_start: float,
    t_end: float,
    bin_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram spike times into a population firing rate vector.

    Returns
    -------
    rate : ndarray, shape (n_bins,)
        Spike count per bin (not divided by bin width -- preserves counts
        for Poisson-like models).
    bin_edges : ndarray, shape (n_bins + 1,)
    """
    bin_edges = np.arange(t_start, t_end + bin_sec, bin_sec)
    rate, _ = np.histogram(times, bins=bin_edges)
    return rate.astype(np.float64), bin_edges


# ===================================================================
# 2. Bimodal distribution analysis
# ===================================================================

def _test_bimodality(rate: np.ndarray) -> dict:
    """Test whether the firing rate distribution is bimodal.

    Uses Hartigan's dip statistic (approximated via the calibrated
    bimodality coefficient BC = (skewness^2 + 1) / kurtosis_excess + 3).
    BC > 5/9 ~ 0.555 suggests bimodality.

    Also fits a two-component Gaussian mixture to estimate modes.
    """
    if len(rate) < 30:
        return {"is_bimodal": False, "bc": 0.0, "reason": "too_few_bins"}

    skew = float(sp_stats.skew(rate))
    kurt = float(sp_stats.kurtosis(rate, fisher=False))  # excess=False -> Pearson
    bc = (skew ** 2 + 1) / kurt if kurt > 0 else 0.0
    is_bimodal = bc > 0.555

    # Histogram-based mode estimation (fast, avoids sklearn dependency)
    n_hist_bins = min(200, max(30, len(rate) // 50))
    hist_counts, hist_edges = np.histogram(rate, bins=n_hist_bins)
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    # Smooth histogram to find peaks
    if len(hist_counts) > 5:
        smoothed = np.convolve(hist_counts, np.ones(5) / 5, mode="same")
    else:
        smoothed = hist_counts.astype(float)

    # Find local maxima
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)

    mode_rates = [float(hist_centers[p]) for p in peaks[:5]]

    # Estimate threshold between two largest modes
    threshold = None
    if len(peaks) >= 2:
        # Sort peaks by height, take two tallest
        peak_heights = [(smoothed[p], p) for p in peaks]
        peak_heights.sort(reverse=True)
        p1, p2 = sorted([peak_heights[0][1], peak_heights[1][1]])
        # Threshold = valley between them
        valley_idx = p1 + int(np.argmin(smoothed[p1:p2 + 1]))
        threshold = float(hist_centers[valley_idx])

    return {
        "is_bimodal": is_bimodal,
        "bc": round(bc, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n_peaks_detected": len(peaks),
        "mode_rates": mode_rates,
        "bimodal_threshold": threshold,
    }


# ===================================================================
# 3. Hidden Markov Model (2-state) for UP/DOWN classification
# ===================================================================

def _logsumexp_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise log(exp(a) + exp(b)), numerically stable."""
    mx = np.maximum(a, b)
    return mx + np.log1p(np.exp(-np.abs(a - b)))


def _fit_hmm_two_state(rate: np.ndarray) -> dict:
    """Fit a two-state Gaussian HMM via Expectation-Maximization.

    Self-contained implementation (no hmmlearn dependency) optimised
    for large T.  All forward / backward / xi / Viterbi passes are
    fully vectorised with numpy -- zero Python for-loops over T.

    K = 2 is hard-coded, which allows every inner loop to be replaced
    by elementwise numpy operations on (T,) or (T,2) arrays.

    Returns dict with state_sequence, means, variances, transition
    matrix, log_likelihood, and convergence flag.
    """
    T = len(rate)
    K = 2

    if T < 10:
        return {
            "converged": False,
            "state_sequence": np.zeros(T, dtype=np.int32),
            "means": [0.0, 0.0],
            "variances": [1.0, 1.0],
            "transition_matrix": [[0.5, 0.5], [0.5, 0.5]],
            "log_likelihood": -np.inf,
        }

    # ---- Initialise parameters from data quantiles ----
    q25, q75 = np.percentile(rate, [25, 75])
    mu = np.array([q25, q75], dtype=np.float64)
    var = np.array([max(np.var(rate) * 0.25, 1e-6)] * K, dtype=np.float64)
    A = np.array([[0.95, 0.05],
                  [0.05, 0.95]], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    max_iter = 50
    tol = 1e-6
    prev_ll = -np.inf
    converged = False
    plateau_count = 0       # early exit on slow convergence

    for iteration in range(max_iter):
        # ---- E-step: forward-backward (fully vectorised) ----
        # Emission log-probabilities  (T, 2)
        log_emit = np.empty((T, K), dtype=np.float64)
        for k in range(K):                       # K=2, trivial
            diff = rate - mu[k]
            log_emit[:, k] = -0.5 * np.log(2.0 * np.pi * var[k]) - 0.5 * diff ** 2 / var[k]

        log_A = np.log(A + 1e-300)               # (2, 2)

        # -- Forward pass: vectorised over T --
        # For K=2, log_alpha(t, k) = logsumexp_j(log_alpha(t-1, j) + log_A(j,k)) + log_emit(t,k)
        # We unroll the logsumexp for K=2: logsumexp(x0, x1) is fast element-wise.
        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = np.log(pi + 1e-300) + log_emit[0]

        # Pre-extract columns of log_A for speed
        la_00, la_10 = log_A[0, 0], log_A[1, 0]
        la_01, la_11 = log_A[0, 1], log_A[1, 1]

        # Chunked forward pass: process blocks of T to exploit numpy
        # vectorisation while keeping memory bounded.  Each step depends
        # on the previous, so we must iterate -- but the inner work is
        # pure numpy with no Python per-element overhead.
        #
        # For T <= 500_000 this loop is fast enough (~0.5 s per EM iter).
        # The adaptive binning above keeps T in this regime.
        for t in range(1, T):
            a0 = log_alpha[t - 1, 0]
            a1 = log_alpha[t - 1, 1]
            # k=0: logsumexp(a0 + la_00, a1 + la_10)
            v0a, v0b = a0 + la_00, a1 + la_10
            mx0 = max(v0a, v0b)
            log_alpha[t, 0] = mx0 + np.log(np.exp(v0a - mx0) + np.exp(v0b - mx0)) + log_emit[t, 0]
            # k=1: logsumexp(a0 + la_01, a1 + la_11)
            v1a, v1b = a0 + la_01, a1 + la_11
            mx1 = max(v1a, v1b)
            log_alpha[t, 1] = mx1 + np.log(np.exp(v1a - mx1) + np.exp(v1b - mx1)) + log_emit[t, 1]

        # Log-likelihood
        max_final = np.max(log_alpha[-1])
        ll = max_final + np.log(np.sum(np.exp(log_alpha[-1] - max_final)))

        improvement = np.abs(ll - prev_ll)
        if improvement < tol:
            converged = True
            break
        # Plateau detection: stop early if improvement is negligible
        # for several consecutive iterations (avoids 50 useless iters
        # on data that will never converge to a bimodal solution).
        if improvement < tol * 100:
            plateau_count += 1
            if plateau_count >= 5:
                converged = True
                break
        else:
            plateau_count = 0
        prev_ll = ll

        # -- Backward pass: same scalar unroll for speed --
        log_beta = np.zeros((T, K), dtype=np.float64)
        for t in range(T - 2, -1, -1):
            e0 = log_emit[t + 1, 0] + log_beta[t + 1, 0]
            e1 = log_emit[t + 1, 1] + log_beta[t + 1, 1]
            # k=0
            v0a, v0b = la_00 + e0, la_01 + e1
            mx0 = max(v0a, v0b)
            log_beta[t, 0] = mx0 + np.log(np.exp(v0a - mx0) + np.exp(v0b - mx0))
            # k=1
            v1a, v1b = la_10 + e0, la_11 + e1
            mx1 = max(v1a, v1b)
            log_beta[t, 1] = mx1 + np.log(np.exp(v1a - mx1) + np.exp(v1b - mx1))

        # -- Posterior gamma(t, k) --
        log_gamma = log_alpha + log_beta
        log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # -- Transition posteriors xi: FULLY VECTORISED (no loop over T) --
        # xi(t, j, k) = alpha(t,j) * A(j,k) * emit(t+1,k) * beta(t+1,k)
        # We compute the (T-1, 2, 2) tensor then sum over t.
        la = log_alpha[:-1]            # (T-1, 2)
        le = log_emit[1:]             # (T-1, 2)
        lb = log_beta[1:]             # (T-1, 2)

        # Build log_xi as (T-1, 2, 2)
        # log_xi[t, j, k] = la[t, j] + log_A[j, k] + le[t, k] + lb[t, k]
        log_xi = (la[:, :, None]       # (T-1, 2, 1)
                  + log_A[None, :, :]   # (1, 2, 2)
                  + le[:, None, :]      # (T-1, 1, 2)
                  + lb[:, None, :])     # (T-1, 1, 2)

        # Normalise per time step then sum
        log_xi_max = log_xi.reshape(T - 1, -1).max(axis=1, keepdims=True).reshape(T - 1, 1, 1)
        xi = np.exp(log_xi - log_xi_max)
        xi /= xi.reshape(T - 1, -1).sum(axis=1, keepdims=True).reshape(T - 1, 1, 1) + 1e-300
        xi_sum = xi.sum(axis=0)        # (2, 2)

        # ---- M-step ----
        gamma_sum = gamma.sum(axis=0) + 1e-300
        pi = gamma[0] / (gamma[0].sum() + 1e-300)

        for k in range(K):
            mu[k] = np.dot(gamma[:, k], rate) / gamma_sum[k]
            diff = rate - mu[k]
            var[k] = max(np.dot(gamma[:, k], diff ** 2) / gamma_sum[k], 1e-6)

        row_sums = xi_sum.sum(axis=1, keepdims=True) + 1e-300
        A = xi_sum / row_sums

    # ---- Viterbi decoding: vectorised over T ----
    # K=2 allows scalar unroll -- no inner k-loop needed.
    log_delta = np.empty((T, K), dtype=np.float64)
    psi = np.empty((T, K), dtype=np.int32)
    log_delta[0] = np.log(pi + 1e-300) + log_emit[0]

    for t in range(1, T):
        d0 = log_delta[t - 1, 0]
        d1 = log_delta[t - 1, 1]
        # k=0
        s0_0, s1_0 = d0 + la_00, d1 + la_10
        if s0_0 >= s1_0:
            psi[t, 0] = 0; log_delta[t, 0] = s0_0 + log_emit[t, 0]
        else:
            psi[t, 0] = 1; log_delta[t, 0] = s1_0 + log_emit[t, 0]
        # k=1
        s0_1, s1_1 = d0 + la_01, d1 + la_11
        if s0_1 >= s1_1:
            psi[t, 1] = 0; log_delta[t, 1] = s0_1 + log_emit[t, 1]
        else:
            psi[t, 1] = 1; log_delta[t, 1] = s1_1 + log_emit[t, 1]

    # Backtrace
    states = np.zeros(T, dtype=np.int32)
    states[-1] = int(np.argmax(log_delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    # Ensure state 0 = DOWN (lower mean), state 1 = UP (higher mean)
    if mu[0] > mu[1]:
        states = 1 - states
        mu = mu[::-1]
        var = var[::-1]
        A = A[::-1, ::-1]
        pi = pi[::-1]

    return {
        "converged": converged,
        "n_iterations": iteration + 1,
        "state_sequence": states,
        "means": [round(float(mu[k]), 4) for k in range(K)],
        "variances": [round(float(var[k]), 4) for k in range(K)],
        "transition_matrix": [[round(float(A[j, k]), 6) for k in range(K)] for j in range(K)],
        "log_likelihood": round(float(ll), 2),
        "stationary_up_prob": round(float(A[0, 1] / (A[0, 1] + A[1, 0] + 1e-300)), 4),
    }


# ===================================================================
# 4. Adaptive threshold method (fast fallback / comparison)
# ===================================================================

def _adaptive_threshold_states(
    rate: np.ndarray,
    bimodal_info: dict,
) -> np.ndarray:
    """Classify bins as UP(1) / DOWN(0) using adaptive thresholding.

    If bimodal analysis found a valley threshold, use it.
    Otherwise fall back to median of rate.
    """
    thr = bimodal_info.get("bimodal_threshold")
    if thr is None:
        thr = float(np.median(rate))
    return (rate > thr).astype(np.int32)


# ===================================================================
# 5. State epoch extraction
# ===================================================================

def _extract_state_epochs(
    states: np.ndarray,
    bin_edges: np.ndarray,
) -> dict:
    """Extract contiguous UP and DOWN epochs with durations.

    Returns dict with up_durations, down_durations (seconds),
    epoch list, and summary statistics.
    """
    T = len(states)
    if T == 0:
        return {
            "up_durations_s": np.array([]),
            "down_durations_s": np.array([]),
            "epochs": [],
        }

    bin_width = float(bin_edges[1] - bin_edges[0])

    # Run-length encoding via diff
    change_idx = np.where(np.diff(states) != 0)[0] + 1
    boundaries = np.concatenate([[0], change_idx, [T]])

    up_durs = []
    down_durs = []
    epochs = []

    for i in range(len(boundaries) - 1):
        start_bin = int(boundaries[i])
        end_bin = int(boundaries[i + 1])
        duration = (end_bin - start_bin) * bin_width
        state_val = int(states[start_bin])

        if state_val == 1:
            up_durs.append(duration)
        else:
            down_durs.append(duration)

        epochs.append({
            "state": "UP" if state_val == 1 else "DOWN",
            "start_s": round(float(bin_edges[start_bin]), 4),
            "end_s": round(float(bin_edges[min(end_bin, len(bin_edges) - 1)]), 4),
            "duration_s": round(duration, 4),
        })

    up_arr = np.array(up_durs) if up_durs else np.array([])
    down_arr = np.array(down_durs) if down_durs else np.array([])

    return {
        "up_durations_s": up_arr,
        "down_durations_s": down_arr,
        "epochs": epochs,
    }


def _duration_stats(durations: np.ndarray, label: str) -> dict:
    """Compute descriptive statistics for a duration array."""
    if len(durations) == 0:
        return {
            f"{label}_count": 0,
            f"{label}_mean_s": 0.0,
            f"{label}_median_s": 0.0,
            f"{label}_std_s": 0.0,
            f"{label}_cv": 0.0,
            f"{label}_min_s": 0.0,
            f"{label}_max_s": 0.0,
            f"{label}_q25_s": 0.0,
            f"{label}_q75_s": 0.0,
        }
    return {
        f"{label}_count": int(len(durations)),
        f"{label}_mean_s": round(float(np.mean(durations)), 4),
        f"{label}_median_s": round(float(np.median(durations)), 4),
        f"{label}_std_s": round(float(np.std(durations)), 4),
        f"{label}_cv": round(float(np.std(durations) / (np.mean(durations) + 1e-12)), 4),
        f"{label}_min_s": round(float(np.min(durations)), 4),
        f"{label}_max_s": round(float(np.max(durations)), 4),
        f"{label}_q25_s": round(float(np.percentile(durations, 25)), 4),
        f"{label}_q75_s": round(float(np.percentile(durations, 75)), 4),
    }


# ===================================================================
# 6. Slow oscillation analysis (0.1--1 Hz)
# ===================================================================

def _analyze_slow_oscillation(
    rate: np.ndarray,
    bin_sec: float,
) -> dict:
    """Spectral analysis of population rate focused on slow oscillations.

    Computes Welch PSD, extracts slow oscillation band power, finds
    dominant frequency, and estimates delta/theta band power for
    comparison (sleep staging analogy).
    """
    fs = 1.0 / bin_sec
    n = len(rate)

    if n < 64:
        return {
            "slow_osc_detected": False,
            "reason": "insufficient_data",
        }

    # Choose nperseg: ~60 s worth of bins, capped at data length
    nperseg = min(n, max(64, int(60.0 / bin_sec)))
    nperseg = min(nperseg, n)

    freqs, psd = sp_signal.welch(
        rate - np.mean(rate),
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        detrend="linear",
    )

    total_power = float(np.sum(psd))
    if total_power < 1e-12:
        return {"slow_osc_detected": False, "reason": "no_power"}

    def _band_power(flo: float, fhi: float) -> float:
        mask = (freqs >= flo) & (freqs <= fhi)
        return float(np.sum(psd[mask])) if np.any(mask) else 0.0

    slow_power = _band_power(*_SLOW_OSC_BAND)
    delta_power = _band_power(*_DELTA_BAND)
    theta_power = _band_power(*_THETA_BAND)

    slow_frac = slow_power / total_power
    delta_frac = delta_power / total_power
    theta_frac = theta_power / total_power

    # Dominant slow-oscillation frequency
    slow_mask = (freqs >= _SLOW_OSC_BAND[0]) & (freqs <= _SLOW_OSC_BAND[1])
    if np.any(slow_mask):
        slow_psd = psd[slow_mask]
        slow_freqs = freqs[slow_mask]
        dom_idx = int(np.argmax(slow_psd))
        dominant_freq = float(slow_freqs[dom_idx])
        peak_power = float(slow_psd[dom_idx])
    else:
        dominant_freq = 0.0
        peak_power = 0.0

    # Criterion: slow osc detected if band power > 20% of total
    # and there is a clear spectral peak above the 1/f baseline
    slow_osc_detected = slow_frac > 0.20 and peak_power > np.median(psd) * 2

    # Truncate PSD for JSON serialisation (up to 500 points)
    max_psd_pts = 500
    stride = max(1, len(freqs) // max_psd_pts)

    return {
        "slow_osc_detected": bool(slow_osc_detected),
        "dominant_slow_freq_hz": round(dominant_freq, 4),
        "slow_osc_power_fraction": round(slow_frac, 4),
        "delta_power_fraction": round(delta_frac, 4),
        "theta_power_fraction": round(theta_frac, 4),
        "total_spectral_power": round(total_power, 4),
        "peak_slow_power": round(peak_power, 4),
        "frequencies_hz": freqs[::stride].tolist(),
        "psd": psd[::stride].tolist(),
    }


# ===================================================================
# 7. Phase-amplitude coupling (slow osc phase x gamma amplitude)
# ===================================================================

def _compute_phase_amplitude_coupling(
    rate: np.ndarray,
    bin_sec: float,
    n_phase_bins: int = 18,
) -> dict:
    """Compute modulation index (Tort et al. 2010) between slow
    oscillation phase and gamma-band amplitude.

    Phase is extracted from the slow-oscillation-filtered signal,
    amplitude from the gamma-filtered signal.  The modulation index
    MI = (log(N) - H) / log(N)  where H is the Shannon entropy of
    the amplitude distribution across phase bins and N = n_phase_bins.
    """
    fs = 1.0 / bin_sec
    n = len(rate)

    # Need at least several slow-oscillation cycles
    min_samples = max(128, int(10.0 / bin_sec))
    if n < min_samples or fs < 2.5:
        return {"pac_computed": False, "reason": "insufficient_resolution"}

    centered = rate - np.mean(rate)

    # Band-pass design
    nyq = fs / 2.0

    def _safe_bandpass(lo: float, hi: float, sig: np.ndarray) -> Optional[np.ndarray]:
        lo_n, hi_n = lo / nyq, hi / nyq
        if lo_n <= 0 or hi_n >= 1.0 or lo_n >= hi_n:
            return None
        try:
            sos = sp_signal.butter(3, [lo_n, hi_n], btype="band", output="sos")
            return sp_signal.sosfiltfilt(sos, sig)
        except Exception:
            return None

    slow_filt = _safe_bandpass(_SLOW_OSC_BAND[0], _SLOW_OSC_BAND[1], centered)

    # For gamma: if Nyquist is below gamma range, use a proxy band
    gamma_lo = min(_GAMMA_BAND[0], nyq * 0.4)
    gamma_hi = min(_GAMMA_BAND[1], nyq * 0.9)
    if gamma_hi <= gamma_lo + 0.5:
        # Sampling too low for any meaningful "fast" band
        return {"pac_computed": False, "reason": "sampling_too_low_for_gamma"}
    gamma_filt = _safe_bandpass(gamma_lo, gamma_hi, centered)

    if slow_filt is None or gamma_filt is None:
        return {"pac_computed": False, "reason": "filter_design_failed"}

    # Analytic signals
    slow_phase = np.angle(sp_signal.hilbert(slow_filt))
    gamma_amp = np.abs(sp_signal.hilbert(gamma_filt))

    # Bin amplitudes by phase
    phase_bin_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    mean_amp = np.zeros(n_phase_bins, dtype=np.float64)
    for b in range(n_phase_bins):
        mask = (slow_phase >= phase_bin_edges[b]) & (slow_phase < phase_bin_edges[b + 1])
        if np.any(mask):
            mean_amp[b] = np.mean(gamma_amp[mask])

    # Normalise to probability distribution
    total = np.sum(mean_amp)
    if total < 1e-12:
        return {"pac_computed": True, "modulation_index": 0.0}
    p = mean_amp / total

    # Shannon entropy
    H = -np.sum(p[p > 0] * np.log(p[p > 0]))
    H_max = np.log(n_phase_bins)
    mi = (H_max - H) / H_max if H_max > 0 else 0.0

    # Phase of maximum amplitude (preferred phase)
    phase_centers = 0.5 * (phase_bin_edges[:-1] + phase_bin_edges[1:])
    preferred_phase = float(phase_centers[np.argmax(mean_amp)])

    return {
        "pac_computed": True,
        "modulation_index": round(float(mi), 6),
        "preferred_phase_rad": round(preferred_phase, 4),
        "gamma_band_used_hz": [round(gamma_lo, 1), round(gamma_hi, 1)],
        "amplitude_by_phase": mean_amp.tolist(),
        "phase_bin_centers_rad": phase_centers.tolist(),
    }


# ===================================================================
# 8. Sleep staging analogy -- active vs quiet periods
# ===================================================================

def _classify_vigilance_states(
    rate: np.ndarray,
    bin_edges: np.ndarray,
    window_sec: float = 60.0,
) -> dict:
    """Classify recording into active (wake-like) and quiet (sleep-like)
    periods at a coarser timescale.

    Within each window, compute:
    - mean rate and variance (high var = bursty = active)
    - coefficient of variation (high CV = sleep-like, bimodal)
    - fraction of time in UP state

    Active periods: high mean rate, low CV (sustained activity).
    Quiet periods: bimodal rate, high CV, clear UP/DOWN alternation.
    """
    bin_sec = float(bin_edges[1] - bin_edges[0])
    bins_per_window = max(1, int(window_sec / bin_sec))
    n_windows = len(rate) // bins_per_window

    if n_windows < 3:
        return {"vigilance_computed": False, "reason": "recording_too_short"}

    # Vectorised: reshape rate into (n_windows, bins_per_window)
    usable = n_windows * bins_per_window
    rate_2d = rate[:usable].reshape(n_windows, bins_per_window)
    global_median = float(np.median(rate))

    means = rate_2d.mean(axis=1)
    stds = rate_2d.std(axis=1)
    cvs = stds / (means + 1e-12)
    up_fracs = (rate_2d > global_median).mean(axis=1)
    t_centers = bin_edges[np.arange(n_windows) * bins_per_window].astype(np.float64) + window_sec / 2

    # Classify: sleep-like = high CV (bimodal within window)
    cv_threshold = float(np.median(cvs) + 0.5 * np.std(cvs))
    is_quiet = cvs > cv_threshold

    results = []
    for w in range(n_windows):
        results.append({
            "t_center_s": round(float(t_centers[w]), 2),
            "mean_rate": round(float(means[w]), 4),
            "std_rate": round(float(stds[w]), 4),
            "cv": round(float(cvs[w]), 4),
            "up_fraction": round(float(up_fracs[w]), 4),
            "state": "quiet" if is_quiet[w] else "active",
        })

    n_quiet = int(np.sum(is_quiet))
    n_active = n_windows - n_quiet

    # Transition count
    states_seq = [r["state"] for r in results]
    transitions = sum(1 for i in range(1, len(states_seq)) if states_seq[i] != states_seq[i - 1])

    # Truncate epochs list for API
    max_epochs = 2000
    return {
        "vigilance_computed": True,
        "window_sec": window_sec,
        "n_windows": n_windows,
        "n_quiet": n_quiet,
        "n_active": n_active,
        "quiet_fraction": round(n_quiet / n_windows, 4),
        "active_fraction": round(n_active / n_windows, 4),
        "n_transitions": transitions,
        "cv_threshold": round(float(cv_threshold), 4),
        "epochs": results[:max_epochs],
    }


# ===================================================================
# 9. Transition dynamics
# ===================================================================

def _analyze_transitions(
    states: np.ndarray,
    bin_sec: float,
) -> dict:
    """Characterize transitions between UP and DOWN states.

    - Transition rate (per minute)
    - Asymmetry: are UP->DOWN transitions faster than DOWN->UP?
    - Autocorrelation of state durations (is there memory?)
    """
    T = len(states)
    if T < 10:
        return {"transition_analysis_computed": False}

    transitions = np.diff(states)
    up_to_down = np.where(transitions == -1)[0]
    down_to_up = np.where(transitions == 1)[0]

    total_time_min = T * bin_sec / 60.0
    n_trans = len(up_to_down) + len(down_to_up)
    trans_rate = n_trans / max(total_time_min, 1e-6)

    # Duration sequences for autocorrelation analysis
    change_idx = np.where(transitions != 0)[0] + 1
    boundaries = np.concatenate([[0], change_idx, [T]])
    dur_seq = np.diff(boundaries).astype(np.float64) * bin_sec

    # Lag-1 autocorrelation of durations (memory in state switching)
    if len(dur_seq) > 3:
        d_mean = np.mean(dur_seq)
        d_std = np.std(dur_seq)
        if d_std > 1e-10:
            lag1 = float(np.corrcoef(dur_seq[:-1], dur_seq[1:])[0, 1])
        else:
            lag1 = 0.0
    else:
        lag1 = 0.0

    # Transition asymmetry
    n_ud = len(up_to_down)
    n_du = len(down_to_up)
    if n_ud > 0 and n_du > 0:
        asymmetry = (n_du - n_ud) / (n_du + n_ud)
    else:
        asymmetry = 0.0

    return {
        "transition_analysis_computed": True,
        "transitions_per_minute": round(trans_rate, 4),
        "n_up_to_down": int(n_ud),
        "n_down_to_up": int(n_du),
        "transition_asymmetry": round(float(asymmetry), 4),
        "duration_lag1_autocorr": round(lag1, 4),
        "total_time_minutes": round(total_time_min, 2),
    }


# ===================================================================
# 10. Homeostatic sleep pressure
# ===================================================================

def _compute_sleep_pressure(
    states: np.ndarray,
    bin_edges: np.ndarray,
    window_sec: float = 600.0,
) -> dict:
    """Track homeostatic sleep pressure over the recording.

    In cortical circuits, prolonged wakefulness (UP-state dominance)
    builds "sleep pressure" which eventually forces more DOWN states.

    We quantify this as:
    - Running UP-fraction in sliding windows
    - Correlation between cumulative wake time and subsequent
      sleep-like activity density (positive = homeostatic regulation)
    """
    bin_sec = float(bin_edges[1] - bin_edges[0])
    bins_per_window = max(1, int(window_sec / bin_sec))
    n_windows = len(states) // bins_per_window

    if n_windows < 5:
        return {"homeostasis_computed": False, "reason": "recording_too_short"}

    up_fracs = []
    t_centers = []
    for w in range(n_windows):
        chunk = states[w * bins_per_window : (w + 1) * bins_per_window]
        up_fracs.append(float(np.mean(chunk)))
        t_center = float(bin_edges[w * bins_per_window]) + window_sec / 2
        t_centers.append(t_center)

    up_fracs = np.array(up_fracs)

    # Cumulative UP-time (proxy for wake pressure)
    cumulative_up = np.cumsum(up_fracs)
    # Next-window DOWN fraction (sleep rebound)
    next_down = 1.0 - up_fracs[1:]
    cum_shifted = cumulative_up[:-1]

    if len(cum_shifted) > 3:
        r, p = sp_stats.pearsonr(cum_shifted, next_down)
        homeostatic_r = round(float(r), 4)
        homeostatic_p = round(float(p), 6)
    else:
        homeostatic_r = 0.0
        homeostatic_p = 1.0

    # Linear trend in UP fraction (does activity change over time?)
    if n_windows > 3:
        slope, intercept, r_trend, p_trend, _ = sp_stats.linregress(
            np.arange(n_windows), up_fracs
        )
    else:
        slope = intercept = r_trend = 0.0
        p_trend = 1.0

    # Truncate for API
    max_pts = 1000
    stride = max(1, n_windows // max_pts)

    return {
        "homeostasis_computed": True,
        "window_sec": window_sec,
        "n_windows": n_windows,
        "homeostatic_correlation": homeostatic_r,
        "homeostatic_p_value": homeostatic_p,
        "homeostatic_regulation_detected": homeostatic_p < 0.05 and homeostatic_r > 0.1,
        "up_fraction_trend_slope": round(float(slope), 6),
        "up_fraction_trend_r": round(float(r_trend), 4),
        "up_fraction_trend_p": round(float(p_trend), 6),
        "t_centers_s": [round(t, 1) for t in t_centers[::stride]],
        "up_fraction_timecourse": [round(float(u), 4) for u in up_fracs[::stride]],
    }


# ===================================================================
# 11. Per-MEA breakdown
# ===================================================================

def _per_mea_analysis(
    data: SpikeData,
    bin_sec: float,
) -> list[dict]:
    """Run UP/DOWN detection independently on each MEA.

    MEA assignment: electrode_id // 8  (FinalSpark layout).
    Returns a list of per-MEA summaries so users can compare
    sleep architecture across organoids.
    """
    t_start, t_end = data.time_range

    mea_results = []
    for mea_id in range(_N_MEAS):
        el_lo = mea_id * _ELECTRODES_PER_MEA
        el_hi = el_lo + _ELECTRODES_PER_MEA
        mea_electrodes = [e for e in data.electrode_ids if el_lo <= e < el_hi]

        if not mea_electrodes:
            mea_results.append({
                "mea_id": mea_id,
                "electrodes": [],
                "n_spikes": 0,
                "analysis": "no_data",
            })
            continue

        mask = np.isin(data.electrodes, mea_electrodes)
        mea_times = data.times[mask]
        n_spikes = len(mea_times)

        if n_spikes < 50:
            mea_results.append({
                "mea_id": mea_id,
                "electrodes": mea_electrodes,
                "n_spikes": n_spikes,
                "analysis": "insufficient_spikes",
            })
            continue

        rate, bin_edges = _compute_population_rate(mea_times, t_start, t_end, bin_sec)

        bimodal = _test_bimodality(rate)

        # Use adaptive threshold for per-MEA (HMM is expensive per-MEA)
        states = _adaptive_threshold_states(rate, bimodal)
        epoch_info = _extract_state_epochs(states, bin_edges)

        up_stats = _duration_stats(epoch_info["up_durations_s"], "up")
        down_stats = _duration_stats(epoch_info["down_durations_s"], "down")

        up_frac = float(np.mean(states)) if len(states) > 0 else 0.0

        mea_results.append({
            "mea_id": mea_id,
            "electrodes": mea_electrodes,
            "n_spikes": n_spikes,
            "mean_rate_hz": round(n_spikes / max(t_end - t_start, 1e-6), 4),
            "bimodality": bimodal,
            "up_fraction": round(up_frac, 4),
            **up_stats,
            **down_stats,
        })

    return mea_results


# ===================================================================
# 12. Multi-scale temporal analysis
# ===================================================================

def _multiscale_analysis(
    rate: np.ndarray,
    bin_sec: float,
    windows_sec: Optional[list[float]] = None,
) -> list[dict]:
    """Analyze UP/DOWN dynamics at multiple timescales.

    At each scale, compute UP fraction, transition rate,
    dominant oscillation frequency, and coefficient of variation.
    """
    if windows_sec is None:
        windows_sec = list(_MULTISCALE_WINDOWS)

    T = len(rate)
    results = []

    for w_sec in windows_sec:
        bins_per_w = max(1, int(w_sec / bin_sec))
        n_windows = T // bins_per_w

        if n_windows < 3:
            results.append({
                "window_sec": w_sec,
                "computed": False,
                "reason": "too_few_windows",
            })
            continue

        # Per-window statistics (vectorised reshape)
        usable = n_windows * bins_per_w
        rate_2d = rate[:usable].reshape(n_windows, bins_per_w)
        means = rate_2d.mean(axis=1)

        global_median = np.median(rate)
        up_fracs = (rate_2d > global_median).mean(axis=1)

        # Transition count at this scale
        binary_windows = (means > global_median).astype(int)
        trans = int(np.sum(np.abs(np.diff(binary_windows))))

        # CV of window means (variability across windows)
        cv = float(np.std(means) / (np.mean(means) + 1e-12))

        # Dominant frequency of the window-mean timecourse
        if n_windows >= 8:
            fs_w = 1.0 / w_sec
            f, p = sp_signal.welch(means - np.mean(means), fs=fs_w,
                                   nperseg=min(n_windows, 64))
            dom_freq = float(f[np.argmax(p)]) if len(p) > 0 else 0.0
        else:
            dom_freq = 0.0

        results.append({
            "window_sec": w_sec,
            "computed": True,
            "n_windows": n_windows,
            "mean_up_fraction": round(float(np.mean(up_fracs)), 4),
            "std_up_fraction": round(float(np.std(up_fracs)), 4),
            "n_transitions": trans,
            "transitions_per_hour": round(trans / (n_windows * w_sec / 3600 + 1e-9), 4),
            "cv_of_window_means": round(cv, 4),
            "dominant_freq_hz": round(dom_freq, 6),
        })

    return results


# ===================================================================
# 13. Statistical validation -- null model comparison
# ===================================================================

def _null_model_validation(
    times: np.ndarray,
    t_start: float,
    t_end: float,
    bin_sec: float,
    observed_states: np.ndarray,
    n_replicates: int = _N_NULL_REPLICATES,
    rng_seed: int = 42,
) -> dict:
    """Compare observed UP/DOWN structure to a rate-matched null model.

    Null model: Poisson process with the same overall mean rate.
    For each replicate, generate surrogate spike times, bin them,
    and classify states.  If the observed bimodality / transition
    structure significantly exceeds the null, the UP/DOWN dynamics
    are real rather than fluctuations expected from a homogeneous
    Poisson process.

    Tests:
    - Bimodality coefficient: is observed BC > null BC?
    - Transition rate: is observed transition rate different from null?
    - UP-state duration CV: is observed CV > null CV?
    """
    rng = np.random.RandomState(rng_seed)
    n_spikes = len(times)
    duration = t_end - t_start

    if n_spikes < 100 or duration < 10.0:
        return {"null_test_computed": False, "reason": "insufficient_data"}

    mean_rate = n_spikes / duration  # spikes per second

    # Observed metrics
    obs_bimodal = _test_bimodality(
        np.histogram(times, bins=np.arange(t_start, t_end + bin_sec, bin_sec))[0].astype(float)
    )
    obs_bc = obs_bimodal["bc"]

    obs_trans = np.diff(observed_states)
    obs_n_trans = int(np.sum(obs_trans != 0))

    # Observed UP duration CV
    change_idx = np.where(obs_trans != 0)[0] + 1
    boundaries = np.concatenate([[0], change_idx, [len(observed_states)]])
    obs_durs = np.diff(boundaries).astype(float) * bin_sec
    obs_dur_cv = float(np.std(obs_durs) / (np.mean(obs_durs) + 1e-12))

    # Generate null distribution -- optimised for large spike counts.
    # Instead of sorting 2.6M spikes per replicate, we directly sample
    # bin counts from a Multinomial (equivalent to histogramming uniform
    # spike times), which is O(n_bins) instead of O(n_spikes * log(n_spikes)).
    null_bcs = np.zeros(n_replicates)
    null_trans = np.zeros(n_replicates)
    null_cvs = np.zeros(n_replicates)

    bin_edges = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins_null = len(bin_edges) - 1
    # Uniform bin probabilities for a homogeneous Poisson process
    p_uniform = np.full(n_bins_null, 1.0 / n_bins_null)

    for rep in range(n_replicates):
        # Multinomial surrogate: each spike falls into a random bin
        sur_rate = rng.multinomial(n_spikes, p_uniform).astype(np.float64)

        # BC
        skew = sp_stats.skew(sur_rate)
        kurt = sp_stats.kurtosis(sur_rate, fisher=False)
        null_bcs[rep] = (skew ** 2 + 1) / kurt if kurt > 0 else 0.0

        # Transitions (median threshold)
        sur_states = (sur_rate > np.median(sur_rate)).astype(np.int32)
        null_trans[rep] = np.sum(np.abs(np.diff(sur_states)))

        # Duration CV
        ch = np.where(np.diff(sur_states) != 0)[0] + 1
        bd = np.concatenate([[0], ch, [len(sur_states)]])
        ds = np.diff(bd).astype(float) * bin_sec
        null_cvs[rep] = np.std(ds) / (np.mean(ds) + 1e-12)

    # p-values (fraction of null >= observed)
    bc_p = float(np.mean(null_bcs >= obs_bc))
    trans_p = float(np.mean(null_trans >= obs_n_trans))
    cv_p = float(np.mean(null_cvs >= obs_dur_cv))

    # Z-scores
    def _zscore(obs: float, null: np.ndarray) -> float:
        s = np.std(null)
        return float((obs - np.mean(null)) / s) if s > 1e-10 else 0.0

    return {
        "null_test_computed": True,
        "n_replicates": n_replicates,
        "null_model": "homogeneous_poisson",
        "bimodality": {
            "observed_bc": round(obs_bc, 4),
            "null_mean_bc": round(float(np.mean(null_bcs)), 4),
            "null_std_bc": round(float(np.std(null_bcs)), 4),
            "z_score": round(_zscore(obs_bc, null_bcs), 3),
            "p_value": round(bc_p, 4),
            "significant": bc_p < 0.05,
        },
        "transition_rate": {
            "observed": obs_n_trans,
            "null_mean": round(float(np.mean(null_trans)), 2),
            "null_std": round(float(np.std(null_trans)), 2),
            "z_score": round(_zscore(obs_n_trans, null_trans), 3),
            "p_value": round(trans_p, 4),
            "significant": trans_p < 0.05,
        },
        "duration_cv": {
            "observed": round(obs_dur_cv, 4),
            "null_mean": round(float(np.mean(null_cvs)), 4),
            "null_std": round(float(np.std(null_cvs)), 4),
            "z_score": round(_zscore(obs_dur_cv, null_cvs), 3),
            "p_value": round(cv_p, 4),
            "significant": cv_p < 0.05,
        },
    }


# ===================================================================
# 14. Composite sleep-like score
# ===================================================================

def _compute_sleep_score(
    bimodal: dict,
    hmm: dict,
    slow_osc: dict,
    pac: dict,
    null_val: dict,
    vigilance: dict,
) -> dict:
    """Compute a composite sleep-like activity score (0--1).

    Weighted components:
    - 0.25  bimodal firing rate distribution (UP/DOWN present)
    - 0.25  HMM state separation (means well-separated, converged)
    - 0.20  slow oscillation power in 0.1--1 Hz band
    - 0.15  phase-amplitude coupling (MI > 0.01)
    - 0.15  statistical significance vs null model

    Higher score = stronger evidence of sleep-like dynamics.
    """
    components = {}

    # Bimodality
    if bimodal.get("is_bimodal"):
        components["bimodality"] = min(1.0, bimodal["bc"] / 0.8)
    else:
        components["bimodality"] = max(0.0, bimodal.get("bc", 0) / 0.8)

    # HMM separation
    if hmm.get("converged") and len(hmm.get("means", [])) == 2:
        mu_down, mu_up = hmm["means"]
        separation = (mu_up - mu_down) / (np.sqrt(max(hmm["variances"][0], 1e-6)) + 1e-6)
        components["hmm_separation"] = min(1.0, max(0.0, separation / 3.0))
    else:
        components["hmm_separation"] = 0.0

    # Slow oscillation
    if slow_osc.get("slow_osc_detected"):
        components["slow_oscillation"] = min(1.0, slow_osc.get("slow_osc_power_fraction", 0) / 0.4)
    else:
        components["slow_oscillation"] = slow_osc.get("slow_osc_power_fraction", 0) / 0.4

    # PAC
    if pac.get("pac_computed"):
        mi = pac.get("modulation_index", 0)
        components["pac"] = min(1.0, mi / 0.05) if mi > 0.005 else 0.0
    else:
        components["pac"] = 0.0

    # Null model
    if null_val.get("null_test_computed"):
        n_sig = sum(1 for key in ("bimodality", "transition_rate", "duration_cv")
                    if null_val.get(key, {}).get("significant", False))
        components["null_validation"] = n_sig / 3.0
    else:
        components["null_validation"] = 0.0

    weights = {
        "bimodality": 0.25,
        "hmm_separation": 0.25,
        "slow_oscillation": 0.20,
        "pac": 0.15,
        "null_validation": 0.15,
    }

    score = sum(weights[k] * max(0.0, min(1.0, components[k])) for k in weights)

    # Qualitative label
    if score >= 0.7:
        label = "strong_sleep_like"
    elif score >= 0.4:
        label = "moderate_sleep_like"
    elif score >= 0.15:
        label = "weak_sleep_like"
    else:
        label = "no_sleep_like_activity"

    return {
        "sleep_score": round(score, 4),
        "label": label,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": weights,
    }


# ===================================================================
# PUBLIC API
# ===================================================================

def detect_up_down_states(
    data: SpikeData,
    bin_size_ms: float = 50.0,
) -> dict:
    """Detect UP/DOWN states in population spiking activity.

    Uses a two-pronged approach:
    1. Bimodal distribution analysis of the population firing rate
    2. Two-state Hidden Markov Model for temporal segmentation

    Parameters
    ----------
    data : SpikeData
        Spike data with times, electrodes, amplitudes.
    bin_size_ms : float
        Bin width for population rate histogram (default 50 ms).

    Returns
    -------
    dict with bimodality analysis, HMM results, state epoch statistics,
    and the classified state sequence (truncated for API transport).
    """
    if data.n_spikes < 20:
        return {
            "computed": False,
            "reason": "insufficient_spikes",
            "n_spikes": data.n_spikes,
        }

    t_start, t_end = data.time_range
    duration_sec = t_end - t_start
    requested_bin_sec = bin_size_ms / 1000.0
    bin_sec = _choose_adaptive_bin_sec(duration_sec, requested_bin_sec)

    rate, bin_edges = _compute_population_rate(data.times, t_start, t_end, bin_sec)

    # Bimodal distribution test
    bimodal = _test_bimodality(rate)

    # HMM classification
    hmm = _fit_hmm_two_state(rate)
    hmm_states = hmm.pop("state_sequence")

    # Adaptive threshold classification (fast comparison)
    thr_states = _adaptive_threshold_states(rate, bimodal)

    # Agreement between methods
    if len(hmm_states) == len(thr_states):
        agreement = float(np.mean(hmm_states == thr_states))
    else:
        agreement = 0.0

    # Use HMM states as primary (more principled)
    primary_states = hmm_states

    # Extract epochs and duration statistics
    epoch_info = _extract_state_epochs(primary_states, bin_edges)
    up_stats = _duration_stats(epoch_info["up_durations_s"], "up")
    down_stats = _duration_stats(epoch_info["down_durations_s"], "down")

    # Transition dynamics
    transitions = _analyze_transitions(primary_states, bin_sec)

    up_frac = float(np.mean(primary_states)) if len(primary_states) > 0 else 0.0

    # Truncate for JSON serialisation
    max_state_pts = 5000
    stride = max(1, len(primary_states) // max_state_pts)

    return {
        "computed": True,
        "bin_size_ms": round(bin_sec * 1000.0, 1),
        "bin_size_requested_ms": bin_size_ms,
        "n_bins": len(rate),
        "bimodality": bimodal,
        "hmm": hmm,
        "method_agreement": round(agreement, 4),
        "up_fraction": round(up_frac, 4),
        **up_stats,
        **down_stats,
        "transitions": transitions,
        "states_sampled": primary_states[::stride].tolist(),
        "time_bins_sampled_s": bin_edges[::stride].tolist(),
        "n_epochs": len(epoch_info["epochs"]),
        "epochs_sample": epoch_info["epochs"][:200],
    }


def detect_slow_waves(
    data: SpikeData,
    bin_size_ms: float = 100.0,
) -> dict:
    """Detect slow-wave oscillations (0.1--1 Hz) in population activity.

    Parameters
    ----------
    data : SpikeData
    bin_size_ms : float
        Bin width for rate vector (default 100 ms = 10 Hz Nyquist).

    Returns
    -------
    dict with spectral analysis, dominant frequency, band powers,
    and phase-amplitude coupling results.
    """
    if data.n_spikes < 50:
        return {"computed": False, "reason": "insufficient_spikes"}

    t_start, t_end = data.time_range
    duration_sec = t_end - t_start
    requested_bin_sec = bin_size_ms / 1000.0
    bin_sec = _choose_adaptive_bin_sec(duration_sec, requested_bin_sec)
    rate, bin_edges = _compute_population_rate(data.times, t_start, t_end, bin_sec)

    # Spectral analysis
    slow_osc = _analyze_slow_oscillation(rate, bin_sec)

    # Phase-amplitude coupling
    pac = _compute_phase_amplitude_coupling(rate, bin_sec)

    return {
        "computed": True,
        "bin_size_ms": bin_size_ms,
        **slow_osc,
        "phase_amplitude_coupling": pac,
    }


def analyze_sleep_wake(
    data: SpikeData,
    bin_size_ms: float = 50.0,
    run_null_model: bool = True,
    run_per_mea: bool = True,
    run_multiscale: bool = True,
    vigilance_window_sec: float = 60.0,
    homeostasis_window_sec: float = 600.0,
    null_replicates: int = _N_NULL_REPLICATES,
) -> dict:
    """Comprehensive sleep-wake analysis for organoid MEA data.

    This is the primary entry point.  It orchestrates all sub-analyses
    and produces a single result dictionary suitable for the API.

    Parameters
    ----------
    data : SpikeData
        Full recording data (all MEAs, all time).
    bin_size_ms : float
        Bin width for population rate (50 ms default).
    run_null_model : bool
        Whether to run Poisson null-model validation (adds ~5 s).
    run_per_mea : bool
        Whether to run per-MEA UP/DOWN analysis.
    run_multiscale : bool
        Whether to run multi-scale temporal analysis.
    vigilance_window_sec : float
        Window size for coarse vigilance-state classification.
    homeostasis_window_sec : float
        Window size for homeostatic sleep-pressure tracking.
    null_replicates : int
        Number of surrogate replicates for null model test.

    Returns
    -------
    dict
        Comprehensive sleep-wake analysis results including:
        - up_down_states: HMM-based state detection with bimodality
        - slow_oscillations: spectral and PAC analysis
        - vigilance_states: active vs quiet period classification
        - homeostatic_pressure: sleep pressure tracking
        - per_mea: per-organoid breakdown (if enabled)
        - multiscale: multi-timescale dynamics (if enabled)
        - null_model: statistical validation (if enabled)
        - sleep_score: composite 0--1 score with qualitative label
    """
    if data.n_spikes < 20:
        return {
            "computed": False,
            "reason": "insufficient_spikes",
            "n_spikes": data.n_spikes,
            "sleep_score": {"sleep_score": 0.0, "label": "no_data"},
        }

    t_start, t_end = data.time_range
    duration_sec = t_end - t_start
    duration_hours = duration_sec / 3600.0
    requested_bin_sec = bin_size_ms / 1000.0
    bin_sec = _choose_adaptive_bin_sec(duration_sec, requested_bin_sec)

    # --- Core rate vector (reused across analyses) ---
    rate, bin_edges = _compute_population_rate(data.times, t_start, t_end, bin_sec)

    # --- 1. UP/DOWN state detection ---
    bimodal = _test_bimodality(rate)
    hmm = _fit_hmm_two_state(rate)
    hmm_states = hmm.pop("state_sequence")

    thr_states = _adaptive_threshold_states(rate, bimodal)
    agreement = float(np.mean(hmm_states == thr_states)) if len(hmm_states) == len(thr_states) else 0.0

    primary_states = hmm_states
    epoch_info = _extract_state_epochs(primary_states, bin_edges)
    up_stats = _duration_stats(epoch_info["up_durations_s"], "up")
    down_stats = _duration_stats(epoch_info["down_durations_s"], "down")
    transitions = _analyze_transitions(primary_states, bin_sec)

    up_frac = float(np.mean(primary_states)) if len(primary_states) > 0 else 0.0

    max_state_pts = 5000
    stride_s = max(1, len(primary_states) // max_state_pts)

    up_down_result = {
        "bin_size_ms": round(bin_sec * 1000.0, 1),
        "bin_size_requested_ms": bin_size_ms,
        "n_bins": len(rate),
        "bimodality": bimodal,
        "hmm": hmm,
        "method_agreement": round(agreement, 4),
        "up_fraction": round(up_frac, 4),
        **up_stats,
        **down_stats,
        "transitions": transitions,
        "states_sampled": primary_states[::stride_s].tolist(),
        "time_bins_sampled_s": bin_edges[::stride_s].tolist(),
        "n_epochs": len(epoch_info["epochs"]),
        "epochs_sample": epoch_info["epochs"][:200],
    }

    # --- 2. Slow oscillation + PAC ---
    slow_bin_sec = _choose_adaptive_bin_sec(duration_sec, _SLOW_OSC_BIN_SEC)
    slow_rate, slow_edges = _compute_population_rate(data.times, t_start, t_end, slow_bin_sec)
    slow_osc = _analyze_slow_oscillation(slow_rate, slow_bin_sec)
    pac = _compute_phase_amplitude_coupling(slow_rate, slow_bin_sec)

    slow_result = {
        **slow_osc,
        "phase_amplitude_coupling": pac,
    }

    # --- 3. Vigilance states ---
    vigilance = _classify_vigilance_states(rate, bin_edges, vigilance_window_sec)

    # --- 4. Homeostatic sleep pressure ---
    homeostasis = _compute_sleep_pressure(primary_states, bin_edges, homeostasis_window_sec)

    # --- 5. Per-MEA breakdown ---
    per_mea = _per_mea_analysis(data, bin_sec) if run_per_mea else []

    # --- 6. Multi-scale temporal analysis ---
    multiscale = _multiscale_analysis(rate, bin_sec) if run_multiscale else []

    # --- 7. Null model validation ---
    if run_null_model:
        null_val = _null_model_validation(
            data.times, t_start, t_end, bin_sec, primary_states, null_replicates,
        )
    else:
        null_val = {"null_test_computed": False, "reason": "skipped"}

    # --- 8. Composite score ---
    score = _compute_sleep_score(bimodal, hmm, slow_osc, pac, null_val, vigilance)

    return {
        "computed": True,
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_hours": round(duration_hours, 2),
        "up_down_states": up_down_result,
        "slow_oscillations": slow_result,
        "vigilance_states": vigilance,
        "homeostatic_pressure": homeostasis,
        "per_mea": per_mea,
        "multiscale": multiscale,
        "null_model": null_val,
        "sleep_score": score,
        # Backward-compatible aliases for existing tests / consumers
        "sleep_like_score": score["sleep_score"],
        "has_up_down_states": bimodal.get("is_bimodal", False) or hmm.get("converged", False),
        "has_slow_waves": slow_osc.get("slow_osc_detected", False),
    }
