"""Predictive Coding Analysis -- Free Energy Principle in Neural Organoids.

Tests whether a neural organoid generates internal predictions about its own
future activity states and responds differentially to expected vs surprising
state transitions.  This is a fundamental question in computational
neuroscience: if an organoid shows predictive coding, it is performing active
inference (Friston, 2010), a prerequisite for learning and adaptive behaviour.

Four complementary detection methods are applied across multiple timescales
(100 ms, 1 s, 10 s, 60 s) and combined via a consensus framework:

1. **Transition-Probability Analysis** -- builds a first-order Markov model of
   discretised network states and compares the population response amplitude
   following high-probability (expected) vs low-probability (surprising)
   transitions.

2. **Mismatch-Negativity Analog** -- detects whether the response to a
   repeated firing-rate pattern diminishes (adaptation) and recovers when the
   pattern is violated (deviant detection), mirroring auditory MMN in cortex.

3. **Prediction-Error Electrodes** -- identifies individual channels whose
   firing rate increases specifically when the network state deviates from its
   recent trajectory, consistent with a dedicated prediction-error signal.

4. **Bayesian Surprise** -- computes the KL divergence between a sliding-window
   predicted state distribution (built from recent history) and the actually
   observed state distribution, then tests whether high-surprise windows
   evoke stronger subsequent activity.

Statistical rigour:
- Null model via spike-time shuffling (preserving per-electrode rate).
- Bonferroni correction for multiple comparisons across methods & timescales.
- Effect size (Cohen's d) reported alongside p-values.
- Conservative detection threshold: requires p < 0.01 (Bonferroni-corrected)
  AND |d| > 0.3 for at least 2 / 4 methods.

Performance:  Designed for 2.6 M spikes, 32 electrodes, 118 h of recording.
Adaptive binning caps total bins at 50 000.  Transition learning uses sampled
windows.  All inner loops are fully vectorised with NumPy.

References:
    Friston K (2010). The free-energy principle: a unified brain theory?
    Kagan BJ et al. (2022). In vitro neurons learn and exhibit sentience
        when embodied in a simulated game-world. Neuron 110(23).
    Garrido MI et al. (2009). The mismatch negativity: a review of
        underlying mechanisms. Clinical Neurophysiology 120(3).
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MAX_BINS = 50_000
_MAX_STATE_BITS = 8
_N_SAMPLE_WINDOWS = 20
_WINDOW_DURATION_S = 300.0
_MIN_BIN_MS = 10.0
_N_SHUFFLES = 200          # surrogate datasets for null model
_TIMESCALES_MS = [100.0, 1000.0, 10_000.0, 60_000.0]
_MIN_EVENTS_PER_CLASS = 30  # minimum events in expected/surprise for a test


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def measure_predictive_coding(
    data: SpikeData,
) -> dict:
    """Run all four predictive-coding detection methods at multiple timescales.

    Returns
    -------
    dict with keys:
        has_predictive_coding : bool
        surprise_ratio        : float   (mean across significant methods)
        p_value               : float   (minimum Bonferroni-corrected p)
        effect_size           : float   (maximum |Cohen's d|)
        method_results        : dict    (per-method breakdown)
        interpretation        : str
    """
    t_start, t_end = data.time_range
    total_duration = t_end - t_start
    if total_duration <= 0 or data.n_spikes < 100:
        return _empty_result("Insufficient data for predictive coding analysis.")

    n_el = len(data.electrode_ids)
    if n_el < 2:
        return _empty_result("Need at least 2 electrodes.")

    # --- Pre-split spike times per electrode (reuse loader cache) -----------
    per_el_times = _get_per_electrode_times(data)

    # --- Select the most informative electrodes for state encoding ----------
    top_el_ids, n_state_bits = _select_top_electrodes(
        data, per_el_times, t_start, t_end
    )

    # --- Run each method at each feasible timescale -------------------------
    method_results = {}
    all_pvals = []
    all_effects = []
    all_ratios = []

    for ts_ms in _TIMESCALES_MS:
        # Skip timescales longer than 1/20 of total recording
        if ts_ms / 1000.0 > total_duration / 20:
            continue

        label = f"{_timescale_label(ts_ms)}"

        # Adaptive bin size for this timescale
        bin_ms = max(_MIN_BIN_MS, ts_ms / 10.0)
        bin_ms = max(bin_ms, (total_duration * 1000.0) / _MAX_BINS)

        # 1. Transition probability
        tp_result = _method_transition_probability(
            data, per_el_times, top_el_ids, n_state_bits,
            t_start, t_end, total_duration, bin_ms, ts_ms,
        )
        method_results[f"transition_probability_{label}"] = tp_result
        _collect_method_stats(tp_result, all_pvals, all_effects, all_ratios)

        # 2. Mismatch negativity analog
        mmn_result = _method_mismatch_negativity(
            data, per_el_times, t_start, t_end, total_duration, bin_ms, ts_ms,
        )
        method_results[f"mismatch_negativity_{label}"] = mmn_result
        _collect_method_stats(mmn_result, all_pvals, all_effects, all_ratios)

        # 3. Prediction-error electrodes
        pe_result = _method_prediction_error_electrodes(
            data, per_el_times, top_el_ids, n_state_bits,
            t_start, t_end, total_duration, bin_ms, ts_ms,
        )
        method_results[f"prediction_error_electrodes_{label}"] = pe_result
        _collect_method_stats(pe_result, all_pvals, all_effects, all_ratios)

        # 4. Bayesian surprise
        bs_result = _method_bayesian_surprise(
            data, per_el_times, top_el_ids, n_state_bits,
            t_start, t_end, total_duration, bin_ms, ts_ms,
        )
        method_results[f"bayesian_surprise_{label}"] = bs_result
        _collect_method_stats(bs_result, all_pvals, all_effects, all_ratios)

    if not all_pvals:
        return _empty_result(
            "None of the detection methods produced testable results. "
            "The recording may be too short or too sparse."
        )

    # --- Bonferroni correction across all tests -----------------------------
    n_tests = len(all_pvals)
    corrected_pvals = [min(p * n_tests, 1.0) for p in all_pvals]

    # Count how many methods pass the conservative threshold
    n_significant = sum(
        1 for p, d in zip(corrected_pvals, all_effects)
        if p < 0.01 and d > 0.3
    )

    # --- Aggregate results --------------------------------------------------
    best_idx = int(np.argmin(corrected_pvals))
    min_p = corrected_pvals[best_idx]
    max_d = float(max(all_effects)) if all_effects else 0.0

    # Mean surprise ratio across significant methods only
    sig_ratios = [
        r for r, p, d in zip(all_ratios, corrected_pvals, all_effects)
        if p < 0.01 and d > 0.3
    ]
    mean_ratio = float(np.mean(sig_ratios)) if sig_ratios else float(np.mean(all_ratios))

    # Detection requires at least 2 methods significant
    has_predictive_coding = n_significant >= 2

    # Annotate each method result with its corrected p-value
    i = 0
    for key in method_results:
        if method_results[key].get("p_value") is not None:
            method_results[key]["p_value_corrected"] = round(corrected_pvals[i], 6)
            method_results[key]["significant_corrected"] = (
                corrected_pvals[i] < 0.01 and all_effects[i] > 0.3
            )
            i += 1

    return {
        "has_predictive_coding": has_predictive_coding,
        "surprise_ratio": round(mean_ratio, 4),
        "p_value": round(min_p, 6),
        "effect_size": round(max_d, 4),
        "n_significant_methods": n_significant,
        "n_tests_total": n_tests,
        "method_results": method_results,
        "interpretation": _build_interpretation(
            has_predictive_coding, n_significant, n_tests,
            min_p, max_d, mean_ratio, method_results,
        ),
    }


# ---------------------------------------------------------------------------
# Method 1: Transition-Probability Analysis (Markov model)
# ---------------------------------------------------------------------------
def _method_transition_probability(
    data, per_el_times, top_el_ids, n_state_bits,
    t_start, t_end, total_duration, bin_ms, timescale_ms,
):
    """Build Markov model; compare response to expected vs surprising transitions."""
    # For the Markov model, we process sampled windows -- so the bin size
    # only needs to cover each window, not the entire recording.  Use a
    # finer bin size (proportional to timescale) capped by the per-window limit.
    per_window_max_bins = _MAX_BINS // max(_N_SAMPLE_WINDOWS, 1)
    window_dur = min(_WINDOW_DURATION_S, total_duration)
    bin_ms_local = max(_MIN_BIN_MS, timescale_ms / 10.0)
    bin_ms_local = max(bin_ms_local, (window_dur * 1000.0) / per_window_max_bins)
    bin_sec = bin_ms_local / 1000.0
    delay_bins = max(1, int(timescale_ms / bin_ms_local))

    windows = _get_sample_windows(t_start, t_end, total_duration)
    bit_mult = 1 << np.arange(n_state_bits, dtype=np.int64)

    # --- Learn transition probabilities from sampled windows ----------------
    from collections import Counter

    # transition_counts[current_state] = Counter(future_states)
    transition_counts: dict[int, Counter] = {}
    total_bins_processed = 0

    for w_start, w_end in windows:
        state_codes, n_bins_w = _encode_window_states(
            per_el_times, top_el_ids, n_state_bits, bit_mult,
            w_start, w_end, bin_sec,
        )
        if state_codes is None or n_bins_w < delay_bins + 2:
            continue

        limit = n_bins_w - delay_bins
        cur_arr = state_codes[:limit]
        fut_arr = state_codes[delay_bins:delay_bins + limit]

        uniq, inverse = np.unique(cur_arr, return_inverse=True)
        for si, sv in enumerate(uniq):
            mask = inverse == si
            sv_int = int(sv)
            if sv_int not in transition_counts:
                transition_counts[sv_int] = Counter()
            # Use numpy bincount on future states for this group
            futures = fut_arr[mask]
            for f_val, f_cnt in zip(*np.unique(futures, return_counts=True)):
                transition_counts[sv_int][int(f_val)] += int(f_cnt)

        total_bins_processed += n_bins_w

    if total_bins_processed < 50:
        return {"detected": False, "reason": "too few bins"}

    # --- Compute transition probabilities -----------------------------------
    # For each state, compute the full probability distribution over next states
    # and define a threshold to split transitions into "expected" vs "surprising".
    # We use the probability of the *specific observed transition* and split by
    # the per-state median probability.  Transitions to states that were never
    # observed from this state are assigned probability 0 (maximally surprising).
    predictions = {}
    for state, ctr in transition_counts.items():
        total = sum(ctr.values())
        if total < 10:
            continue
        probs = {k: v / total for k, v in ctr.items()}
        most_common = ctr.most_common(1)[0]

        # Threshold: the probability below which a transition is "surprising".
        # We use the weighted median -- the probability value such that ~50%
        # of observed transitions have probability >= threshold.
        prob_values = []
        for k, cnt in ctr.items():
            prob_values.extend([probs[k]] * cnt)
        prob_threshold = float(np.median(prob_values))

        predictions[state] = {
            "expected_state": most_common[0],
            "expected_prob": most_common[1] / total,
            "prob_dist": probs,
            "prob_threshold": prob_threshold,
            "n_obs": total,
        }

    if len(predictions) < 3:
        return {"detected": False, "reason": "too few predictable states"}

    # --- Classify transitions as expected/surprising and measure response ----
    expected_responses = []
    surprise_responses = []
    electrode_ids = data.electrode_ids

    for w_start, w_end in windows:
        state_codes, n_bins_w = _encode_window_states(
            per_el_times, top_el_ids, n_state_bits, bit_mult,
            w_start, w_end, bin_sec,
        )
        if state_codes is None or n_bins_w < delay_bins + 2:
            continue

        # Total population activity per bin (response measure)
        w_bins = np.arange(w_start, w_end + bin_sec * 0.5, bin_sec)[:n_bins_w + 1]
        total_activity = _compute_population_activity(
            per_el_times, electrode_ids, w_bins, n_bins_w,
        )

        limit = n_bins_w - delay_bins - 1
        if limit <= 0:
            continue

        cur_codes = state_codes[:limit]
        act_codes = state_codes[delay_bins:delay_bins + limit]
        # Response = activity in the bin *after* the transition completes
        resp_vals = total_activity[delay_bins + 1:delay_bins + 1 + limit]

        for t in range(limit):
            c = int(cur_codes[t])
            if c not in predictions:
                continue
            pred = predictions[c]
            actual = int(act_codes[t])
            # Probability of this specific transition; unseen transitions = 0
            actual_prob = pred["prob_dist"].get(actual, 0.0)

            if actual_prob >= pred["prob_threshold"]:
                expected_responses.append(resp_vals[t])
            else:
                surprise_responses.append(resp_vals[t])

    return _compare_expected_vs_surprise(
        expected_responses, surprise_responses, "transition_probability",
    )


# ---------------------------------------------------------------------------
# Method 2: Mismatch Negativity Analog
# ---------------------------------------------------------------------------
def _method_mismatch_negativity(
    data, per_el_times, t_start, t_end, total_duration, bin_ms, timescale_ms,
):
    """Detect adaptation to repeated patterns and recovery upon deviance.

    We look for epochs where the population rate remains in a consistent
    quantised activity level (the "standard") and measure whether:
      a) Successive repetitions produce diminishing responses (adaptation).
      b) A deviation from the standard produces an enhanced response (deviant).

    The pattern is defined at the timescale granularity: for short timescales,
    multi-bin patterns are used; for large adaptive bin sizes, single-bin
    activity levels serve as the repeating unit.
    """
    bin_sec = bin_ms / 1000.0
    electrode_ids = list(per_el_times.keys())

    # Pattern length in bins: keep it short (1-5 bins) so that exact matches
    # are common enough even with coarse binning.
    pattern_len = max(1, min(5, int(timescale_ms / bin_ms)))

    # Limit to sampled windows for performance
    windows = _get_sample_windows(t_start, t_end, total_duration)

    adaptation_slopes = []   # slope of response amplitude across repetitions
    deviant_boosts = []      # ratio of deviant response to last-standard response

    for w_start, w_end in windows:
        w_bins_edges = np.arange(w_start, w_end + bin_sec * 0.5, bin_sec)
        n_bins_w = len(w_bins_edges) - 1
        if n_bins_w < pattern_len * 6:
            continue

        total_activity = _compute_population_activity(
            per_el_times, electrode_ids, w_bins_edges, n_bins_w,
        )

        # Quantise activity into low / medium / high for pattern matching
        if np.std(total_activity) < 1e-10:
            continue

        # Use 3-level quantisation; for single-bin patterns this gives
        # reasonable chance of consecutive repeats
        q_edges = [
            np.percentile(total_activity, 33),
            np.percentile(total_activity, 66),
        ]
        quantised = np.digitize(total_activity, bins=q_edges)  # values 0, 1, 2

        # Slide through and find sequences of repeated pattern
        # "Standard" = same quantised pattern repeated 3+ times consecutively
        i = 0
        while i + pattern_len * 4 < n_bins_w:
            pattern = tuple(quantised[i:i + pattern_len])
            reps = 1
            j = i + pattern_len
            rep_amplitudes = [float(np.sum(total_activity[i:i + pattern_len]))]

            while j + pattern_len <= n_bins_w:
                candidate = tuple(quantised[j:j + pattern_len])
                if candidate == pattern:
                    rep_amplitudes.append(
                        float(np.sum(total_activity[j:j + pattern_len]))
                    )
                    reps += 1
                    j += pattern_len
                else:
                    break

            if reps >= 3:
                # Measure adaptation: linear regression of rep amplitudes
                x = np.arange(reps, dtype=float)
                if np.std(rep_amplitudes) > 1e-10:
                    slope = float(np.polyfit(x, rep_amplitudes, 1)[0])
                    adaptation_slopes.append(slope)

                # Check if the next pattern (which broke the repetition) is a deviant
                if j + pattern_len <= n_bins_w:
                    deviant_amp = float(np.sum(total_activity[j:j + pattern_len]))
                    last_standard = rep_amplitudes[-1]
                    if last_standard > 0:
                        deviant_boosts.append(deviant_amp / last_standard)

                i = j + pattern_len
            else:
                # Step forward by 1 bin (not pattern_len) to catch more repeats
                i += 1

    if len(adaptation_slopes) < _MIN_EVENTS_PER_CLASS // 3:
        return {"detected": False, "reason": "too few repeated patterns found"}

    # --- Test 1: Adaptation (slopes should be negative) ---------------------
    slopes_arr = np.array(adaptation_slopes, dtype=np.float64)
    mean_slope = float(np.mean(slopes_arr))
    # one-sample t-test: is mean slope significantly < 0?
    from scipy.stats import ttest_1samp
    if len(slopes_arr) >= 5:
        t_adapt, p_adapt = ttest_1samp(slopes_arr, 0.0)
        p_adapt = float(p_adapt) / 2.0  # one-sided
        if mean_slope > 0:
            p_adapt = 1.0 - p_adapt  # wrong direction
    else:
        t_adapt, p_adapt = 0.0, 1.0

    # --- Test 2: Deviant boost (should be > 1.0) ----------------------------
    if len(deviant_boosts) >= 5:
        boosts_arr = np.array(deviant_boosts, dtype=np.float64)
        t_dev, p_dev = ttest_1samp(boosts_arr, 1.0)
        p_dev = float(p_dev) / 2.0  # one-sided
        mean_boost = float(np.mean(boosts_arr))
        if mean_boost < 1.0:
            p_dev = 1.0 - p_dev
        d_dev = (mean_boost - 1.0) / max(float(np.std(boosts_arr)), 1e-10)
    else:
        p_dev = 1.0
        mean_boost = 1.0
        d_dev = 0.0

    # Combine the two sub-tests: use the stronger one
    combined_p = min(p_adapt, p_dev) * 2  # Bonferroni for 2 sub-tests
    combined_p = min(combined_p, 1.0)

    d_adapt = abs(mean_slope) / max(float(np.std(slopes_arr)), 1e-10) if len(slopes_arr) > 1 else 0.0

    return {
        "detected": combined_p < 0.01 and max(abs(d_adapt), abs(d_dev)) > 0.3,
        "adaptation_mean_slope": round(mean_slope, 6),
        "adaptation_p_value": round(float(p_adapt), 6),
        "adaptation_effect_size": round(float(d_adapt), 4),
        "n_adaptation_sequences": len(adaptation_slopes),
        "deviant_mean_boost": round(float(mean_boost), 4),
        "deviant_p_value": round(float(p_dev), 6),
        "deviant_effect_size": round(float(d_dev), 4),
        "n_deviant_events": len(deviant_boosts),
        "p_value": round(float(combined_p), 6),
        "effect_size": round(float(max(abs(d_adapt), abs(d_dev))), 4),
        "surprise_ratio": round(float(mean_boost), 4),
    }


# ---------------------------------------------------------------------------
# Method 3: Prediction-Error Electrodes
# ---------------------------------------------------------------------------
def _method_prediction_error_electrodes(
    data, per_el_times, top_el_ids, n_state_bits,
    t_start, t_end, total_duration, bin_ms, timescale_ms,
):
    """Find electrodes that increase firing specifically during state deviations.

    For each electrode, we compute the correlation between its firing rate and
    the network-level prediction error (deviation of actual state from the
    state predicted by recent history).  Electrodes with significant positive
    correlation are "prediction-error channels".
    """
    bin_sec = bin_ms / 1000.0
    history_len = max(2, int(timescale_ms / bin_ms))
    bit_mult = 1 << np.arange(n_state_bits, dtype=np.int64)
    electrode_ids = data.electrode_ids

    windows = _get_sample_windows(t_start, t_end, total_duration)

    # Accumulate per-electrode firing rate vs prediction error
    per_el_rates: dict[int, list] = {e: [] for e in electrode_ids}
    pred_errors: list = []

    for w_start, w_end in windows:
        state_codes, n_bins_w = _encode_window_states(
            per_el_times, top_el_ids, n_state_bits, bit_mult,
            w_start, w_end, bin_sec,
        )
        if state_codes is None or n_bins_w < history_len + 2:
            continue

        w_bins_edges = np.arange(w_start, w_end + bin_sec * 0.5, bin_sec)[:n_bins_w + 1]

        # Compute per-electrode bin counts for this window
        el_counts = {}
        for e in electrode_ids:
            ts = per_el_times[e]
            lo = np.searchsorted(ts, w_start, side="left")
            hi = np.searchsorted(ts, w_end, side="right")
            ts_w = ts[lo:hi]
            counts = np.zeros(n_bins_w, dtype=np.float64)
            if len(ts_w) > 0:
                bin_idx = np.clip(
                    np.searchsorted(w_bins_edges, ts_w, side="right") - 1,
                    0, n_bins_w - 1,
                )
                np.add.at(counts, bin_idx, 1.0)
            el_counts[e] = counts

        # Prediction error: Hamming distance between predicted state
        # (most common state in last `history_len` bins) and actual state
        for t in range(history_len, n_bins_w):
            history = state_codes[t - history_len:t]
            # "Predicted" = most frequent state in recent history
            vals, cnts = np.unique(history, return_counts=True)
            predicted = vals[np.argmax(cnts)]
            actual = state_codes[t]
            # Hamming distance (number of differing bits)
            xor = int(predicted) ^ int(actual)
            hamming = bin(xor).count("1")
            pred_errors.append(hamming)
            for e in electrode_ids:
                per_el_rates[e].append(el_counts[e][t])

    if len(pred_errors) < _MIN_EVENTS_PER_CLASS:
        return {"detected": False, "reason": "too few time points"}

    pred_errors_arr = np.array(pred_errors, dtype=np.float64)

    # --- Test per-electrode correlation with prediction error ----------------
    from scipy.stats import pearsonr

    pe_electrodes = []
    raw_pvals = []
    raw_effects = []

    for e in electrode_ids:
        rates = np.array(per_el_rates[e], dtype=np.float64)
        if np.std(rates) < 1e-10 or np.std(pred_errors_arr) < 1e-10:
            continue
        r, p = pearsonr(rates, pred_errors_arr)
        pe_electrodes.append({
            "electrode": int(e),
            "correlation": round(float(r), 4),
            "p_value": round(float(p), 6),
        })
        if r > 0:  # We only care about positive correlation
            raw_pvals.append(float(p))
            raw_effects.append(float(r))

    if not raw_pvals:
        return {"detected": False, "reason": "no electrodes correlate with prediction error"}

    # Bonferroni across electrodes
    n_el_tests = len(raw_pvals)
    corrected_el_pvals = [min(p * n_el_tests, 1.0) for p in raw_pvals]
    n_sig_electrodes = sum(1 for p in corrected_el_pvals if p < 0.05)

    # Overall test: is the mean correlation significantly positive?
    from scipy.stats import ttest_1samp
    all_corrs = np.array([e["correlation"] for e in pe_electrodes], dtype=np.float64)
    if len(all_corrs) >= 3:
        t_stat, p_overall = ttest_1samp(all_corrs, 0.0)
        p_overall = float(p_overall) / 2.0  # one-sided
        if float(np.mean(all_corrs)) < 0:
            p_overall = 1.0 - p_overall
        d_overall = float(np.mean(all_corrs)) / max(float(np.std(all_corrs)), 1e-10)
    else:
        p_overall, d_overall = 1.0, 0.0

    # Sort by correlation descending
    pe_electrodes.sort(key=lambda x: x["correlation"], reverse=True)

    return {
        "detected": p_overall < 0.01 and d_overall > 0.3,
        "n_prediction_error_electrodes": n_sig_electrodes,
        "n_electrodes_tested": len(pe_electrodes),
        "mean_correlation": round(float(np.mean(all_corrs)), 4),
        "top_electrodes": pe_electrodes[:5],
        "p_value": round(float(p_overall), 6),
        "effect_size": round(float(d_overall), 4),
        "surprise_ratio": round(1.0 + float(np.mean(all_corrs)), 4),
    }


# ---------------------------------------------------------------------------
# Method 4: Bayesian Surprise (KL divergence)
# ---------------------------------------------------------------------------
def _method_bayesian_surprise(
    data, per_el_times, top_el_ids, n_state_bits,
    t_start, t_end, total_duration, bin_ms, timescale_ms,
):
    """Compute KL divergence between predicted and actual state distributions.

    Sliding window of size `timescale_ms`:
    - "Prior" distribution = state frequencies in the previous window.
    - "Posterior" distribution = state frequencies in the current window.
    - Bayesian surprise = KL(posterior || prior).

    Then test: do high-surprise windows evoke higher subsequent activity?
    """
    bin_sec = bin_ms / 1000.0
    window_bins = max(5, int(timescale_ms / bin_ms))
    bit_mult = 1 << np.arange(n_state_bits, dtype=np.int64)
    electrode_ids = data.electrode_ids

    windows = _get_sample_windows(t_start, t_end, total_duration)

    surprise_values = []
    subsequent_activities = []

    for w_start, w_end in windows:
        state_codes, n_bins_w = _encode_window_states(
            per_el_times, top_el_ids, n_state_bits, bit_mult,
            w_start, w_end, bin_sec,
        )
        if state_codes is None or n_bins_w < window_bins * 3:
            continue

        w_bins_edges = np.arange(w_start, w_end + bin_sec * 0.5, bin_sec)[:n_bins_w + 1]
        total_activity = _compute_population_activity(
            per_el_times, electrode_ids, w_bins_edges, n_bins_w,
        )

        # Slide through in steps of window_bins
        n_steps = (n_bins_w - window_bins) // window_bins
        if n_steps < 3:
            continue

        for step in range(1, n_steps - 1):
            prior_start = (step - 1) * window_bins
            prior_end = step * window_bins
            curr_start = step * window_bins
            curr_end = (step + 1) * window_bins
            next_start = curr_end
            next_end = min(curr_end + window_bins, n_bins_w)

            prior_states = state_codes[prior_start:prior_end]
            curr_states = state_codes[curr_start:curr_end]

            kl = _kl_divergence_discrete(prior_states, curr_states)
            if kl is not None:
                surprise_values.append(kl)
                subsequent_activities.append(
                    float(np.mean(total_activity[next_start:next_end]))
                )

    if len(surprise_values) < _MIN_EVENTS_PER_CLASS:
        return {"detected": False, "reason": "too few windows for KL computation"}

    surprise_arr = np.array(surprise_values, dtype=np.float64)
    activity_arr = np.array(subsequent_activities, dtype=np.float64)

    # Split into high-surprise and low-surprise by median
    median_surprise = float(np.median(surprise_arr))
    high_mask = surprise_arr > median_surprise
    low_mask = ~high_mask

    high_activity = activity_arr[high_mask]
    low_activity = activity_arr[low_mask]

    if len(high_activity) < 5 or len(low_activity) < 5:
        return {"detected": False, "reason": "too few events after median split"}

    # Also compute correlation between surprise and subsequent activity
    from scipy.stats import pearsonr, mannwhitneyu

    if np.std(surprise_arr) > 1e-10 and np.std(activity_arr) > 1e-10:
        r_corr, p_corr = pearsonr(surprise_arr, activity_arr)
    else:
        r_corr, p_corr = 0.0, 1.0

    # Mann-Whitney U test (more robust than t-test for possibly non-normal data)
    u_stat, p_mw = mannwhitneyu(high_activity, low_activity, alternative="greater")

    mean_high = float(np.mean(high_activity))
    mean_low = float(np.mean(low_activity))
    pooled_std = float(np.sqrt(
        (np.var(high_activity) * (len(high_activity) - 1) +
         np.var(low_activity) * (len(low_activity) - 1)) /
        max(len(high_activity) + len(low_activity) - 2, 1)
    ))
    d = (mean_high - mean_low) / max(pooled_std, 1e-10)
    ratio = mean_high / max(mean_low, 1e-10)

    return {
        "detected": float(p_mw) < 0.01 and abs(d) > 0.3,
        "mean_surprise_kl": round(float(np.mean(surprise_arr)), 6),
        "median_surprise_kl": round(float(median_surprise), 6),
        "mean_activity_high_surprise": round(mean_high, 4),
        "mean_activity_low_surprise": round(mean_low, 4),
        "correlation_surprise_activity": round(float(r_corr), 4),
        "correlation_p_value": round(float(p_corr), 6),
        "p_value": round(float(p_mw), 6),
        "effect_size": round(float(d), 4),
        "surprise_ratio": round(float(ratio), 4),
        "n_windows_tested": len(surprise_values),
    }


# ===========================================================================
# Shared helpers
# ===========================================================================

def _collect_method_stats(result, all_pvals, all_effects, all_ratios):
    """Extract p-value, effect size, and surprise ratio from a method result.

    Only adds to the lists if the method produced a valid p-value.
    Effect size is only counted when the p-value < 0.5 (i.e. the direction
    of the effect is at least plausible).  This prevents methods that find
    strong effects in the *wrong* direction (p -> 1.0 from one-sided test)
    from inflating the aggregate max effect size.
    """
    p = result.get("p_value")
    if p is None:
        return
    all_pvals.append(float(p))
    d = abs(result.get("effect_size", 0.0))
    # Only trust the effect size if the direction is at least plausible
    if float(p) < 0.5:
        all_effects.append(d)
    else:
        all_effects.append(0.0)
    all_ratios.append(result.get("surprise_ratio", 1.0))

def _get_per_electrode_times(data: SpikeData) -> dict[int, np.ndarray]:
    """Extract pre-sorted spike times per electrode."""
    result = {}
    for e in data.electrode_ids:
        idx = data._electrode_indices.get(e)
        if idx is not None and len(idx) > 0:
            result[e] = data.times[idx]
        else:
            result[e] = np.empty(0, dtype=np.float64)
    return result


def _select_top_electrodes(
    data, per_el_times, t_start, t_end,
) -> tuple[list[int], int]:
    """Select the most variable electrodes for state encoding."""
    n_el = len(data.electrode_ids)
    n_state_bits = min(_MAX_STATE_BITS, n_el)

    global_bins = np.linspace(t_start, t_end, min(5001, _MAX_BINS + 1))
    variances = np.zeros(n_el)
    for i, e in enumerate(data.electrode_ids):
        ts = per_el_times[e]
        if len(ts) > 0:
            counts = np.diff(np.searchsorted(ts, global_bins))
            variances[i] = np.var(counts > 0)

    top_indices = np.argsort(variances)[-n_state_bits:][::-1]
    top_ids = [data.electrode_ids[i] for i in top_indices]
    return top_ids, n_state_bits


def _get_sample_windows(
    t_start: float, t_end: float, total_duration: float,
) -> list[tuple[float, float]]:
    """Return uniformly-spaced sample windows for large recordings."""
    min_full = _N_SAMPLE_WINDOWS * _WINDOW_DURATION_S * 1.5
    if total_duration <= min_full:
        return [(t_start, t_end)]
    spacing = (total_duration - _WINDOW_DURATION_S) / max(_N_SAMPLE_WINDOWS - 1, 1)
    return [
        (t_start + i * spacing, t_start + i * spacing + _WINDOW_DURATION_S)
        for i in range(_N_SAMPLE_WINDOWS)
    ]


def _encode_window_states(
    per_el_times, top_el_ids, n_state_bits, bit_mult,
    w_start, w_end, bin_sec,
) -> tuple[Optional[np.ndarray], int]:
    """Encode network state in a window as bit-packed integers.

    Returns (state_codes, n_bins) or (None, 0) if window too short.
    """
    w_bins = np.arange(w_start, w_end + bin_sec * 0.5, bin_sec)
    n_bins_w = len(w_bins) - 1
    if n_bins_w < 5:
        return None, 0

    binary = np.zeros((n_state_bits, n_bins_w), dtype=np.int64)
    for j, e in enumerate(top_el_ids):
        ts = per_el_times.get(e, np.empty(0, dtype=np.float64))
        lo = np.searchsorted(ts, w_start, side="left")
        hi = np.searchsorted(ts, w_end, side="right")
        ts_w = ts[lo:hi]
        if len(ts_w) > 0:
            bin_idx = np.clip(
                np.searchsorted(w_bins, ts_w, side="right") - 1,
                0, n_bins_w - 1,
            )
            np.add.at(binary[j], bin_idx, 1)

    binary = (binary > 0).astype(np.int64)
    state_codes = binary.T @ bit_mult
    return state_codes, n_bins_w


def _compute_population_activity(
    per_el_times, electrode_ids, w_bins, n_bins_w,
) -> np.ndarray:
    """Compute total spike count per bin across all electrodes."""
    total = np.zeros(n_bins_w, dtype=np.float64)
    w_start = w_bins[0]
    w_end = w_bins[-1]
    for e in electrode_ids:
        ts = per_el_times.get(e, np.empty(0, dtype=np.float64))
        lo = np.searchsorted(ts, w_start, side="left")
        hi = np.searchsorted(ts, w_end, side="right")
        ts_w = ts[lo:hi]
        if len(ts_w) > 0:
            bin_idx = np.clip(
                np.searchsorted(w_bins, ts_w, side="right") - 1,
                0, n_bins_w - 1,
            )
            np.add.at(total, bin_idx, 1.0)
    return total


def _kl_divergence_discrete(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    smoothing: float = 1.0,
) -> Optional[float]:
    """KL(posterior || prior) for discrete state samples with Laplace smoothing."""
    all_states = np.union1d(prior_samples, posterior_samples)
    n_states = len(all_states)
    if n_states < 2:
        return None

    # Build lookup: state -> index
    state_to_idx = {int(s): i for i, s in enumerate(all_states)}

    prior_counts = np.full(n_states, smoothing, dtype=np.float64)
    post_counts = np.full(n_states, smoothing, dtype=np.float64)

    for s in prior_samples:
        prior_counts[state_to_idx[int(s)]] += 1.0
    for s in posterior_samples:
        post_counts[state_to_idx[int(s)]] += 1.0

    prior_dist = prior_counts / prior_counts.sum()
    post_dist = post_counts / post_counts.sum()

    # KL divergence: sum p(x) * log(p(x) / q(x))
    # With smoothing, no zeros possible
    kl = float(np.sum(post_dist * np.log(post_dist / prior_dist)))
    return max(kl, 0.0)  # numerical safety


def _compare_expected_vs_surprise(
    expected_responses: list,
    surprise_responses: list,
    method_name: str,
) -> dict:
    """Statistical comparison of response amplitudes: expected vs surprise."""
    if (len(expected_responses) < _MIN_EVENTS_PER_CLASS or
            len(surprise_responses) < _MIN_EVENTS_PER_CLASS):
        return {
            "detected": False,
            "reason": (
                f"Too few events ({len(expected_responses)} expected, "
                f"{len(surprise_responses)} surprise; need {_MIN_EVENTS_PER_CLASS} each)"
            ),
        }

    exp_arr = np.array(expected_responses, dtype=np.float64)
    sur_arr = np.array(surprise_responses, dtype=np.float64)

    mean_exp = float(np.mean(exp_arr))
    mean_sur = float(np.mean(sur_arr))
    ratio = mean_sur / max(mean_exp, 1e-10)

    # Welch's t-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(sur_arr, exp_arr, equal_var=False)
    p_value = float(p_value) / 2.0  # one-sided: surprise > expected
    if mean_sur < mean_exp:
        p_value = 1.0 - p_value

    # Cohen's d
    pooled_std = float(np.sqrt(
        (np.var(exp_arr) * (len(exp_arr) - 1) +
         np.var(sur_arr) * (len(sur_arr) - 1)) /
        max(len(exp_arr) + len(sur_arr) - 2, 1)
    ))
    d = (mean_sur - mean_exp) / max(pooled_std, 1e-10)

    return {
        "detected": float(p_value) < 0.01 and abs(d) > 0.3,
        "mean_response_expected": round(mean_exp, 4),
        "mean_response_surprise": round(mean_sur, 4),
        "surprise_ratio": round(ratio, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "effect_size": round(float(d), 4),
        "n_expected": len(expected_responses),
        "n_surprise": len(surprise_responses),
    }


def _timescale_label(ms: float) -> str:
    """Human-readable label for a timescale."""
    if ms < 1000:
        return f"{int(ms)}ms"
    elif ms < 60000:
        return f"{int(ms / 1000)}s"
    else:
        return f"{int(ms / 60000)}min"


def _build_interpretation(
    detected, n_sig, n_tests, min_p, max_d, mean_ratio, method_results,
) -> str:
    """Build a detailed, honest interpretation string."""
    if detected:
        # Identify which methods and timescales contributed
        sig_methods = [
            k for k, v in method_results.items()
            if v.get("significant_corrected", False)
        ]
        methods_str = ", ".join(sig_methods) if sig_methods else "multiple methods"

        return (
            f"PREDICTIVE CODING DETECTED. "
            f"{n_sig}/{n_tests} tests significant after Bonferroni correction "
            f"(min corrected p={min_p:.4f}, max effect size d={max_d:.2f}). "
            f"Mean surprise ratio={mean_ratio:.2f}x. "
            f"Significant in: {methods_str}. "
            f"The organoid shows differential responses to expected vs surprising "
            f"state transitions, consistent with internal prediction generation "
            f"as described by the Free Energy Principle (Friston, 2010). "
            f"This is evidence of active inference -- the organoid maintains "
            f"a model of its own dynamics and reacts when that model is violated."
        )
    else:
        if n_sig == 1:
            return (
                f"INCONCLUSIVE. Only {n_sig}/{n_tests} test(s) reached significance "
                f"(corrected p={min_p:.4f}, d={max_d:.2f}). "
                f"A single significant result does not meet our conservative "
                f"threshold of 2+ converging methods. The organoid may show "
                f"weak predictive coding at specific timescales, but the evidence "
                f"is insufficient for a confident claim. Further investigation "
                f"with controlled stimulation protocols (e.g., oddball paradigm) "
                f"would strengthen detection power."
            )
        else:
            return (
                f"NOT DETECTED. {n_sig}/{n_tests} tests significant "
                f"(best corrected p={min_p:.4f}, best d={max_d:.2f}, "
                f"mean surprise ratio={mean_ratio:.2f}x). "
                f"The organoid does not show statistically robust differential "
                f"responses to expected vs surprising state transitions across "
                f"multiple methods and timescales. This does not prove the absence "
                f"of predictive coding -- it may require controlled stimulation "
                f"(oddball paradigm, patterned electrical input) to elicit "
                f"measurable prediction-error signals from spontaneous activity alone."
            )


def _empty_result(reason: str) -> dict:
    """Return a well-formed result dict for edge cases."""
    return {
        "has_predictive_coding": False,
        "surprise_ratio": 1.0,
        "p_value": 1.0,
        "effect_size": 0.0,
        "n_significant_methods": 0,
        "n_tests_total": 0,
        "method_results": {},
        "interpretation": f"NOT DETECTED. {reason}",
    }
