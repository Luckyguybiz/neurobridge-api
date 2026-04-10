"""Memory consolidation analysis for biological neural networks.

Memory consolidation happens during "offline" periods -- when active
processing pauses and the network replays/rehearses recent patterns.

In mammalian hippocampus, Sharp-Wave Ripples (SWRs, 150-250 Hz)
during sleep/rest are the key mechanism: recently learned sequences
are replayed in compressed form, strengthening synaptic connections.

In organoids, we look for the functional analogue:
1. Periods of LOW overall activity (the "offline" state)
2. Brief HIGH-FREQUENCY bursts during these quiet periods
3. Burst content that resembles recent active-period patterns

If detected, these are "consolidation events" -- evidence that the
organoid has an endogenous memory consolidation mechanism.

This is directly relevant to the 45-minute memory barrier:
if consolidation events are rare or absent, memory fades.
If we can induce them (e.g., via electrical stimulation patterns
mimicking SWRs), we might extend organoid memory.

References:
- Buzsaki (2015) "Hippocampal sharp wave-ripple: A cognitive biomarker
  for episodic memory and planning" Hippocampus 25(10):1073-1188
- Diekelmann & Born (2010) "The memory function of sleep"
  Nature Reviews Neuroscience 11:114-126
- Girardeau et al. (2009) "Selective suppression of hippocampal ripples
  impairs spatial memory" Nature Neuroscience 12:1222-1223
"""

import numpy as np
from scipy.signal import hilbert
from typing import Optional
from .loader import SpikeData


def _compute_population_rate(
    data: SpikeData,
    bin_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population firing rate over time.

    Returns:
        rate: Population rate per bin (spikes/bin across all electrodes).
        bin_centers: Center times of each bin.
    """
    t_start, t_end = data.time_range
    bin_sec = bin_ms / 1000.0
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    counts, _ = np.histogram(data.times, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return counts.astype(np.float64), bin_centers


def _build_activity_fingerprint(
    data: SpikeData,
    bin_ms: float = 10.0,
) -> np.ndarray:
    """Build a normalized activity fingerprint for a data segment.

    Returns a vector of per-electrode firing rates, normalized to unit length.
    """
    electrode_ids = data.electrode_ids
    n = len(electrode_ids)
    duration = data.duration

    if duration < 1e-6 or n == 0:
        return np.zeros(max(n, 1))

    rates = np.zeros(n)
    for i, eid in enumerate(electrode_ids):
        mask = data.electrodes == eid
        rates[i] = float(np.sum(mask)) / duration

    norm = np.linalg.norm(rates)
    if norm > 0:
        rates /= norm
    return rates


def detect_consolidation_events(
    data: SpikeData,
    quiet_percentile: float = 25.0,
    burst_percentile: float = 90.0,
    min_quiet_duration_ms: float = 200.0,
    min_burst_spikes: int = 5,
    similarity_threshold: float = 0.3,
    bin_ms: float = 10.0,
) -> dict:
    """Detect memory consolidation events in spike data.

    A consolidation event is defined as:
    1. A quiet period where population rate is below the quiet_percentile
       threshold for at least min_quiet_duration_ms.
    2. Within or immediately after the quiet period, a brief burst where
       rate exceeds the burst_percentile threshold.
    3. The burst activity pattern (per-electrode rates) is similar to
       a recent active-period pattern (cosine similarity > threshold).

    Args:
        data: Spike data to analyze.
        quiet_percentile: Percentile below which activity is "quiet".
        burst_percentile: Percentile above which activity is a "burst".
        min_quiet_duration_ms: Minimum quiet period before a valid burst.
        min_burst_spikes: Minimum spikes in a burst to be considered.
        similarity_threshold: Minimum cosine similarity to active template.
        bin_ms: Bin size for rate computation.

    Returns:
        Dict with keys:
            - events: List of consolidation events with times, similarity scores.
            - n_events: Number of detected events.
            - quiet_periods: Number of qualifying quiet periods found.
            - has_consolidation: True if any events detected.
            - mean_similarity: Average similarity of burst to active template.
    """
    if data.n_spikes < 50:
        return {
            "error": "Not enough spikes for consolidation analysis",
            "events": [],
            "n_events": 0,
            "has_consolidation": False,
        }

    rate, bin_centers = _compute_population_rate(data, bin_ms=bin_ms)
    bin_sec = bin_ms / 1000.0
    n_bins = len(rate)

    if n_bins < 10:
        return {
            "error": "Recording too short for consolidation analysis",
            "events": [],
            "n_events": 0,
            "has_consolidation": False,
        }

    quiet_thresh = np.percentile(rate, quiet_percentile)
    burst_thresh = np.percentile(rate, burst_percentile)
    min_quiet_bins = max(1, int(min_quiet_duration_ms / bin_ms))

    # Build activity template from high-activity periods (top 25%)
    active_thresh = np.percentile(rate, 75)
    active_mask = rate > active_thresh
    # Collect active period data for template
    active_times = bin_centers[active_mask]
    if len(active_times) < 3:
        return {
            "events": [],
            "n_events": 0,
            "quiet_periods": 0,
            "has_consolidation": False,
            "interpretation": "No active periods found to build consolidation template.",
        }

    # Build template from the most recent active period
    # Find contiguous active segments
    active_segments = []
    seg_start = None
    for i in range(n_bins):
        if active_mask[i]:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                active_segments.append((seg_start, i))
                seg_start = None
    if seg_start is not None:
        active_segments.append((seg_start, n_bins))

    # Build fingerprint from each active segment
    active_fingerprints = []
    t_start_data, _ = data.time_range
    for seg_s, seg_e in active_segments:
        seg_data = data.get_time_range(
            float(bin_centers[seg_s] - bin_sec / 2),
            float(bin_centers[min(seg_e, n_bins - 1)] + bin_sec / 2),
        )
        if seg_data.n_spikes >= min_burst_spikes:
            fp = _build_activity_fingerprint(seg_data, bin_ms=bin_ms)
            active_fingerprints.append({
                "fingerprint": fp,
                "start_time": float(bin_centers[seg_s]),
                "end_time": float(bin_centers[min(seg_e, n_bins - 1)]),
            })

    if not active_fingerprints:
        return {
            "events": [],
            "n_events": 0,
            "quiet_periods": 0,
            "has_consolidation": False,
            "interpretation": "Active segments too sparse for fingerprinting.",
        }

    # Find quiet periods followed by bursts
    events = []
    quiet_periods_found = 0
    i = 0

    while i < n_bins:
        # Look for quiet period
        if rate[i] <= quiet_thresh:
            quiet_start = i
            while i < n_bins and rate[i] <= quiet_thresh:
                i += 1
            quiet_end = i
            quiet_length = quiet_end - quiet_start

            if quiet_length >= min_quiet_bins:
                quiet_periods_found += 1

                # Look for burst within next few bins after quiet
                search_end = min(quiet_end + int(100.0 / bin_ms), n_bins)
                for j in range(quiet_end, search_end):
                    if rate[j] > burst_thresh:
                        # Found a burst -- extract its pattern
                        burst_start = j
                        while j < n_bins and rate[j] > quiet_thresh:
                            j += 1
                        burst_end = j

                        burst_data = data.get_time_range(
                            float(bin_centers[burst_start] - bin_sec / 2),
                            float(bin_centers[min(burst_end, n_bins - 1)] + bin_sec / 2),
                        )

                        if burst_data.n_spikes < min_burst_spikes:
                            break

                        burst_fp = _build_activity_fingerprint(burst_data, bin_ms=bin_ms)

                        # Compare to active-period templates
                        best_sim = 0.0
                        best_template_idx = -1
                        for t_idx, tmpl in enumerate(active_fingerprints):
                            min_len = min(len(burst_fp), len(tmpl["fingerprint"]))
                            if min_len < 1:
                                continue
                            bp = burst_fp[:min_len]
                            tp = tmpl["fingerprint"][:min_len]
                            nb = np.linalg.norm(bp)
                            nt = np.linalg.norm(tp)
                            if nb > 0 and nt > 0:
                                sim = float(np.dot(bp, tp) / (nb * nt))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_template_idx = t_idx

                        if best_sim >= similarity_threshold:
                            events.append({
                                "quiet_start": round(float(bin_centers[quiet_start]), 3),
                                "quiet_end": round(float(bin_centers[min(quiet_end, n_bins - 1)]), 3),
                                "quiet_duration_ms": round(quiet_length * bin_ms, 1),
                                "burst_start": round(float(bin_centers[burst_start]), 3),
                                "burst_end": round(float(bin_centers[min(burst_end, n_bins - 1)]), 3),
                                "burst_spikes": burst_data.n_spikes,
                                "similarity_to_active": round(best_sim, 4),
                                "matched_template_idx": best_template_idx,
                                "matched_template_time": round(
                                    active_fingerprints[best_template_idx]["start_time"], 3
                                ) if best_template_idx >= 0 else None,
                            })
                        break  # One burst per quiet period
        else:
            i += 1

    mean_similarity = float(np.mean([e["similarity_to_active"] for e in events])) if events else 0.0

    return {
        "events": events,
        "n_events": len(events),
        "quiet_periods": quiet_periods_found,
        "n_active_templates": len(active_fingerprints),
        "has_consolidation": len(events) > 0,
        "mean_similarity": round(mean_similarity, 4),
        "quiet_threshold_rate": round(float(quiet_thresh), 2),
        "burst_threshold_rate": round(float(burst_thresh), 2),
        "interpretation": (
            f"CONSOLIDATION DETECTED -- {len(events)} events found across "
            f"{quiet_periods_found} quiet periods. Mean similarity to active "
            f"templates: {mean_similarity:.3f}. These bursts during quiet periods "
            f"resemble recent active patterns -- consistent with memory replay "
            f"and consolidation."
            if len(events) > 2
            else f"WEAK CONSOLIDATION -- {len(events)} event(s) found. "
            f"Some evidence of replay-like bursts during quiet periods, "
            f"but not enough for strong conclusion."
            if len(events) > 0
            else f"NO CONSOLIDATION DETECTED -- {quiet_periods_found} quiet periods "
            f"found but no qualifying replay bursts. The organoid may lack "
            f"endogenous consolidation mechanisms, or stimulation protocol "
            f"may need adjustment."
        ),
    }


def measure_retention(
    data: SpikeData,
    train_window: tuple[float, float],
    test_window: tuple[float, float],
    bin_ms: float = 10.0,
) -> dict:
    """Compare activity patterns between a training window and a later test window.

    Measures how much the neural activity fingerprint from the training
    period is retained in the test period. Uses both cosine similarity
    of firing rate profiles and correlation of cross-electrode patterns.

    Args:
        data: Spike data to analyze.
        train_window: (start, end) in seconds for the training/learning period.
        test_window: (start, end) in seconds for the retention test period.
        bin_ms: Bin size for population rate computation.

    Returns:
        Dict with keys:
            - rate_similarity: Cosine similarity of firing rate profiles.
            - pattern_correlation: Pearson correlation of electrode pair
              activity ratios.
            - retention_score: Combined retention metric (0-1).
            - time_gap_sec: Time between end of train and start of test.
    """
    train_data = data.get_time_range(train_window[0], train_window[1])
    test_data = data.get_time_range(test_window[0], test_window[1])

    if train_data.n_spikes < 10 or test_data.n_spikes < 10:
        return {
            "error": "Not enough spikes in one or both windows",
            "rate_similarity": 0.0,
            "pattern_correlation": 0.0,
            "retention_score": 0.0,
        }

    # 1. Firing rate profile similarity
    train_fp = _build_activity_fingerprint(train_data, bin_ms=bin_ms)
    test_fp = _build_activity_fingerprint(test_data, bin_ms=bin_ms)

    min_len = min(len(train_fp), len(test_fp))
    if min_len < 2:
        return {
            "error": "Too few common electrodes for comparison",
            "rate_similarity": 0.0,
            "pattern_correlation": 0.0,
            "retention_score": 0.0,
        }

    train_fp = train_fp[:min_len]
    test_fp = test_fp[:min_len]
    nt = np.linalg.norm(train_fp)
    ne = np.linalg.norm(test_fp)

    rate_sim = float(np.dot(train_fp, test_fp) / (nt * ne)) if nt > 0 and ne > 0 else 0.0

    # 2. Cross-electrode pattern correlation
    # Compare pairwise activity ratios
    electrode_ids = sorted(set(train_data.electrode_ids) & set(test_data.electrode_ids))
    n_common = len(electrode_ids)

    if n_common < 2:
        pattern_corr = rate_sim  # Fall back to rate similarity
    else:
        train_rates = np.zeros(n_common)
        test_rates = np.zeros(n_common)
        for i, eid in enumerate(electrode_ids):
            train_mask = train_data.electrodes == eid
            test_mask = test_data.electrodes == eid
            train_rates[i] = float(np.sum(train_mask))
            test_rates[i] = float(np.sum(test_mask))

        if np.std(train_rates) > 1e-12 and np.std(test_rates) > 1e-12:
            r, _ = pearsonr(train_rates, test_rates)
            pattern_corr = float(r) if not np.isnan(r) else 0.0
        else:
            pattern_corr = 0.0

    # Combined retention score
    retention_score = 0.5 * max(0, rate_sim) + 0.5 * max(0, pattern_corr)
    time_gap = test_window[0] - train_window[1]

    return {
        "rate_similarity": round(rate_sim, 4),
        "pattern_correlation": round(pattern_corr, 4),
        "retention_score": round(retention_score, 4),
        "time_gap_sec": round(time_gap, 3),
        "train_window": (round(train_window[0], 3), round(train_window[1], 3)),
        "test_window": (round(test_window[0], 3), round(test_window[1], 3)),
        "train_spikes": train_data.n_spikes,
        "test_spikes": test_data.n_spikes,
        "n_common_electrodes": n_common if n_common >= 2 else min_len,
        "interpretation": (
            f"STRONG RETENTION -- score {retention_score:.3f}. "
            f"Activity patterns from the training window are well preserved "
            f"in the test window ({time_gap:.1f}s later). "
            f"Rate similarity: {rate_sim:.3f}, pattern correlation: {pattern_corr:.3f}."
            if retention_score > 0.6
            else f"PARTIAL RETENTION -- score {retention_score:.3f}. "
            f"Some features of the training pattern survive after {time_gap:.1f}s, "
            f"but with degradation."
            if retention_score > 0.3
            else f"WEAK RETENTION -- score {retention_score:.3f}. "
            f"The training pattern is largely lost after {time_gap:.1f}s. "
            f"Consistent with the ~45 minute memory barrier."
        ),
    }
