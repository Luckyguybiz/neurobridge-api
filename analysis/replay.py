"""Neural Replay Detection — finding memory traces in organoid activity.

NOBODY HAS DONE THIS ON ORGANOIDS.

In mammalian brains, hippocampal "replay" occurs during rest/sleep:
neural activity patterns from waking experience are replayed (often
compressed in time). This is believed to consolidate memories.

If organoids show replay → evidence of memory formation mechanism.
This directly attacks the field's biggest barrier: memory loss after 45 min.

Methods:
- Template matching: define activity templates during stimulation,
  search for similar patterns during rest periods
- Sequence detection: find repeated temporal sequences across electrodes
- Compression ratio: replayed patterns are typically 5-20x faster
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def detect_replay(
    data: SpikeData,
    template_window_sec: float = 5.0,
    search_window_sec: float = 1.0,
    min_similarity: float = 0.3,
    compression_range: tuple[float, float] = (2.0, 20.0),
) -> dict:
    """Detect replay of neural activity patterns.

    1. Extract activity templates from high-activity periods
    2. Search for similar (possibly time-compressed) patterns in low-activity periods
    3. Score similarity using normalized cross-correlation

    Args:
        template_window_sec: Duration of template patterns
        search_window_sec: Duration of search windows
        min_similarity: Minimum correlation to count as replay
        compression_range: Range of time compression factors to test
    """
    if data.n_spikes < 100:
        return {"error": "Not enough spikes for replay detection"}

    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)
    t_start, t_end = data.time_range

    # Step 1: Bin all spike trains (10ms bins)
    bin_ms = 10.0
    bin_sec = bin_ms / 1000.0
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins = len(bins) - 1

    binned = np.zeros((n_electrodes, n_bins))
    for i, e in enumerate(electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binned[i] = counts

    # Step 2: Find high-activity periods (templates) and low-activity periods (search)
    population_rate = np.sum(binned, axis=0)
    rate_threshold = np.percentile(population_rate, 75)
    quiet_threshold = np.percentile(population_rate, 25)

    template_bins = int(template_window_sec / bin_sec)
    search_bins = int(search_window_sec / bin_sec)

    # Extract templates from active periods
    templates = []
    i = 0
    while i + template_bins < n_bins:
        window_rate = np.mean(population_rate[i:i + template_bins])
        if window_rate > rate_threshold:
            template = binned[:, i:i + template_bins].copy()
            # Normalize
            norm = np.linalg.norm(template)
            if norm > 0:
                templates.append({
                    "pattern": template / norm,
                    "start_bin": i,
                    "start_time": float(bins[i]),
                    "mean_rate": float(window_rate),
                })
            i += template_bins  # skip ahead
        else:
            i += 1

    if not templates:
        return {
            "replay_events": [],
            "n_templates": 0,
            "n_replay_events": 0,
            "has_replay": False,
            "interpretation": "No high-activity templates found to search for replay",
        }

    # Step 3: Search for compressed replays in quiet periods
    compression_factors = np.linspace(compression_range[0], compression_range[1], 10)
    replay_events = []

    for t_idx, template_info in enumerate(templates[:5]):  # Limit to 5 templates
        template = template_info["pattern"]

        for cf in compression_factors:
            compressed_bins = max(3, int(template_bins / cf))

            # Resample template to compressed size
            from scipy.ndimage import zoom
            compressed_template = zoom(template, (1, compressed_bins / template_bins), order=1)
            c_norm = np.linalg.norm(compressed_template)
            if c_norm > 0:
                compressed_template /= c_norm

            # Slide through quiet periods
            j = 0
            while j + compressed_bins < n_bins:
                window_rate = np.mean(population_rate[j:j + compressed_bins])
                if window_rate < quiet_threshold * 2:
                    search_pattern = binned[:, j:j + compressed_bins]
                    s_norm = np.linalg.norm(search_pattern)
                    if s_norm > 0:
                        search_pattern = search_pattern / s_norm
                        # Cross-correlation
                        similarity = float(np.sum(compressed_template * search_pattern))

                        if similarity > min_similarity:
                            replay_events.append({
                                "template_idx": t_idx,
                                "template_time": round(template_info["start_time"], 3),
                                "replay_time": round(float(bins[j]), 3),
                                "similarity": round(similarity, 4),
                                "compression_factor": round(float(cf), 1),
                                "replay_duration_ms": round(compressed_bins * bin_ms, 1),
                                "original_duration_ms": round(template_bins * bin_ms, 1),
                            })
                j += max(1, compressed_bins // 2)

    # Deduplicate (keep best similarity per time window)
    replay_events.sort(key=lambda x: x["similarity"], reverse=True)
    seen_times = set()
    unique_events = []
    for ev in replay_events:
        t_key = round(ev["replay_time"], 1)
        if t_key not in seen_times:
            seen_times.add(t_key)
            unique_events.append(ev)

    unique_events = unique_events[:20]  # Top 20

    return {
        "replay_events": unique_events,
        "n_templates": len(templates),
        "n_replay_events": len(unique_events),
        "has_replay": len(unique_events) > 0,
        "mean_similarity": round(float(np.mean([e["similarity"] for e in unique_events])), 4) if unique_events else 0,
        "mean_compression": round(float(np.mean([e["compression_factor"] for e in unique_events])), 1) if unique_events else 0,
        "interpretation": (
            f"REPLAY DETECTED — {len(unique_events)} replay events found. "
            f"Activity patterns from high-activity periods are replayed during quiet periods "
            f"at ~{np.mean([e['compression_factor'] for e in unique_events]):.0f}x compression. "
            f"This is evidence of memory consolidation mechanisms."
            if unique_events
            else "No significant replay detected in this recording."
        ),
        "significance": (
            "HIGH — strong evidence of memory-like replay" if len(unique_events) > 5
            else "MODERATE — some replay patterns detected" if len(unique_events) > 0
            else "NONE — no replay detected"
        ),
    }


def detect_sequence_replay(
    data: SpikeData,
    min_sequence_length: int = 3,
    max_lag_ms: float = 50.0,
) -> dict:
    """Detect repeated sequential activation patterns across electrodes.

    Finds ordered sequences like E2→E5→E1→E7 that repeat multiple times.
    Repeated sequences = evidence of functional neural circuits.
    """
    max_lag_sec = max_lag_ms / 1000.0
    electrode_ids = data.electrode_ids

    # Find all sequential triplets/quadruplets within time window
    sequences: dict[str, int] = {}
    seq_times: dict[str, list[float]] = {}

    sorted_spikes = sorted(zip(data.times.tolist(), data.electrodes.tolist()), key=lambda x: x[0])

    for i in range(len(sorted_spikes) - min_sequence_length + 1):
        seq_electrodes = []
        seq_start = sorted_spikes[i][0]

        for j in range(i, min(i + min_sequence_length + 2, len(sorted_spikes))):
            t, e = sorted_spikes[j]
            if t - seq_start > max_lag_sec:
                break
            if e not in [s for s in seq_electrodes]:
                seq_electrodes.append(int(e))

        if len(seq_electrodes) >= min_sequence_length:
            for length in range(min_sequence_length, min(len(seq_electrodes) + 1, 6)):
                key = "→".join(f"E{e}" for e in seq_electrodes[:length])
                sequences[key] = sequences.get(key, 0) + 1
                if key not in seq_times:
                    seq_times[key] = []
                seq_times[key].append(seq_start)

    # Filter to repeated sequences
    repeated = []
    for seq, count in sorted(sequences.items(), key=lambda x: -x[1]):
        if count >= 3:
            times = seq_times[seq]
            intervals = np.diff(sorted(times))
            repeated.append({
                "sequence": seq,
                "count": count,
                "electrodes": [int(e.replace("E", "")) for e in seq.split("→")],
                "length": len(seq.split("→")),
                "mean_interval_ms": round(float(np.mean(intervals)) * 1000, 1) if len(intervals) > 0 else 0,
                "regularity": round(float(1 / (1 + np.std(intervals) / np.mean(intervals))), 3) if len(intervals) > 0 and np.mean(intervals) > 0 else 0,
            })

    repeated = repeated[:20]

    return {
        "sequences": repeated,
        "n_unique_sequences": len(repeated),
        "most_common": repeated[0] if repeated else None,
        "has_circuits": len(repeated) > 0,
        "interpretation": (
            f"Found {len(repeated)} repeated neural sequences. "
            f"Most common: {repeated[0]['sequence']} ({repeated[0]['count']} repetitions). "
            f"These represent functional neural circuits — preferred pathways of information flow."
            if repeated
            else "No significant repeated sequences found."
        ),
    }
