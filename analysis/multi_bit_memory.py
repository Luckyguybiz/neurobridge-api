"""Multi-bit memory capacity analysis for biological neural networks.

Current organoids reliably store ~1 bit of information (stimulated vs.
not stimulated). Multi-bit storage requires distinct population codes --
different input patterns must produce distinguishable neural responses.

We measure information capacity using Shannon's channel capacity:
C = max I(X; Y), where X = input patterns, Y = neural responses.
Mutual information I(X; Y) = H(Y) - H(Y|X), where H is entropy.

Population code diversity tells us how many distinct states the network
can represent. More unique states = higher potential capacity.
This is bounded by log2(n_states) bits.

For NeuroBridge, multi-bit memory is the difference between a novelty
and a useful computing substrate. Going from 1 bit to even 4 bits
would be a landmark result.

References:
- Shannon (1948) "A mathematical theory of communication"
  Bell System Technical Journal 27:379-423
- Quiroga & Panzeri (2009) "Extracting information from neuronal
  populations" Nature Reviews Neuroscience 10:173-185
- Tkacik et al. (2010) "Optimal population coding by noisy spiking
  neurons" PNAS 107(32):14419-14424
- Georgopoulos et al. (1986) "Neuronal population coding of movement
  direction" Science 233(4771):1416-1419
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def _bin_population_vectors(
    data: SpikeData,
    bin_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin spike data into population activity vectors.

    Returns:
        vectors: Array of shape (n_bins, n_electrodes) -- each row is
            a population state vector.
        bin_centers: Array of bin center times.
    """
    t_start, t_end = data.time_range
    bin_sec = bin_ms / 1000.0
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins = len(bins) - 1
    electrode_ids = data.electrode_ids
    n_electrodes = len(electrode_ids)

    vectors = np.zeros((n_bins, n_electrodes))
    for i, eid in enumerate(electrode_ids):
        mask = data.electrodes == eid
        counts, _ = np.histogram(data.times[mask], bins=bins)
        vectors[:, i] = counts

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return vectors, bin_centers


def _estimate_entropy(labels: np.ndarray) -> float:
    """Estimate Shannon entropy of a discrete label array.

    H(X) = -sum(p(x) * log2(p(x)))
    """
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    # Filter out zero probabilities
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


def _discretize_responses(
    vectors: np.ndarray,
    n_levels: int = 4,
) -> np.ndarray:
    """Discretize continuous population vectors into state labels.

    Each electrode's activity is binned into n_levels. The combined state
    across all electrodes forms a unique label.

    Returns:
        Array of integer state labels, one per time bin.
    """
    n_bins, n_electrodes = vectors.shape
    discrete = np.zeros((n_bins, n_electrodes), dtype=np.int32)

    for j in range(n_electrodes):
        col = vectors[:, j]
        if np.max(col) == np.min(col):
            discrete[:, j] = 0
        else:
            # Quantile-based binning for robustness to outliers
            percentiles = np.linspace(0, 100, n_levels + 1)[1:-1]
            thresholds = np.percentile(col, percentiles)
            discrete[:, j] = np.digitize(col, thresholds)

    # Combine electrode states into a single label per time bin
    # Use a unique hash for each combination
    labels = np.zeros(n_bins, dtype=np.int64)
    for j in range(n_electrodes):
        labels = labels * (n_levels + 1) + discrete[:, j]

    return labels


def estimate_channel_capacity(
    data: SpikeData,
    n_patterns: int = 4,
    bin_ms: float = 50.0,
    n_response_levels: int = 4,
) -> dict:
    """Estimate the channel capacity of the neural network.

    Treats different time windows as "input patterns" (X) and measures
    how distinguishable the neural responses (Y) are.

    Channel capacity C = max I(X; Y) where:
    I(X; Y) = H(Y) - H(Y|X)
    H(Y) = entropy of all response states
    H(Y|X) = average entropy of responses given each input pattern

    Args:
        data: Spike data to analyze.
        n_patterns: Number of time segments treated as distinct input patterns.
        bin_ms: Bin size for population vectors.
        n_response_levels: Number of discretization levels per electrode.

    Returns:
        Dict with keys:
            - channel_capacity_bits: Estimated channel capacity.
            - mutual_information_bits: Mutual information I(X; Y).
            - response_entropy_bits: H(Y) -- total response variability.
            - noise_entropy_bits: H(Y|X) -- response noise per pattern.
            - n_unique_states: Number of distinct population states observed.
            - theoretical_max_bits: log2(n_patterns) upper bound.
    """
    if data.n_spikes < 50:
        return {
            "error": "Not enough spikes for channel capacity estimation",
            "channel_capacity_bits": 0.0,
            "mutual_information_bits": 0.0,
        }

    vectors, bin_centers = _bin_population_vectors(data, bin_ms=bin_ms)
    n_bins, n_electrodes = vectors.shape
    seg_len = n_bins // n_patterns

    if seg_len < 5 or n_electrodes < 2:
        return {
            "error": "Recording too short or too few electrodes",
            "channel_capacity_bits": 0.0,
            "mutual_information_bits": 0.0,
        }

    # Discretize all response vectors
    all_labels = _discretize_responses(vectors, n_levels=n_response_levels)

    # Assign input pattern labels (which segment each bin belongs to)
    input_labels = np.zeros(n_bins, dtype=np.int32)
    for p in range(n_patterns):
        start = p * seg_len
        end = (p + 1) * seg_len if p < n_patterns - 1 else n_bins
        input_labels[start:end] = p

    # Only use bins that fall within defined segments
    valid_bins = n_patterns * seg_len
    input_labels = input_labels[:valid_bins]
    response_labels = all_labels[:valid_bins]

    # H(Y) -- marginal entropy of responses
    h_y = _estimate_entropy(response_labels)

    # H(Y|X) -- conditional entropy
    h_y_given_x = 0.0
    for p in range(n_patterns):
        mask = input_labels == p
        if np.sum(mask) < 2:
            continue
        h_y_given_xp = _estimate_entropy(response_labels[mask])
        h_y_given_x += h_y_given_xp * (np.sum(mask) / valid_bins)

    # I(X; Y) = H(Y) - H(Y|X)
    mutual_info = max(0.0, h_y - h_y_given_x)

    # Channel capacity is at most log2(n_patterns)
    theoretical_max = np.log2(n_patterns)
    channel_capacity = min(mutual_info, theoretical_max)

    n_unique = len(np.unique(response_labels))

    # Per-pattern analysis
    pattern_stats = []
    for p in range(n_patterns):
        mask = input_labels == p
        p_responses = response_labels[mask]
        p_unique = len(np.unique(p_responses))
        p_entropy = _estimate_entropy(p_responses)
        pattern_stats.append({
            "pattern_id": p,
            "n_bins": int(np.sum(mask)),
            "n_unique_states": p_unique,
            "response_entropy_bits": round(p_entropy, 4),
        })

    return {
        "channel_capacity_bits": round(channel_capacity, 4),
        "mutual_information_bits": round(mutual_info, 4),
        "response_entropy_bits": round(h_y, 4),
        "noise_entropy_bits": round(h_y_given_x, 4),
        "n_unique_states": n_unique,
        "theoretical_max_bits": round(theoretical_max, 4),
        "efficiency": round(channel_capacity / theoretical_max, 4) if theoretical_max > 0 else 0.0,
        "n_patterns": n_patterns,
        "n_electrodes": n_electrodes,
        "pattern_stats": pattern_stats,
        "interpretation": (
            f"HIGH CAPACITY -- {channel_capacity:.2f} bits (efficiency "
            f"{channel_capacity / theoretical_max:.0%} of {theoretical_max:.2f} bit max). "
            f"The network produces distinguishable responses to different input patterns. "
            f"{n_unique} unique population states observed."
            if channel_capacity > 1.0
            else f"MODERATE CAPACITY -- {channel_capacity:.2f} bits. "
            f"Some discrimination between patterns but significant overlap in responses. "
            f"H(Y)={h_y:.2f}, H(Y|X)={h_y_given_x:.2f}."
            if channel_capacity > 0.3
            else f"LOW CAPACITY -- {channel_capacity:.2f} bits. "
            f"Neural responses are not well differentiated across input patterns. "
            f"Essentially ~1 bit or less of usable information."
        ),
    }


def measure_population_code_diversity(
    data: SpikeData,
    bin_size_ms: float = 50.0,
    n_levels: int = 3,
    min_rate_threshold: float = 0.0,
) -> dict:
    """Measure the diversity of population codes in the neural network.

    How many distinct population states does the network visit?
    More states = higher potential information capacity.

    For each time bin, the activity across all electrodes forms a
    "population vector". We discretize these and count unique states.

    The theoretical maximum is n_levels^n_electrodes, but biological
    networks use only a fraction of the state space (sparse coding).

    Args:
        data: Spike data to analyze.
        bin_size_ms: Time bin for population vectors.
        n_levels: Discretization levels per electrode.
        min_rate_threshold: Minimum total rate to include a bin (filters silence).

    Returns:
        Dict with keys:
            - n_unique_states: Number of distinct population states.
            - theoretical_max_states: n_levels^n_electrodes.
            - state_space_usage: Fraction of theoretical space visited.
            - capacity_bits: log2(n_unique_states).
            - state_distribution: Histogram of state occupancies.
            - sparsity: Fraction of bins in the most common state.
    """
    if data.n_spikes < 20:
        return {
            "error": "Not enough spikes for population code analysis",
            "n_unique_states": 0,
            "capacity_bits": 0.0,
        }

    vectors, bin_centers = _bin_population_vectors(data, bin_size_ms=bin_size_ms)
    n_bins, n_electrodes = vectors.shape

    if n_bins < 5 or n_electrodes < 2:
        return {
            "error": "Too few bins or electrodes for diversity analysis",
            "n_unique_states": 0,
            "capacity_bits": 0.0,
        }

    # Filter out silent bins
    total_rate = np.sum(vectors, axis=1)
    active_mask = total_rate > min_rate_threshold
    active_vectors = vectors[active_mask]
    n_active_bins = active_vectors.shape[0]

    if n_active_bins < 5:
        return {
            "error": "Too few active bins after filtering",
            "n_unique_states": 0,
            "capacity_bits": 0.0,
        }

    # Discretize to state labels
    labels = _discretize_responses(active_vectors, n_levels=n_levels)

    # Count unique states
    unique_states, state_counts = np.unique(labels, return_counts=True)
    n_unique = len(unique_states)

    # Theoretical maximum (capped for display)
    theoretical_max = n_levels ** n_electrodes
    state_space_usage = n_unique / theoretical_max if theoretical_max > 0 else 0.0

    # Capacity in bits
    capacity_bits = np.log2(n_unique) if n_unique > 1 else 0.0

    # State distribution statistics
    sorted_counts = np.sort(state_counts)[::-1]
    most_common_count = int(sorted_counts[0])
    sparsity = most_common_count / n_active_bins

    # Entropy of state distribution (actual information content)
    state_entropy = _estimate_entropy(labels)

    # Temporal structure: do states repeat in patterns?
    # Measure autocorrelation of state sequence
    if n_active_bins > 10:
        state_ints = labels.astype(np.float64)
        state_mean = np.mean(state_ints)
        state_var = np.var(state_ints)
        if state_var > 1e-12:
            lag1_corr = float(np.mean(
                (state_ints[:-1] - state_mean) * (state_ints[1:] - state_mean)
            ) / state_var)
        else:
            lag1_corr = 0.0
    else:
        lag1_corr = 0.0

    # Top 10 most frequent states
    top_states = []
    for idx in np.argsort(state_counts)[::-1][:10]:
        top_states.append({
            "state_id": int(unique_states[idx]),
            "count": int(state_counts[idx]),
            "fraction": round(float(state_counts[idx]) / n_active_bins, 4),
        })

    return {
        "n_unique_states": n_unique,
        "theoretical_max_states": theoretical_max,
        "state_space_usage": round(state_space_usage, 6),
        "capacity_bits": round(float(capacity_bits), 4),
        "state_entropy_bits": round(state_entropy, 4),
        "theoretical_max_bits": round(float(np.log2(theoretical_max)), 4) if theoretical_max > 1 else 0.0,
        "n_active_bins": n_active_bins,
        "n_total_bins": n_bins,
        "sparsity": round(sparsity, 4),
        "temporal_autocorrelation": round(lag1_corr, 4),
        "n_electrodes": n_electrodes,
        "n_levels": n_levels,
        "top_states": top_states,
        "interpretation": (
            f"HIGH DIVERSITY -- {n_unique} unique states observed "
            f"({capacity_bits:.1f} bits capacity). "
            f"State entropy: {state_entropy:.2f} bits. "
            f"The network explores {state_space_usage:.2%} of theoretical state space "
            f"({theoretical_max} possible). Temporal autocorrelation: {lag1_corr:.3f}."
            if capacity_bits > 3.0
            else f"MODERATE DIVERSITY -- {n_unique} states ({capacity_bits:.1f} bits). "
            f"The network uses a limited but non-trivial repertoire of population codes. "
            f"Entropy: {state_entropy:.2f} bits."
            if capacity_bits > 1.5
            else f"LOW DIVERSITY -- only {n_unique} distinct states ({capacity_bits:.1f} bits). "
            f"The network has very limited representational capacity. "
            f"Most common state occupies {sparsity:.0%} of time. "
            f"Consistent with ~1 bit memory limitation."
        ),
    }
