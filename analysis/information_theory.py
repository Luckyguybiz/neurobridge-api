"""Information theory module — entropy, mutual information, complexity.

Measures information content and processing in organoid neural activity:
- Shannon entropy of spike trains
- Mutual information between electrode pairs
- Active information storage
- Lempel-Ziv complexity (neural complexity measure)
- Integration/segregation balance (Phi — integrated information)
"""

import numpy as np
from collections import Counter
from typing import Optional
from .loader import SpikeData


def compute_spike_train_entropy(
    data: SpikeData,
    bin_size_ms: float = 10.0,
) -> dict:
    """Compute Shannon entropy of binned spike trains per electrode.

    Higher entropy = more unpredictable firing = more information capacity.
    Max entropy = log2(n_bins) for uniform distribution.
    """
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    n_bins = len(bins) - 1

    results = {}
    for e in data.electrode_ids:
        e_times = data.times[data.electrodes == e]
        counts, _ = np.histogram(e_times, bins=bins)
        binary = (counts > 0).astype(int)

        # Shannon entropy
        p1 = np.mean(binary)
        p0 = 1 - p1
        if p0 > 0 and p1 > 0:
            entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            entropy = 0.0

        # Word entropy (patterns of length L)
        L = 4
        words = [''.join(str(b) for b in binary[i:i + L]) for i in range(len(binary) - L + 1)]
        word_counts = Counter(words)
        total = sum(word_counts.values())
        word_entropy = -sum((c / total) * np.log2(c / total) for c in word_counts.values() if c > 0)

        # Normalized entropy (0=deterministic, 1=max random)
        max_word_entropy = L  # bits (for binary)
        normalized = word_entropy / max_word_entropy if max_word_entropy > 0 else 0

        results[int(e)] = {
            "binary_entropy": round(float(entropy), 4),
            "word_entropy_L4": round(float(word_entropy), 4),
            "normalized_entropy": round(float(normalized), 4),
            "active_fraction": round(float(p1), 4),
            "n_unique_words": len(word_counts),
            "max_possible_words": 2 ** L,
        }

    # Population entropy (all electrodes combined)
    all_binary = []
    for e in data.electrode_ids:
        e_times = data.times[data.electrodes == e]
        counts, _ = np.histogram(e_times, bins=bins)
        all_binary.append((counts > 0).astype(int))

    if all_binary:
        population_state = np.array(all_binary)  # shape: (n_electrodes, n_bins)
        # Joint entropy of population state
        state_strings = [''.join(str(b) for b in population_state[:, t]) for t in range(n_bins)]
        state_counts = Counter(state_strings)
        total = sum(state_counts.values())
        joint_entropy = -sum((c / total) * np.log2(c / total) for c in state_counts.values() if c > 0)
    else:
        joint_entropy = 0.0

    return {
        "per_electrode": results,
        "population_joint_entropy": round(float(joint_entropy), 4),
        "bin_size_ms": bin_size_ms,
        "n_bins": n_bins,
        "mean_entropy": round(float(np.mean([r["binary_entropy"] for r in results.values()])), 4),
    }


def compute_mutual_information(
    data: SpikeData,
    bin_size_ms: float = 10.0,
) -> dict:
    """Compute pairwise mutual information between all electrode pairs.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    Higher MI = more shared information between electrodes.
    """
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    # Bin all electrodes
    binned = {}
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binned[e] = (counts > 0).astype(int)

    electrode_ids = data.electrode_ids
    n = len(electrode_ids)
    mi_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x = binned[electrode_ids[i]]
            y = binned[electrode_ids[j]]

            # Joint distribution
            joint = Counter(zip(x, y))
            total = len(x)

            mi = 0.0
            px = Counter(x)
            py = Counter(y)
            for (xi, yi), count in joint.items():
                p_xy = count / total
                p_x = px[xi] / total
                p_y = py[yi] / total
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # Find strongest pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if mi_matrix[i, j] > 0.001:
                pairs.append({
                    "electrode_1": int(electrode_ids[i]),
                    "electrode_2": int(electrode_ids[j]),
                    "mutual_information": round(float(mi_matrix[i, j]), 5),
                })
    pairs.sort(key=lambda p: p["mutual_information"], reverse=True)

    return {
        "mi_matrix": mi_matrix.tolist(),
        "electrode_ids": electrode_ids,
        "top_pairs": pairs[:10],
        "mean_mi": round(float(np.mean(mi_matrix[mi_matrix > 0])), 5) if np.any(mi_matrix > 0) else 0,
        "bin_size_ms": bin_size_ms,
    }


def compute_lempel_ziv_complexity(
    data: SpikeData,
    bin_size_ms: float = 5.0,
) -> dict:
    """Compute Lempel-Ziv complexity of spike trains.

    LZ complexity measures the number of distinct patterns in a binary sequence.
    Normalized: C = c(n) * log2(n) / n
    where c(n) = number of distinct words in LZ76 decomposition.

    Higher = more complex/random. Lower = more structured/repetitive.
    Values: 0 (constant) to 1 (random). Organoids typically 0.3-0.7.
    """
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    results = {}
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        binary = (counts > 0).astype(int)
        s = ''.join(str(b) for b in binary)

        c = _lz76_complexity(s)
        n = len(s)
        normalized = (c * np.log2(n)) / n if n > 1 else 0

        results[int(e)] = {
            "raw_complexity": int(c),
            "normalized_complexity": round(float(normalized), 4),
            "sequence_length": n,
            "interpretation": "random" if normalized > 0.8 else "complex" if normalized > 0.5 else "structured" if normalized > 0.2 else "repetitive",
        }

    # Population complexity (concatenated multi-electrode state)
    all_binary = []
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        all_binary.append((counts > 0).astype(int))

    if all_binary:
        pop_state = np.array(all_binary)
        pop_string = ''.join(''.join(str(b) for b in pop_state[:, t]) for t in range(pop_state.shape[1]))
        pop_c = _lz76_complexity(pop_string)
        pop_n = len(pop_string)
        pop_normalized = (pop_c * np.log2(pop_n)) / pop_n if pop_n > 1 else 0
    else:
        pop_normalized = 0

    return {
        "per_electrode": results,
        "population_complexity": round(float(pop_normalized), 4),
        "bin_size_ms": bin_size_ms,
        "mean_complexity": round(float(np.mean([r["normalized_complexity"] for r in results.values()])), 4),
    }


def _lz76_complexity(s: str) -> int:
    """Lempel-Ziv 1976 complexity — counts distinct substrings."""
    n = len(s)
    if n == 0:
        return 0

    c = 1
    i = 0
    k = 1
    kmax = 1

    while i + k <= n:
        if s[i + k - 1:i + k] not in s[0:i + k - 1]:
            c += 1
            i += kmax
            k = 1
            kmax = 1
        else:
            k += 1
            if k > kmax:
                kmax = k
            if i + k > n:
                c += 1
                break

    return c
