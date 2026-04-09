"""Multi-Timescale Complexity — information at different temporal resolutions.

NOVEL: Nobody has done systematic multi-scale analysis on organoid data.

Different computations operate at different timescales:
- 1ms: individual spikes, synaptic transmission
- 10ms: spike patterns, STDP window
- 100ms: bursts, short-term memory
- 1s: behavioral responses, decision-making timescale
- 10s: state transitions, adaptation
- 60s: long-term trends, circadian-like rhythms

By measuring information content at each scale, we find the
organoid's "operating frequency" — the timescale at which it
does the most computation.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def compute_multiscale_complexity(
    data: SpikeData,
    scales_ms: Optional[list[float]] = None,
) -> dict:
    """Compute information-theoretic complexity at multiple timescales.

    For each timescale:
    - Shannon entropy (information content)
    - Lempel-Ziv complexity (algorithmic complexity)
    - Mutual information between consecutive windows
    - Active information storage (how much past predicts future)

    Returns "complexity spectrum" — complexity as function of timescale.
    """
    if scales_ms is None:
        scales_ms = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    t_start, t_end = data.time_range
    electrode_ids = data.electrode_ids
    n_el = len(electrode_ids)

    scale_results = []

    for scale_ms in scales_ms:
        scale_sec = scale_ms / 1000.0
        bins = np.arange(t_start, t_end + scale_sec, scale_sec)
        n_bins = len(bins) - 1

        if n_bins < 10:
            continue

        # Population binary state per bin
        pop_counts, _ = np.histogram(data.times, bins=bins)
        binary = (pop_counts > 0).astype(int)

        # Per-electrode binary
        electrode_binary = np.zeros((n_el, n_bins), dtype=int)
        for i, e in enumerate(electrode_ids):
            counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
            electrode_binary[i] = (counts > 0).astype(int)

        # 1. Shannon entropy of population state
        p1 = np.mean(binary)
        p0 = 1 - p1
        if p0 > 0 and p1 > 0:
            entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            entropy = 0

        # 2. Joint entropy (multi-electrode state)
        from collections import Counter
        state_strings = [''.join(str(b) for b in electrode_binary[:, t]) for t in range(n_bins)]
        state_counts = Counter(state_strings)
        total = sum(state_counts.values())
        joint_entropy = -sum((c / total) * np.log2(c / total) for c in state_counts.values() if c > 0)
        max_entropy = np.log2(min(2 ** n_el, n_bins))
        normalized_joint = joint_entropy / max_entropy if max_entropy > 0 else 0

        # 3. LZ complexity
        s = ''.join(str(b) for b in binary)
        lz = _lz76(s)
        n = len(s)
        lz_normalized = (lz * np.log2(n)) / n if n > 1 else 0

        # 4. Active Information Storage (AIS)
        # How much past predicts future at this timescale
        if n_bins >= 4:
            from collections import Counter as C2
            past_future = C2(zip(state_strings[:-1], state_strings[1:]))
            past_counts = C2(state_strings[:-1])
            future_counts = C2(state_strings[1:])
            total_pf = n_bins - 1

            ais = 0
            for (p, f), count in past_future.items():
                p_pf = count / total_pf
                p_p = past_counts[p] / total_pf
                p_f = future_counts[f] / total_pf
                if p_pf > 0 and p_p > 0 and p_f > 0:
                    ais += p_pf * np.log2(p_pf / (p_p * p_f))
            ais = max(0, ais)
        else:
            ais = 0

        scale_results.append({
            "scale_ms": scale_ms,
            "n_bins": n_bins,
            "entropy": round(float(entropy), 4),
            "joint_entropy": round(float(joint_entropy), 4),
            "normalized_joint_entropy": round(float(normalized_joint), 4),
            "lz_complexity": round(float(lz_normalized), 4),
            "active_information_storage": round(float(ais), 5),
            "active_fraction": round(float(p1), 4),
            "n_unique_states": len(state_counts),
        })

    # Find optimal timescale (maximum complexity)
    if scale_results:
        complexities = [r["lz_complexity"] for r in scale_results]
        optimal_idx = int(np.argmax(complexities))
        optimal_scale = scale_results[optimal_idx]["scale_ms"]

        # Find scale with maximum AIS (best self-prediction)
        ais_values = [r["active_information_storage"] for r in scale_results]
        max_ais_idx = int(np.argmax(ais_values))
        max_ais_scale = scale_results[max_ais_idx]["scale_ms"]

        # Complexity slope: how does complexity change with scale?
        if len(complexities) >= 3:
            log_scales = np.log10([r["scale_ms"] for r in scale_results])
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(log_scales, complexities)
        else:
            slope = 0
    else:
        optimal_scale = 0
        max_ais_scale = 0
        slope = 0

    return {
        "scales": scale_results,
        "optimal_computation_scale_ms": optimal_scale,
        "max_ais_scale_ms": max_ais_scale,
        "complexity_slope": round(float(slope), 4),
        "n_scales_analyzed": len(scale_results),
        "interpretation": (
            f"Peak computational complexity at {optimal_scale:.0f}ms timescale. "
            f"Maximum self-prediction (AIS) at {max_ais_scale:.0f}ms. "
            + (
                f"Complexity increases with scale (slope={slope:.3f}) — long-range temporal structure."
                if slope > 0.05
                else f"Complexity decreases with scale (slope={slope:.3f}) — activity is locally complex but globally simple."
                if slope < -0.05
                else "Relatively flat complexity spectrum — uniform information across scales."
            )
        ),
        "operating_frequency": f"{1000 / optimal_scale:.1f} Hz" if optimal_scale > 0 else "unknown",
    }


def _lz76(s: str) -> int:
    """Lempel-Ziv 1976 complexity."""
    n = len(s)
    if n == 0:
        return 0
    c, i, k, kmax = 1, 0, 1, 1
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
