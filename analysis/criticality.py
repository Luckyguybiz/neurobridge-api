"""Criticality analysis module — neuronal avalanches and power laws.

Tests whether organoid operates near criticality (edge of chaos).
Critical systems maximize information processing, storage, and transfer.

- Neuronal avalanche detection
- Size/duration power law fits
- Branching ratio estimation
- Deviation from criticality (DCC)
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional
from .loader import SpikeData


def detect_avalanches(
    data: SpikeData,
    bin_size_ms: float = 5.0,
    threshold: float = 0.0,
) -> dict:
    """Detect neuronal avalanches — cascades of neural activity.

    Avalanche = sequence of consecutive time bins with activity > threshold.
    At criticality, avalanche sizes follow a power law: P(s) ~ s^(-1.5)
    """
    bin_size_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    # Population activity per bin
    counts, _ = np.histogram(data.times, bins=bins)

    # Detect avalanches (consecutive bins above threshold)
    above = counts > threshold
    avalanches = []
    in_avalanche = False
    start_bin = 0
    size = 0

    for i, active in enumerate(above):
        if active and not in_avalanche:
            in_avalanche = True
            start_bin = i
            size = int(counts[i])
        elif active and in_avalanche:
            size += int(counts[i])
        elif not active and in_avalanche:
            duration = i - start_bin
            avalanches.append({
                "start_bin": int(start_bin),
                "start_time": round(float(bins[start_bin]), 4),
                "duration_bins": int(duration),
                "duration_ms": round(float(duration * bin_size_ms), 2),
                "size": int(size),
            })
            in_avalanche = False
            size = 0

    if in_avalanche:
        duration = len(above) - start_bin
        avalanches.append({
            "start_bin": int(start_bin),
            "start_time": round(float(bins[start_bin]), 4),
            "duration_bins": int(duration),
            "duration_ms": round(float(duration * bin_size_ms), 2),
            "size": int(size),
        })

    sizes = [a["size"] for a in avalanches]
    durations = [a["duration_bins"] for a in avalanches]

    # Power law fit for sizes
    size_fit = _fit_power_law(sizes) if len(sizes) > 10 else None
    duration_fit = _fit_power_law(durations) if len(durations) > 10 else None

    # Branching ratio
    branching = _compute_branching_ratio(counts, above)

    # Criticality assessment
    is_critical = False
    if size_fit and duration_fit:
        # At criticality: size exponent ≈ -1.5, duration exponent ≈ -2.0
        size_ok = 1.2 < abs(size_fit["exponent"]) < 1.8
        dur_ok = 1.5 < abs(duration_fit["exponent"]) < 2.5
        branching_ok = 0.9 < branching["mean_ratio"] < 1.1
        is_critical = size_ok and dur_ok and branching_ok

    return {
        "avalanches": avalanches[:100],  # limit for JSON
        "n_avalanches": len(avalanches),
        "mean_size": round(float(np.mean(sizes)), 2) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "mean_duration_ms": round(float(np.mean([a["duration_ms"] for a in avalanches])), 2) if avalanches else 0,
        "size_distribution": {
            "values": sorted(set(sizes)),
            "counts": [sizes.count(s) for s in sorted(set(sizes))],
        } if sizes else {},
        "size_power_law": size_fit,
        "duration_power_law": duration_fit,
        "branching_ratio": branching,
        "is_critical": is_critical,
        "criticality_assessment": (
            "CRITICAL — organoid operates near edge of chaos (optimal for information processing)"
            if is_critical
            else "SUB-CRITICAL — activity dies out quickly, low information capacity"
            if branching.get("mean_ratio", 0) < 0.9
            else "SUPER-CRITICAL — runaway activity, epileptic-like dynamics"
            if branching.get("mean_ratio", 0) > 1.1
            else "NEAR-CRITICAL — close to optimal, minor deviations"
        ),
        "bin_size_ms": bin_size_ms,
    }


def _fit_power_law(values: list) -> Optional[dict]:
    """Fit power law distribution P(x) ~ x^(-alpha) using MLE."""
    arr = np.array([v for v in values if v > 0])
    if len(arr) < 10:
        return None

    # Maximum likelihood estimate (Clauset et al., 2009)
    x_min = max(1, int(np.min(arr)))
    arr_filtered = arr[arr >= x_min]
    if len(arr_filtered) < 5:
        return None

    n = len(arr_filtered)
    alpha = 1 + n / np.sum(np.log(arr_filtered / (x_min - 0.5)))

    # Kolmogorov-Smirnov test for goodness of fit
    # Compare empirical CDF with power law CDF
    sorted_data = np.sort(arr_filtered)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1 - (sorted_data / x_min) ** (-(alpha - 1))
    ks_stat = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    return {
        "exponent": round(float(-alpha), 3),
        "x_min": int(x_min),
        "n_samples": int(n),
        "ks_statistic": round(ks_stat, 4),
        "good_fit": ks_stat < 0.1,
    }


def _compute_branching_ratio(counts: np.ndarray, above: np.ndarray) -> dict:
    """Compute branching ratio — ratio of activity in bin t+1 to bin t.

    At criticality, branching ratio ≈ 1.0
    < 1.0 = subcritical (activity dies)
    > 1.0 = supercritical (activity explodes)
    """
    ratios = []
    for i in range(len(counts) - 1):
        if counts[i] > 0 and above[i]:
            ratios.append(counts[i + 1] / counts[i])

    if not ratios:
        return {"mean_ratio": 0, "std_ratio": 0}

    return {
        "mean_ratio": round(float(np.mean(ratios)), 4),
        "std_ratio": round(float(np.std(ratios)), 4),
        "median_ratio": round(float(np.median(ratios)), 4),
    }
