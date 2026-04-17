"""Neuronal avalanche analysis and criticality assessment.

Determines whether an organoid operates at the critical point ("edge of chaos"),
the dynamical regime that maximizes information transmission, storage capacity,
and dynamic range in neural systems (Beggs & Plenz 2003, Shew & Plenz 2013).

Pipeline
--------
1. Adaptive time-bin selection via geometric-mean ISI (Beggs & Plenz 2003).
2. Population-level avalanche detection with vectorised run-length encoding.
3. Maximum-likelihood power-law fitting with optimal x_min (Clauset et al. 2009).
4. Statistical validation: KS test, likelihood-ratio tests vs exponential and
   lognormal alternatives, bootstrap confidence intervals on exponents.
5. Scaling relation: size ~ duration^gamma and crackling-noise consistency check.
6. Branching ratio sigma and Deviation from Criticality Coefficient (DCC, Ma 2019).
7. Temporal evolution of criticality over the full recording span.
8. Final classification: subcritical / critical / supercritical with confidence.

Performance target: < 60 s on 2.6 M spikes, 32 electrodes, 118 hours.

References
----------
- Beggs & Plenz, J Neurosci 23(35):11167-11177, 2003.
- Clauset, Shalizi & Newman, SIAM Review 51(4):661-703, 2009.
- Klaus, Yu & Bhatt, PLoS ONE 6(5):e19779, 2011.
- Friedman et al., PRL 108:208102, 2012.
- Shew & Plenz, The Neuroscientist 19(1):88-100, 2013.
- Ma et al., J Neurosci 39(21):3088-3099, 2019.
- Sethna, Dahmen & Myers, Nature 410:242-250, 2001 (crackling noise).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .loader import SpikeData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Theoretical critical exponents for mean-field directed percolation
# (universality class expected for cortical avalanches)
CRITICAL_TAU = 1.5          # P(size) ~ s^{-tau}
CRITICAL_ALPHA = 2.0        # P(duration) ~ d^{-alpha}
CRITICAL_GAMMA_PRED = 2.0   # gamma = (alpha-1)/(tau-1)  [crackling noise]
CRITICAL_SIGMA = 1.0        # branching ratio at criticality

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------
@dataclass
class PowerLawFit:
    """Result of a maximum-likelihood power-law fit (Clauset et al. 2009)."""
    exponent: float              # alpha (positive; P(x) ~ x^{-alpha})
    x_min: int                   # optimal lower cutoff
    n_tail: int                  # samples in the tail (x >= x_min)
    ks_statistic: float          # KS distance for the fitted power law
    ks_p_value: float            # p-value from semi-parametric bootstrap
    # Likelihood-ratio tests vs alternatives
    lr_exp_statistic: float      # LR stat vs exponential
    lr_exp_p_value: float
    lr_lognormal_statistic: float
    lr_lognormal_p_value: float
    # Bootstrap CI
    ci_low: float                # 2.5th percentile of alpha
    ci_high: float               # 97.5th percentile of alpha
    is_power_law: bool           # passes KS + beats alternatives


@dataclass
class ScalingRelation:
    """Size-duration scaling: <S>(D) ~ D^gamma."""
    gamma: float
    gamma_ci_low: float
    gamma_ci_high: float
    gamma_predicted: float       # (alpha-1)/(tau-1) from crackling noise
    crackling_consistent: bool   # |gamma - gamma_pred| within CI


@dataclass
class BranchingRatio:
    """Population branching ratio sigma."""
    mean: float
    std: float
    median: float
    ci_low: float
    ci_high: float


@dataclass
class DCC:
    """Deviation from Criticality Coefficient (Ma et al. 2019)."""
    kappa: float                 # DCC value; 1 = critical
    interpretation: str          # human-readable


@dataclass
class TemporalWindow:
    """Criticality metrics for one temporal window."""
    window_index: int
    start_time_h: float
    end_time_h: float
    n_avalanches: int
    branching_ratio: float
    tau: Optional[float]
    alpha: Optional[float]
    dcc_kappa: Optional[float]


@dataclass
class CriticalityResult:
    """Complete criticality analysis output."""
    # Bin selection
    bin_size_ms: float
    adaptive_method: str

    # Avalanche statistics
    n_avalanches: int
    mean_size: float
    median_size: float
    max_size: int
    mean_duration_bins: float
    max_duration_bins: int

    # Power-law fits
    size_fit: Optional[PowerLawFit]
    duration_fit: Optional[PowerLawFit]

    # Scaling
    scaling: Optional[ScalingRelation]

    # Branching ratio
    branching: BranchingRatio

    # DCC
    dcc: DCC

    # Temporal evolution
    temporal_windows: list[TemporalWindow]

    # Classification
    classification: str       # "SUBCRITICAL" / "CRITICAL" / "SUPERCRITICAL"
    confidence: float         # 0-1
    evidence_summary: str     # human-readable justification

    # Size / duration distributions for downstream plotting
    size_distribution: dict   # {"values": [...], "counts": [...]}
    duration_distribution: dict

    # First N avalanches for inspection
    sample_avalanches: list[dict]

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return _dataclass_to_dict(self)


# ============================================================================
# PUBLIC API
# ============================================================================
def analyse_criticality(
    data: SpikeData,
    *,
    bin_size_ms: Optional[float] = None,
    n_temporal_windows: int = 10,
    n_bootstrap: int = 500,
    max_sample_avalanches: int = 200,
) -> CriticalityResult:
    """Run the full criticality analysis pipeline.

    Parameters
    ----------
    data : SpikeData
        Spike train (times in seconds, electrode ids).
    bin_size_ms : float or None
        Time-bin width in ms.  If None, adaptive selection via geometric-mean
        ISI across the population (Beggs & Plenz 2003).
    n_temporal_windows : int
        Number of equal-duration windows for temporal evolution tracking.
    n_bootstrap : int
        Bootstrap resamples for confidence intervals on power-law exponents
        and for the KS p-value.
    max_sample_avalanches : int
        How many individual avalanche records to include in the output.

    Returns
    -------
    CriticalityResult
    """
    if data.n_spikes == 0:
        return _empty_result()

    # ------------------------------------------------------------------
    # 1. Adaptive bin-size selection
    # ------------------------------------------------------------------
    if bin_size_ms is None:
        bin_size_ms = _adaptive_bin_size(data)
        adaptive_method = "geometric_mean_ISI"
    else:
        adaptive_method = "user_specified"

    bin_sec = bin_size_ms / 1000.0

    # ------------------------------------------------------------------
    # 2. Population activity histogram & avalanche detection
    # ------------------------------------------------------------------
    t0, t1 = data.time_range
    bins_edges = np.arange(t0, t1 + bin_sec, bin_sec)
    pop_counts = np.empty(len(bins_edges) - 1, dtype=np.int32)
    # np.histogram is fine for 2.6 M spikes; takes ~100 ms
    pop_counts[:], _ = np.histogram(data.times, bins=bins_edges)

    active = pop_counts > 0
    sizes, durations, starts = _detect_avalanche_runs(pop_counts, active)

    n_av = len(sizes)
    if n_av == 0:
        return _empty_result(bin_size_ms=bin_size_ms, adaptive_method=adaptive_method)

    # ------------------------------------------------------------------
    # 3. Power-law fits for size and duration distributions
    # ------------------------------------------------------------------
    size_fit = _fit_power_law_clauset(sizes, n_bootstrap=n_bootstrap) if n_av >= 50 else None
    dur_fit = _fit_power_law_clauset(durations, n_bootstrap=n_bootstrap) if n_av >= 50 else None

    # ------------------------------------------------------------------
    # 4. Size-duration scaling relation
    # ------------------------------------------------------------------
    scaling = _compute_scaling_relation(
        sizes, durations, size_fit, dur_fit, n_bootstrap=n_bootstrap,
    ) if n_av >= 50 else None

    # ------------------------------------------------------------------
    # 5. Branching ratio
    # ------------------------------------------------------------------
    branching = _compute_branching_ratio(pop_counts, active)

    # ------------------------------------------------------------------
    # 6. DCC (Ma et al. 2019)
    # ------------------------------------------------------------------
    dcc = _compute_dcc(sizes, durations, size_fit, dur_fit)

    # ------------------------------------------------------------------
    # 7. Temporal evolution
    # ------------------------------------------------------------------
    temporal_windows = _temporal_evolution(
        data, bin_size_ms=bin_size_ms, n_windows=n_temporal_windows,
    )

    # ------------------------------------------------------------------
    # 8. Classification
    # ------------------------------------------------------------------
    classification, confidence, evidence = _classify_criticality(
        size_fit, dur_fit, scaling, branching, dcc,
    )

    # ------------------------------------------------------------------
    # 9. Build output
    # ------------------------------------------------------------------
    # Size / duration distributions
    uq_s, ct_s = np.unique(sizes, return_counts=True)
    uq_d, ct_d = np.unique(durations, return_counts=True)

    sample = []
    for i in range(min(n_av, max_sample_avalanches)):
        sb = int(starts[i])
        dur = int(durations[i])
        sample.append({
            "start_bin": sb,
            "start_time": round(float(bins_edges[sb]), 4),
            "duration_bins": dur,
            "duration_ms": round(float(dur * bin_size_ms), 2),
            "size": int(sizes[i]),
        })

    return CriticalityResult(
        bin_size_ms=round(bin_size_ms, 4),
        adaptive_method=adaptive_method,
        n_avalanches=n_av,
        mean_size=round(float(np.mean(sizes)), 2),
        median_size=round(float(np.median(sizes)), 2),
        max_size=int(np.max(sizes)),
        mean_duration_bins=round(float(np.mean(durations)), 2),
        max_duration_bins=int(np.max(durations)),
        size_fit=size_fit,
        duration_fit=dur_fit,
        scaling=scaling,
        branching=branching,
        dcc=dcc,
        temporal_windows=temporal_windows,
        classification=classification,
        confidence=round(confidence, 3),
        evidence_summary=evidence,
        size_distribution={"values": uq_s.tolist(), "counts": ct_s.tolist()},
        duration_distribution={"values": uq_d.tolist(), "counts": ct_d.tolist()},
        sample_avalanches=sample,
    )


# Keep the old name as an alias for backwards compatibility
detect_avalanches = analyse_criticality


# ============================================================================
# ADAPTIVE BIN SIZE (Beggs & Plenz 2003)
# ============================================================================
def _adaptive_bin_size(data: SpikeData) -> float:
    """Select time-bin width as the geometric mean of per-electrode mean ISIs.

    This is the standard Beggs & Plenz (2003) prescription: the bin should
    approximate the average time for activity to propagate from one electrode
    to the next.  The geometric mean avoids bias from electrodes with very
    low or very high firing rates.
    """
    mean_isis: list[float] = []
    for eid in data.electrode_ids:
        idx = data._electrode_indices[eid]
        if len(idx) < 2:
            continue
        t_e = data.times[idx]
        isi = np.diff(t_e)
        if len(isi) > 0:
            m = float(np.mean(isi))
            if m > 0:
                mean_isis.append(m)

    if not mean_isis:
        # Fallback: population-level ISI
        if data.n_spikes < 2:
            return 5.0  # conservative default (ms)
        isi_all = np.diff(data.times)
        return max(0.5, float(np.mean(isi_all)) * 1000.0)

    # Geometric mean in seconds, convert to ms
    log_mean = float(np.mean(np.log(np.array(mean_isis))))
    geom_mean_sec = np.exp(log_mean)
    bin_ms = geom_mean_sec * 1000.0

    # Clamp to reasonable range [0.5, 50] ms
    return float(np.clip(bin_ms, 0.5, 50.0))


# ============================================================================
# AVALANCHE DETECTION (vectorised)
# ============================================================================
def _detect_avalanche_runs(
    counts: NDArray[np.int32],
    active: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Detect contiguous runs of active bins (vectorised run-length encoding).

    An avalanche starts when a bin has >= 1 spike after a silent bin, and
    ends when the next bin is silent.

    Returns
    -------
    sizes : int64 array
        Total spike count in each avalanche.
    durations : int64 array
        Number of active bins in each avalanche.
    starts : int64 array
        Index of the first active bin of each avalanche.
    """
    empty = np.array([], dtype=np.int64)
    if len(active) == 0 or not np.any(active):
        return empty, empty, empty

    padded = np.empty(len(active) + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = active.view(np.uint8)

    d = np.diff(padded)
    run_starts = np.where(d == 1)[0]
    run_ends = np.where(d == -1)[0]

    durations_out = (run_ends - run_starts).astype(np.int64)

    # Cumulative-sum trick to get sizes without a Python loop
    cs = np.empty(len(counts) + 1, dtype=np.int64)
    cs[0] = 0
    np.cumsum(counts, out=cs[1:])
    sizes_out = (cs[run_ends] - cs[run_starts]).astype(np.int64)

    return sizes_out, durations_out, run_starts.astype(np.int64)


# ============================================================================
# POWER-LAW FITTING (Clauset, Shalizi & Newman 2009)
# ============================================================================
def _fit_power_law_clauset(
    values: NDArray[np.int64],
    *,
    n_bootstrap: int = 500,
    x_min_candidates: int = 50,
) -> Optional[PowerLawFit]:
    """Maximum-likelihood discrete power-law fit with optimal x_min.

    Algorithm
    ---------
    1. For each candidate x_min, compute MLE alpha and KS distance.
    2. Choose x_min that minimises KS distance.
    3. Semi-parametric bootstrap for KS p-value (Clauset et al. 2009, sec 4).
    4. Likelihood-ratio tests vs exponential and lognormal.
    5. Bootstrap CI on alpha.
    """
    arr = values[values > 0]
    if len(arr) < 50:
        return None

    unique_vals = np.unique(arr)
    if len(unique_vals) < 5:
        return None

    # --- Step 1-2: Optimal x_min via KS minimisation ---
    # Limit candidate x_min values for performance
    if len(unique_vals) > x_min_candidates:
        # Logarithmically-spaced subset to cover the range efficiently
        idx = np.unique(np.geomspace(1, len(unique_vals) - 1, x_min_candidates).astype(int))
        idx = np.concatenate(([0], idx))  # include first element
        candidates = unique_vals[idx]
    else:
        candidates = unique_vals

    # Exclude candidates that leave fewer than 30 tail samples
    best_ks = np.inf
    best_xmin = int(candidates[0])
    best_alpha = 2.0

    for xm in candidates:
        xm = int(xm)
        tail = arr[arr >= xm]
        n_t = len(tail)
        if n_t < 30:
            continue
        a = _discrete_mle_alpha(tail, xm)
        ks = _ks_distance_discrete(tail, xm, a)
        if ks < best_ks:
            best_ks = ks
            best_xmin = xm
            best_alpha = a

    tail = arr[arr >= best_xmin]
    n_tail = len(tail)
    if n_tail < 30:
        return None

    # --- Step 3: Semi-parametric bootstrap KS p-value ---
    ks_p = _bootstrap_ks_pvalue(arr, best_xmin, best_alpha, best_ks, n_boot=n_bootstrap)

    # --- Step 4: Likelihood-ratio tests ---
    lr_exp_stat, lr_exp_p = _lr_test_vs_exponential(tail, best_xmin, best_alpha)
    lr_ln_stat, lr_ln_p = _lr_test_vs_lognormal(tail, best_xmin, best_alpha)

    # --- Step 5: Bootstrap CI on alpha ---
    ci_low, ci_high = _bootstrap_alpha_ci(tail, best_xmin, n_boot=n_bootstrap)

    # Classify as power law if KS p >= 0.1 and LR favours power law over
    # at least one alternative (or alternatives are indistinguishable).
    is_pl = (
        ks_p >= 0.1
        and n_tail >= 50
        and (lr_exp_stat > 0 or lr_exp_p > 0.1)
    )

    return PowerLawFit(
        exponent=round(best_alpha, 4),
        x_min=best_xmin,
        n_tail=n_tail,
        ks_statistic=round(best_ks, 5),
        ks_p_value=round(ks_p, 4),
        lr_exp_statistic=round(lr_exp_stat, 4),
        lr_exp_p_value=round(lr_exp_p, 4),
        lr_lognormal_statistic=round(lr_ln_stat, 4),
        lr_lognormal_p_value=round(lr_ln_p, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        is_power_law=is_pl,
    )


def _discrete_mle_alpha(tail: NDArray, x_min: int) -> float:
    """MLE for discrete power-law exponent (Clauset et al. 2009, eq. 3.7).

    alpha = 1 + n * [ sum_i ln(x_i / (x_min - 0.5)) ]^{-1}
    """
    n = len(tail)
    if n == 0:
        return 2.0
    s = float(np.sum(np.log(tail.astype(np.float64) / (x_min - 0.5))))
    if s <= 0:
        return 2.0
    return 1.0 + n / s


def _ks_distance_discrete(tail: NDArray, x_min: int, alpha: float) -> float:
    """KS distance between empirical CDF and discrete power-law CDF."""
    n = len(tail)
    sorted_t = np.sort(tail)
    ecdf = np.arange(1, n + 1, dtype=np.float64) / n
    # Approximate discrete power-law CDF using continuous form
    # CDF(x) = 1 - (x / x_min)^{-(alpha - 1)}
    # This approximation is standard and accurate for x_min >= 1
    tcdf = 1.0 - (sorted_t.astype(np.float64) / x_min) ** (-(alpha - 1.0))
    return float(np.max(np.abs(ecdf - tcdf)))


def _bootstrap_ks_pvalue(
    all_values: NDArray,
    x_min: int,
    alpha: float,
    observed_ks: float,
    n_boot: int,
) -> float:
    """Semi-parametric bootstrap for KS goodness-of-fit p-value.

    Procedure (Clauset et al. 2009, section 4):
    - Below x_min: resample from empirical data.
    - At or above x_min: draw from the fitted power-law.
    - Refit alpha on each synthetic sample, compute KS, count how often
      synthetic KS >= observed KS.
    """
    below = all_values[all_values < x_min]
    n_total = len(all_values)
    n_tail = int(np.sum(all_values >= x_min))
    n_below = n_total - n_tail

    if n_tail < 30 or n_boot < 10:
        return 0.0

    count_ge = 0
    for _ in range(n_boot):
        # Resample below x_min
        if n_below > 0:
            syn_below = _RNG.choice(below, size=n_below, replace=True)
        else:
            syn_below = np.array([], dtype=all_values.dtype)

        # Draw from power law above x_min via inverse-CDF
        u = _RNG.uniform(0, 1, size=n_tail)
        syn_above = np.floor(x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))).astype(
            all_values.dtype,
        )
        syn_above = np.clip(syn_above, x_min, None)

        syn = np.concatenate([syn_below, syn_above])
        syn_tail = syn[syn >= x_min]
        if len(syn_tail) < 10:
            continue

        a_syn = _discrete_mle_alpha(syn_tail, x_min)
        ks_syn = _ks_distance_discrete(syn_tail, x_min, a_syn)
        if ks_syn >= observed_ks:
            count_ge += 1

    return count_ge / n_boot


def _bootstrap_alpha_ci(
    tail: NDArray,
    x_min: int,
    n_boot: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval on the MLE alpha."""
    n = len(tail)
    if n < 30 or n_boot < 10:
        a0 = _discrete_mle_alpha(tail, x_min)
        return (a0, a0)

    alphas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _RNG.integers(0, n, size=n)
        alphas[i] = _discrete_mle_alpha(tail[idx], x_min)

    lo = (1.0 - ci) / 2.0
    return (float(np.quantile(alphas, lo)), float(np.quantile(alphas, 1.0 - lo)))


# ============================================================================
# LIKELIHOOD-RATIO TESTS (Clauset et al. 2009, section 5)
# ============================================================================
def _loglik_powerlaw(tail: NDArray, x_min: int, alpha: float) -> float:
    """Log-likelihood for discrete power law (continuous approximation)."""
    n = len(tail)
    if n == 0:
        return -np.inf
    # L_PL = -alpha * sum(ln x_i) - n * ln(zeta(alpha, x_min))
    # Approximate zeta with continuous: n*ln(alpha-1) - n*ln(x_min) (for alpha>1)
    # More precisely: ln L ~ -alpha * sum(ln x) + n*ln(alpha-1) - n*(alpha-1)*ln(x_min)
    t = tail.astype(np.float64)
    return float(
        -alpha * np.sum(np.log(t))
        + n * np.log(alpha - 1.0)
        + n * (alpha - 1.0) * np.log(x_min - 0.5)
    )


def _lr_test_vs_exponential(
    tail: NDArray, x_min: int, alpha: float,
) -> tuple[float, float]:
    """Likelihood-ratio test: power law vs exponential.

    Returns (R, p): R > 0 favours power law, R < 0 favours exponential.
    """
    t = tail.astype(np.float64)
    n = len(t)
    if n < 10:
        return (0.0, 1.0)

    ll_pl = _loglik_powerlaw(tail, x_min, alpha)

    # MLE for exponential: lambda = 1 / mean(x - x_min + 0.5)
    shifted = t - x_min + 0.5
    shifted = np.clip(shifted, 1e-12, None)
    lam = 1.0 / float(np.mean(shifted))
    ll_exp = float(n * np.log(lam) - lam * np.sum(shifted))

    R = ll_pl - ll_exp
    # Normalised test statistic
    sigma = np.sqrt(float(np.sum((
        -alpha * np.log(t) + np.log(alpha - 1.0) + (alpha - 1.0) * np.log(x_min - 0.5)
        - np.log(lam) + lam * shifted
    ) ** 2)) / n - (R / n) ** 2)

    if sigma < 1e-12:
        return (float(np.sign(R)), 1.0)

    z = R / (sigma * np.sqrt(n))
    p = float(sp_stats.norm.sf(abs(z)) * 2)  # two-sided
    return (round(float(R), 4), round(p, 4))


def _lr_test_vs_lognormal(
    tail: NDArray, x_min: int, alpha: float,
) -> tuple[float, float]:
    """Likelihood-ratio test: power law vs lognormal."""
    t = tail.astype(np.float64)
    n = len(t)
    if n < 10:
        return (0.0, 1.0)

    ll_pl = _loglik_powerlaw(tail, x_min, alpha)

    # MLE for lognormal
    log_t = np.log(t)
    mu = float(np.mean(log_t))
    sigma_ln = float(np.std(log_t, ddof=1))
    if sigma_ln < 1e-12:
        return (0.0, 1.0)

    ll_ln = float(np.sum(sp_stats.lognorm.logpdf(t, s=sigma_ln, scale=np.exp(mu))))

    R = ll_pl - ll_ln
    # Approximate p-value via Vuong's test
    per_point = (
        -alpha * log_t + np.log(alpha - 1.0) + (alpha - 1.0) * np.log(x_min - 0.5)
        - sp_stats.lognorm.logpdf(t, s=sigma_ln, scale=np.exp(mu))
    )
    sig = float(np.std(per_point, ddof=1))
    if sig < 1e-12:
        return (float(np.sign(R)), 1.0)
    z = float(np.mean(per_point)) / (sig / np.sqrt(n))
    p = float(sp_stats.norm.sf(abs(z)) * 2)
    return (round(float(R), 4), round(p, 4))


# ============================================================================
# SCALING RELATION & CRACKLING NOISE
# ============================================================================
def _compute_scaling_relation(
    sizes: NDArray[np.int64],
    durations: NDArray[np.int64],
    size_fit: Optional[PowerLawFit],
    dur_fit: Optional[PowerLawFit],
    *,
    n_bootstrap: int = 500,
) -> Optional[ScalingRelation]:
    """Compute size-duration scaling: <S>(D) ~ D^gamma.

    At criticality with crackling-noise universality (Sethna et al. 2001,
    Friedman et al. 2012):
        gamma = (alpha - 1) / (tau - 1)

    where tau is the size exponent and alpha is the duration exponent.
    """
    if len(sizes) < 50:
        return None

    # Group sizes by duration, compute mean size for each duration
    unique_d = np.unique(durations)
    # Need at least 3 distinct durations for a regression
    if len(unique_d) < 3:
        return None

    log_d_list = []
    log_mean_s_list = []
    for d in unique_d:
        mask = durations == d
        count = np.sum(mask)
        if count < 3:
            continue
        mean_s = float(np.mean(sizes[mask]))
        if mean_s > 0 and d > 0:
            log_d_list.append(np.log(float(d)))
            log_mean_s_list.append(np.log(mean_s))

    if len(log_d_list) < 3:
        return None

    log_d_arr = np.array(log_d_list)
    log_s_arr = np.array(log_mean_s_list)

    # OLS in log-log space
    if len(log_d_arr) < 2 or np.all(log_d_arr == log_d_arr[0]):
        return None
    slope, intercept, r, p, se = sp_stats.linregress(log_d_arr, log_s_arr)
    gamma = slope

    # Bootstrap CI on gamma
    n_pts = len(log_d_arr)
    gammas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = _RNG.integers(0, n_pts, size=n_pts)
        sl, *_ = sp_stats.linregress(log_d_arr[idx], log_s_arr[idx])
        gammas[i] = sl
    ci_lo = float(np.quantile(gammas, 0.025))
    ci_hi = float(np.quantile(gammas, 0.975))

    # Predicted gamma from crackling-noise relation
    if size_fit is not None and dur_fit is not None:
        tau = size_fit.exponent
        alpha_d = dur_fit.exponent
        if tau > 1.0:
            gamma_pred = (alpha_d - 1.0) / (tau - 1.0)
        else:
            gamma_pred = CRITICAL_GAMMA_PRED
    else:
        gamma_pred = CRITICAL_GAMMA_PRED

    consistent = ci_lo <= gamma_pred <= ci_hi

    return ScalingRelation(
        gamma=round(gamma, 4),
        gamma_ci_low=round(ci_lo, 4),
        gamma_ci_high=round(ci_hi, 4),
        gamma_predicted=round(gamma_pred, 4),
        crackling_consistent=consistent,
    )


# ============================================================================
# BRANCHING RATIO
# ============================================================================
def _compute_branching_ratio(
    counts: NDArray[np.int32],
    active: NDArray[np.bool_],
) -> BranchingRatio:
    """Population branching ratio sigma = <n_{t+1} / n_t> over active bins.

    sigma = 1  at criticality (Haldeman & Beggs 2005).
    sigma < 1  subcritical (activity dies).
    sigma > 1  supercritical (activity explodes).
    """
    mask = active[:-1] & (counts[:-1] > 0)
    if not np.any(mask):
        return BranchingRatio(mean=0.0, std=0.0, median=0.0, ci_low=0.0, ci_high=0.0)

    numer = counts[1:][mask].astype(np.float64)
    denom = counts[:-1][mask].astype(np.float64)
    ratios = numer / denom

    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    med_r = float(np.median(ratios))

    # Bootstrap 95% CI on the mean
    n_boot = min(1000, max(100, len(ratios)))
    boot_means = np.empty(n_boot)
    n_r = len(ratios)
    for i in range(n_boot):
        idx = _RNG.integers(0, n_r, size=n_r)
        boot_means[i] = np.mean(ratios[idx])
    ci_lo = float(np.quantile(boot_means, 0.025))
    ci_hi = float(np.quantile(boot_means, 0.975))

    return BranchingRatio(
        mean=round(mean_r, 5),
        std=round(std_r, 5),
        median=round(med_r, 5),
        ci_low=round(ci_lo, 5),
        ci_high=round(ci_hi, 5),
    )


# ============================================================================
# DEVIATION FROM CRITICALITY COEFFICIENT (Ma et al. 2019)
# ============================================================================
def _compute_dcc(
    sizes: NDArray[np.int64],
    durations: NDArray[np.int64],
    size_fit: Optional[PowerLawFit],
    dur_fit: Optional[PowerLawFit],
) -> DCC:
    r"""Deviation from Criticality Coefficient (kappa).

    kappa = 1 + sum of three normalised deviations:
      1. |tau - 1.5| / 0.5
      2. |alpha - 2.0| / 1.0
      3. |sigma - 1| / 0.5  (branching ratio is handled separately below)

    Simplified: we compute kappa from how far the exponents deviate from
    criticality. At the critical point, kappa ~ 1.

    Following Ma et al. (2019): kappa is computed by comparing the predicted
    and empirical mean size given duration, normalised.
    """
    if size_fit is None or dur_fit is None or len(sizes) < 50:
        return DCC(kappa=float("nan"), interpretation="insufficient_data")

    tau = size_fit.exponent
    alpha_d = dur_fit.exponent

    # Predicted <S>(D) from the scaling relation
    unique_d = np.unique(durations)
    if len(unique_d) < 3 or tau <= 1.0:
        return DCC(kappa=float("nan"), interpretation="insufficient_data")

    gamma_pred = (alpha_d - 1.0) / (tau - 1.0)

    # For each unique duration, compute predicted and empirical mean size
    empirical_means = []
    predicted_means = []
    for d_val in unique_d:
        mask = durations == d_val
        count = np.sum(mask)
        if count < 2:
            continue
        emp = float(np.mean(sizes[mask]))
        pred = float(d_val) ** gamma_pred
        empirical_means.append(emp)
        predicted_means.append(pred)

    if len(empirical_means) < 3:
        return DCC(kappa=float("nan"), interpretation="insufficient_data")

    emp_arr = np.array(empirical_means)
    pred_arr = np.array(predicted_means)

    # Normalise predicted to same scale as empirical
    scale = np.sum(emp_arr) / np.sum(pred_arr) if np.sum(pred_arr) > 0 else 1.0
    pred_arr = pred_arr * scale

    # kappa = mean(pred / emp) -- at criticality this is ~1
    ratios = pred_arr / np.clip(emp_arr, 1e-12, None)
    kappa = float(np.mean(ratios))

    if 0.9 <= kappa <= 1.1:
        interp = "critical (kappa near 1)"
    elif kappa < 0.9:
        interp = "subcritical (predicted < empirical)"
    else:
        interp = "supercritical (predicted > empirical)"

    return DCC(kappa=round(kappa, 4), interpretation=interp)


# ============================================================================
# TEMPORAL EVOLUTION
# ============================================================================
def _temporal_evolution(
    data: SpikeData,
    *,
    bin_size_ms: float,
    n_windows: int,
) -> list[TemporalWindow]:
    """Track how criticality metrics evolve over the recording duration.

    Splits the recording into `n_windows` equal segments and independently
    analyses each one.
    """
    t0, t1 = data.time_range
    total_dur = t1 - t0
    if total_dur <= 0 or n_windows < 1:
        return []

    window_sec = total_dur / n_windows
    bin_sec = bin_size_ms / 1000.0
    results: list[TemporalWindow] = []

    for w in range(n_windows):
        w_start = t0 + w * window_sec
        w_end = w_start + window_sec

        # Select spikes in this window (binary search for speed)
        i_lo = int(np.searchsorted(data.times, w_start, side="left"))
        i_hi = int(np.searchsorted(data.times, w_end, side="right"))
        if i_hi - i_lo < 10:
            results.append(TemporalWindow(
                window_index=w,
                start_time_h=round((w_start - t0) / 3600, 2),
                end_time_h=round((w_end - t0) / 3600, 2),
                n_avalanches=0,
                branching_ratio=float("nan"),
                tau=None,
                alpha=None,
                dcc_kappa=None,
            ))
            continue

        # Build population histogram for this window
        bins_edges = np.arange(w_start, w_end + bin_sec, bin_sec)
        window_times = data.times[i_lo:i_hi]
        pop_counts, _ = np.histogram(window_times, bins=bins_edges)
        pop_counts = pop_counts.astype(np.int32)
        active_w = pop_counts > 0

        sizes_w, durs_w, _ = _detect_avalanche_runs(pop_counts, active_w)
        n_av = len(sizes_w)

        # Branching ratio
        br = _compute_branching_ratio(pop_counts, active_w)

        # Quick power-law fit (reduced bootstrap for speed)
        tau_val = None
        alpha_val = None
        kappa_val = None
        if n_av >= 50:
            sf = _fit_power_law_clauset(sizes_w, n_bootstrap=50, x_min_candidates=20)
            df = _fit_power_law_clauset(durs_w, n_bootstrap=50, x_min_candidates=20)
            if sf is not None:
                tau_val = sf.exponent
            if df is not None:
                alpha_val = df.exponent
            dcc_w = _compute_dcc(sizes_w, durs_w, sf, df)
            if not np.isnan(dcc_w.kappa):
                kappa_val = dcc_w.kappa

        results.append(TemporalWindow(
            window_index=w,
            start_time_h=round((w_start - t0) / 3600, 2),
            end_time_h=round((w_end - t0) / 3600, 2),
            n_avalanches=n_av,
            branching_ratio=round(br.mean, 5),
            tau=round(tau_val, 4) if tau_val is not None else None,
            alpha=round(alpha_val, 4) if alpha_val is not None else None,
            dcc_kappa=round(kappa_val, 4) if kappa_val is not None else None,
        ))

    return results


# ============================================================================
# CLASSIFICATION
# ============================================================================
def _classify_criticality(
    size_fit: Optional[PowerLawFit],
    dur_fit: Optional[PowerLawFit],
    scaling: Optional[ScalingRelation],
    branching: BranchingRatio,
    dcc: DCC,
) -> tuple[str, float, str]:
    """Classify the dynamical regime and assign a confidence score.

    Uses a multi-evidence scoring approach:
    - Power-law fits (tau ~ 1.5, alpha ~ 2.0, passes KS)
    - Scaling relation consistency (crackling noise)
    - Branching ratio ~ 1
    - DCC ~ 1

    Returns (classification, confidence, evidence_summary).
    """
    evidence: list[str] = []
    score = 0.0  # positive = critical, negative = non-critical
    max_score = 0.0

    # --- Evidence 1: Size exponent tau ---
    max_score += 2.0
    if size_fit is not None:
        tau = size_fit.exponent
        if size_fit.is_power_law:
            dev = abs(tau - CRITICAL_TAU)
            if dev < 0.2:
                score += 2.0
                evidence.append(f"Size exponent tau={tau:.2f} (near 1.5, good power-law fit)")
            elif dev < 0.5:
                score += 1.0
                evidence.append(f"Size exponent tau={tau:.2f} (moderate deviation from 1.5)")
            else:
                evidence.append(f"Size exponent tau={tau:.2f} (far from 1.5)")
        else:
            evidence.append(f"Size distribution: power-law hypothesis rejected (KS p={size_fit.ks_p_value:.3f})")
    else:
        evidence.append("Size distribution: insufficient data for power-law fit")

    # --- Evidence 2: Duration exponent alpha ---
    max_score += 2.0
    if dur_fit is not None:
        alpha_d = dur_fit.exponent
        if dur_fit.is_power_law:
            dev = abs(alpha_d - CRITICAL_ALPHA)
            if dev < 0.3:
                score += 2.0
                evidence.append(f"Duration exponent alpha={alpha_d:.2f} (near 2.0, good power-law fit)")
            elif dev < 0.7:
                score += 1.0
                evidence.append(f"Duration exponent alpha={alpha_d:.2f} (moderate deviation from 2.0)")
            else:
                evidence.append(f"Duration exponent alpha={alpha_d:.2f} (far from 2.0)")
        else:
            evidence.append(f"Duration distribution: power-law hypothesis rejected (KS p={dur_fit.ks_p_value:.3f})")
    else:
        evidence.append("Duration distribution: insufficient data for power-law fit")

    # --- Evidence 3: Crackling-noise scaling ---
    max_score += 2.0
    if scaling is not None:
        if scaling.crackling_consistent:
            score += 2.0
            evidence.append(
                f"Scaling gamma={scaling.gamma:.2f} consistent with crackling-noise prediction "
                f"{scaling.gamma_predicted:.2f} (within 95% CI [{scaling.gamma_ci_low:.2f}, {scaling.gamma_ci_high:.2f}])"
            )
        else:
            dev = abs(scaling.gamma - scaling.gamma_predicted)
            if dev < 0.5:
                score += 0.5
            evidence.append(
                f"Scaling gamma={scaling.gamma:.2f} deviates from predicted {scaling.gamma_predicted:.2f}"
            )
    else:
        evidence.append("Scaling relation: insufficient data")

    # --- Evidence 4: Branching ratio ---
    max_score += 2.0
    sigma = branching.mean
    if sigma > 0:
        dev = abs(sigma - CRITICAL_SIGMA)
        if dev < 0.05:
            score += 2.0
            evidence.append(f"Branching ratio sigma={sigma:.4f} (very close to 1.0)")
        elif dev < 0.1:
            score += 1.5
            evidence.append(f"Branching ratio sigma={sigma:.4f} (near 1.0)")
        elif dev < 0.2:
            score += 0.5
            evidence.append(f"Branching ratio sigma={sigma:.4f} (moderate deviation from 1.0)")
        else:
            evidence.append(f"Branching ratio sigma={sigma:.4f} (far from 1.0)")
    else:
        evidence.append("Branching ratio: could not compute (no active bins)")

    # --- Evidence 5: DCC ---
    max_score += 2.0
    if not np.isnan(dcc.kappa):
        dev = abs(dcc.kappa - 1.0)
        if dev < 0.1:
            score += 2.0
            evidence.append(f"DCC kappa={dcc.kappa:.3f} (near 1.0, consistent with criticality)")
        elif dev < 0.2:
            score += 1.0
            evidence.append(f"DCC kappa={dcc.kappa:.3f} (small deviation from 1.0)")
        else:
            evidence.append(f"DCC kappa={dcc.kappa:.3f} ({dcc.interpretation})")
    else:
        evidence.append("DCC: insufficient data")

    # --- Classification ---
    confidence = score / max_score if max_score > 0 else 0.0

    if confidence >= 0.7:
        classification = "CRITICAL"
    elif sigma < 0.9 or confidence < 0.3:
        classification = "SUBCRITICAL"
    elif sigma > 1.1:
        classification = "SUPERCRITICAL"
    elif confidence >= 0.4:
        classification = "NEAR_CRITICAL"
    else:
        classification = "SUBCRITICAL"

    summary = (
        f"{classification} (confidence {confidence:.0%}). "
        + " | ".join(evidence)
    )

    return classification, confidence, summary


# ============================================================================
# HELPERS
# ============================================================================
def _empty_result(
    bin_size_ms: float = 5.0,
    adaptive_method: str = "none",
) -> CriticalityResult:
    """Return a minimal result when there are no spikes or avalanches."""
    return CriticalityResult(
        bin_size_ms=bin_size_ms,
        adaptive_method=adaptive_method,
        n_avalanches=0,
        mean_size=0.0,
        median_size=0.0,
        max_size=0,
        mean_duration_bins=0.0,
        max_duration_bins=0,
        size_fit=None,
        duration_fit=None,
        scaling=None,
        branching=BranchingRatio(mean=0.0, std=0.0, median=0.0, ci_low=0.0, ci_high=0.0),
        dcc=DCC(kappa=float("nan"), interpretation="no_data"),
        temporal_windows=[],
        classification="SUBCRITICAL",
        confidence=0.0,
        evidence_summary="No avalanches detected.",
        size_distribution={},
        duration_distribution={},
        sample_avalanches=[],
    )


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert a dataclass (or nested structure) to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        out = {}
        for f_name in obj.__dataclass_fields__:
            val = getattr(obj, f_name)
            out[f_name] = _dataclass_to_dict(val)
        return out
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
