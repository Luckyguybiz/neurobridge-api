"""Metastability analysis for MEA neural organoid recordings.

Implements publication-quality metastability metrics grounded in the
computational neuroscience literature:

  - Kuramoto order parameter R(t) and metastability index (Shanahan, 2010)
  - Coalition entropy (Shanahan, 2010; Deco & Kringelbach, 2016)
  - Functional Connectivity Dynamics (FCD) matrix (Hansen et al., 2015)
  - Integration-segregation balance (Tononi, Sporns & Edelman, 1994)
  - Chimera state detection (Abrams & Strogatz, 2004)
  - Dwell-time distributions in synchronised / desynchronised regimes
  - Multi-scale analysis across temporal resolutions
  - Per-MEA vs cross-MEA metastability decomposition
  - Null-model statistics via phase-shuffled surrogates

Designed for FinalSpark organoid data:
  4 MEAs x 8 electrodes = 32 channels, up to 118 h, 2.6 M spikes.
  Completes full analysis in <60 s on commodity hardware.

References
----------
Shanahan, M. (2010). Metastable chimera states in community-structured
    oscillator networks. Chaos, 20(1), 013108.
Deco, G. & Kringelbach, M. L. (2016). Metastability and coherence:
    extending the communication through coherence hypothesis using a
    whole-brain computational perspective. Trends in Neurosciences, 39(3).
Hansen, E. C. A. et al. (2015). Functional connectivity dynamics:
    modeling the switching behavior of the resting state. NeuroImage, 105.
Tononi, G., Sporns, O. & Edelman, G. M. (1994). A measure for brain
    complexity: relating functional segregation and integration in the
    nervous system. PNAS, 91(11), 5033-5037.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert
from scipy.stats import entropy as scipy_entropy

from .loader import SpikeData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BINS: int = 50_000          # hard cap on time bins (Hilbert input length)
MAX_WINDOWS_TIMESERIES: int = 500  # max points returned in JSON timeseries
N_SURROGATES: int = 200        # null-model replicates for significance tests
FCD_MAX_WINDOWS: int = 1_000   # cap for FCD matrix size (1000x1000 = 1M floats)
FINALSPARK_MEA_SIZE: int = 8   # electrodes per MEA in FinalSpark hardware
FINALSPARK_N_MEAS: int = 4     # number of MEAs per device

# Multi-scale temporal resolutions (seconds)
MULTISCALE_WINDOWS: tuple[float, ...] = (1.0, 10.0, 60.0)


# ---------------------------------------------------------------------------
# Data classes for typed, self-documenting results
# ---------------------------------------------------------------------------
@dataclass
class KuramotoResult:
    """Result of Kuramoto order parameter analysis."""
    R_mean: float                        # mean synchronisation level
    R_std: float                         # fluctuation magnitude
    R_min: float
    R_max: float
    metastability_index: float           # var(R) -- Shanahan 2010
    synchronisation_level: str           # "high" / "moderate" / "low"
    R_timeseries: list[float]            # down-sampled R(t)
    time_axis_s: list[float]             # time axis for R(t) in seconds
    bin_size_ms: float
    window_size_s: float
    n_windows: int
    # Null model
    null_mean: float = 0.0               # surrogate mean(metastability)
    null_std: float = 0.0                # surrogate std(metastability)
    p_value: float = 1.0                 # one-sided p vs shuffled phases


@dataclass
class CoalitionResult:
    """Coalition entropy and synchronisation pattern diversity."""
    coalition_entropy: float             # bits -- how many distinct patterns
    n_coalitions: int                    # number of identified clusters
    coalition_sizes: list[int]           # sizes of each cluster
    coalition_labels: list[int]          # cluster assignment per window


@dataclass
class FCDResult:
    """Functional Connectivity Dynamics."""
    fcd_variance: float                  # variance of upper-tri FCD entries
    fcd_mean: float                      # mean correlation between FC windows
    fcd_matrix_shape: tuple[int, int]    # shape of FCD matrix
    speed_of_fc_change: float            # mean abs(diff) along diagonal


@dataclass
class DwellTimeResult:
    """Dwell-time statistics in sync / desync regimes."""
    sync_threshold: float                # R threshold used
    mean_sync_dwell_s: float             # mean dwell in synchronised state
    mean_desync_dwell_s: float           # mean dwell in desynchronised state
    median_sync_dwell_s: float
    median_desync_dwell_s: float
    n_sync_epochs: int
    n_desync_epochs: int
    fraction_time_synchronised: float    # fraction of total time in sync


@dataclass
class ChimeraResult:
    """Chimera state detection -- partial synchronisation."""
    chimera_index: float                 # 0 = full sync/desync, 1 = chimera
    n_chimera_epochs: int                # windows with chimera detected
    fraction_chimera: float              # fraction of time in chimera state
    local_order_params: dict[str, float] # per-MEA mean R


@dataclass
class IntegrationSegregationResult:
    """Integration-segregation dynamics (Tononi & Edelman, 1994)."""
    integration: float                   # mutual information among electrodes
    segregation: float                   # sum of individual entropies
    complexity: float                    # neural complexity = integration * segregation balance
    balance_ratio: float                 # integration / (integration + segregation)


@dataclass
class MultiScaleResult:
    """Metastability at multiple temporal resolutions."""
    scales: dict[str, KuramotoResult]    # window_size_s -> KuramotoResult


@dataclass
class MEADecompositionResult:
    """Within-MEA vs cross-MEA metastability."""
    within_mea_metastability: dict[str, float]   # per-MEA metastability index
    across_mea_metastability: float              # cross-MEA metastability
    within_mean: float
    ratio: float                                 # across / within_mean


@dataclass
class StateTransitionResult:
    """Network state clustering and transition dynamics."""
    n_states: int
    transition_matrix: list[list[float]]
    state_labels: list[int]                      # truncated to MAX_WINDOWS_TIMESERIES
    dwell_times_ms: dict[str, list[float]]
    mean_dwell_ms: dict[str, float]
    n_transitions: int
    bin_size_ms: float


@dataclass
class MetastabilityReport:
    """Complete metastability analysis report."""
    kuramoto: KuramotoResult
    coalition: CoalitionResult
    fcd: FCDResult
    dwell_times: DwellTimeResult
    chimera: ChimeraResult
    integration_segregation: IntegrationSegregationResult
    multiscale: MultiScaleResult
    mea_decomposition: MEADecompositionResult
    state_transitions: StateTransitionResult
    is_metastable: bool
    computation_time_s: float


# ---------------------------------------------------------------------------
# Adaptive helpers
# ---------------------------------------------------------------------------

def _adaptive_bin_size(duration_s: float, default_ms: float) -> float:
    """Return bin size (seconds) capped so total bins <= MAX_BINS."""
    default_s = default_ms / 1000.0
    if duration_s / default_s <= MAX_BINS:
        return default_s
    return duration_s / MAX_BINS


def _adaptive_window_size(duration_s: float) -> float:
    """Choose Kuramoto analysis window appropriate for recording length."""
    if duration_s <= 3600.0:
        return 1.0
    if duration_s <= 12 * 3600:
        return 60.0
    return 300.0


# ---------------------------------------------------------------------------
# Core: build rate matrix and extract instantaneous phases
# ---------------------------------------------------------------------------

def _build_rate_matrix(
    data: SpikeData,
    bin_size_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin spikes into per-electrode firing rates.

    Returns
    -------
    rates : (n_electrodes, n_bins) float64
    bin_edges : (n_bins + 1,) float64
    """
    t0, t1 = data.time_range
    bin_edges = np.arange(t0, t1 + bin_size_s, bin_size_s)
    n_bins = len(bin_edges) - 1
    if n_bins < 2:
        return np.empty((data.n_electrodes, 0)), bin_edges

    rates = np.zeros((data.n_electrodes, n_bins), dtype=np.float64)
    for e_idx, e_id in enumerate(data.electrode_ids):
        idx = data._electrode_indices.get(e_id)
        if idx is not None and len(idx) > 0:
            rates[e_idx] = np.histogram(data.times[idx], bins=bin_edges)[0]
    return rates, bin_edges


def _extract_phases(rates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hilbert transform -> instantaneous phase for each electrode.

    Mean-centres each row, applies analytic signal, returns angles.
    Silent electrodes (zero variance) get phases = 0.
    """
    n_elec, n_bins = rates.shape
    phases = np.zeros_like(rates)
    for i in range(n_elec):
        row = rates[i]
        if np.std(row) > 1e-12:
            analytic = hilbert(row - np.mean(row))
            phases[i] = np.angle(analytic)
    return phases


# ---------------------------------------------------------------------------
# Kuramoto order parameter
# ---------------------------------------------------------------------------

def _kuramoto_r_timeseries(
    phases: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute instantaneous Kuramoto order parameter R(t).

    R(t) = | (1/N) sum_j exp(i * phi_j(t)) |

    Fully vectorised -- no Python loops over time.
    """
    # phases shape: (n_electrodes, n_bins)
    return np.abs(np.mean(np.exp(1j * phases), axis=0))


def _window_average_r(
    R_t: NDArray[np.float64],
    bins_per_window: int,
) -> NDArray[np.float64]:
    """Average R(t) within non-overlapping windows.

    Uses reshape trick for zero-copy windowing; truncates trailing bins.
    """
    n_windows = len(R_t) // bins_per_window
    if n_windows < 1:
        return np.array([float(np.mean(R_t))])
    truncated = R_t[: n_windows * bins_per_window]
    return truncated.reshape(n_windows, bins_per_window).mean(axis=1)


# ---------------------------------------------------------------------------
# Null model: phase-shuffled surrogates
# ---------------------------------------------------------------------------

def _surrogate_metastability(
    phases: NDArray[np.float64],
    bins_per_window: int,
    n_surrogates: int = N_SURROGATES,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Compute metastability index for phase-shuffled surrogates.

    For each surrogate, independently circularly shifts each electrode's
    phase time series by a random offset, then recomputes var(R_window).
    This preserves autocorrelation within each channel but destroys
    inter-channel synchronisation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_elec, n_bins = phases.shape
    meta_surr = np.empty(n_surrogates, dtype=np.float64)

    for s in range(n_surrogates):
        shifts = rng.integers(0, n_bins, size=n_elec)
        shuffled = np.empty_like(phases)
        for e in range(n_elec):
            shuffled[e] = np.roll(phases[e], shifts[e])
        R_surr = _kuramoto_r_timeseries(shuffled)
        R_win = _window_average_r(R_surr, bins_per_window)
        meta_surr[s] = float(np.var(R_win))

    return meta_surr


def compute_kuramoto(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    window_size_s: Optional[float] = None,
    run_null_model: bool = True,
) -> KuramotoResult:
    """Compute Kuramoto order parameter with optional null-model statistics.

    Parameters
    ----------
    data : SpikeData
    bin_size_ms : float
        Bin width for spike histograms (auto-grows for long recordings).
    window_size_s : float or None
        Window for averaging R(t). None = auto-select.
    run_null_model : bool
        If True, run surrogate analysis for significance testing.
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    actual_bin_ms = bin_size_s * 1000.0

    rates, bin_edges = _build_rate_matrix(data, bin_size_s)
    n_bins = rates.shape[1]

    if n_bins < 4:
        return KuramotoResult(
            R_mean=0.0, R_std=0.0, R_min=0.0, R_max=0.0,
            metastability_index=0.0, synchronisation_level="insufficient_data",
            R_timeseries=[], time_axis_s=[], bin_size_ms=actual_bin_ms,
            window_size_s=0.0, n_windows=0,
        )

    phases = _extract_phases(rates)
    R_t = _kuramoto_r_timeseries(phases)

    if window_size_s is None:
        window_size_s = _adaptive_window_size(data.duration)
    bins_per_window = max(1, int(round(window_size_s / bin_size_s)))

    R_win = _window_average_r(R_t, bins_per_window)
    n_windows = len(R_win)

    mean_R = float(np.mean(R_win))
    std_R = float(np.std(R_win))
    metastability = float(np.var(R_win))

    # Null model
    null_mean, null_std, p_value = 0.0, 0.0, 1.0
    if run_null_model and n_bins >= 20:
        surr_meta = _surrogate_metastability(phases, bins_per_window)
        null_mean = float(np.mean(surr_meta))
        null_std = float(np.std(surr_meta))
        # One-sided p-value: fraction of surrogates >= observed
        p_value = float(np.mean(surr_meta >= metastability))

    # Sync level label
    if mean_R > 0.7:
        sync_level = "high"
    elif mean_R > 0.4:
        sync_level = "moderate"
    else:
        sync_level = "low"

    # Down-sample timeseries for JSON output
    ts_len = min(n_windows, MAX_WINDOWS_TIMESERIES)
    if n_windows > MAX_WINDOWS_TIMESERIES:
        idx = np.linspace(0, n_windows - 1, ts_len, dtype=int)
        R_out = R_win[idx].tolist()
    else:
        R_out = R_win.tolist()

    t0 = data.time_range[0]
    time_axis = [(t0 + (i + 0.5) * window_size_s) for i in range(ts_len)]

    return KuramotoResult(
        R_mean=mean_R,
        R_std=std_R,
        R_min=float(np.min(R_win)),
        R_max=float(np.max(R_win)),
        metastability_index=metastability,
        synchronisation_level=sync_level,
        R_timeseries=R_out,
        time_axis_s=time_axis,
        bin_size_ms=actual_bin_ms,
        window_size_s=window_size_s,
        n_windows=n_windows,
        null_mean=null_mean,
        null_std=null_std,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Coalition entropy
# ---------------------------------------------------------------------------

def compute_coalition_entropy(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    window_size_s: Optional[float] = None,
    n_coalitions: int = 8,
) -> CoalitionResult:
    """Coalition entropy -- diversity of synchronisation patterns.

    For each window, compute the binarised synchronisation matrix
    (which electrode pairs are in phase), then cluster windows to find
    recurrent coalitions.  Coalition entropy = Shannon entropy of the
    cluster-frequency distribution.

    Uses MiniBatchKMeans for scalability.
    """
    from sklearn.cluster import MiniBatchKMeans

    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_elec, n_bins = rates.shape

    if n_bins < 10 or n_elec < 3:
        return CoalitionResult(
            coalition_entropy=0.0, n_coalitions=0,
            coalition_sizes=[], coalition_labels=[],
        )

    phases = _extract_phases(rates)

    if window_size_s is None:
        window_size_s = _adaptive_window_size(data.duration)
    bpw = max(1, int(round(window_size_s / bin_size_s)))
    n_windows = n_bins // bpw

    if n_windows < n_coalitions * 2:
        n_coalitions = max(2, n_windows // 2)

    # Build phase-locking matrix per window -- vectorised
    # Pairwise phase difference (upper triangle indices)
    i_idx, j_idx = np.triu_indices(n_elec, k=1)
    n_pairs = len(i_idx)
    sync_features = np.empty((n_windows, n_pairs), dtype=np.float64)

    for w in range(n_windows):
        s = w * bpw
        e = s + bpw
        # Mean phase-locking value for each pair in this window
        dphase = phases[i_idx, s:e] - phases[j_idx, s:e]
        plv = np.abs(np.mean(np.exp(1j * dphase), axis=1))
        sync_features[w] = plv

    # Cluster synchronisation patterns
    km = MiniBatchKMeans(
        n_clusters=n_coalitions,
        random_state=42,
        batch_size=min(4096, n_windows),
        n_init=3,
    )
    labels = km.fit_predict(sync_features)

    # Coalition entropy
    counts = np.bincount(labels, minlength=n_coalitions)
    freqs = counts / counts.sum()
    coal_entropy = float(scipy_entropy(freqs, base=2))

    return CoalitionResult(
        coalition_entropy=coal_entropy,
        n_coalitions=n_coalitions,
        coalition_sizes=counts.tolist(),
        coalition_labels=labels[:MAX_WINDOWS_TIMESERIES].tolist(),
    )


# ---------------------------------------------------------------------------
# Functional Connectivity Dynamics (FCD)
# ---------------------------------------------------------------------------

def compute_fcd(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    window_size_s: Optional[float] = None,
) -> FCDResult:
    """Functional Connectivity Dynamics matrix (Hansen et al., 2015).

    Computes FC (correlation matrix) in sliding windows, then computes
    FCD as the correlation between vectorised FC at different times.
    Captures how the connectivity pattern changes over the recording.
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_elec, n_bins = rates.shape

    if window_size_s is None:
        window_size_s = _adaptive_window_size(data.duration)
    bpw = max(1, int(round(window_size_s / bin_size_s)))
    n_windows = n_bins // bpw

    # Cap FCD matrix size
    step = max(1, n_windows // FCD_MAX_WINDOWS)
    window_indices = list(range(0, n_windows, step))
    n_fcd = len(window_indices)

    if n_fcd < 3 or n_elec < 3:
        return FCDResult(
            fcd_variance=0.0, fcd_mean=0.0,
            fcd_matrix_shape=(0, 0), speed_of_fc_change=0.0,
        )

    # Upper-triangle indices for vectorising FC matrices
    triu_i, triu_j = np.triu_indices(n_elec, k=1)
    n_pairs = len(triu_i)
    fc_vectors = np.empty((n_fcd, n_pairs), dtype=np.float64)

    for wi, w in enumerate(window_indices):
        s = w * bpw
        e = s + bpw
        chunk = rates[:, s:e]  # (n_elec, bpw)
        # Pearson correlation matrix
        stds = np.std(chunk, axis=1)
        valid = stds > 1e-12
        if np.sum(valid) < 2:
            fc_vectors[wi] = 0.0
            continue
        # Centre rows
        centred = chunk - chunk.mean(axis=1, keepdims=True)
        # Correlation via dot product
        norms = np.sqrt(np.sum(centred ** 2, axis=1, keepdims=True))
        norms = np.where(norms > 1e-12, norms, 1.0)
        normed = centred / norms
        corr = normed @ normed.T
        fc_vectors[wi] = corr[triu_i, triu_j]

    # FCD matrix: correlation between FC vectors at different times
    # Normalise each FC vector
    fc_norms = np.linalg.norm(fc_vectors, axis=1, keepdims=True)
    fc_norms = np.where(fc_norms > 1e-12, fc_norms, 1.0)
    fc_normed = fc_vectors / fc_norms
    fcd_matrix = fc_normed @ fc_normed.T

    # Statistics from upper triangle of FCD
    fcd_upper = fcd_matrix[np.triu_indices(n_fcd, k=1)]
    fcd_variance = float(np.var(fcd_upper)) if len(fcd_upper) > 0 else 0.0
    fcd_mean = float(np.mean(fcd_upper)) if len(fcd_upper) > 0 else 0.0

    # Speed of FC change: mean absolute diff along the main diagonal
    diag_diffs = np.abs(np.diff(np.diag(fcd_matrix, k=1)))
    speed = float(np.mean(diag_diffs)) if len(diag_diffs) > 0 else 0.0

    return FCDResult(
        fcd_variance=fcd_variance,
        fcd_mean=fcd_mean,
        fcd_matrix_shape=(n_fcd, n_fcd),
        speed_of_fc_change=speed,
    )


# ---------------------------------------------------------------------------
# Dwell-time analysis
# ---------------------------------------------------------------------------

def compute_dwell_times(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    window_size_s: Optional[float] = None,
    sync_threshold: float = 0.5,
) -> DwellTimeResult:
    """Dwell times in synchronised vs desynchronised states.

    Thresholds R(t) at `sync_threshold` and computes epoch durations.
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_bins = rates.shape[1]

    if n_bins < 4:
        return DwellTimeResult(
            sync_threshold=sync_threshold,
            mean_sync_dwell_s=0.0, mean_desync_dwell_s=0.0,
            median_sync_dwell_s=0.0, median_desync_dwell_s=0.0,
            n_sync_epochs=0, n_desync_epochs=0,
            fraction_time_synchronised=0.0,
        )

    phases = _extract_phases(rates)
    R_t = _kuramoto_r_timeseries(phases)

    if window_size_s is None:
        window_size_s = _adaptive_window_size(data.duration)
    bpw = max(1, int(round(window_size_s / bin_size_s)))
    R_win = _window_average_r(R_t, bpw)

    is_sync = R_win >= sync_threshold
    changes = np.where(np.diff(is_sync.astype(np.int8)) != 0)[0]
    boundaries = np.concatenate([[0], changes + 1, [len(R_win)]])

    sync_dwells: list[float] = []
    desync_dwells: list[float] = []
    for k in range(len(boundaries) - 1):
        state_sync = bool(is_sync[boundaries[k]])
        length_s = float(boundaries[k + 1] - boundaries[k]) * window_size_s
        if state_sync:
            sync_dwells.append(length_s)
        else:
            desync_dwells.append(length_s)

    total_sync = sum(sync_dwells)
    total_time = float(len(R_win)) * window_size_s

    return DwellTimeResult(
        sync_threshold=sync_threshold,
        mean_sync_dwell_s=float(np.mean(sync_dwells)) if sync_dwells else 0.0,
        mean_desync_dwell_s=float(np.mean(desync_dwells)) if desync_dwells else 0.0,
        median_sync_dwell_s=float(np.median(sync_dwells)) if sync_dwells else 0.0,
        median_desync_dwell_s=float(np.median(desync_dwells)) if desync_dwells else 0.0,
        n_sync_epochs=len(sync_dwells),
        n_desync_epochs=len(desync_dwells),
        fraction_time_synchronised=total_sync / total_time if total_time > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Chimera state detection
# ---------------------------------------------------------------------------

def _electrode_to_mea(electrode_ids: list[int]) -> dict[int, list[int]]:
    """Map MEA index -> list of electrode indices (position in electrode_ids).

    FinalSpark convention: electrodes 0-7 = MEA 0, 8-15 = MEA 1, etc.
    Falls back to groups of FINALSPARK_MEA_SIZE if IDs don't follow convention.
    """
    mea_map: dict[int, list[int]] = {}
    for pos, eid in enumerate(electrode_ids):
        mea_idx = eid // FINALSPARK_MEA_SIZE
        mea_map.setdefault(mea_idx, []).append(pos)
    return mea_map


def compute_chimera(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    window_size_s: Optional[float] = None,
    chimera_threshold: float = 0.3,
) -> ChimeraResult:
    """Detect chimera states -- coexistence of synchronised and desynchronised subpopulations.

    For each window, compute local order parameter per MEA.
    A chimera state exists when some MEAs have high R while others have low R.
    Chimera index = std(R_local) / mean(R_local) -- coefficient of variation.
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_elec, n_bins = rates.shape

    mea_map = _electrode_to_mea(data.electrode_ids)
    n_meas = len(mea_map)

    if n_bins < 4 or n_meas < 2:
        return ChimeraResult(
            chimera_index=0.0, n_chimera_epochs=0, fraction_chimera=0.0,
            local_order_params={},
        )

    phases = _extract_phases(rates)

    if window_size_s is None:
        window_size_s = _adaptive_window_size(data.duration)
    bpw = max(1, int(round(window_size_s / bin_size_s)))
    n_windows = n_bins // bpw

    if n_windows < 1:
        return ChimeraResult(
            chimera_index=0.0, n_chimera_epochs=0, fraction_chimera=0.0,
            local_order_params={},
        )

    # Local R per MEA per window
    local_R = np.zeros((n_meas, n_windows), dtype=np.float64)
    mea_keys = sorted(mea_map.keys())

    for mi, mea_idx in enumerate(mea_keys):
        elec_positions = mea_map[mea_idx]
        if len(elec_positions) < 2:
            continue
        mea_phases = phases[elec_positions]  # (n_elec_in_mea, n_bins)
        R_mea = np.abs(np.mean(np.exp(1j * mea_phases), axis=0))
        R_win_mea = _window_average_r(R_mea, bpw)
        local_R[mi, : len(R_win_mea)] = R_win_mea

    # Chimera index per window: CV of local R values
    mean_local = np.mean(local_R, axis=0)
    std_local = np.std(local_R, axis=0)
    cv = np.where(mean_local > 1e-6, std_local / mean_local, 0.0)

    chimera_detected = cv > chimera_threshold
    chimera_index = float(np.mean(cv))
    n_chimera = int(np.sum(chimera_detected))

    # Per-MEA mean R across all windows
    local_means = {
        f"MEA_{mea_keys[mi]}": float(np.mean(local_R[mi]))
        for mi in range(n_meas)
    }

    return ChimeraResult(
        chimera_index=chimera_index,
        n_chimera_epochs=n_chimera,
        fraction_chimera=n_chimera / n_windows if n_windows > 0 else 0.0,
        local_order_params=local_means,
    )


# ---------------------------------------------------------------------------
# Integration-segregation dynamics
# ---------------------------------------------------------------------------

def compute_integration_segregation(
    data: SpikeData,
    bin_size_ms: float = 20.0,
) -> IntegrationSegregationResult:
    """Integration-segregation balance (Tononi, Sporns & Edelman, 1994).

    Uses the covariance structure of electrode firing rates.
    Integration ~ mutual information among all electrodes.
    Segregation ~ sum of individual channel entropies.
    Complexity ~ integration when the system has maximum segregation.
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_elec, n_bins = rates.shape

    if n_bins < 10 or n_elec < 2:
        return IntegrationSegregationResult(
            integration=0.0, segregation=0.0, complexity=0.0, balance_ratio=0.0,
        )

    # Covariance matrix of electrode rates
    cov = np.cov(rates)  # (n_elec, n_elec)

    # Entropy of multivariate Gaussian: 0.5 * ln(det(2*pi*e*Sigma))
    # For numerical stability, use log-determinant
    sign, logdet_full = np.linalg.slogdet(cov + np.eye(n_elec) * 1e-10)
    if sign <= 0:
        logdet_full = 0.0

    # Integration: sum of individual entropies - joint entropy
    # Individual entropy_i = 0.5 * ln(2*pi*e*var_i)
    variances = np.diag(cov)
    variances = np.maximum(variances, 1e-10)
    sum_individual_logdet = float(np.sum(np.log(variances)))

    # Integration I = sum(H_i) - H_joint
    # In Gaussian case: I = 0.5 * (sum(log(var_i)) - log(det(Sigma)))
    integration = 0.5 * (sum_individual_logdet - logdet_full)
    integration = max(0.0, float(integration))

    # Segregation: average individual entropy
    segregation = 0.5 * sum_individual_logdet / n_elec
    segregation = max(0.0, float(segregation))

    # Neural complexity (Tononi et al.): sum over all bipartitions
    # Approximation: integration * (1 - integration/max_integration)
    max_possible = 0.5 * n_elec * np.log(np.max(variances) + 1e-10)
    if max_possible > 0:
        complexity = float(integration * (1.0 - integration / max_possible))
    else:
        complexity = 0.0
    complexity = max(0.0, complexity)

    total = integration + segregation
    balance = integration / total if total > 0 else 0.0

    return IntegrationSegregationResult(
        integration=integration,
        segregation=segregation,
        complexity=complexity,
        balance_ratio=float(balance),
    )


# ---------------------------------------------------------------------------
# Multi-scale analysis
# ---------------------------------------------------------------------------

def compute_multiscale(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    scales: tuple[float, ...] = MULTISCALE_WINDOWS,
) -> MultiScaleResult:
    """Compute Kuramoto metastability at multiple temporal scales.

    Runs Kuramoto analysis at each window size in `scales` (1s, 10s, 60s).
    Null model is skipped at multi-scale to stay within time budget.
    """
    results: dict[str, KuramotoResult] = {}
    for ws in scales:
        if data.duration < ws * 3:
            continue
        kr = compute_kuramoto(
            data, bin_size_ms=bin_size_ms,
            window_size_s=ws, run_null_model=False,
        )
        results[f"{ws}s"] = kr

    return MultiScaleResult(scales=results)


# ---------------------------------------------------------------------------
# Per-MEA decomposition
# ---------------------------------------------------------------------------

def compute_mea_decomposition(
    data: SpikeData,
    bin_size_ms: float = 20.0,
) -> MEADecompositionResult:
    """Compare within-MEA vs across-MEA metastability.

    Answers: is metastability generated locally within MEAs or by
    inter-MEA interactions?
    """
    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_elec, n_bins = rates.shape

    mea_map = _electrode_to_mea(data.electrode_ids)
    window_size_s = _adaptive_window_size(data.duration)
    bpw = max(1, int(round(window_size_s / bin_size_s)))

    if n_bins < 4 or len(mea_map) < 2:
        return MEADecompositionResult(
            within_mea_metastability={}, across_mea_metastability=0.0,
            within_mean=0.0, ratio=0.0,
        )

    phases = _extract_phases(rates)
    mea_keys = sorted(mea_map.keys())

    # Within-MEA metastability
    within_meta: dict[str, float] = {}
    for mea_idx in mea_keys:
        elec_pos = mea_map[mea_idx]
        if len(elec_pos) < 2:
            within_meta[f"MEA_{mea_idx}"] = 0.0
            continue
        mea_phases = phases[elec_pos]
        R_mea = _kuramoto_r_timeseries(mea_phases)
        R_win = _window_average_r(R_mea, bpw)
        within_meta[f"MEA_{mea_idx}"] = float(np.var(R_win))

    # Across-MEA: use one representative per MEA (mean phase)
    n_meas = len(mea_keys)
    mea_mean_phases = np.zeros((n_meas, n_bins), dtype=np.float64)
    for mi, mea_idx in enumerate(mea_keys):
        elec_pos = mea_map[mea_idx]
        # Circular mean of phases within MEA
        mean_vec = np.mean(np.exp(1j * phases[elec_pos]), axis=0)
        mea_mean_phases[mi] = np.angle(mean_vec)

    R_across = _kuramoto_r_timeseries(mea_mean_phases)
    R_across_win = _window_average_r(R_across, bpw)
    across_meta = float(np.var(R_across_win))

    within_vals = [v for v in within_meta.values() if v > 0]
    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    ratio = across_meta / within_mean if within_mean > 1e-12 else 0.0

    return MEADecompositionResult(
        within_mea_metastability=within_meta,
        across_mea_metastability=across_meta,
        within_mean=within_mean,
        ratio=ratio,
    )


# ---------------------------------------------------------------------------
# State transitions (KMeans clustering)
# ---------------------------------------------------------------------------

def compute_state_transitions(
    data: SpikeData,
    n_states: int = 4,
    bin_size_ms: float = 50.0,
) -> StateTransitionResult:
    """Cluster network activity into discrete states and compute transitions.

    Uses MiniBatchKMeans for datasets with >20k bins.
    """
    from sklearn.cluster import MiniBatchKMeans, KMeans

    bin_size_s = _adaptive_bin_size(data.duration, bin_size_ms)
    actual_bin_ms = bin_size_s * 1000.0

    rates, _ = _build_rate_matrix(data, bin_size_s)
    n_bins = rates.shape[1]
    rates_T = rates.T  # (n_bins, n_electrodes)

    if n_bins < n_states * 2:
        return StateTransitionResult(
            n_states=0, transition_matrix=[], state_labels=[],
            dwell_times_ms={}, mean_dwell_ms={}, n_transitions=0,
            bin_size_ms=actual_bin_ms,
        )

    n_states = min(n_states, n_bins // 2)

    if n_bins > 20_000:
        km = MiniBatchKMeans(
            n_clusters=n_states, random_state=42,
            batch_size=min(4096, n_bins), n_init=3,
        )
    else:
        km = KMeans(n_clusters=n_states, random_state=42, n_init=10)

    labels = km.fit_predict(rates_T)

    # Transition matrix -- vectorised
    trans = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(trans, (labels[:-1], labels[1:]), 1)
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(
        trans, row_sums, where=row_sums > 0, out=np.zeros_like(trans),
    )

    # Dwell times -- vectorised via np.diff
    changes = np.where(np.diff(labels) != 0)[0]
    boundaries = np.concatenate([[0], changes + 1, [len(labels)]])
    dwell_times: dict[str, list[float]] = {str(s): [] for s in range(n_states)}

    for k in range(len(boundaries) - 1):
        state = int(labels[boundaries[k]])
        length_ms = float(boundaries[k + 1] - boundaries[k]) * actual_bin_ms
        dwell_times[str(state)].append(length_ms)

    mean_dwell = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in dwell_times.items()
    }

    return StateTransitionResult(
        n_states=n_states,
        transition_matrix=trans_prob.tolist(),
        state_labels=labels[:MAX_WINDOWS_TIMESERIES].tolist(),
        dwell_times_ms=dwell_times,
        mean_dwell_ms=mean_dwell,
        n_transitions=int(np.sum(np.diff(labels) != 0)),
        bin_size_ms=actual_bin_ms,
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def analyze_metastability(data: SpikeData) -> dict:
    """Full metastability analysis -- single entry point.

    Runs all sub-analyses and returns a unified dict suitable for
    JSON serialisation.  Designed to complete in <60 s on 2.6 M spikes,
    32 electrodes, 118 hours.
    """
    import time as _time
    t0 = _time.perf_counter()

    logger.info(
        "Starting metastability analysis: %d spikes, %d electrodes, %.1f h",
        data.n_spikes, data.n_electrodes, data.duration / 3600,
    )

    kuramoto = compute_kuramoto(data, run_null_model=True)
    coalition = compute_coalition_entropy(data)
    fcd = compute_fcd(data)
    dwell = compute_dwell_times(data)
    chimera = compute_chimera(data)
    int_seg = compute_integration_segregation(data)
    multiscale = compute_multiscale(data)
    mea_decomp = compute_mea_decomposition(data)
    state_trans = compute_state_transitions(data)

    elapsed = _time.perf_counter() - t0
    logger.info("Metastability analysis completed in %.2f s", elapsed)

    # Determine if system is genuinely metastable:
    # metastability index > null model + 2 sigma AND p < 0.05
    is_metastable = (
        kuramoto.metastability_index > kuramoto.null_mean + 2 * kuramoto.null_std
        and kuramoto.p_value < 0.05
    )

    report = MetastabilityReport(
        kuramoto=kuramoto,
        coalition=coalition,
        fcd=fcd,
        dwell_times=dwell,
        chimera=chimera,
        integration_segregation=int_seg,
        multiscale=multiscale,
        mea_decomposition=mea_decomp,
        state_transitions=state_trans,
        is_metastable=is_metastable,
        computation_time_s=elapsed,
    )

    return _report_to_dict(report)


def _report_to_dict(report: MetastabilityReport) -> dict:
    """Convert MetastabilityReport to a JSON-serialisable dict."""
    from dataclasses import asdict

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(item) for item in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    raw = asdict(report)

    # Flatten multiscale: convert KuramotoResult dicts
    if "multiscale" in raw and "scales" in raw["multiscale"]:
        raw["multiscale"] = {
            k: _convert(v) for k, v in raw["multiscale"]["scales"].items()
        }

    return _convert(raw)
