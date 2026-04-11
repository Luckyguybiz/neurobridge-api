"""NeuroBridge API — Backend for biocomputing data analysis.

FastAPI server that provides endpoints for:
- Data upload and management
- Spike detection and sorting
- Burst detection
- Functional connectivity analysis
- Comprehensive statistics
- Export (CSV, JSON)
"""

import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

from analysis.loader import SpikeData, load_file
from analysis.spikes import compute_isi, compute_firing_rates, compute_amplitude_stats, sort_spikes
from analysis.bursts import detect_bursts, compute_burst_profiles, detect_single_channel_bursts
from analysis.connectivity import compute_cross_correlation, compute_connectivity_graph, compute_transfer_entropy
from analysis.stats import compute_full_summary, compute_temporal_dynamics, compute_quality_metrics
from analysis.information_theory import compute_spike_train_entropy, compute_mutual_information, compute_lempel_ziv_complexity
from analysis.spectral import compute_power_spectrum, compute_coherence
from analysis.criticality import detect_avalanches
from analysis.digital_twin import fit_lif_parameters, simulate_lif_network, compare_real_vs_simulated
from analysis.ml_pipeline import detect_anomalies, classify_states, compute_pca_embedding, extract_features
from analysis.report import generate_full_report
from analysis.plasticity import compute_stdp_matrix, detect_learning_episodes
from analysis.organoid_iq import compute_organoid_iq
from analysis.predictions import predict_firing_rates, predict_burst_probability, estimate_organoid_health
from analysis.replay import detect_replay, detect_sequence_replay
from analysis.reservoir import estimate_memory_capacity, benchmark_nonlinear_computation
from analysis.fingerprint import compute_fingerprint
from analysis.sonification import generate_sonification, compute_rhythmic_analysis
from analysis.emergence import compute_integrated_information
from analysis.attractors import map_attractor_landscape, compute_state_space_geometry
from analysis.phase_transitions import detect_phase_transitions
from analysis.predictive_coding import measure_predictive_coding
from analysis.weight_inference import infer_synaptic_weights, track_weight_changes
from analysis.multiscale import compute_multiscale_complexity
from analysis.sleep_wake import analyze_sleep_wake
from analysis.habituation import detect_repeated_patterns
from analysis.metastability import analyze_metastability
from analysis.information_flow import compute_granger_causality
from analysis.network_motifs import enumerate_motifs
from analysis.energy_landscape import fit_ising_model
from analysis.stimulus_design import evolve_protocol
from analysis.consciousness_metrics import compute_consciousness_score
from analysis.comparative import compare_with_references
from analysis.protocol_library import list_protocols, get_protocol, suggest_protocol
from analysis.closed_loop import run_dishbrain_session, run_cartpole_benchmark, compare_reward_strategies
from analysis.curriculum import run_curriculum, get_current_stage, simulate_stage
from analysis.memory_tests import run_memory_battery, test_working_memory, test_short_term_memory, test_long_term_memory, test_associative_memory
from analysis.pong_engine import simulate_pong, encode_ball_state
from analysis.xor_benchmark import run_full_benchmark, run_gate_benchmark, compute_xor_difficulty
from analysis.japanese_vowels import generate_synthetic_vowels, build_reservoir, reservoir_transform, train_linear_readout
from analysis.experiment_tracker import start_experiment, end_experiment, get_experiment as tracker_get_experiment, get_history as get_experiment_history, compute_delta as tracker_compute_delta, clear_experiments as tracker_clear_experiments
from analysis.publication_generator import generate_draft, generate_abstract_only, generate_methods_section
from analysis.ethical_assessment import assess_ethics, assess_consciousness_indicators, compute_sentience_risk_score
from analysis.grant_matcher import match_grants, get_grant_details, list_grants
from analysis.turing_test import run_turing_test
from analysis.neural_architecture_search import search_optimal_protocol
from analysis.hybrid_ai import benchmark_hybrid
from analysis.genetic_programming import evolve_programs
from analysis.homeostatic_plasticity import monitor_homeostasis
from analysis.suffering_detector import detect_suffering
from analysis.welfare_report import generate_welfare_report
from analysis.swarm_organoid import simulate_swarm
from analysis.morphological_computing import analyze_morphological_computation
from analysis.catastrophic_forgetting import measure_forgetting, compute_retention_curve
from analysis.transfer_learning import measure_transfer, compute_representational_similarity
from analysis.functional_connectome import build_full_connectome, detect_communities, compute_graph_theory_metrics
from analysis.effective_connectivity import estimate_effective_connectivity, compute_causal_hierarchy
from analysis.consolidation import detect_consolidation_events, measure_retention
from analysis.multi_bit_memory import estimate_channel_capacity, measure_population_code_diversity
from analysis.topology import compute_betti_numbers, compute_topological_complexity
from analysis.multi_organoid import split_into_organoids, compare_organoids
from analysis.temporal_evolution import track_evolution, detect_trends, find_critical_moments
from analysis.stim_response import detect_response, compute_dose_response, estimate_stim_times
from analysis.llm_optimizer import generate_optimization_prompt, parse_llm_suggestion, run_optimization_loop
from models.schemas import (
    SpikeDetectionParams, BurstDetectionParams, SpikeSortingParams,
    ConnectivityParams, TimeRangeFilter, DatasetInfo,
)

# Directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

import math

app = FastAPI(
    title="NeuroBridge API",
    description="Backend for biocomputing data analysis — spike detection, burst analysis, connectivity mapping",
    version="0.2.0",
)


# Global exception handler — catch inf/nan JSON errors
from fastapi.responses import JSONResponse as _JSONResponse
import json as _json_mod

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    if "Out of range float" in str(exc) or "inf" in str(exc).lower():
        return _JSONResponse(
            status_code=200,
            content={"error": "Result contained infinity values (replaced with 0)", "partial": True},
        )
    return _JSONResponse(status_code=500, content={"detail": str(exc)})


# ─── Analysis cache (LRU, max 100 results, keyed by dataset_id + analysis_name) ───
from collections import OrderedDict

_analysis_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 100

def _cache_key(dataset_id: str, analysis: str) -> str:
    return f"{dataset_id}:{analysis}"

def _cache_get(dataset_id: str, analysis: str):
    key = _cache_key(dataset_id, analysis)
    if key in _analysis_cache:
        _analysis_cache.move_to_end(key)
        return _analysis_cache[key]
    return None

def _cache_set(dataset_id: str, analysis: str, result: dict):
    key = _cache_key(dataset_id, analysis)
    _analysis_cache[key] = result
    _analysis_cache.move_to_end(key)
    while len(_analysis_cache) > _CACHE_MAX:
        _analysis_cache.popitem(last=False)


# Fix numpy serialization
import json as _json

class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            v = float(obj)
            if np.isinf(v) or np.isnan(v):
                return 0.0
            return v
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from fastapi.responses import ORJSONResponse

def _sanitize(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isinf(v) or np.isnan(v):
            return 0.0
        return v
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neurocomputers.io",
        "https://www.neurocomputers.io",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataset store (production would use Redis/DB)
datasets: dict[str, SpikeData] = {}


# ═══════════ HEALTH ═══════════

@app.get("/health")
async def health():
    return {"status": "ok", "datasets_loaded": len(datasets), "timestamp": datetime.now(timezone.utc).isoformat()}


# ═══════════ DATA MANAGEMENT ═══════════

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    sampling_rate: float = Query(30000.0, description="Sampling rate in Hz"),
):
    """Upload a spike data file (CSV, HDF5, Parquet, JSON)."""
    dataset_id = str(uuid.uuid4())[:8]
    filepath = UPLOAD_DIR / f"{dataset_id}_{file.filename}"

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    try:
        t0 = time.time()
        data = load_file(str(filepath), sampling_rate=sampling_rate)
        load_time = (time.time() - t0) * 1000

        datasets[dataset_id] = data

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "n_spikes": data.n_spikes,
            "n_electrodes": data.n_electrodes,
            "duration_s": round(data.duration, 3),
            "time_range": data.time_range,
            "electrode_ids": data.electrode_ids,
            "file_size_mb": round(len(content) / 1024 / 1024, 2),
            "load_time_ms": round(load_time, 1),
            "metadata": data.metadata,
        }
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@app.post("/api/generate")
async def generate_synthetic(
    duration: float = Query(30.0, ge=1.0, le=600.0),
    n_electrodes: int = Query(8, ge=1, le=32),
    base_rate_min: float = Query(2.0),
    base_rate_max: float = Query(15.0),
    burst_probability: float = Query(0.15, ge=0.0, le=1.0),
):
    """Generate synthetic spike data for testing."""
    dataset_id = str(uuid.uuid4())[:8]

    # Generate using Poisson process + bursts
    spikes_times, spikes_electrodes, spikes_amplitudes = [], [], []
    base_rates = np.random.uniform(base_rate_min, base_rate_max, n_electrodes)

    for e in range(n_electrodes):
        t = np.random.exponential(1.0 / base_rates[e])
        while t < duration:
            spikes_times.append(t)
            spikes_electrodes.append(e)
            spikes_amplitudes.append(-(60 + np.random.random() * 140))
            t += np.random.exponential(1.0 / base_rates[e])

    # Inject bursts
    n_bursts = int(duration * burst_probability)
    for _ in range(n_bursts):
        burst_time = np.random.random() * duration
        burst_electrodes = np.random.choice(n_electrodes, size=min(3 + int(np.random.random() * (n_electrodes - 3)), n_electrodes), replace=False)
        for e in burst_electrodes:
            for _ in range(3 + int(np.random.random() * 5)):
                t = burst_time + (np.random.random() - 0.5) * 0.05
                if 0 <= t < duration:
                    spikes_times.append(t)
                    spikes_electrodes.append(int(e))
                    spikes_amplitudes.append(-(100 + np.random.random() * 100))

    data = SpikeData(
        times=np.array(spikes_times),
        electrodes=np.array(spikes_electrodes, dtype=np.int32),
        amplitudes=np.array(spikes_amplitudes),
        sampling_rate=30000.0,
        metadata={"source": "synthetic", "duration": duration, "n_electrodes": n_electrodes},
    )
    datasets[dataset_id] = data

    return {
        "dataset_id": dataset_id,
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(data.duration, 3),
    }


@app.get("/api/datasets")
async def list_datasets():
    """List all loaded datasets."""
    return {
        dataset_id: {
            "n_spikes": data.n_spikes,
            "n_electrodes": data.n_electrodes,
            "duration_s": round(data.duration, 3),
            "metadata": data.metadata,
        }
        for dataset_id, data in datasets.items()
    }


@app.get("/api/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed info about a dataset."""
    data = _get_dataset(dataset_id)
    return data.to_dict()


@app.get("/api/datasets/{dataset_id}/spikes")
async def get_spikes(
    dataset_id: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    electrodes: Optional[str] = None,
    limit: int = Query(10000, le=100000),
):
    """Get spike data with optional filtering."""
    data = _get_dataset(dataset_id)

    el_list = [int(e) for e in electrodes.split(",")] if electrodes else None
    filtered = data.get_filtered(electrodes=el_list, start=start, end=end)

    if filtered.n_spikes > limit:
        # Downsample
        idx = np.linspace(0, filtered.n_spikes - 1, limit, dtype=int)
        return {
            "times": filtered.times[idx].tolist(),
            "electrodes": filtered.electrodes[idx].tolist(),
            "amplitudes": filtered.amplitudes[idx].tolist(),
            "n_total": filtered.n_spikes,
            "n_returned": limit,
            "downsampled": True,
        }

    return {
        "times": filtered.times.tolist(),
        "electrodes": filtered.electrodes.tolist(),
        "amplitudes": filtered.amplitudes.tolist(),
        "n_total": filtered.n_spikes,
        "n_returned": filtered.n_spikes,
        "downsampled": False,
    }


# ═══════════ ANALYSIS ENDPOINTS ═══════════

@app.get("/api/analysis/{dataset_id}/summary")
async def analyze_summary(dataset_id: str):
    """Full dataset summary — per-electrode stats, population metrics, quality."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_full_summary(data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/api/analysis/{dataset_id}/quality")
async def analyze_quality(dataset_id: str):
    """Data quality assessment — SNR, violations, gaps, issues."""
    data = _get_dataset(dataset_id)
    return compute_quality_metrics(data)


@app.get("/api/analysis/{dataset_id}/firing-rates")
async def analyze_firing_rates(
    dataset_id: str,
    bin_size: float = Query(1.0, ge=0.1, le=60.0, description="Bin size in seconds"),
):
    """Compute time-binned firing rates per electrode."""
    data = _get_dataset(dataset_id)
    return compute_firing_rates(data, bin_size_sec=bin_size)


@app.get("/api/analysis/{dataset_id}/isi")
async def analyze_isi(
    dataset_id: str,
    electrode: Optional[int] = None,
):
    """Inter-spike interval analysis."""
    data = _get_dataset(dataset_id)
    return compute_isi(data, electrode=electrode)


@app.get("/api/analysis/{dataset_id}/amplitudes")
async def analyze_amplitudes(dataset_id: str):
    """Amplitude distribution statistics per electrode."""
    data = _get_dataset(dataset_id)
    return compute_amplitude_stats(data)


@app.get("/api/analysis/{dataset_id}/temporal")
async def analyze_temporal(
    dataset_id: str,
    bin_size: float = Query(60.0, ge=1.0, le=3600.0),
):
    """Temporal dynamics — trends, stationarity, Fano factors."""
    data = _get_dataset(dataset_id)
    return compute_temporal_dynamics(data, bin_size_sec=bin_size)


@app.post("/api/analysis/{dataset_id}/spike-sorting")
async def analyze_spike_sorting(
    dataset_id: str,
    params: SpikeSortingParams = SpikeSortingParams(),
):
    """Spike sorting — cluster spikes by waveform shape."""
    data = _get_dataset(dataset_id)
    if data.waveforms is None or len(data.waveforms) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no waveform data. Upload data with waveforms for spike sorting.")
    return sort_spikes(data.waveforms, method=params.method, n_components=params.n_components, n_clusters=params.n_clusters, min_cluster_size=params.min_cluster_size)


@app.get("/api/analysis/{dataset_id}/bursts")
async def analyze_bursts(
    dataset_id: str,
    min_electrodes: int = Query(3, ge=2),
    window_ms: float = Query(50.0, ge=10.0),
    min_spikes: int = Query(2, ge=1),
):
    """Network burst detection — synchronized multi-electrode firing."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = detect_bursts(data, min_electrodes=min_electrodes, window_ms=window_ms, min_spikes_per_electrode=min_spikes)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/api/analysis/{dataset_id}/bursts/profiles")
async def analyze_burst_profiles(
    dataset_id: str,
    min_electrodes: int = Query(3, ge=2),
    window_ms: float = Query(50.0),
):
    """Detailed burst profiles — recruitment order, temporal shape."""
    data = _get_dataset(dataset_id)
    burst_result = detect_bursts(data, min_electrodes=min_electrodes, window_ms=window_ms)
    return compute_burst_profiles(data, burst_result["bursts"])


@app.get("/api/analysis/{dataset_id}/bursts/single-channel")
async def analyze_single_channel_bursts(
    dataset_id: str,
    electrode: int = Query(..., ge=0),
    max_isi_ms: float = Query(100.0),
    min_spikes: int = Query(3),
):
    """Single-channel burst detection using ISI method."""
    data = _get_dataset(dataset_id)
    spike_times = data.times[data.electrodes == electrode]
    bursts = detect_single_channel_bursts(spike_times, max_isi_ms=max_isi_ms, min_spikes=min_spikes)
    return {"electrode": electrode, "bursts": bursts, "n_bursts": len(bursts)}


@app.get("/api/analysis/{dataset_id}/connectivity")
async def analyze_connectivity(
    dataset_id: str,
    window_ms: float = Query(10.0, ge=1.0),
    min_strength: float = Query(0.02, ge=0.0),
):
    """Functional connectivity graph — co-firing analysis."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_connectivity_graph(data, coincidence_window_ms=window_ms, min_strength=min_strength)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/api/analysis/{dataset_id}/cross-correlation")
async def analyze_cross_correlation(
    dataset_id: str,
    max_lag_ms: float = Query(50.0),
    bin_size_ms: float = Query(1.0),
):
    """Pairwise cross-correlograms between all electrodes."""
    data = _get_dataset(dataset_id)
    return compute_cross_correlation(data, max_lag_ms=max_lag_ms, bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/transfer-entropy")
async def analyze_transfer_entropy(
    dataset_id: str,
    bin_size_ms: float = Query(5.0),
    history_bins: int = Query(5, ge=1, le=20),
):
    """Transfer entropy — directional information flow between electrodes."""
    data = _get_dataset(dataset_id)
    return compute_transfer_entropy(data, bin_size_ms=bin_size_ms, history_bins=history_bins)


# ═══════════ INFORMATION THEORY ═══════════

@app.get("/api/analysis/{dataset_id}/entropy")
async def analyze_entropy(dataset_id: str, bin_size_ms: float = Query(10.0)):
    """Shannon entropy of spike trains per electrode."""
    return compute_spike_train_entropy(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/mutual-information")
async def analyze_mutual_info(dataset_id: str, bin_size_ms: float = Query(10.0)):
    """Pairwise mutual information between electrodes."""
    return compute_mutual_information(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/complexity")
async def analyze_complexity(dataset_id: str, bin_size_ms: float = Query(5.0)):
    """Lempel-Ziv complexity of spike trains."""
    return compute_lempel_ziv_complexity(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


# ═══════════ SPECTRAL ═══════════

@app.get("/api/analysis/{dataset_id}/power-spectrum")
async def analyze_power_spectrum(dataset_id: str, bin_size_ms: float = Query(1.0)):
    """Power spectral density and frequency band analysis."""
    return compute_power_spectrum(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/coherence")
async def analyze_coherence(dataset_id: str, bin_size_ms: float = Query(1.0)):
    """Spectral coherence between electrode pairs."""
    return compute_coherence(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


# ═══════════ CRITICALITY ═══════════

@app.get("/api/analysis/{dataset_id}/avalanches")
async def analyze_avalanches(dataset_id: str, bin_size_ms: float = Query(5.0)):
    """Neuronal avalanche detection and criticality assessment."""
    return detect_avalanches(_get_dataset(dataset_id), bin_size_ms=bin_size_ms)


# ═══════════ DIGITAL TWIN ═══════════

@app.get("/api/analysis/{dataset_id}/digital-twin/fit")
async def fit_twin(dataset_id: str):
    """Fit LIF neuron model parameters from recorded data."""
    return fit_lif_parameters(_get_dataset(dataset_id))


@app.post("/api/analysis/{dataset_id}/digital-twin/simulate")
async def simulate_twin(dataset_id: str, duration_ms: float = Query(5000.0)):
    """Simulate digital twin and compare with real data."""
    data = _get_dataset(dataset_id)
    params = fit_lif_parameters(data)
    sim = simulate_lif_network(params, duration_ms=duration_ms)
    comparison = compare_real_vs_simulated(data, sim)
    return {"simulation": sim, "comparison": comparison, "parameters": params}


# ═══════════ ML PIPELINE ═══════════

@app.get("/api/analysis/{dataset_id}/anomalies")
async def analyze_anomalies(dataset_id: str, window_sec: float = Query(1.0)):
    """Anomaly detection using Isolation Forest."""
    return detect_anomalies(_get_dataset(dataset_id), window_sec=window_sec)


@app.get("/api/analysis/{dataset_id}/states")
async def analyze_states(dataset_id: str, window_sec: float = Query(2.0)):
    """Neural activity state classification (resting, active, bursting)."""
    return classify_states(_get_dataset(dataset_id), window_sec=window_sec)


@app.get("/api/analysis/{dataset_id}/pca")
async def analyze_pca(dataset_id: str, n_components: int = Query(3)):
    """PCA embedding of neural state space."""
    return compute_pca_embedding(_get_dataset(dataset_id), n_components=n_components)


@app.get("/api/analysis/{dataset_id}/features")
async def analyze_features(dataset_id: str, window_sec: float = Query(1.0)):
    """Extract multi-scale features from spike data."""
    return extract_features(_get_dataset(dataset_id), window_sec=window_sec)


# ═══════════ PLASTICITY & LEARNING ═══════════

@app.get("/api/analysis/{dataset_id}/stdp")
async def analyze_stdp(dataset_id: str, max_lag_ms: float = Query(30.0)):
    """STDP analysis — spike-timing dependent plasticity curves."""
    return _sanitize(compute_stdp_matrix(_get_dataset(dataset_id), max_lag_ms=max_lag_ms))


@app.get("/api/analysis/{dataset_id}/learning")
async def analyze_learning(dataset_id: str, window_sec: float = Query(60.0)):
    """Detect learning episodes — changes in plasticity over time."""
    return _sanitize(detect_learning_episodes(_get_dataset(dataset_id), window_sec=window_sec))


# ═══════════ ORGANOID IQ ═══════════

@app.get("/api/analysis/{dataset_id}/iq")
async def analyze_iq(dataset_id: str):
    """Organoid Intelligence Quotient — composite computational capacity score."""
    return _sanitize(compute_organoid_iq(_get_dataset(dataset_id)))


# ═══════════ PREDICTIONS ═══════════

@app.get("/api/analysis/{dataset_id}/predict/firing-rates")
async def predict_rates(dataset_id: str, forecast_sec: float = Query(300.0)):
    """Predict future firing rates with confidence intervals."""
    return _sanitize(predict_firing_rates(_get_dataset(dataset_id), forecast_sec=forecast_sec))


@app.get("/api/analysis/{dataset_id}/predict/bursts")
async def predict_bursts(dataset_id: str, window_sec: float = Query(10.0)):
    """Predict burst probability in next time window."""
    return _sanitize(predict_burst_probability(_get_dataset(dataset_id), window_sec=window_sec))


@app.get("/api/analysis/{dataset_id}/health")
async def analyze_health(dataset_id: str):
    """Estimate organoid health and viability."""
    return _sanitize(estimate_organoid_health(_get_dataset(dataset_id)))


# ═══════════ NEURAL REPLAY ═══════════

@app.get("/api/analysis/{dataset_id}/replay")
async def analyze_replay(dataset_id: str, min_similarity: float = Query(0.3)):
    """Detect neural replay — memory consolidation signatures."""
    return _sanitize(detect_replay(_get_dataset(dataset_id), min_similarity=min_similarity))


@app.get("/api/analysis/{dataset_id}/sequences")
async def analyze_sequences(dataset_id: str, min_length: int = Query(3)):
    """Detect repeated neural sequences — functional circuits."""
    return _sanitize(detect_sequence_replay(_get_dataset(dataset_id), min_sequence_length=min_length))


# ═══════════ RESERVOIR COMPUTING ═══════════

@app.get("/api/analysis/{dataset_id}/memory-capacity")
async def analyze_memory_capacity(dataset_id: str, max_delay: int = Query(20)):
    """Estimate memory capacity of neural network as reservoir computer."""
    return _sanitize(estimate_memory_capacity(_get_dataset(dataset_id), max_delay=max_delay))


@app.get("/api/analysis/{dataset_id}/nonlinearity")
async def analyze_nonlinearity(dataset_id: str):
    """Benchmark nonlinear computational capability."""
    return _sanitize(benchmark_nonlinear_computation(_get_dataset(dataset_id)))


# ═══════════ FINGERPRINTING ═══════════

@app.get("/api/analysis/{dataset_id}/fingerprint")
async def analyze_fingerprint(dataset_id: str):
    """Compute unique organoid fingerprint — identity signature."""
    return _sanitize(compute_fingerprint(_get_dataset(dataset_id)))


# ═══════════ SONIFICATION ═══════════

@app.get("/api/analysis/{dataset_id}/sonify")
async def sonify(dataset_id: str, speed: float = Query(10.0), duration: Optional[float] = None):
    """Convert neural activity to audio WAV (base64 encoded)."""
    return _sanitize(generate_sonification(_get_dataset(dataset_id), speed_factor=speed, duration_sec=duration))


@app.get("/api/analysis/{dataset_id}/rhythms")
async def analyze_rhythms(dataset_id: str):
    """Analyze rhythmic structure of neural activity."""
    return _sanitize(compute_rhythmic_analysis(_get_dataset(dataset_id)))


# ═══════════ CAUSAL EMERGENCE ═══════════

@app.get("/api/analysis/{dataset_id}/emergence")
async def analyze_emergence(dataset_id: str):
    """Compute integrated information (Phi) and causal emergence."""
    return _sanitize(compute_integrated_information(_get_dataset(dataset_id)))


# ═══════════ BREAKTHROUGH MODULES ═══════════

@app.get("/api/analysis/{dataset_id}/attractors")
async def analyze_attractors(dataset_id: str, min_visits: int = Query(3)):
    """Map attractor landscape — find memory traces as dynamical states."""
    return _sanitize(map_attractor_landscape(_get_dataset(dataset_id), min_visits=min_visits))


@app.get("/api/analysis/{dataset_id}/state-space")
async def analyze_state_space(dataset_id: str):
    """Analyze geometry of neural state space."""
    return _sanitize(compute_state_space_geometry(_get_dataset(dataset_id)))


@app.get("/api/analysis/{dataset_id}/phase-transitions")
async def analyze_phase_transitions(dataset_id: str, window_sec: float = Query(5.0)):
    """Detect phase transitions — moments of neural reorganization."""
    return _sanitize(detect_phase_transitions(_get_dataset(dataset_id), window_sec=window_sec))


@app.get("/api/analysis/{dataset_id}/predictive-coding")
async def analyze_predictive_coding(dataset_id: str):
    """Measure predictive coding — does the organoid generate predictions?"""
    return _sanitize(measure_predictive_coding(_get_dataset(dataset_id)))


@app.get("/api/analysis/{dataset_id}/weights")
async def analyze_weights(dataset_id: str):
    """Infer synaptic weight matrix from spike timing."""
    return _sanitize(infer_synaptic_weights(_get_dataset(dataset_id)))


@app.get("/api/analysis/{dataset_id}/weight-tracking")
async def analyze_weight_tracking(dataset_id: str, window_sec: float = Query(30.0)):
    """Track synaptic weight changes over time — watch learning happen."""
    return _sanitize(track_weight_changes(_get_dataset(dataset_id), window_sec=window_sec))


@app.get("/api/analysis/{dataset_id}/multiscale")
async def analyze_multiscale(dataset_id: str):
    """Multi-timescale complexity analysis — find operating frequency."""
    return _sanitize(compute_multiscale_complexity(_get_dataset(dataset_id)))


# ═══════════ FULL REPORT ═══════════

@app.get("/api/analysis/{dataset_id}/full-report")
async def full_report(dataset_id: str):
    """Run ALL 21 analyses and generate comprehensive report."""
    data = _get_dataset(dataset_id)
    report = generate_full_report(data)
    return _sanitize(report)


# ═══════════ DISCOVERY ANALYSIS ═══════════

@app.get("/api/analysis/{dataset_id}/sleep-wake")
async def api_sleep_wake(dataset_id: str):
    """Detect sleep-wake-like cycles in organoid activity."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = analyze_sleep_wake(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/habituation")
async def api_habituation(dataset_id: str):
    """Detect habituation — response decay to repeated patterns."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = detect_repeated_patterns(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/metastability")
async def api_metastability(dataset_id: str):
    """Analyze metastability — brain-like state switching dynamics."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = analyze_metastability(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/information-flow")
async def api_information_flow(dataset_id: str):
    """Map directed information flow using Granger causality."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_granger_causality(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/motifs")
async def api_motifs(dataset_id: str):
    """Enumerate network motifs (3-node subgraph patterns)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = enumerate_motifs(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/energy-landscape")
async def api_energy_landscape(dataset_id: str):
    """Compute Ising model energy landscape."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = fit_ising_model(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.post("/api/analysis/{dataset_id}/design-stimulus")
async def api_design_stimulus(dataset_id: str, generations: int = 15, population_size: int = 10):
    """Evolve optimal stimulation protocol using digital twin."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = evolve_protocol(data, generations=generations, population_size=population_size)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consciousness")
async def api_consciousness(dataset_id: str):
    """Composite consciousness assessment score."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_consciousness_score(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/comparative")
async def api_comparative(dataset_id: str):
    """Compare organoid with reference neural systems."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compare_with_references(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ PROTOCOLS ═══════════

@app.get("/api/protocols")
async def api_protocols():
    """List all available stimulation protocols."""
    return list_protocols()


@app.get("/api/protocols/{name}")
async def api_protocol_detail(name: str):
    """Get full protocol details by name."""
    result = get_protocol(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/analysis/{dataset_id}/suggest-protocol")
async def api_suggest_protocol(dataset_id: str):
    """Suggest best protocol based on organoid state."""
    data = _get_dataset(dataset_id)
    return suggest_protocol(data)


# ═══════════ EXPERIMENTS ═══════════

# ═══════════ CLOSED-LOOP EXPERIMENTS ═══════════

@app.get("/api/experiments/{dataset_id}/closed-loop")
async def api_closed_loop(
    dataset_id: str,
    n_episodes: int = Query(20, ge=1, le=100),
    reward_strategy: str = Query("hebbian"),
    game: str = Query("pong"),
):
    """Run DishBrain-style closed-loop session (pong or cartpole)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_dishbrain_session(data, n_episodes=n_episodes, reward_strategy=reward_strategy, game=game)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/closed-loop/cartpole")
async def api_cartpole(
    dataset_id: str,
    n_trials: int = Query(50, ge=1, le=200),
    reward_strategy: str = Query("dopamine"),
):
    """CartPole benchmark adapted for biological neural networks."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_cartpole_benchmark(data, n_trials=n_trials, reward_strategy=reward_strategy)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/closed-loop/strategies")
async def api_compare_strategies(dataset_id: str, n_episodes: int = Query(15, ge=5, le=50)):
    """Compare all 4 reward strategies (hebbian/dopamine/contrastive/reinforce)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compare_reward_strategies(data, n_episodes=n_episodes)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ CURRICULUM LEARNING ═══════════

@app.get("/api/experiments/{dataset_id}/curriculum")
async def api_curriculum(dataset_id: str):
    """Run full 4-stage adaptive curriculum learning protocol."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_curriculum(data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/curriculum/stage")
async def api_curriculum_stage(dataset_id: str):
    """Assess which curriculum stage the organoid is ready for."""
    data = _get_dataset(dataset_id)
    return _sanitize(get_current_stage(data))


@app.get("/api/experiments/{dataset_id}/curriculum/simulate")
async def api_curriculum_simulate(
    dataset_id: str,
    stage: int = Query(1, ge=1, le=4),
    n_trials: int = Query(30, ge=10, le=100),
):
    """Simulate a single curriculum stage with trial-by-trial output."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = simulate_stage(data, stage=stage, n_trials=n_trials)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ MEMORY BATTERY ═══════════

@app.get("/api/experiments/{dataset_id}/memory")
async def api_memory_battery(dataset_id: str):
    """Run complete memory battery: working + short-term + long-term + associative."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_memory_battery(data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/memory/working")
async def api_memory_working(dataset_id: str, n_trials: int = Query(40, ge=10, le=100)):
    """Delay-match-to-sample working memory test."""
    data = _get_dataset(dataset_id)
    return _sanitize(test_working_memory(data, n_trials=n_trials))


@app.get("/api/experiments/{dataset_id}/memory/short-term")
async def api_memory_stm(dataset_id: str, max_span: int = Query(9, ge=3, le=15)):
    """Serial-span short-term memory test."""
    data = _get_dataset(dataset_id)
    return _sanitize(test_short_term_memory(data, max_span=max_span))


@app.get("/api/experiments/{dataset_id}/memory/long-term")
async def api_memory_ltm(dataset_id: str, n_patterns: int = Query(5, ge=2, le=20)):
    """Spaced-retention long-term memory test."""
    data = _get_dataset(dataset_id)
    return _sanitize(test_long_term_memory(data, n_patterns=n_patterns))


@app.get("/api/experiments/{dataset_id}/memory/associative")
async def api_memory_am(dataset_id: str, n_pairs: int = Query(6, ge=2, le=20)):
    """Paired-associate / Hopfield-style associative memory test."""
    data = _get_dataset(dataset_id)
    return _sanitize(test_associative_memory(data, n_pairs=n_pairs))


# ═══════════ PONG ENGINE ═══════════

@app.get("/api/experiments/{dataset_id}/pong")
async def api_pong(
    dataset_id: str,
    n_games: int = Query(20, ge=1, le=100),
    encoding: str = Query("rate"),
    decoding: str = Query("population_vector"),
    reward_rule: str = Query("dishbrain"),
):
    """Simulate N games of Pong using organoid spike data as controller."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = simulate_pong(data, n_games=n_games, encoding=encoding, decoding=decoding, reward_rule=reward_rule)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/pong/encode")
async def api_pong_encode(
    ball_x: float = Query(0.5, ge=0.0, le=1.0),
    ball_y: float = Query(0.5, ge=0.0, le=1.0),
    ball_vx: float = Query(0.05),
    ball_vy: float = Query(0.02),
    n_electrodes: int = Query(8, ge=2, le=64),
    encoding: str = Query("rate"),
):
    """Compute stimulation pattern for a given Pong ball state."""
    return _sanitize(encode_ball_state(ball_x, ball_y, ball_vx, ball_vy, n_electrodes, encoding))


# ═══════════ XOR BENCHMARK ═══════════

@app.get("/api/experiments/{dataset_id}/xor")
async def api_xor_benchmark(
    dataset_id: str,
    n_trials_per_gate: int = Query(60, ge=20, le=200),
    readout: str = Query("logistic"),
):
    """Run full logical gate benchmark suite (AND, OR, XOR, NAND, XNOR)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_full_benchmark(data, n_trials_per_gate=n_trials_per_gate, readout=readout)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/xor/gate")
async def api_xor_gate(
    dataset_id: str,
    gate: str = Query("XOR"),
    n_trials: int = Query(60, ge=20, le=200),
    readout: str = Query("logistic"),
):
    """Run a single logical gate benchmark."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_gate_benchmark(data, gate=gate.upper(), n_trials=n_trials, readout=readout)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/xor/difficulty")
async def api_xor_difficulty(dataset_id: str):
    """Estimate XOR difficulty for this organoid (Fisher discriminant)."""
    data = _get_dataset(dataset_id)
    return _sanitize(compute_xor_difficulty(data))


# ═══════════ JAPANESE VOWELS ═══════════

@app.get("/api/experiments/vowels/classify")
async def api_vowels_classify(
    n_samples: int = Query(240, ge=40, le=1000),
    n_classes: int = Query(8, ge=2, le=8),
    reservoir_size: int = Query(256, ge=64, le=1024),
    spectral_radius: float = Query(0.9, ge=0.1, le=1.5),
    seed: int = Query(42),
):
    """Run Brainoware-inspired vowel recognition via reservoir computing.

    Generates synthetic Japanese vowel features, processes through
    an echo-state reservoir, and trains a linear readout classifier.
    """
    t0 = time.time()
    vowels = generate_synthetic_vowels(n_samples=n_samples, n_classes=n_classes, seed=seed)
    reservoir = build_reservoir(
        input_dim=vowels["n_features"],
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        seed=seed,
    )
    states = reservoir_transform(vowels["features"], reservoir)
    readout = train_linear_readout(states, vowels["labels"], n_classes=n_classes)
    result = {
        "task": "japanese_vowel_recognition",
        "n_samples": vowels["n_samples"],
        "n_classes": n_classes,
        "class_names": vowels["class_names"],
        "reservoir_size": reservoir_size,
        "spectral_radius": spectral_radius,
        "train_accuracy": readout["train_accuracy"],
        "test_accuracy": readout["test_accuracy"],
        "per_class_metrics": readout["per_class"],
        "n_train": readout["n_train"],
        "n_test": readout["n_test"],
        "above_chance": round(readout["test_accuracy"] - 1.0 / n_classes, 3),
        "_computation_time_ms": round((time.time() - t0) * 1000, 1),
        "interpretation": (
            f"Vowel recognition: {readout['test_accuracy']:.1%} accuracy "
            f"({readout['test_accuracy'] - 1/n_classes:+.1%} vs chance). "
            + ("Replicates Brainoware result!" if readout["test_accuracy"] > 0.7
               else "Above chance — reservoir dynamics separate vowel classes."
               if readout["test_accuracy"] > 1 / n_classes + 0.1
               else "Near chance — try larger reservoir or more samples.")
        ),
    }
    return _sanitize(result)


# ═══════════ EXPERIMENT TRACKER ═══════════

@app.post("/api/experiments/tracker/{experiment_id}/start/{dataset_id}")
async def api_tracker_start(
    experiment_id: str,
    dataset_id: str,
    name: Optional[str] = None,
    experiment_type: str = Query("stimulation"),
):
    """Start a new experiment — record pre-intervention baseline."""
    data = _get_dataset(dataset_id)
    return _sanitize(start_experiment(experiment_id, data, name=name, experiment_type=experiment_type))


@app.post("/api/experiments/tracker/{experiment_id}/end/{dataset_id}")
async def api_tracker_end(
    experiment_id: str,
    dataset_id: str,
    notes: Optional[str] = None,
):
    """End experiment — record post-intervention and compute delta."""
    data = _get_dataset(dataset_id)
    return _sanitize(end_experiment(experiment_id, data, notes=notes))


@app.get("/api/experiments/tracker/{experiment_id}")
async def api_tracker_get(experiment_id: str):
    """Get full experiment record including delta analysis."""
    result = tracker_get_experiment(experiment_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _sanitize(result)


@app.get("/api/experiments/tracker")
async def api_tracker_history(
    experiment_type: Optional[str] = None,
    status: Optional[str] = None,
):
    """List all tracked experiments."""
    return _sanitize(get_experiment_history(experiment_type=experiment_type, status=status))


@app.post("/api/experiments/tracker/compare/{pre_dataset_id}/{post_dataset_id}")
async def api_tracker_delta(pre_dataset_id: str, post_dataset_id: str):
    """Compute pre/post delta between two datasets without creating an experiment."""
    pre = _get_dataset(pre_dataset_id)
    post = _get_dataset(post_dataset_id)
    return _sanitize(tracker_compute_delta(pre, post))


# ═══════════ PUBLICATION GENERATOR ═══════════

@app.post("/api/publish/{dataset_id}")
async def api_publish_draft(dataset_id: str):
    """Generate full paper draft (title, abstract, methods, results, discussion)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = generate_draft(data, analyses={})
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/publish/{dataset_id}/abstract")
async def api_publish_abstract(dataset_id: str):
    """Generate abstract only — fast 250-word structured abstract."""
    data = _get_dataset(dataset_id)
    return _sanitize(generate_abstract_only(data, analyses={}))


@app.post("/api/publish/{dataset_id}/methods")
async def api_publish_methods(dataset_id: str):
    """Generate methods section only."""
    data = _get_dataset(dataset_id)
    return _sanitize(generate_methods_section(data, analyses={}))


# ═══════════ ETHICS ═══════════

@app.get("/api/analysis/{dataset_id}/ethics")
async def api_ethics(
    dataset_id: str,
    culture_age_days: Optional[int] = None,
    organoid_type: Optional[str] = None,
):
    """Full ethical assessment: consciousness indicators, sentience risk, guidelines, compliance."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = assess_ethics(data, culture_age_days=culture_age_days, organoid_type=organoid_type)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consciousness")
async def api_consciousness_indicators(dataset_id: str):
    """Consciousness proxy indicators: Phi, global workspace, criticality, complexity."""
    data = _get_dataset(dataset_id)
    return _sanitize(assess_consciousness_indicators(data))


@app.get("/api/analysis/{dataset_id}/sentience-risk")
async def api_sentience_risk(dataset_id: str):
    """Sentience risk score: nociception proxy, stress, valence, distress signals."""
    data = _get_dataset(dataset_id)
    return _sanitize(compute_sentience_risk_score(data))


# ═══════════ FUNDING ═══════════

@app.get("/api/funding/match")
async def api_funding_match(
    country_filter: Optional[str] = None,
    min_funding_usd: Optional[int] = None,
):
    """Find matching grants (NSF, NIH, DARPA, EU Horizon, Astana Hub, IndieBio…)."""
    return _sanitize(match_grants(data=None, analyses={}, country_filter=country_filter, min_funding_usd=min_funding_usd))


@app.get("/api/funding/match/{dataset_id}")
async def api_funding_match_with_data(
    dataset_id: str,
    country_filter: Optional[str] = None,
    min_funding_usd: Optional[int] = None,
):
    """Match grants using actual organoid data for scoring."""
    data = _get_dataset(dataset_id)
    return _sanitize(match_grants(data=data, analyses={}, country_filter=country_filter, min_funding_usd=min_funding_usd))


@app.get("/api/funding/grants")
async def api_funding_list(country_filter: Optional[str] = None):
    """List all available grant programs."""
    return _sanitize(list_grants(country_filter=country_filter))


@app.get("/api/funding/grants/{grant_id}")
async def api_funding_grant_detail(grant_id: str):
    """Get full details for a specific grant program."""
    result = get_grant_details(grant_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _sanitize(result)


# ═══════════ COMPUTATION & AI ═══════════

@app.get("/api/analysis/{dataset_id}/turing-test")
async def api_turing_test(dataset_id: str):
    """Run organoid Turing test — compare real data vs Poisson and LIF models."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_turing_test(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.post("/api/analysis/{dataset_id}/architecture-search")
async def api_architecture_search(dataset_id: str, generations: int = 15, population_size: int = 10):
    """Neural Architecture Search for optimal stimulation protocol."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = search_optimal_protocol(data, population_size=population_size, generations=generations)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/hybrid-benchmark")
async def api_hybrid_benchmark(dataset_id: str):
    """Benchmark hybrid bio-digital AI vs pure digital vs pure biological."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = benchmark_hybrid(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ LEARNING & MEMORY ═══════════

@app.get("/api/analysis/{dataset_id}/forgetting")
async def api_forgetting(dataset_id: str):
    """Measure catastrophic forgetting — do early patterns survive?"""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_retention_curve(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/channel-capacity")
async def api_channel_capacity(dataset_id: str):
    """Estimate information channel capacity (multi-bit memory)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = estimate_channel_capacity(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/population-codes")
async def api_population_codes(dataset_id: str):
    """Measure population code diversity (distinct neural states)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = measure_population_code_diversity(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/topology")
async def api_topology(dataset_id: str):
    """Compute Betti numbers (topological data analysis)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_betti_numbers(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/topological-complexity")
async def api_topo_complexity(dataset_id: str):
    """Compute topological complexity score."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_topological_complexity(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consolidation")
async def api_consolidation(dataset_id: str):
    """Detect memory consolidation events during offline periods."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = detect_consolidation_events(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/transfer")
async def api_transfer(dataset_id: str):
    """Measure transfer learning between recording segments."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = measure_transfer(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/rsa")
async def api_rsa(dataset_id: str):
    """Representational Similarity Analysis."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_representational_similarity(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ CONNECTOMICS ═══════════

@app.get("/api/analysis/{dataset_id}/connectome")
async def api_connectome(dataset_id: str):
    """Build full functional connectome with graph theory metrics."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = build_full_connectome(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/communities")
async def api_communities(dataset_id: str):
    """Detect network communities (modules) via spectral clustering."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = detect_communities(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/graph-theory")
async def api_graph_theory(dataset_id: str):
    """Compute rich-club, small-world, efficiency, betweenness centrality."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_graph_theory_metrics(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/effective-connectivity")
async def api_effective_connectivity(dataset_id: str):
    """Estimate directed causal connections (effective connectivity)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = estimate_effective_connectivity(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/causal-hierarchy")
async def api_causal_hierarchy(dataset_id: str):
    """Order electrodes by causal influence hierarchy."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = compute_causal_hierarchy(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ BIO-INSPIRED & ETHICS ═══════════

@app.post("/api/analysis/{dataset_id}/evolve-programs")
async def api_evolve_programs(dataset_id: str, generations: int = 20):
    """Genetic programming — evolve stimulation program trees."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = evolve_programs(data, generations=generations)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/homeostasis")
async def api_homeostasis(dataset_id: str):
    """Monitor homeostatic plasticity — firing rate self-regulation."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = monitor_homeostasis(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/suffering")
async def api_suffering(dataset_id: str):
    """Detect distress/suffering patterns — ethical monitoring."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = detect_suffering(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/welfare")
async def api_welfare(dataset_id: str):
    """Generate comprehensive welfare report."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = generate_welfare_report(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/swarm")
async def api_swarm(dataset_id: str, n_organoids: int = 4):
    """Simulate multi-organoid swarm intelligence."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = simulate_swarm(data, n_organoids=n_organoids)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/morphology")
async def api_morphology(dataset_id: str):
    """Analyze morphological computing — structure vs function."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = analyze_morphological_computation(data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ WEBSOCKET — REAL-TIME STREAMING ═══════════

import asyncio

@app.websocket("/ws/spikes")
async def ws_live_spikes(ws: WebSocket):
    """Stream synthetic spike data in real-time via WebSocket.

    Sends JSON frames every 100ms with spike events.
    Protocol: connect → receive frames → disconnect.
    Each frame: {"spikes": [{"time": float, "electrode": int, "amplitude": float}], "timestamp": float}
    """
    await ws.accept()
    t = 0.0
    dt = 0.1  # 100ms intervals
    try:
        while True:
            # Generate spikes for this time window
            n_spikes = np.random.poisson(5)  # ~50 Hz total
            spikes = []
            for _ in range(n_spikes):
                spikes.append({
                    "time": round(t + np.random.uniform(0, dt), 4),
                    "electrode": int(np.random.randint(0, 8)),
                    "amplitude": round(float(np.random.normal(-50, 15)), 1),
                })
            await ws.send_json({
                "spikes": spikes,
                "timestamp": round(t, 3),
                "n_spikes": len(spikes),
            })
            t += dt
            await asyncio.sleep(dt)
    except WebSocketDisconnect:
        pass


# ═══════════ MULTI-ORGANOID ═══════════

@app.get("/api/analysis/{dataset_id}/multi-organoid")
async def analyze_multi_organoid(
    dataset_id: str,
    electrodes_per_organoid: int = Query(8, ge=1, le=32, description="Electrodes per virtual organoid"),
):
    """Compare multiple virtual organoids from a multi-MEA dataset."""
    data = _get_dataset(dataset_id)
    cached = _cache_get(dataset_id, f"multi-organoid-{electrodes_per_organoid}")
    if cached:
        return cached
    t0 = time.time()
    result = compare_organoids(data, electrodes_per_organoid=electrodes_per_organoid)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    _cache_set(dataset_id, f"multi-organoid-{electrodes_per_organoid}", result)
    return _sanitize(result)


# ═══════════ TEMPORAL EVOLUTION ═══════════

@app.get("/api/analysis/{dataset_id}/temporal-evolution")
async def analyze_temporal_evolution(
    dataset_id: str,
    window_sec: float = Query(60.0, ge=1.0, le=3600.0, description="Window size in seconds"),
    mode: str = Query("full", description="Mode: full, trends, critical"),
):
    """Track how organoid properties evolve over time."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    if mode == "trends":
        result = detect_trends(data, window_sec=window_sec)
    elif mode == "critical":
        result = find_critical_moments(data, window_sec=window_sec)
    else:
        evolution = track_evolution(data, window_sec=window_sec)
        trends = detect_trends(data, window_sec=window_sec)
        critical = find_critical_moments(data, window_sec=window_sec)
        result = {
            "evolution": evolution,
            "trends": trends,
            "critical_moments": critical,
        }
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ STIMULATION RESPONSE ═══════════

@app.get("/api/analysis/{dataset_id}/stim-response")
async def analyze_stim_response(
    dataset_id: str,
    stim_times: Optional[str] = Query(None, description="Comma-separated stimulation times in seconds"),
    window_ms: float = Query(200.0, ge=10.0, le=5000.0, description="Post-stimulus window in ms"),
):
    """Analyze organoid response to stimulation events."""
    data = _get_dataset(dataset_id)
    t0 = time.time()

    if stim_times:
        stim_list = [float(t.strip()) for t in stim_times.split(",")]
        result = detect_response(data, stim_times=stim_list, window_ms=window_ms)
    else:
        # Auto-detect stimulation times
        estimated = estimate_stim_times(data)
        stim_list = estimated.get("estimated_stim_times", [])
        if stim_list:
            response = detect_response(data, stim_times=stim_list, window_ms=window_ms)
            result = {
                "stim_detection": estimated,
                "response_analysis": response,
                "note": "Stimulation times were auto-detected from spike patterns",
            }
        else:
            result = {
                "stim_detection": estimated,
                "note": "No stimulation events detected. Provide stim_times parameter manually.",
            }

    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ LLM OPTIMIZER ═══════════

@app.post("/api/analysis/{dataset_id}/llm-optimize")
async def analyze_llm_optimize(
    dataset_id: str,
    n_iterations: int = Query(5, ge=1, le=20, description="Number of optimization iterations"),
    objective: str = Query("maximize_complexity", description="Optimization objective"),
):
    """Run LLM-in-the-loop protocol optimization (simulated)."""
    data = _get_dataset(dataset_id)
    t0 = time.time()
    result = run_optimization_loop(data, n_iterations=n_iterations, objective=objective)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ EXPORT ═══════════

@app.get("/api/export/{dataset_id}/csv")
async def export_csv(
    dataset_id: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    electrodes: Optional[str] = None,
):
    """Export spike data as CSV."""
    data = _get_dataset(dataset_id)
    el_list = [int(e) for e in electrodes.split(",")] if electrodes else None
    filtered = data.get_filtered(electrodes=el_list, start=start, end=end)
    df = filtered.to_dataframe()

    import io
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=neurobridge_{dataset_id}.csv"},
    )


@app.get("/api/export/{dataset_id}/json")
async def export_json(dataset_id: str):
    """Export full dataset as JSON."""
    data = _get_dataset(dataset_id)
    return data.to_dict()


# ═══════════ HELPERS ═══════════

def _get_dataset(dataset_id: str) -> SpikeData:
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found. Upload data first via POST /api/upload")
    return datasets[dataset_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8847)
