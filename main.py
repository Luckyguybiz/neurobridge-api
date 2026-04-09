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

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
from models.schemas import (
    SpikeDetectionParams, BurstDetectionParams, SpikeSortingParams,
    ConnectivityParams, TimeRangeFilter, DatasetInfo,
)

# Directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="NeuroBridge API",
    description="Backend for biocomputing data analysis — spike detection, burst analysis, connectivity mapping",
    version="0.2.0",
)


# Fix numpy serialization
import json as _json

class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
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
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
async def api_design_stimulus(dataset_id: str, generations: int = 30, population_size: int = 15):
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
