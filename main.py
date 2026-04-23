"""NeuroBridge API — Backend for biocomputing data analysis.

FastAPI server that provides endpoints for:
- Data upload and management
- Spike detection and sorting
- Burst detection
- Functional connectivity analysis
- Comprehensive statistics
- Export (CSV, JSON)
"""

import logging
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# PM2 captures stdout/stderr; INFO-level logging surfaces in `pm2 logs`.
# Format has timestamp + level + message so post-mortem analysis is easy.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("neurobridge")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

from analysis.loader import SpikeData, load_file
from analysis.spikes import compute_isi, compute_firing_rates, compute_amplitude_stats, sort_spikes
from analysis.bursts import analyze_bursts as detect_bursts, detect_bursts_max_interval as detect_single_channel_bursts, detect_network_bursts as compute_burst_profiles
from analysis.connectivity import compute_cross_correlation, compute_connectivity_graph, compute_transfer_entropy, connectivity_to_dict
from analysis.stats import compute_full_summary, compute_temporal_dynamics, compute_quality_metrics
from analysis.information_theory import compute_spike_train_entropy, compute_mutual_information, compute_lempel_ziv_complexity
from analysis.spectral import compute_power_spectrum, compute_coherence
from analysis.criticality import analyse_criticality as detect_avalanches
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
from analysis.protocols.center_activity import compute_center_of_activity, simulate_ca_shift
from analysis.protocols.dishbrain_pong import simulate_pong_game
from analysis.protocols.brainoware_reservoir import simulate_reservoir_classification
from analysis.protocols.cartpole_coaching import simulate_cartpole
from analysis.protocols.dopamine_reinforcement import simulate_dopamine_training
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
    """Recursively convert numpy types and dataclasses to Python natives for JSON serialization."""
    if obj is None:
        return None
    if hasattr(obj, 'to_dict'):
        return _sanitize(obj.to_dict())
    if hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return _sanitize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
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
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════ RATE LIMITING ═══════════
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse as _RateLimitResponse
import threading as _rl_threading

_RATE_LIMIT_REGULAR = 60       # requests per minute for regular endpoints
_RATE_LIMIT_HEAVY = 5          # requests per minute for heavy endpoints
_HEAVY_ENDPOINTS = {"bursts", "connectivity", "cross-correlation", "transfer-entropy", "full-report"}
_rate_buckets: dict[str, list[float]] = {}   # key -> list of timestamps
_rate_lock = _rl_threading.Lock()


def _is_heavy_endpoint(path: str) -> bool:
    """Check if the request path matches a heavy analysis endpoint."""
    parts = path.rstrip("/").split("/")
    # Pattern: /api/analysis/{dataset_id}/{endpoint}
    if len(parts) >= 5 and parts[1] == "api" and parts[2] == "analysis":
        return parts[4] in _HEAVY_ENDPOINTS
    return False


def _check_rate_limit(key: str, limit: int) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_lock:
        timestamps = _rate_buckets.get(key)
        if timestamps is None:
            _rate_buckets[key] = [now]
            # Opportunistic cleanup — every ~1 in 50 requests, drop empty buckets.
            # Without this, each unique IP leaves a dict entry forever (leak on 1M+ IPs).
            if len(_rate_buckets) > 200 and uuid.uuid4().int % 50 == 0:
                _gc_rate_buckets(now)
            return True
        # Purge entries older than 60 seconds
        cutoff = now - 60.0
        _rate_buckets[key] = timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
        return True


def _gc_rate_buckets(now: float) -> None:
    """Drop bucket entries with no recent activity. Caller must hold _rate_lock."""
    cutoff = now - 120.0  # 2× rate window — gives slack on lock contention
    stale = [k for k, ts in _rate_buckets.items() if not ts or ts[-1] < cutoff]
    for k in stale:
        del _rate_buckets[k]


def _client_ip_from_request(request: Request) -> str:
    """Extract real client IP through Caddy reverse proxy.

    Caddy sets `X-Forwarded-For: <client-ip>` (see /etc/caddy/Caddyfile).
    Without reading that header, `request.client.host` is always 127.0.0.1
    and all users share one rate-limit bucket — first user to hit 60/min
    locks out everyone else.

    We trust the leftmost entry because only Caddy directly connects to us
    on localhost; it overwrites any forged upstream header.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip non-API and OPTIONS requests
        path = request.url.path
        if not path.startswith("/api/") or request.method == "OPTIONS":
            return await call_next(request)

        client_ip = _client_ip_from_request(request)
        heavy = _is_heavy_endpoint(path)
        limit = _RATE_LIMIT_HEAVY if heavy else _RATE_LIMIT_REGULAR
        bucket_key = f"{client_ip}:{'heavy' if heavy else 'regular'}"

        if not _check_rate_limit(bucket_key, limit):
            logger.warning("rate_limit.hit ip=%s path=%s heavy=%s", client_ip, path, heavy)
            return _RateLimitResponse(
                {"error": f"Rate limit exceeded. Max {limit} requests per minute for {'heavy analysis' if heavy else 'regular'} endpoints. Please wait and try again."},
                status_code=429,
                headers={"Access-Control-Allow-Origin": "*", "Retry-After": "60"},
            )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

# Ensure CORS headers on unhandled exceptions (500s)
from starlette.responses import JSONResponse as StarletteJSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # logger.exception includes traceback — more structured than print
    logger.exception("unhandled_exception path=%s method=%s err=%s", request.url.path, request.method, str(exc)[:200])
    return StarletteJSONResponse(
        {"error": str(exc)[:200]},
        status_code=500,
        headers={"Access-Control-Allow-Origin": "*"},
    )

# In-memory dataset store (production would use Redis/DB)
datasets: dict[str, SpikeData] = {}
_MAX_DATASETS = 2  # Auto-evict oldest when exceeded (OOM protection)

# Memory-based eviction thresholds. Without these, the count-based limit
# (_MAX_DATASETS) is blind to actual RAM pressure — two small uploads of
# 10MB each = fine, two loads of 2.6M FinalSpark + running analyses = OOM.
_MEMORY_SOFT_LIMIT_PCT = 75.0   # Trigger gc + malloc_trim
_MEMORY_HARD_LIMIT_PCT = 85.0   # Evict datasets even if under _MAX_DATASETS

# Remember evicted dataset IDs (bounded set) so /get_dataset can return
# 410 Gone instead of a confusing 404 — the frontend shows "your session
# was replaced by another user's data" and offers to reload.
_evicted_ids: list[str] = []  # kept as list for FIFO bounded history
_EVICTED_HISTORY = 64

# Semaphore: max 1 heavy computation at a time (prevents 100% CPU lockup)
import asyncio as _asyncio
_compute_semaphore = _asyncio.Semaphore(1)

# Tiered timeouts. Caddy upstream timeout is 300s, so we can go up to 270s
# safely. Old default was 45s for everything — too aggressive for O(N²) ops
# that genuinely need 60-90s to converge. Split into three classes so fast
# ops still fail fast (firing-rates should not take 2 min = something wrong).
_TIMEOUT_NORMAL_SEC = 45.0   # firing-rates, bursts, isi, summary, etc.
_TIMEOUT_HEAVY_SEC = 120.0   # connectivity, cross-correlation, IQ (O(N²))
_TIMEOUT_VERY_HEAVY_SEC = 240.0  # metastability, transfer-entropy, IIT, turing
_HEAVY_TIMEOUT_SEC = _TIMEOUT_NORMAL_SEC  # back-compat alias


async def _run_heavy(func, *args, timeout_sec: float = _TIMEOUT_NORMAL_SEC, **kwargs):
    """Run CPU-heavy function with semaphore (max 1 at a time) and timeout.

    Pass timeout_sec=_TIMEOUT_HEAVY_SEC or _TIMEOUT_VERY_HEAVY_SEC for
    endpoints known to need more than 45s on FinalSpark-sized inputs.
    """
    async with _compute_semaphore:
        try:
            return await _asyncio.wait_for(
                _asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout_sec,
            )
        except _asyncio.TimeoutError:
            logger.warning("heavy.timeout func=%s timeout=%.0fs", getattr(func, "__name__", "?"), timeout_sec)
            raise HTTPException(
                status_code=504,
                detail=f"Analysis took longer than {int(timeout_sec)}s and was abandoned. Try a smaller time range (e.g. subset=1h) or a faster endpoint.",
            )


# ─── Memory management helpers ───
import gc as _gc
import ctypes as _ctypes

# glibc's malloc holds freed memory in per-thread arenas and doesn't return it
# to the OS by default. This matters for long-running Python processes that
# allocate + free large numpy arrays — the RSS grows monotonically even after
# `del`. malloc_trim(0) walks the arenas and madvise(MADV_DONTNEED) unused
# chunks, so the kernel can reclaim pages. Loaded lazily so we degrade
# gracefully on non-glibc systems (e.g. macOS dev machines).
try:
    _libc = _ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [_ctypes.c_int]
    _libc.malloc_trim.restype = _ctypes.c_int
    _HAS_MALLOC_TRIM = True
except (OSError, AttributeError):
    _libc = None
    _HAS_MALLOC_TRIM = False


def _reclaim_memory(reason: str = "eviction") -> dict:
    """Force GC + return freed pages to the OS. Returns before/after metrics."""
    import psutil
    proc = psutil.Process()
    before_rss_mb = proc.memory_info().rss / 1024 / 1024
    collected = _gc.collect()
    trimmed = 0
    if _HAS_MALLOC_TRIM:
        trimmed = _libc.malloc_trim(0)
    after_rss_mb = proc.memory_info().rss / 1024 / 1024
    logger.info(
        "memory.reclaim reason=%s collected=%d trimmed=%d rss_before=%.0fmb rss_after=%.0fmb delta=%+.0fmb",
        reason, collected, trimmed, before_rss_mb, after_rss_mb, after_rss_mb - before_rss_mb,
    )
    return {
        "reason": reason,
        "gc_collected": collected,
        "malloc_trim_ret": trimmed,
        "rss_before_mb": round(before_rss_mb),
        "rss_after_mb": round(after_rss_mb),
        "delta_mb": round(after_rss_mb - before_rss_mb),
    }


def _current_memory_percent() -> float:
    """System-wide memory usage %. Used for pressure-based eviction decisions."""
    import psutil
    return psutil.virtual_memory().percent


def _evict_one_oldest(reason: str) -> Optional[str]:
    """Remove oldest dataset and record it in evicted history. Returns evicted id or None."""
    if not datasets:
        return None
    oldest_id = next(iter(datasets))
    del datasets[oldest_id]
    _evicted_ids.append(oldest_id)
    if len(_evicted_ids) > _EVICTED_HISTORY:
        _evicted_ids.pop(0)
    logger.info("dataset.evict id=%s reason=%s remaining=%d", oldest_id, reason, len(datasets))
    return oldest_id


def _store_dataset(dataset_id: str, data: SpikeData) -> None:
    """Store dataset with auto-eviction to prevent OOM.

    Two eviction triggers:
      1. Count-based: len(datasets) >= _MAX_DATASETS
      2. Memory-based: system RAM >= _MEMORY_HARD_LIMIT_PCT (85%)

    After any eviction, we force GC + malloc_trim to actually return
    freed pages to the OS. Without this, RSS grows unboundedly as glibc
    holds arenas for reuse (classic Python long-runner pattern).
    """
    evicted_any = False
    while len(datasets) >= _MAX_DATASETS:
        if _evict_one_oldest("count_limit"):
            evicted_any = True
        else:
            break
    # Memory pressure check — evict even if count is OK
    if _current_memory_percent() >= _MEMORY_HARD_LIMIT_PCT and datasets:
        _evict_one_oldest("memory_pressure")
        evicted_any = True
    if evicted_any:
        _reclaim_memory("post_eviction")
    datasets[dataset_id] = data


# ═══════════ HEALTH ═══════════

@app.get("/health")
async def health():
    """Minimal public health check — just confirms the API is alive.
    Internal metrics moved to /health/detailed (operator-only)."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health metrics — for monitoring dashboards, not public UI."""
    import psutil
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    proc = psutil.Process()
    proc_mem = proc.memory_info()
    return {
        "status": "ok",
        "datasets_loaded": len(datasets),
        "max_datasets": _MAX_DATASETS,
        "evicted_history_size": len(_evicted_ids),
        "rate_buckets": len(_rate_buckets),
        "ws_connections": sum(_ws_connections.values()),
        "memory_used_mb": round(mem.used / 1024 / 1024),
        "memory_available_mb": round(mem.available / 1024 / 1024),
        "memory_percent": round(mem.percent, 1),
        "swap_used_mb": round(swap.used / 1024 / 1024),
        "swap_percent": round(swap.percent, 1),
        "process_rss_mb": round(proc_mem.rss / 1024 / 1024),
        "process_vms_mb": round(proc_mem.vms / 1024 / 1024),
        "soft_limit_pct": _MEMORY_SOFT_LIMIT_PCT,
        "hard_limit_pct": _MEMORY_HARD_LIMIT_PCT,
        "malloc_trim_available": _HAS_MALLOC_TRIM,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/_debug/reclaim-memory")
async def debug_reclaim_memory():
    """Operator endpoint: force gc.collect() + malloc_trim(0) and return diff.
    Safe to call anytime — idempotent. Useful for investigating leaks."""
    return _reclaim_memory("manual")


@app.post("/api/_debug/evict-all")
async def debug_evict_all():
    """Operator endpoint: drop every loaded dataset and reclaim memory.
    Use when memory is pinned and you want to force a clean state.
    Frontend will get 410 on next fetch and gracefully re-upload."""
    n_before = len(datasets)
    evicted = list(datasets.keys())
    for ds_id in evicted:
        _evicted_ids.append(ds_id)
    datasets.clear()
    while len(_evicted_ids) > _EVICTED_HISTORY:
        _evicted_ids.pop(0)
    reclaim_stats = _reclaim_memory("manual_evict_all")
    logger.warning("dataset.evict_all n=%d ids=%s", n_before, evicted[:5])
    return {"evicted_count": n_before, **reclaim_stats}


# ═══════════ DATA MANAGEMENT ═══════════

UPLOAD_MAX_BYTES = 100 * 1024 * 1024  # 100 MB hard cap

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    sampling_rate: float = Query(30000.0, description="Sampling rate in Hz"),
):
    """Upload a spike data file (CSV, HDF5, Parquet, JSON)."""
    dataset_id = str(uuid.uuid4())[:8]
    filepath = UPLOAD_DIR / f"{dataset_id}_{file.filename}"

    # Stream-read with running size check — don't await .read() on a 500MB file
    # (would load entire content in RAM, OOM risk). Abort as soon as we cross cap.
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(1024 * 1024)  # 1MB chunks
        if not chunk:
            break
        size += len(chunk)
        if size > UPLOAD_MAX_BYTES:
            logger.warning("upload.rejected filename=%r size_mb=%.1f reason=too_large", file.filename, size / 1024 / 1024)
            raise HTTPException(status_code=413, detail=f"File too large. Max {UPLOAD_MAX_BYTES // 1024 // 1024}MB.")
        chunks.append(chunk)
    content = b"".join(chunks)

    with open(filepath, "wb") as f:
        f.write(content)

    try:
        t0 = time.time()
        data = load_file(str(filepath), sampling_rate=sampling_rate)
        load_time = (time.time() - t0) * 1000

        _store_dataset(dataset_id, data)

        logger.info(
            "upload.ok id=%s filename=%r size_mb=%.2f n_spikes=%d n_electrodes=%d duration_s=%.1f load_ms=%.0f",
            dataset_id, file.filename, len(content) / 1024 / 1024, data.n_spikes, data.n_electrodes, data.duration, load_time,
        )
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
        logger.warning("upload.parse_failed filename=%r err=%s", file.filename, str(e)[:200])
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
    _store_dataset(dataset_id, data)

    return {
        "dataset_id": dataset_id,
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(data.duration, 3),
    }


# Cache for parsed local files — avoid re-reading 129MB CSV every click
_local_file_cache: dict[str, SpikeData] = {}

@app.post("/api/load-local")
async def load_local_file(
    filename: str = Query("SpikeDataToShare_fs437data.csv"),
    sampling_rate: float = Query(437.0),
    mea: Optional[int] = Query(None, ge=0, le=3, description="Load specific MEA (0-3). Omit for ALL data."),
    max_duration: Optional[float] = Query(None, ge=60, description="Max duration in seconds. Omit for full recording."),
):
    """Load a CSV file from server's data/ directory.

    Caches parsed data so repeated loads are instant (~0ms vs ~10s).
    """
    import asyncio

    cache_key = f"{filename}:{mea}:{max_duration}"
    dataset_id = str(uuid.uuid4())[:8]

    # Check cache first
    if cache_key in _local_file_cache:
        data = _local_file_cache[cache_key]
        _store_dataset(dataset_id, data)
        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "n_spikes": data.n_spikes,
            "n_electrodes": data.n_electrodes,
            "duration_s": round(data.duration, 3),
            "mea_filter": mea,
            "max_duration_filter": max_duration,
            "_cached": True,
        }

    # Parse from disk in thread pool (don't block event loop)
    data = await _run_heavy(_parse_local_file, filename, sampling_rate, mea, max_duration)
    _local_file_cache[cache_key] = data
    _store_dataset(dataset_id, data)

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(data.duration, 3),
        "mea_filter": mea,
        "max_duration_filter": max_duration,
    }


def _parse_local_file(filename: str, sampling_rate: float, mea: Optional[int], max_duration: Optional[float]) -> SpikeData:
    """Parse CSV file — runs in thread pool."""
    import pandas as pd
    filepath = Path("data") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found in data/")

    usecols = None
    df_peek = pd.read_csv(filepath, nrows=2)
    is_finalspark = '_time' in df_peek.columns and '_value' in df_peek.columns and 'index' in df_peek.columns
    if is_finalspark:
        usecols = ['_time', '_value', 'index']

    df = pd.read_csv(filepath, usecols=usecols)

    if is_finalspark:
        times_dt = pd.to_datetime(df['_time'], utc=True, format='mixed')
        t0 = times_dt.min()
        time_sec = (times_dt - t0).dt.total_seconds().values
        electrodes = df['index'].values % 32
        amplitudes = df['_value'].values

        mask = np.ones(len(time_sec), dtype=bool)
        if mea is not None:
            mea_start = mea * 8
            mea_end = mea_start + 8
            mask &= (electrodes >= mea_start) & (electrodes < mea_end)
        if max_duration is not None:
            mask &= time_sec <= max_duration

        if not mask.all():
            time_sec = time_sec[mask]
            electrodes = electrodes[mask]
            amplitudes = amplitudes[mask]

        data = SpikeData(times=time_sec, electrodes=electrodes, amplitudes=amplitudes, sampling_rate=sampling_rate)
    else:
        data = load_file(str(filepath), sampling_rate=sampling_rate)

    return data


@app.get("/api/local-files")
async def list_local_files():
    """List CSV files available in data/ directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return {"files": []}
    files = [f.name for f in data_dir.iterdir() if f.suffix in ('.csv', '.h5', '.hdf5', '.parquet', '.json')]
    return {"files": files}


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
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)

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
    import asyncio
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_full_summary, data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/api/analysis/{dataset_id}/quality")
async def analyze_quality(dataset_id: str):
    """Data quality assessment — SNR, violations, gaps, issues."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_quality_metrics, data)


@app.get("/api/analysis/{dataset_id}/firing-rates")
async def analyze_firing_rates(
    dataset_id: str,
    bin_size: float = Query(1.0, ge=0.1, le=60.0, description="Bin size in seconds"),
):
    """Compute time-binned firing rates per electrode.

    Guards bin_size against explosive output shape. FinalSpark 2.6M spikes
    over 118h with bin_size=0.1s = 4.2M bins × 32 electrodes = 1.1B floats
    (8.7GB array). Auto-widen the bin for long recordings so the output stays
    under ~5M cells and computation stays under the 45s heavy timeout.
    """
    import asyncio
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=500_000)

    # Adaptive bin guard: target max 150K time bins regardless of requested size.
    # Far below the O(N²) thresholds but prevents ambiguous timeouts when users
    # pass small bin_size on long recordings.
    _MAX_BINS = 150_000
    requested_bin = bin_size
    n_bins_requested = data.duration / max(bin_size, 0.001)
    if n_bins_requested > _MAX_BINS:
        bin_size = max(data.duration / _MAX_BINS, 0.1)
        logger.info(
            "firing_rates.bin_widen duration=%.0fs requested_bin=%.3fs widened_bin=%.3fs spikes=%d",
            data.duration, requested_bin, bin_size, data.n_spikes,
        )

    result = await _run_heavy(compute_firing_rates, data, bin_size_sec=bin_size)
    result = _sanitize(result)
    if subsampled:
        result["_subsampled"] = True
        result["_subsampled_spikes"] = data.n_spikes
    if bin_size != requested_bin:
        result["_bin_widened"] = True
        result["_bin_requested"] = requested_bin
        result["_bin_actual"] = bin_size
    return result


@app.get("/api/analysis/{dataset_id}/isi")
async def analyze_isi(
    dataset_id: str,
    electrode: Optional[int] = None,
):
    """Inter-spike interval analysis."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_isi, data, electrode=electrode)


@app.get("/api/analysis/{dataset_id}/amplitudes")
async def analyze_amplitudes(dataset_id: str):
    """Amplitude distribution statistics per electrode."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_amplitude_stats, data)


@app.get("/api/analysis/{dataset_id}/temporal")
async def analyze_temporal(
    dataset_id: str,
    bin_size: float = Query(60.0, ge=1.0, le=3600.0),
):
    """Temporal dynamics — trends, stationarity, Fano factors."""
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=500_000)
    try:
        result = _sanitize(await _run_heavy(compute_temporal_dynamics, data, bin_size_sec=bin_size))
        if subsampled:
            result["_subsampled"] = True
            result["_subsampled_spikes"] = data.n_spikes
        return result
    except Exception as e:
        return {"error": str(e), "partial": True}


@app.post("/api/analysis/{dataset_id}/spike-sorting")
async def analyze_spike_sorting(
    dataset_id: str,
    params: SpikeSortingParams = SpikeSortingParams(),
):
    """Spike sorting — cluster spikes by waveform shape."""
    data, _ = _get_dataset_capped(dataset_id)
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
    import asyncio
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=30_000)
    t0 = time.time()
    result = await _run_heavy(detect_bursts, data)
    result = _sanitize(result)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    if subsampled:
        result["_subsampled"] = True
        result["_subsampled_spikes"] = data.n_spikes
    return result


@app.get("/api/analysis/{dataset_id}/bursts/profiles")
async def analyze_burst_profiles(
    dataset_id: str,
    min_electrodes: int = Query(3, ge=2),
    window_ms: float = Query(50.0),
):
    """Detailed burst profiles — recruitment order, temporal shape."""
    data, _ = _get_dataset_capped(dataset_id)
    result = _sanitize(await _run_heavy(detect_bursts, data))
    return result.get("network", result)


@app.get("/api/analysis/{dataset_id}/bursts/single-channel")
async def analyze_single_channel_bursts(
    dataset_id: str,
    electrode: int = Query(..., ge=0),
    max_isi_ms: float = Query(100.0),
    min_spikes: int = Query(3),
):
    """Single-channel burst detection using ISI method."""
    data, _ = _get_dataset_capped(dataset_id)
    def _compute():
        spike_times = data.times[data.electrodes == electrode]
        bursts = detect_single_channel_bursts(spike_times, max_isi_ms=max_isi_ms, min_spikes=min_spikes)
        return {"electrode": electrode, "bursts": bursts, "n_bursts": len(bursts)}
    return await _run_heavy(_compute)


@app.get("/api/analysis/{dataset_id}/connectivity")
async def analyze_connectivity(
    dataset_id: str,
    window_ms: float = Query(10.0, ge=1.0),
    min_strength: float = Query(0.02, ge=0.0),
    subset: Optional[str] = Query(None, description="Time subset: 1h, 10h, full, or Ns/Nmin/Nh"),
):
    """Functional connectivity graph -- fast default (CCG + co-firing).

    Add query params include_te=true, include_plv=true, include_mi=true,
    include_granger=true, or include_all=true for expensive measures.
    """
    import asyncio
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset)
    t0 = time.time()
    conn = await _run_heavy(compute_connectivity_graph, data, window_ms, timeout_sec=_TIMEOUT_HEAVY_SEC)
    try:
        result = connectivity_to_dict(conn)
    except Exception:
        result = {"nodes": [], "edges": [], "n_edges": 0, "graph_metrics": {}, "measures_computed": []}
    result = _sanitize(result)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    if subsampled:
        result["_subsampled"] = True
        result["_subsampled_spikes"] = data.n_spikes
    return result


@app.get("/api/analysis/{dataset_id}/cross-correlation")
async def analyze_cross_correlation(
    dataset_id: str,
    max_lag_ms: float = Query(50.0),
    bin_size_ms: float = Query(1.0),
    subset: Optional[str] = Query(None),
):
    """Pairwise cross-correlograms between all electrodes."""
    import asyncio
    # O(pairs × lag_bins) — 600s duration cap stays comfortably under timeout
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset, max_duration_sec=600)
    raw = await _run_heavy(compute_cross_correlation, data, max_lag_ms, bin_size_ms, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result = _sanitize(raw)
    if subsampled:
        result["_subsampled"] = True
    return result


@app.get("/api/analysis/{dataset_id}/transfer-entropy")
async def analyze_transfer_entropy(
    dataset_id: str,
    bin_size_ms: float = Query(5.0),
    history_bins: int = Query(5, ge=1, le=20),
    n_surrogates: int = Query(50, ge=10, le=500, description="Surrogates per pair for significance testing"),
    subset: Optional[str] = Query(None),
):
    """Transfer entropy — directional information flow between electrodes.

    Complexity = PAIRS × SURROGATES × BINS. On 32 electrodes = 1024 pairs.
    Default n_surrogates=50 (lowered from 200) → 50K TE calls, runs in ~60s
    on 300s duration. Users can request more surrogates for tighter p-values.
    """
    import asyncio
    data, subsampled = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset, max_duration_sec=300)
    raw = await _run_heavy(
        compute_transfer_entropy, data, bin_size_ms, history_bins,
        n_surrogates=n_surrogates,
        timeout_sec=_TIMEOUT_VERY_HEAVY_SEC,
    )
    result = _sanitize(raw)
    if subsampled:
        result["_subsampled"] = True
    result["_n_surrogates"] = n_surrogates
    return result


# ═══════════ INFORMATION THEORY ═══════════

@app.get("/api/analysis/{dataset_id}/entropy")
async def analyze_entropy(dataset_id: str, bin_size_ms: float = Query(10.0)):
    """Shannon entropy of spike trains per electrode."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_spike_train_entropy, data, bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/mutual-information")
async def analyze_mutual_info(dataset_id: str, bin_size_ms: float = Query(10.0)):
    """Pairwise mutual information between electrodes."""
    # Pairwise O(N²) — tighter cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000)
    return await _run_heavy(compute_mutual_information, data, bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/complexity")
async def analyze_complexity(dataset_id: str, bin_size_ms: float = Query(5.0)):
    """Lempel-Ziv complexity of spike trains."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_lempel_ziv_complexity, data, bin_size_ms=bin_size_ms)


# ═══════════ SPECTRAL ═══════════

@app.get("/api/analysis/{dataset_id}/power-spectrum")
async def analyze_power_spectrum(dataset_id: str, bin_size_ms: float = Query(1.0)):
    """Power spectral density and frequency band analysis."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_power_spectrum, data, bin_size_ms=bin_size_ms)


@app.get("/api/analysis/{dataset_id}/coherence")
async def analyze_coherence(dataset_id: str, bin_size_ms: float = Query(1.0)):
    """Spectral coherence between electrode pairs."""
    # Pairwise spectral — tighter cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000)
    return await _run_heavy(compute_coherence, data, bin_size_ms=bin_size_ms)


# ═══════════ CRITICALITY ═══════════

@app.get("/api/analysis/{dataset_id}/avalanches")
async def analyze_avalanches(dataset_id: str, bin_size_ms: float = Query(5.0)):
    """Neuronal avalanche detection and criticality assessment."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(detect_avalanches, data, bin_size_ms=bin_size_ms))


# ═══════════ DIGITAL TWIN ═══════════

@app.get("/api/analysis/{dataset_id}/digital-twin/fit")
async def fit_twin(dataset_id: str):
    """Fit LIF neuron model parameters from recorded data."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(fit_lif_parameters, data)


@app.post("/api/analysis/{dataset_id}/digital-twin/simulate")
async def simulate_twin(dataset_id: str, duration_ms: float = Query(5000.0)):
    """Simulate digital twin and compare with real data."""
    data, _ = _get_dataset_capped(dataset_id)
    def _compute():
        params = fit_lif_parameters(data)
        sim = simulate_lif_network(params, duration_ms=duration_ms)
        comparison = compare_real_vs_simulated(data, sim)
        return {"simulation": sim, "comparison": comparison, "parameters": params}
    return await _run_heavy(_compute)


# ═══════════ ML PIPELINE ═══════════

@app.get("/api/analysis/{dataset_id}/anomalies")
async def analyze_anomalies(dataset_id: str, window_sec: float = Query(1.0)):
    """Anomaly detection using Isolation Forest."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(detect_anomalies, data, window_sec=window_sec)


@app.get("/api/analysis/{dataset_id}/states")
async def analyze_states(dataset_id: str, window_sec: float = Query(2.0)):
    """Neural activity state classification (resting, active, bursting)."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(classify_states, data, window_sec=window_sec)


@app.get("/api/analysis/{dataset_id}/pca")
async def analyze_pca(dataset_id: str, n_components: int = Query(3)):
    """PCA embedding of neural state space."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(compute_pca_embedding, data, n_components=n_components)


@app.get("/api/analysis/{dataset_id}/features")
async def analyze_features(dataset_id: str, window_sec: float = Query(1.0)):
    """Extract multi-scale features from spike data."""
    data, _ = _get_dataset_capped(dataset_id)
    return await _run_heavy(extract_features, data, window_sec=window_sec)


# ═══════════ PLASTICITY & LEARNING ═══════════

@app.get("/api/analysis/{dataset_id}/stdp")
async def analyze_stdp(dataset_id: str, max_lag_ms: float = Query(30.0)):
    """STDP analysis — spike-timing dependent plasticity curves."""
    # O(pairs × lag bins) — cap same as other pairwise analyses
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000)
    return _sanitize(await _run_heavy(compute_stdp_matrix, data, max_lag_ms=max_lag_ms))


@app.get("/api/analysis/{dataset_id}/learning")
async def analyze_learning(dataset_id: str, window_sec: float = Query(60.0)):
    """Detect learning episodes — changes in plasticity over time."""
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000)
    return _sanitize(await _run_heavy(detect_learning_episodes, data, window_sec=window_sec))


# ═══════════ ORGANOID IQ ═══════════

@app.get("/api/analysis/{dataset_id}/iq")
async def analyze_iq(dataset_id: str, subset: Optional[str] = Query(None)):
    """Organoid Intelligence Quotient — composite computational capacity score."""
    import asyncio
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    return _sanitize(await _run_heavy(compute_organoid_iq, data, timeout_sec=_TIMEOUT_HEAVY_SEC))


# ═══════════ PREDICTIONS ═══════════

@app.get("/api/analysis/{dataset_id}/predict/firing-rates")
async def predict_rates(dataset_id: str, forecast_sec: float = Query(300.0)):
    """Predict future firing rates with confidence intervals."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(predict_firing_rates, data, forecast_sec=forecast_sec))


@app.get("/api/analysis/{dataset_id}/predict/bursts")
async def predict_bursts(dataset_id: str, window_sec: float = Query(10.0)):
    """Predict burst probability in next time window."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(predict_burst_probability, data, window_sec=window_sec))


@app.get("/api/analysis/{dataset_id}/health")
async def analyze_health(dataset_id: str):
    """Estimate organoid health and viability."""
    import asyncio
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(estimate_organoid_health, data))


# ═══════════ NEURAL REPLAY ═══════════

@app.get("/api/analysis/{dataset_id}/replay")
async def analyze_replay(dataset_id: str, min_similarity: float = Query(0.3), subset: Optional[str] = Query(None)):
    """Detect neural replay — memory consolidation signatures."""
    # O(N²) sequence matching — cap at 50K to stay under ~10s on FinalSpark
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000, subset=subset)
    return _sanitize(await _run_heavy(detect_replay, data, min_similarity=min_similarity, timeout_sec=_TIMEOUT_HEAVY_SEC))


@app.get("/api/analysis/{dataset_id}/sequences")
async def analyze_sequences(dataset_id: str, min_length: int = Query(3), subset: Optional[str] = Query(None)):
    """Detect repeated neural sequences — functional circuits."""
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000, subset=subset)
    return _sanitize(await _run_heavy(detect_sequence_replay, data, min_sequence_length=min_length, timeout_sec=_TIMEOUT_HEAVY_SEC))


# ═══════════ RESERVOIR COMPUTING ═══════════

@app.get("/api/analysis/{dataset_id}/memory-capacity")
async def analyze_memory_capacity(dataset_id: str, max_delay: int = Query(20)):
    """Estimate memory capacity of neural network as reservoir computer."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(estimate_memory_capacity, data, max_delay=max_delay))


@app.get("/api/analysis/{dataset_id}/nonlinearity")
async def analyze_nonlinearity(dataset_id: str):
    """Benchmark nonlinear computational capability."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(benchmark_nonlinear_computation, data))


# ═══════════ FINGERPRINTING ═══════════

@app.get("/api/analysis/{dataset_id}/fingerprint")
async def analyze_fingerprint(dataset_id: str, subset: Optional[str] = Query(None)):
    """Compute unique organoid fingerprint — identity signature."""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    return _sanitize(await _run_heavy(compute_fingerprint, data, timeout_sec=_TIMEOUT_HEAVY_SEC))


# ═══════════ SONIFICATION ═══════════

@app.get("/api/analysis/{dataset_id}/sonify")
async def sonify(dataset_id: str, speed: float = Query(10.0), duration: Optional[float] = None):
    """Convert neural activity to audio WAV (base64 encoded)."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(generate_sonification, data, speed_factor=speed, duration_sec=duration))


@app.get("/api/analysis/{dataset_id}/rhythms")
async def analyze_rhythms(dataset_id: str):
    """Analyze rhythmic structure of neural activity."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(compute_rhythmic_analysis, data))


# ═══════════ CAUSAL EMERGENCE ═══════════

@app.get("/api/analysis/{dataset_id}/emergence")
async def analyze_emergence(dataset_id: str, subset: Optional[str] = Query(None)):
    """Compute integrated information (Phi) and causal emergence."""
    import asyncio
    # IIT Phi is O(2^N electrodes), mostly sensitive to duration via bin count
    data, _ = _get_dataset_capped(dataset_id, max_spikes=20_000, subset=subset, max_duration_sec=300)
    return _sanitize(await _run_heavy(compute_integrated_information, data, timeout_sec=_TIMEOUT_VERY_HEAVY_SEC))


# ═══════════ BREAKTHROUGH MODULES ═══════════

@app.get("/api/analysis/{dataset_id}/attractors")
async def analyze_attractors(dataset_id: str, min_visits: int = Query(3)):
    """Map attractor landscape — find memory traces as dynamical states."""
    try:
        data, _ = _get_dataset_capped(dataset_id)
        return _sanitize(await _run_heavy(map_attractor_landscape, data, min_visits=min_visits))
    except Exception as e:
        return {"error": str(e), "partial": True}


@app.get("/api/analysis/{dataset_id}/state-space")
async def analyze_state_space(dataset_id: str):
    """Analyze geometry of neural state space."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(await _run_heavy(compute_state_space_geometry, data))


@app.get("/api/analysis/{dataset_id}/phase-transitions")
async def analyze_phase_transitions(dataset_id: str, window_sec: float = Query(5.0), subset: Optional[str] = Query(None)):
    """Detect phase transitions — moments of neural reorganization."""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    return _sanitize(await _run_heavy(detect_phase_transitions, data, window_sec=window_sec, timeout_sec=_TIMEOUT_HEAVY_SEC))


@app.get("/api/analysis/{dataset_id}/predictive-coding")
async def analyze_predictive_coding(dataset_id: str, subset: Optional[str] = Query(None)):
    """Measure predictive coding — does the organoid generate predictions?"""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    return _sanitize(await _run_heavy(measure_predictive_coding, data, timeout_sec=_TIMEOUT_HEAVY_SEC))


@app.get("/api/analysis/{dataset_id}/weights")
async def analyze_weights(dataset_id: str, subset: Optional[str] = Query(None)):
    """Infer synaptic weight matrix from spike timing."""
    # O(N²) pairwise correlation, 300s duration cap keeps 1h subset tractable
    data, _ = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset, max_duration_sec=300)
    return _sanitize(await _run_heavy(infer_synaptic_weights, data, timeout_sec=_TIMEOUT_HEAVY_SEC))


@app.get("/api/analysis/{dataset_id}/weight-tracking")
async def analyze_weight_tracking(dataset_id: str, window_sec: float = Query(30.0), subset: Optional[str] = Query(None)):
    """Track synaptic weight changes over time — watch learning happen."""
    try:
        # O(N²) per-window × N windows — 600s cap (20 windows of 30s) keeps tractable
        data, _ = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset, max_duration_sec=600)
        return _sanitize(await _run_heavy(track_weight_changes, data, window_sec=window_sec, timeout_sec=_TIMEOUT_HEAVY_SEC))
    except Exception as e:
        return {"error": str(e), "partial": True}


@app.get("/api/analysis/{dataset_id}/multiscale")
async def analyze_multiscale(dataset_id: str, subset: Optional[str] = Query(None)):
    """Multi-timescale complexity analysis — find operating frequency."""
    # Iterates over multiple bin scales; 600s duration keeps under 90s even with 6-8 scales
    data, _ = _get_dataset_capped(dataset_id, subset=subset, max_duration_sec=600)
    return _sanitize(await _run_heavy(compute_multiscale_complexity, data, timeout_sec=_TIMEOUT_HEAVY_SEC))


# ═══════════ FULL REPORT ═══════════

@app.get("/api/analysis/{dataset_id}/full-report")
async def full_report(dataset_id: str, subset: Optional[str] = Query(None)):
    """Run ALL 21 analyses and generate comprehensive report."""
    import asyncio
    # Full report chains 21 analyses; must stay under 240s — clamp duration to 180s
    data, subsampled = _get_dataset_capped(dataset_id, subset=subset, max_duration_sec=180)
    report = await _run_heavy(generate_full_report, data, timeout_sec=_TIMEOUT_VERY_HEAVY_SEC)
    report = _sanitize(report)
    if subsampled:
        report["_subsampled"] = True
        report["_subsampled_spikes"] = data.n_spikes
    return report


# ═══════════ DISCOVERY ANALYSIS ═══════════

@app.get("/api/analysis/{dataset_id}/sleep-wake")
async def api_sleep_wake(dataset_id: str, subset: Optional[str] = Query(None)):
    """Detect sleep-wake-like cycles in organoid activity."""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    t0 = time.time()
    result = await _run_heavy(analyze_sleep_wake, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/habituation")
async def api_habituation(dataset_id: str, subset: Optional[str] = Query(None)):
    """Detect habituation — response decay to repeated patterns."""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    t0 = time.time()
    result = await _run_heavy(detect_repeated_patterns, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/metastability")
async def api_metastability(dataset_id: str, subset: Optional[str] = Query(None)):
    """Analyze metastability — brain-like state switching dynamics."""
    # Kuramoto computation previously took 406s on full FinalSpark — clamp to 5min
    data, _ = _get_dataset_capped(dataset_id, subset=subset, max_duration_sec=300)
    t0 = time.time()
    result = await _run_heavy(analyze_metastability, data, timeout_sec=_TIMEOUT_VERY_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/information-flow")
async def api_information_flow(dataset_id: str, subset: Optional[str] = Query(None)):
    """Map directed information flow using Granger causality."""
    # Granger is O(pairs × history) — tightest cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset)
    t0 = time.time()
    result = await _run_heavy(compute_granger_causality, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/motifs")
async def api_motifs(dataset_id: str, subset: Optional[str] = Query(None)):
    """Enumerate network motifs (3-node subgraph patterns)."""
    # 3-node motif enumeration is O(V³) on the connectivity graph — tightest cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=10_000, subset=subset)
    t0 = time.time()
    result = await _run_heavy(enumerate_motifs, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/energy-landscape")
async def api_energy_landscape(dataset_id: str):
    """Compute Ising model energy landscape."""
    # Ising fit is expensive per-spike — tighter cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=50_000)
    t0 = time.time()
    result = await _run_heavy(fit_ising_model, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.post("/api/analysis/{dataset_id}/design-stimulus")
async def api_design_stimulus(dataset_id: str, generations: int = 15, population_size: int = 10):
    """Evolve optimal stimulation protocol using digital twin."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(evolve_protocol, data, generations=generations, population_size=population_size)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consciousness")
async def api_consciousness(dataset_id: str, subset: Optional[str] = Query(None)):
    """Composite consciousness assessment score.
    Internally calls IIT Phi + PCI + transfer entropy — three heavies
    compound. 120s cap + TE sub-call uses n_surrogates=30 (see
    consciousness_metrics.py). Keeps total runtime under the 240s
    very-heavy timeout on any input."""
    data, _ = _get_dataset_capped(dataset_id, max_spikes=20_000, subset=subset, max_duration_sec=120)
    t0 = time.time()
    result = await _run_heavy(compute_consciousness_score, data, timeout_sec=_TIMEOUT_VERY_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/comparative")
async def api_comparative(dataset_id: str):
    """Compare organoid with reference neural systems."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compare_with_references, data)
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
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_dishbrain_session, data, n_episodes=n_episodes, reward_strategy=reward_strategy, game=game)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/closed-loop/cartpole")
async def api_cartpole(
    dataset_id: str,
    n_trials: int = Query(50, ge=1, le=200),
    reward_strategy: str = Query("dopamine"),
):
    """CartPole benchmark adapted for biological neural networks."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_cartpole_benchmark, data, n_trials=n_trials, reward_strategy=reward_strategy)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/closed-loop/strategies")
async def api_compare_strategies(dataset_id: str, n_episodes: int = Query(15, ge=5, le=50)):
    """Compare all 4 reward strategies (hebbian/dopamine/contrastive/reinforce)."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compare_reward_strategies, data, n_episodes=n_episodes)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ CURRICULUM LEARNING ═══════════

@app.get("/api/experiments/{dataset_id}/curriculum")
async def api_curriculum(dataset_id: str):
    """Run full 4-stage adaptive curriculum learning protocol."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_curriculum, data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/curriculum/stage")
async def api_curriculum_stage(dataset_id: str):
    """Assess which curriculum stage the organoid is ready for."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(get_current_stage(data))


@app.get("/api/experiments/{dataset_id}/curriculum/simulate")
async def api_curriculum_simulate(
    dataset_id: str,
    stage: int = Query(1, ge=1, le=4),
    n_trials: int = Query(30, ge=10, le=100),
):
    """Simulate a single curriculum stage with trial-by-trial output."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_stage, data, stage=stage, n_trials=n_trials)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ MEMORY BATTERY ═══════════

@app.get("/api/experiments/{dataset_id}/memory")
async def api_memory_battery(dataset_id: str):
    """Run complete memory battery: working + short-term + long-term + associative."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_memory_battery, data)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/memory/working")
async def api_memory_working(dataset_id: str, n_trials: int = Query(40, ge=10, le=100)):
    """Delay-match-to-sample working memory test."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(test_working_memory(data, n_trials=n_trials))


@app.get("/api/experiments/{dataset_id}/memory/short-term")
async def api_memory_stm(dataset_id: str, max_span: int = Query(9, ge=3, le=15)):
    """Serial-span short-term memory test."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(test_short_term_memory(data, max_span=max_span))


@app.get("/api/experiments/{dataset_id}/memory/long-term")
async def api_memory_ltm(dataset_id: str, n_patterns: int = Query(5, ge=2, le=20)):
    """Spaced-retention long-term memory test."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(test_long_term_memory(data, n_patterns=n_patterns))


@app.get("/api/experiments/{dataset_id}/memory/associative")
async def api_memory_am(dataset_id: str, n_pairs: int = Query(6, ge=2, le=20)):
    """Paired-associate / Hopfield-style associative memory test."""
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_pong, data, n_games=n_games, encoding=encoding, decoding=decoding, reward_rule=reward_rule)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_full_benchmark, data, n_trials_per_gate=n_trials_per_gate, readout=readout)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_gate_benchmark, data, gate=gate.upper(), n_trials=n_trials, readout=readout)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/experiments/{dataset_id}/xor/difficulty")
async def api_xor_difficulty(dataset_id: str):
    """Estimate XOR difficulty for this organoid (Fisher discriminant)."""
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(start_experiment(experiment_id, data, name=name, experiment_type=experiment_type))


@app.post("/api/experiments/tracker/{experiment_id}/end/{dataset_id}")
async def api_tracker_end(
    experiment_id: str,
    dataset_id: str,
    notes: Optional[str] = None,
):
    """End experiment — record post-intervention and compute delta."""
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(generate_draft, data, analyses={})
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/publish/{dataset_id}/abstract")
async def api_publish_abstract(dataset_id: str):
    """Generate abstract only — fast 250-word structured abstract."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(generate_abstract_only(data, analyses={}))


@app.post("/api/publish/{dataset_id}/methods")
async def api_publish_methods(dataset_id: str):
    """Generate methods section only."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(generate_methods_section(data, analyses={}))


# ═══════════ ETHICS ═══════════

@app.get("/api/analysis/{dataset_id}/ethics")
async def api_ethics(
    dataset_id: str,
    culture_age_days: Optional[int] = None,
    organoid_type: Optional[str] = None,
):
    """Full ethical assessment: consciousness indicators, sentience risk, guidelines, compliance."""
    # Internally calls the full consciousness/Phi pipeline — tight cap
    data, _ = _get_dataset_capped(dataset_id, max_spikes=20_000)
    t0 = time.time()
    result = await _run_heavy(assess_ethics, data, culture_age_days=culture_age_days, organoid_type=organoid_type)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consciousness")
async def api_consciousness_indicators(dataset_id: str):
    """Consciousness proxy indicators: Phi, global workspace, criticality, complexity."""
    data, _ = _get_dataset_capped(dataset_id)
    return _sanitize(assess_consciousness_indicators(data))


@app.get("/api/analysis/{dataset_id}/sentience-risk")
async def api_sentience_risk(dataset_id: str):
    """Sentience risk score: nociception proxy, stress, valence, distress signals."""
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
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
async def api_turing_test(dataset_id: str, subset: Optional[str] = Query(None)):
    """Run organoid Turing test — compare real data vs Poisson and LIF models."""
    # Internally runs LIF simulation + statistical tests — clamp to 5min
    data, _ = _get_dataset_capped(dataset_id, subset=subset, max_duration_sec=300)
    t0 = time.time()
    result = await _run_heavy(run_turing_test, data, timeout_sec=_TIMEOUT_VERY_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.post("/api/analysis/{dataset_id}/architecture-search")
async def api_architecture_search(dataset_id: str, generations: int = 15, population_size: int = 10):
    """Neural Architecture Search for optimal stimulation protocol."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(search_optimal_protocol, data, population_size=population_size, generations=generations)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/hybrid-benchmark")
async def api_hybrid_benchmark(dataset_id: str):
    """Benchmark hybrid bio-digital AI vs pure digital vs pure biological."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(benchmark_hybrid, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ LEARNING & MEMORY ═══════════

@app.get("/api/analysis/{dataset_id}/forgetting")
async def api_forgetting(dataset_id: str, subset: Optional[str] = Query(None)):
    """Measure catastrophic forgetting — do early patterns survive?"""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    t0 = time.time()
    result = await _run_heavy(compute_retention_curve, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/channel-capacity")
async def api_channel_capacity(dataset_id: str):
    """Estimate information channel capacity (multi-bit memory)."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(estimate_channel_capacity, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/population-codes")
async def api_population_codes(dataset_id: str):
    """Measure population code diversity (distinct neural states)."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(measure_population_code_diversity, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/topology")
async def api_topology(dataset_id: str):
    """Compute Betti numbers (topological data analysis)."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_betti_numbers, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/topological-complexity")
async def api_topo_complexity(dataset_id: str):
    """Compute topological complexity score."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_topological_complexity, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/consolidation")
async def api_consolidation(dataset_id: str):
    """Detect memory consolidation events during offline periods."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(detect_consolidation_events, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/transfer")
async def api_transfer(dataset_id: str):
    """Measure transfer learning between recording segments."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(measure_transfer, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/rsa")
async def api_rsa(dataset_id: str):
    """Representational Similarity Analysis."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_representational_similarity, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ CONNECTOMICS ═══════════

@app.get("/api/analysis/{dataset_id}/connectome")
async def api_connectome(dataset_id: str):
    """Build full functional connectome with graph theory metrics."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(build_full_connectome, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/communities")
async def api_communities(dataset_id: str):
    """Detect network communities (modules) via spectral clustering."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(detect_communities, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/graph-theory")
async def api_graph_theory(dataset_id: str):
    """Compute rich-club, small-world, efficiency, betweenness centrality."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_graph_theory_metrics, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/effective-connectivity")
async def api_effective_connectivity(dataset_id: str):
    """Estimate directed causal connections (effective connectivity)."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(estimate_effective_connectivity, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/causal-hierarchy")
async def api_causal_hierarchy(dataset_id: str):
    """Order electrodes by causal influence hierarchy."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(compute_causal_hierarchy, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ BIO-INSPIRED & ETHICS ═══════════

@app.post("/api/analysis/{dataset_id}/evolve-programs")
async def api_evolve_programs(dataset_id: str, generations: int = 20):
    """Genetic programming — evolve stimulation program trees."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(evolve_programs, data, generations=generations)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/homeostasis")
async def api_homeostasis(dataset_id: str, subset: Optional[str] = Query(None)):
    """Monitor homeostatic plasticity — firing rate self-regulation."""
    data, _ = _get_dataset_capped(dataset_id, subset=subset)
    t0 = time.time()
    result = await _run_heavy(monitor_homeostasis, data, timeout_sec=_TIMEOUT_HEAVY_SEC)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/suffering")
async def api_suffering(dataset_id: str):
    """Detect distress/suffering patterns — ethical monitoring."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(detect_suffering, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/welfare")
async def api_welfare(dataset_id: str):
    """Generate comprehensive welfare report."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(generate_welfare_report, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/swarm")
async def api_swarm(dataset_id: str, n_organoids: int = 4):
    """Simulate multi-organoid swarm intelligence."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_swarm, data, n_organoids=n_organoids)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


@app.get("/api/analysis/{dataset_id}/morphology")
async def api_morphology(dataset_id: str):
    """Analyze morphological computing — structure vs function."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(analyze_morphological_computation, data)
    result["_computation_time_ms"] = (time.time() - t0) * 1000
    return _sanitize(result)


# ═══════════ WEBSOCKET — REAL-TIME STREAMING ═══════════

import asyncio

# Per-IP WebSocket connection tracker — stops one user from opening 10K sockets.
# Keyed by IP (same extraction logic as HTTP rate limiter).
_ws_connections: dict[str, int] = {}
_ws_lock = _rl_threading.Lock()
_WS_MAX_PER_IP = 3


@app.websocket("/ws/spikes")
async def ws_live_spikes(ws: WebSocket):
    """Stream synthetic spike data in real-time via WebSocket.

    Sends JSON frames every 100ms with spike events.
    Protocol: connect → receive frames → disconnect.
    Each frame: {"spikes": [{"time": float, "electrode": int, "amplitude": float}], "timestamp": float}

    Connections per IP are capped — WebSockets are cheap to open but expensive
    to serve (each spawns an async loop that runs forever). Without a cap, a
    single abusive client could exhaust memory.
    """
    # Extract client IP through Caddy (same logic as _client_ip_from_request).
    xff = ws.headers.get("x-forwarded-for", "")
    client_ip = (xff.split(",")[0].strip() if xff else ws.headers.get("x-real-ip", "").strip()) or (ws.client.host if ws.client else "unknown")

    with _ws_lock:
        if _ws_connections.get(client_ip, 0) >= _WS_MAX_PER_IP:
            logger.warning("ws.rejected ip=%s reason=per_ip_limit current=%d", client_ip, _ws_connections[client_ip])
            await ws.close(code=1008, reason="Too many connections from your IP")
            return
        _ws_connections[client_ip] = _ws_connections.get(client_ip, 0) + 1

    logger.info("ws.open ip=%s total=%d", client_ip, _ws_connections[client_ip])
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
    finally:
        with _ws_lock:
            remaining = _ws_connections.get(client_ip, 1) - 1
            if remaining <= 0:
                _ws_connections.pop(client_ip, None)
            else:
                _ws_connections[client_ip] = remaining
        logger.info("ws.close ip=%s remaining=%d", client_ip, remaining if remaining > 0 else 0)


# ═══════════ MULTI-ORGANOID ═══════════

@app.get("/api/analysis/{dataset_id}/multi-organoid")
async def analyze_multi_organoid(
    dataset_id: str,
    electrodes_per_organoid: int = Query(8, ge=1, le=32, description="Electrodes per virtual organoid"),
):
    """Compare multiple virtual organoids from a multi-MEA dataset."""
    data, _ = _get_dataset_capped(dataset_id)
    cached = _cache_get(dataset_id, f"multi-organoid-{electrodes_per_organoid}")
    if cached:
        return cached
    t0 = time.time()
    result = await _run_heavy(compare_organoids, data, electrodes_per_organoid=electrodes_per_organoid)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    if mode == "trends":
        result = await _run_heavy(detect_trends, data, window_sec=window_sec)
    elif mode == "critical":
        result = await _run_heavy(find_critical_moments, data, window_sec=window_sec)
    else:
        evolution = await _run_heavy(track_evolution, data, window_sec=window_sec)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()

    if stim_times:
        stim_list = [float(t.strip()) for t in stim_times.split(",")]
        result = await _run_heavy(detect_response, data, stim_times=stim_list, window_ms=window_ms)
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
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(run_optimization_loop, data, n_iterations=n_iterations, objective=objective)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


# ═══════════ EXPERIMENT PROTOCOLS ═══════════

@app.post("/api/protocols/center-activity/{dataset_id}/simulate")
async def protocol_center_activity(dataset_id: str, n_steps: int = Query(20)):
    """Center of Activity Protocol -- shift neural activity via distant electrode stimulation."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    ca = compute_center_of_activity(data)
    shift = simulate_ca_shift(data, n_steps=n_steps)
    result = {"center_of_activity": ca, "simulation": shift}
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/protocols/dishbrain-pong/{dataset_id}/simulate")
async def protocol_dishbrain_pong(dataset_id: str, n_trials: int = Query(100)):
    """DishBrain Pong Protocol -- train organoid to play Pong via free energy principle."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_pong_game, data, n_trials=n_trials)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/protocols/brainoware/{dataset_id}/simulate")
async def protocol_brainoware(dataset_id: str, n_classes: int = Query(4), n_samples: int = Query(100)):
    """Brainoware Reservoir Computing -- organoid as nonlinear reservoir for classification."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_reservoir_classification, data, n_classes=n_classes, n_samples=n_samples)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/protocols/cartpole/{dataset_id}/simulate")
async def protocol_cartpole(dataset_id: str, n_episodes: int = Query(50), max_steps: int = Query(200)):
    """Cart-Pole Coaching -- balance inverted pendulum with adaptive coaching."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_cartpole, data, n_episodes=n_episodes, max_steps=max_steps)
    result["_computation_time_ms"] = round((time.time() - t0) * 1000, 1)
    return _sanitize(result)


@app.post("/api/protocols/dopamine/{dataset_id}/simulate")
async def protocol_dopamine(dataset_id: str, n_trials: int = Query(100)):
    """Dopamine UV Reinforcement -- chemical reward via UV-activated dopamine release."""
    data, _ = _get_dataset_capped(dataset_id)
    t0 = time.time()
    result = await _run_heavy(simulate_dopamine_training, data, n_trials=n_trials)
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
    data, _ = _get_dataset_capped(dataset_id)
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
    data, _ = _get_dataset_capped(dataset_id)
    return data.to_dict()


# ═══════════ HELPERS ═══════════

def _get_dataset(dataset_id: str) -> SpikeData:
    if dataset_id in datasets:
        return datasets[dataset_id]
    if dataset_id in _evicted_ids:
        # 410 Gone signals "it existed but was released" vs "never existed".
        # Frontend uses the distinction to show "session replaced, reload" UX
        # instead of the generic "dataset not found".
        raise HTTPException(
            status_code=410,
            detail=f"Dataset '{dataset_id}' was evicted to free memory. Reload the page to start a new session.",
        )
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found. Upload data first via POST /api/upload")


# Auto-subsample for heavy endpoints: cap at MAX_SPIKES for O(N²) analyses.
# 20K is a compromise: keeps most pairwise analyses under 30s on the single-
# worker VPS while still giving ~600 spikes per electrode on a 32-channel
# MEA (enough for meaningful statistics). O(N²) endpoints like weights,
# transfer-entropy, motifs, information-flow override this further down.
_MAX_SPIKES_HEAVY = 20_000


# Subset presets — user-selectable time ranges for heavy analyses.
# FinalSpark is 118h; 1h gives ~22K spikes (fast), 10h ~220K (slower but
# richer), full is everything (bounded further by max_spikes cap).
_SUBSET_PRESETS = {
    "1h": 3600.0,
    "10h": 36000.0,
    "full": None,
}


def _parse_subset(subset: Optional[str]) -> Optional[float]:
    """Parse subset query param to seconds. Returns None for 'full' / unrecognized."""
    if not subset:
        return None
    subset = subset.strip().lower()
    if subset in _SUBSET_PRESETS:
        return _SUBSET_PRESETS[subset]
    # Free-form: accept numeric seconds ("3600") or "Nh"/"Nmin"/"Ns"
    try:
        if subset.endswith("h"):
            return float(subset[:-1]) * 3600.0
        if subset.endswith("min"):
            return float(subset[:-3]) * 60.0
        if subset.endswith("s"):
            return float(subset[:-1])
        return float(subset)
    except ValueError:
        return None


def _get_dataset_capped(
    dataset_id: str,
    max_spikes: int = _MAX_SPIKES_HEAVY,
    subset: Optional[str] = None,
    max_duration_sec: Optional[float] = None,
) -> tuple[SpikeData, bool]:
    """Get dataset, optionally slicing to a user-requested time range, then
    auto-subsampling if still too large.

    Order of operations:
      1. User's `subset` preset narrows the time window first (if provided).
      2. `max_duration_sec` enforces a hard upper bound on duration — some
         algorithms (transfer-entropy, consciousness, weights) scale with
         BINS × PAIRS, so duration matters more than spike count. A 1h slice
         with 5ms bins = 720K bins × 1024 pairs which never completes. Cap
         at 300s for those endpoints via max_duration_sec=300.
      3. `max_spikes` cap enforces an upper bound on what the algorithm sees.

    Returns (data, was_subsampled). was_subsampled is True if ANY step
    reduced the data — frontend shows a "subsampled" badge in all cases.
    """
    data = _get_dataset(dataset_id)
    was_subsampled = False

    # Step 1: honour user-requested time slice
    subset_duration = _parse_subset(subset)
    if subset_duration is not None and subset_duration < data.duration:
        t_start = data.time_range[0]
        data = data.get_time_range(t_start, t_start + subset_duration)
        was_subsampled = True

    # Step 2: enforce endpoint-specific duration cap
    if max_duration_sec is not None and data.duration > max_duration_sec:
        t_start = data.time_range[0]
        data = data.get_time_range(t_start, t_start + max_duration_sec)
        was_subsampled = True

    if data.n_spikes <= max_spikes:
        return data, was_subsampled

    # Step 3: safety cap if still over max_spikes
    # Find time cutoff that gives approximately max_spikes
    rate = data.n_spikes / max(data.duration, 1)
    target_duration = max_spikes / max(rate, 0.001)
    t_start = data.time_range[0]
    subset_data = data.get_time_range(t_start, t_start + target_duration)
    return subset_data, True


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8847)
