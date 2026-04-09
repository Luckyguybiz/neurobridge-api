"""Experiment tracking system for organoid biocomputing experiments.

Scientific basis:
    Longitudinal tracking of organoid experiments is critical for
    understanding neural development, plasticity, and computational
    capability over time. This module provides a lightweight, in-memory
    experiment tracker that records pre/post analysis snapshots,
    computes deltas (changes in neural metrics), and maintains a
    history of all experiments.

    Key metrics tracked:
    - Firing rate changes (indicate health and excitability)
    - Burst frequency evolution (network maturation marker)
    - Connectivity changes (synaptic development)
    - Information capacity shifts (computational capability)
    - Criticality measures (edge-of-chaos dynamics)

    Delta computation follows standard neuroscience protocols for
    paired comparisons (pre vs post stimulus/treatment), enabling
    assessment of:
    - Stimulus response magnitude
    - Learning-induced plasticity
    - Drug effects on neural dynamics
    - Long-term organoid maturation
"""

import numpy as np
import time
from typing import Optional
from .loader import SpikeData


# Module-level experiment store
_experiments: dict = {}


def start_experiment(
    experiment_id: str,
    data: SpikeData,
    name: Optional[str] = None,
    description: Optional[str] = None,
    experiment_type: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> dict:
    """Start a new experiment by recording the pre-condition snapshot.

    Captures baseline neural metrics from the organoid before any
    intervention (stimulus, drug application, training protocol, etc.).

    Args:
        experiment_id: Unique identifier for this experiment.
        data: SpikeData captured before the intervention.
        name: Human-readable experiment name.
        description: Description of the experiment and hypothesis.
        experiment_type: Type of experiment (stimulus, drug, training, maturation).
        parameters: Dict of experimental parameters (e.g., stimulus frequency,
            drug concentration, training protocol).

    Returns:
        Dict with experiment metadata and pre-condition snapshot.
    """
    pre_snapshot = _compute_snapshot(data)

    experiment = {
        "experiment_id": experiment_id,
        "name": name or experiment_id,
        "description": description or "",
        "experiment_type": experiment_type or "unspecified",
        "parameters": parameters or {},
        "status": "running",
        "started_at": time.time(),
        "ended_at": None,
        "pre_snapshot": pre_snapshot,
        "post_snapshot": None,
        "delta": None,
    }

    _experiments[experiment_id] = experiment

    return {
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": experiment["started_at"],
        "pre_snapshot": pre_snapshot,
        "message": (
            f"Experiment '{name or experiment_id}' started. "
            f"Baseline: {pre_snapshot['n_spikes']} spikes, "
            f"{pre_snapshot['mean_firing_rate']:.2f} Hz mean rate."
        ),
    }


def end_experiment(
    experiment_id: str,
    data: SpikeData,
    notes: Optional[str] = None,
) -> dict:
    """End an experiment by recording the post-condition snapshot.

    Captures neural metrics after the intervention and computes
    the delta (change) between pre and post conditions.

    Args:
        experiment_id: ID of the experiment to end.
        data: SpikeData captured after the intervention.
        notes: Optional notes about observations during the experiment.

    Returns:
        Dict with pre/post snapshots, delta, and interpretation.
    """
    if experiment_id not in _experiments:
        return {"error": f"Experiment '{experiment_id}' not found."}

    experiment = _experiments[experiment_id]
    if experiment["status"] != "running":
        return {"error": f"Experiment '{experiment_id}' is not running (status: {experiment['status']})."}

    post_snapshot = _compute_snapshot(data)
    delta = _compute_delta_internal(experiment["pre_snapshot"], post_snapshot)

    experiment["status"] = "completed"
    experiment["ended_at"] = time.time()
    experiment["post_snapshot"] = post_snapshot
    experiment["delta"] = delta
    if notes:
        experiment["notes"] = notes

    duration_sec = experiment["ended_at"] - experiment["started_at"]

    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "duration_seconds": round(duration_sec, 2),
        "pre_snapshot": experiment["pre_snapshot"],
        "post_snapshot": post_snapshot,
        "delta": delta,
        "interpretation": _interpret_delta(delta, experiment["experiment_type"]),
    }


def get_experiment(experiment_id: str) -> dict:
    """Retrieve a specific experiment's full record.

    Args:
        experiment_id: ID of the experiment.

    Returns:
        Full experiment dict or error.
    """
    if experiment_id not in _experiments:
        return {"error": f"Experiment '{experiment_id}' not found."}
    exp = _experiments[experiment_id]
    return {
        "experiment_id": exp["experiment_id"],
        "name": exp["name"],
        "description": exp["description"],
        "experiment_type": exp["experiment_type"],
        "parameters": exp["parameters"],
        "status": exp["status"],
        "started_at": exp["started_at"],
        "ended_at": exp["ended_at"],
        "pre_snapshot": exp["pre_snapshot"],
        "post_snapshot": exp["post_snapshot"],
        "delta": exp["delta"],
    }


def get_history(
    experiment_type: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """Get history of all experiments with optional filtering.

    Args:
        experiment_type: Filter by type (stimulus, drug, training, maturation).
        status: Filter by status (running, completed).

    Returns:
        Dict with list of experiments and summary statistics.
    """
    experiments = list(_experiments.values())

    if experiment_type:
        experiments = [e for e in experiments if e["experiment_type"] == experiment_type]
    if status:
        experiments = [e for e in experiments if e["status"] == status]

    summaries = []
    for exp in experiments:
        summary = {
            "experiment_id": exp["experiment_id"],
            "name": exp["name"],
            "experiment_type": exp["experiment_type"],
            "status": exp["status"],
            "started_at": exp["started_at"],
            "ended_at": exp["ended_at"],
        }
        if exp["delta"]:
            summary["firing_rate_delta_pct"] = exp["delta"].get("firing_rate_change_pct", 0)
            summary["activity_delta_pct"] = exp["delta"].get("total_spikes_change_pct", 0)
        summaries.append(summary)

    # Aggregate stats across completed experiments
    completed = [e for e in experiments if e["status"] == "completed" and e["delta"]]
    agg_stats = {}
    if completed:
        fr_deltas = [e["delta"]["firing_rate_change_pct"] for e in completed]
        agg_stats = {
            "n_completed": len(completed),
            "mean_firing_rate_change_pct": round(float(np.mean(fr_deltas)), 2),
            "std_firing_rate_change_pct": round(float(np.std(fr_deltas)), 2),
            "max_firing_rate_change_pct": round(float(np.max(fr_deltas)), 2),
            "min_firing_rate_change_pct": round(float(np.min(fr_deltas)), 2),
        }

    return {
        "total_experiments": len(experiments),
        "running": sum(1 for e in experiments if e["status"] == "running"),
        "completed": sum(1 for e in experiments if e["status"] == "completed"),
        "experiments": summaries,
        "aggregate_stats": agg_stats,
    }


def compute_delta(
    pre_data: SpikeData,
    post_data: SpikeData,
) -> dict:
    """Compute delta between two SpikeData snapshots without experiment context.

    Standalone function for ad-hoc pre/post comparisons when formal
    experiment tracking is not needed.

    Args:
        pre_data: SpikeData from before intervention.
        post_data: SpikeData from after intervention.

    Returns:
        Dict with all computed deltas and interpretation.
    """
    pre = _compute_snapshot(pre_data)
    post = _compute_snapshot(post_data)
    delta = _compute_delta_internal(pre, post)
    return {
        "pre_snapshot": pre,
        "post_snapshot": post,
        "delta": delta,
        "interpretation": _interpret_delta(delta, "unspecified"),
    }


def clear_experiments() -> dict:
    """Clear all experiments from the store.

    Returns:
        Dict confirming the operation.
    """
    n = len(_experiments)
    _experiments.clear()
    return {"cleared": n, "message": f"Cleared {n} experiments."}


def _compute_snapshot(data: SpikeData) -> dict:
    """Compute a snapshot of key neural metrics from SpikeData.

    Metrics captured:
    - Total spike count and per-electrode counts
    - Firing rates (mean, std, per-electrode)
    - Amplitude statistics
    - Inter-spike interval statistics
    - Active electrode count
    - Recording duration
    """
    duration = max(data.duration, 1e-6)
    n_spikes = data.n_spikes

    # Per-electrode firing rates
    electrode_rates = {}
    electrode_counts = {}
    all_isis = []
    for eid in data.electrode_ids:
        e_mask = data.electrodes == eid
        e_times = data.times[e_mask]
        count = int(np.sum(e_mask))
        electrode_counts[eid] = count
        electrode_rates[eid] = round(count / duration, 4)
        if len(e_times) > 1:
            isi = np.diff(e_times)
            all_isis.extend(isi.tolist())

    rates = list(electrode_rates.values())
    mean_rate = float(np.mean(rates)) if rates else 0.0
    std_rate = float(np.std(rates)) if rates else 0.0

    # Amplitude stats
    mean_amp = float(np.mean(np.abs(data.amplitudes))) if n_spikes > 0 else 0.0
    std_amp = float(np.std(data.amplitudes)) if n_spikes > 0 else 0.0

    # ISI stats
    isi_mean = float(np.mean(all_isis)) if all_isis else 0.0
    isi_std = float(np.std(all_isis)) if all_isis else 0.0
    isi_cv = float(isi_std / isi_mean) if isi_mean > 0 else 0.0

    # Burst detection (simple: >3 spikes within 10ms window)
    burst_count = 0
    if n_spikes > 3:
        sorted_times = np.sort(data.times)
        for i in range(len(sorted_times) - 3):
            if sorted_times[i + 3] - sorted_times[i] < 0.01:
                burst_count += 1

    return {
        "n_spikes": n_spikes,
        "n_active_electrodes": data.n_electrodes,
        "duration_sec": round(duration, 4),
        "mean_firing_rate": round(mean_rate, 4),
        "std_firing_rate": round(std_rate, 4),
        "mean_amplitude": round(mean_amp, 4),
        "std_amplitude": round(std_amp, 4),
        "isi_mean": round(isi_mean, 6),
        "isi_cv": round(isi_cv, 4),
        "burst_count": burst_count,
        "electrode_rates": {str(k): v for k, v in electrode_rates.items()},
        "electrode_counts": {str(k): v for k, v in electrode_counts.items()},
    }


def _compute_delta_internal(pre: dict, post: dict) -> dict:
    """Compute deltas between pre and post snapshots."""
    def _pct_change(old: float, new: float) -> float:
        if abs(old) < 1e-10:
            return 0.0 if abs(new) < 1e-10 else 100.0
        return round(((new - old) / abs(old)) * 100.0, 2)

    def _abs_change(old: float, new: float) -> float:
        return round(new - old, 6)

    delta = {
        "total_spikes_change": post["n_spikes"] - pre["n_spikes"],
        "total_spikes_change_pct": _pct_change(pre["n_spikes"], post["n_spikes"]),
        "firing_rate_change": _abs_change(pre["mean_firing_rate"], post["mean_firing_rate"]),
        "firing_rate_change_pct": _pct_change(pre["mean_firing_rate"], post["mean_firing_rate"]),
        "amplitude_change": _abs_change(pre["mean_amplitude"], post["mean_amplitude"]),
        "amplitude_change_pct": _pct_change(pre["mean_amplitude"], post["mean_amplitude"]),
        "isi_cv_change": _abs_change(pre["isi_cv"], post["isi_cv"]),
        "burst_count_change": post["burst_count"] - pre["burst_count"],
        "electrode_count_change": post["n_active_electrodes"] - pre["n_active_electrodes"],
    }

    # Per-electrode rate changes
    pre_rates = pre.get("electrode_rates", {})
    post_rates = post.get("electrode_rates", {})
    all_eids = set(list(pre_rates.keys()) + list(post_rates.keys()))
    electrode_deltas = {}
    for eid in all_eids:
        old_r = pre_rates.get(eid, 0.0)
        new_r = post_rates.get(eid, 0.0)
        electrode_deltas[eid] = {
            "pre_rate": old_r,
            "post_rate": new_r,
            "change": _abs_change(old_r, new_r),
            "change_pct": _pct_change(old_r, new_r),
        }
    delta["per_electrode"] = electrode_deltas

    # Effect size (Cohen's d approximation using firing rate)
    pooled_std = (pre["std_firing_rate"] + post["std_firing_rate"]) / 2.0
    if pooled_std > 1e-10:
        delta["effect_size_cohens_d"] = round(
            abs(post["mean_firing_rate"] - pre["mean_firing_rate"]) / pooled_std, 4
        )
    else:
        delta["effect_size_cohens_d"] = 0.0

    return delta


def _interpret_delta(delta: dict, experiment_type: str) -> str:
    """Generate human-readable interpretation of the delta."""
    parts = []

    fr_pct = delta["firing_rate_change_pct"]
    if abs(fr_pct) < 5:
        parts.append(f"Minimal firing rate change ({fr_pct:+.1f}%)")
    elif fr_pct > 0:
        parts.append(f"Firing rate increased by {fr_pct:.1f}% (excitation/activation)")
    else:
        parts.append(f"Firing rate decreased by {abs(fr_pct):.1f}% (inhibition/suppression)")

    amp_pct = delta["amplitude_change_pct"]
    if abs(amp_pct) > 10:
        direction = "increased" if amp_pct > 0 else "decreased"
        parts.append(f"Amplitude {direction} by {abs(amp_pct):.1f}%")

    burst_d = delta["burst_count_change"]
    if burst_d > 0:
        parts.append(f"Burst activity increased (+{burst_d} bursts)")
    elif burst_d < 0:
        parts.append(f"Burst activity decreased ({burst_d} bursts)")

    d = delta.get("effect_size_cohens_d", 0)
    if d > 0.8:
        parts.append(f"Large effect size (Cohen's d = {d:.2f})")
    elif d > 0.5:
        parts.append(f"Medium effect size (Cohen's d = {d:.2f})")
    elif d > 0.2:
        parts.append(f"Small effect size (Cohen's d = {d:.2f})")

    if experiment_type == "stimulus":
        parts.append("Stimulus-evoked changes detected." if abs(fr_pct) > 10 else "Weak stimulus response.")
    elif experiment_type == "drug":
        parts.append("Pharmacological effect observed." if abs(fr_pct) > 15 else "Minimal drug effect.")
    elif experiment_type == "training":
        parts.append("Plasticity signature present." if abs(fr_pct) > 5 else "No clear learning signal.")

    return " ".join(parts)
