"""LLM-in-the-loop protocol optimization module.

Framework for using a large language model to iteratively optimize
stimulation protocols for biological neural networks. Generates
structured prompts describing organoid state, parses LLM suggestions
into protocol parameters, and simulates optimization loops using
a digital twin for rapid evaluation.
"""

import numpy as np
import json
import hashlib
from typing import Optional
from .loader import SpikeData


def _compute_organoid_summary(data: SpikeData) -> dict:
    """Quick summary of organoid state for prompt generation."""
    if data.n_spikes == 0:
        return {"status": "silent", "n_spikes": 0}

    t_start, t_end = data.time_range
    duration = t_end - t_start

    firing_rate = data.n_spikes / duration if duration > 0 else 0

    # ISI stats
    isi_cv = 0.0
    if data.n_spikes >= 2:
        isi = np.diff(data.times) * 1000
        isi_cv = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0.0

    # Burst count
    burst_count = 0
    if data.n_spikes >= 3:
        isi = np.diff(data.times) * 1000
        in_burst = isi < 10.0
        transitions = np.diff(in_burst.astype(int))
        burst_count = int(np.sum(transitions == 1))

    # Synchrony estimate
    sync = 0.0
    if data.n_spikes >= 2 and data.n_electrodes >= 2:
        sample_size = min(data.n_spikes, 500)
        indices = np.random.choice(data.n_spikes, sample_size, replace=False)
        sync_count = 0
        for idx in indices:
            t = data.times[idx]
            nearby = np.abs(data.times - t) < 0.005
            diff_e = data.electrodes != data.electrodes[idx]
            if np.any(nearby & diff_e):
                sync_count += 1
        sync = sync_count / sample_size

    return {
        "status": "active",
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(duration, 2),
        "firing_rate_hz": round(firing_rate, 3),
        "isi_cv": round(isi_cv, 4),
        "burst_count": burst_count,
        "burst_rate_per_min": round(burst_count / (duration / 60), 3) if duration > 0 else 0,
        "synchrony": round(sync, 4),
        "mean_amplitude_uv": round(float(np.mean(np.abs(data.amplitudes))), 2),
    }


def generate_optimization_prompt(
    data: SpikeData,
    current_protocol: dict,
    iteration: int,
    objective: str = "maximize_complexity",
    history: Optional[list[dict]] = None,
) -> dict:
    """Build a text prompt describing organoid state + protocol + results.

    The prompt is suitable for sending to an LLM (GPT-4, Claude, etc.)
    to get protocol optimization suggestions.
    """
    summary = _compute_organoid_summary(data)

    # Compute a score based on objective
    score = _compute_score(summary, objective)

    objectives_description = {
        "maximize_complexity": "Maximize neural complexity (ISI variability, burst diversity, information content)",
        "maximize_firing_rate": "Maximize overall firing rate while maintaining healthy patterns",
        "maximize_synchrony": "Maximize cross-electrode synchronization",
        "maximize_bursting": "Maximize organized burst activity",
        "minimize_noise": "Minimize noise while maintaining healthy activity levels",
    }

    history_text = ""
    if history:
        history_text = "\n\n## Previous iterations:\n"
        for h in history[-5:]:  # Last 5 iterations
            history_text += (
                f"- Iteration {h['iteration']}: "
                f"score={h['score']:.4f}, "
                f"protocol={json.dumps(h['protocol'])}\n"
            )

    prompt = f"""You are an expert in biological neural network optimization.
You are optimizing a stimulation protocol for a living brain organoid on a multi-electrode array (MEA).

## Objective
{objectives_description.get(objective, objective)}

## Current organoid state (iteration {iteration}):
- Firing rate: {summary['firing_rate_hz']} Hz
- ISI coefficient of variation: {summary['isi_cv']}
- Burst count: {summary['burst_count']} ({summary['burst_rate_per_min']} per minute)
- Cross-electrode synchrony: {summary['synchrony']}
- Mean amplitude: {summary['mean_amplitude_uv']} uV
- Active electrodes: {summary['n_electrodes']}
- Recording duration: {summary['duration_s']} seconds
- Current score: {score:.4f}

## Current protocol:
{json.dumps(current_protocol, indent=2)}
{history_text}

## Instructions
Suggest a NEW stimulation protocol to improve the score. Respond with a JSON object containing:
{{
  "stim_amplitude_uv": <float, 100-1000>,
  "stim_frequency_hz": <float, 0.1-100>,
  "stim_duration_ms": <float, 0.1-10>,
  "stim_pattern": <"regular"|"burst"|"theta"|"gamma"|"random">,
  "inter_burst_interval_ms": <float, 50-5000, only if pattern=burst>,
  "n_pulses_per_burst": <int, 1-20, only if pattern=burst>,
  "electrode_selection": <"all"|"most_active"|"least_active"|"alternating">,
  "rest_period_s": <float, 0-60>,
  "reasoning": "<brief explanation of why this should improve the score>"
}}
"""

    return {
        "prompt": prompt,
        "organoid_summary": summary,
        "current_protocol": current_protocol,
        "current_score": round(score, 4),
        "objective": objective,
        "iteration": iteration,
        "prompt_length_chars": len(prompt),
    }


def parse_llm_suggestion(suggestion_text: str) -> dict:
    """Parse a text suggestion from an LLM into structured protocol parameters.

    Handles various formats: raw JSON, JSON within markdown code blocks,
    or natural language with embedded numbers.
    """
    # Try to extract JSON from the text
    parsed = None

    # Strategy 1: direct JSON parse
    try:
        parsed = json.loads(suggestion_text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: find JSON block in markdown
    if parsed is None:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', suggestion_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: find any JSON-like object
    if parsed is None:
        import re
        obj_match = re.search(r'\{[^{}]*"stim_amplitude_uv"[^{}]*\}', suggestion_text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

    # Strategy 4: extract key-value pairs from natural language
    if parsed is None:
        import re
        parsed = {}
        patterns = {
            "stim_amplitude_uv": r'amplitude[:\s]*(\d+\.?\d*)',
            "stim_frequency_hz": r'frequency[:\s]*(\d+\.?\d*)',
            "stim_duration_ms": r'duration[:\s]*(\d+\.?\d*)',
            "stim_pattern": r'pattern[:\s]*["\']?(\w+)["\']?',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, suggestion_text, re.IGNORECASE)
            if match:
                val = match.group(1)
                if key != "stim_pattern":
                    parsed[key] = float(val)
                else:
                    parsed[key] = val

    if not parsed:
        return {
            "success": False,
            "error": "Could not parse suggestion into protocol parameters",
            "raw_text": suggestion_text[:500],
            "protocol": _default_protocol(),
        }

    # Validate and clamp values
    protocol = _validate_protocol(parsed)

    return {
        "success": True,
        "protocol": protocol,
        "reasoning": parsed.get("reasoning", ""),
        "raw_parsed": parsed,
    }


def run_optimization_loop(
    data: SpikeData,
    n_iterations: int = 5,
    objective: str = "maximize_complexity",
    initial_protocol: Optional[dict] = None,
) -> dict:
    """Simulate optimization iterations using a digital twin.

    Since we can't actually call an LLM or stimulate a real organoid,
    this simulates the loop: generate protocol -> simulate effect on
    digital twin -> evaluate score -> repeat.

    Returns history of protocols and scores.
    """
    if data.n_spikes == 0:
        return {"error": "No spikes in dataset"}

    protocol = initial_protocol or _default_protocol()
    rng = np.random.RandomState(42)

    history: list[dict] = []
    best_score = -float("inf")
    best_protocol = protocol.copy()

    for iteration in range(n_iterations):
        # Simulate effect of protocol on spike data (digital twin)
        simulated = _simulate_protocol_effect(data, protocol, rng)
        summary = _compute_organoid_summary(simulated)
        score = _compute_score(summary, objective)

        history.append({
            "iteration": iteration,
            "protocol": protocol.copy(),
            "score": round(score, 4),
            "summary": summary,
        })

        if score > best_score:
            best_score = score
            best_protocol = protocol.copy()

        # Generate next protocol (simulated LLM suggestion)
        protocol = _evolve_protocol(protocol, score, history, rng)

    # Final evaluation
    improvement = history[-1]["score"] - history[0]["score"] if history else 0

    return {
        "n_iterations": n_iterations,
        "objective": objective,
        "history": history,
        "best_protocol": best_protocol,
        "best_score": round(best_score, 4),
        "initial_score": round(history[0]["score"], 4) if history else 0,
        "final_score": round(history[-1]["score"], 4) if history else 0,
        "improvement": round(improvement, 4),
        "improved": improvement > 0,
        "convergence": _check_convergence(history),
    }


# ─── Internal helpers ────────────────────────────────────────────


def _default_protocol() -> dict:
    """Default stimulation protocol."""
    return {
        "stim_amplitude_uv": 300.0,
        "stim_frequency_hz": 1.0,
        "stim_duration_ms": 1.0,
        "stim_pattern": "regular",
        "electrode_selection": "all",
        "rest_period_s": 5.0,
    }


def _validate_protocol(raw: dict) -> dict:
    """Validate and clamp protocol parameters to safe ranges."""
    protocol = _default_protocol()  # Start with defaults

    if "stim_amplitude_uv" in raw:
        protocol["stim_amplitude_uv"] = float(np.clip(float(raw["stim_amplitude_uv"]), 100, 1000))
    if "stim_frequency_hz" in raw:
        protocol["stim_frequency_hz"] = float(np.clip(float(raw["stim_frequency_hz"]), 0.1, 100))
    if "stim_duration_ms" in raw:
        protocol["stim_duration_ms"] = float(np.clip(float(raw["stim_duration_ms"]), 0.1, 10))
    if "stim_pattern" in raw:
        valid_patterns = {"regular", "burst", "theta", "gamma", "random"}
        protocol["stim_pattern"] = raw["stim_pattern"] if raw["stim_pattern"] in valid_patterns else "regular"
    if "electrode_selection" in raw:
        valid_sel = {"all", "most_active", "least_active", "alternating"}
        protocol["electrode_selection"] = raw["electrode_selection"] if raw["electrode_selection"] in valid_sel else "all"
    if "rest_period_s" in raw:
        protocol["rest_period_s"] = float(np.clip(float(raw["rest_period_s"]), 0, 60))
    if "inter_burst_interval_ms" in raw:
        protocol["inter_burst_interval_ms"] = float(np.clip(float(raw["inter_burst_interval_ms"]), 50, 5000))
    if "n_pulses_per_burst" in raw:
        protocol["n_pulses_per_burst"] = int(np.clip(int(raw["n_pulses_per_burst"]), 1, 20))

    return protocol


def _compute_score(summary: dict, objective: str) -> float:
    """Compute optimization score based on objective."""
    if summary.get("status") == "silent":
        return 0.0

    rate = summary.get("firing_rate_hz", 0)
    cv = summary.get("isi_cv", 0)
    burst_rate = summary.get("burst_rate_per_min", 0)
    sync = summary.get("synchrony", 0)

    if objective == "maximize_complexity":
        return 0.3 * min(cv, 2.0) / 2.0 + 0.3 * min(burst_rate / 10, 1.0) + 0.2 * min(rate / 50, 1.0) + 0.2 * sync
    elif objective == "maximize_firing_rate":
        return min(rate / 100, 1.0)
    elif objective == "maximize_synchrony":
        return sync
    elif objective == "maximize_bursting":
        return min(burst_rate / 20, 1.0)
    elif objective == "minimize_noise":
        return max(0, 1.0 - cv / 3.0) * 0.5 + min(rate / 50, 1.0) * 0.5
    return 0.0


def _simulate_protocol_effect(
    data: SpikeData,
    protocol: dict,
    rng: np.random.RandomState,
) -> SpikeData:
    """Simulate how a protocol would affect the organoid (digital twin).

    Simple model: protocol parameters modulate firing rate, burstiness, etc.
    """
    amp_factor = protocol.get("stim_amplitude_uv", 300) / 300.0
    freq_factor = protocol.get("stim_frequency_hz", 1.0)

    # Higher amplitude -> more spikes (with saturation)
    rate_multiplier = 1.0 + 0.3 * np.tanh(amp_factor - 1.0) + 0.1 * rng.randn()
    rate_multiplier = max(0.3, min(3.0, rate_multiplier))

    # Subsample or duplicate spikes to match new rate
    n_target = int(data.n_spikes * rate_multiplier)
    n_target = max(10, min(n_target, data.n_spikes * 5))

    if n_target <= data.n_spikes:
        indices = rng.choice(data.n_spikes, n_target, replace=False)
    else:
        indices = rng.choice(data.n_spikes, n_target, replace=True)

    indices = np.sort(indices)

    # Add jitter proportional to stimulation effect
    jitter = rng.randn(n_target) * 0.001 * freq_factor
    new_times = data.times[indices] + jitter
    new_times = np.sort(new_times)

    return SpikeData(
        times=new_times,
        electrodes=data.electrodes[indices],
        amplitudes=data.amplitudes[indices] * amp_factor,
        sampling_rate=data.sampling_rate,
        metadata=data.metadata,
    )


def _evolve_protocol(
    current: dict,
    score: float,
    history: list[dict],
    rng: np.random.RandomState,
) -> dict:
    """Simulated LLM suggestion: perturb protocol parameters."""
    new = current.copy()

    # Perturb amplitude
    new["stim_amplitude_uv"] = float(np.clip(
        current["stim_amplitude_uv"] + rng.randn() * 50,
        100, 1000,
    ))

    # Perturb frequency
    new["stim_frequency_hz"] = float(np.clip(
        current["stim_frequency_hz"] * (1.0 + rng.randn() * 0.3),
        0.1, 100,
    ))

    # Occasionally switch pattern
    if rng.random() < 0.2:
        patterns = ["regular", "burst", "theta", "gamma", "random"]
        new["stim_pattern"] = patterns[rng.randint(len(patterns))]

    return _validate_protocol(new)


def _check_convergence(history: list[dict]) -> dict:
    """Check if optimization has converged."""
    if len(history) < 3:
        return {"converged": False, "reason": "too_few_iterations"}

    scores = [h["score"] for h in history]
    recent = scores[-3:]
    score_range = max(recent) - min(recent)

    if score_range < 0.01:
        return {"converged": True, "reason": "scores_plateau", "final_range": round(score_range, 4)}

    # Check monotonic improvement
    improving = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    return {
        "converged": False,
        "monotonically_improving": improving,
        "score_range": round(float(max(scores) - min(scores)), 4),
    }
