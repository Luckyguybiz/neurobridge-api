"""Dopamine UV Reinforcement -- chemical reward via UV-activated dopamine release.

Simulates a training paradigm where UV light triggers caged dopamine release
as a chemical reward signal. The organoid is trained to produce a target
activity pattern. When the output matches the target, a UV pulse delivers
dopamine, reinforcing the matching neurons. Over trials, the organoid should
converge toward the desired pattern.
"""

import numpy as np
from typing import Optional
from ..loader import SpikeData


N_ELECTRODES = 8


def _get_base_rates(data: SpikeData) -> np.ndarray:
    """Extract baseline firing rates for 8 electrodes."""
    rates = np.zeros(N_ELECTRODES)
    duration = max(data.duration, 0.001)
    for e in data.electrode_ids:
        idx = e % N_ELECTRODES
        rates[idx] += int(np.sum(data.electrodes == e)) / duration
    if np.sum(rates) == 0:
        rates = np.ones(N_ELECTRODES) * 1.0
    return rates


def _normalize_pattern(pattern: np.ndarray) -> np.ndarray:
    """Normalize pattern to unit vector for comparison."""
    norm = np.linalg.norm(pattern)
    if norm < 1e-10:
        return np.ones_like(pattern) / np.sqrt(len(pattern))
    return pattern / norm


def _compute_match_score(output: np.ndarray, target: np.ndarray) -> float:
    """Compute cosine similarity between output and target patterns."""
    out_norm = _normalize_pattern(output)
    tgt_norm = _normalize_pattern(target)
    similarity = float(np.dot(out_norm, tgt_norm))
    return float(np.clip(similarity, 0, 1))


def _generate_target_pattern(rng: np.random.Generator) -> np.ndarray:
    """Generate a random but structured target activity pattern.

    Creates a pattern with some electrodes highly active and others quiet,
    simulating a desired functional pattern.
    """
    pattern = np.zeros(N_ELECTRODES)
    # Select 3-4 "active" electrodes
    n_active = rng.integers(3, 5)
    active_idx = rng.choice(N_ELECTRODES, size=n_active, replace=False)
    for idx in active_idx:
        pattern[idx] = rng.uniform(2.0, 5.0)
    # Low background for others
    for i in range(N_ELECTRODES):
        if i not in active_idx:
            pattern[i] = rng.uniform(0.1, 0.5)
    return pattern


def simulate_dopamine_training(
    data: SpikeData,
    n_trials: int = 100,
    target_pattern: Optional[list] = None,
) -> dict:
    """Simulate dopamine-reinforced training via UV activation.

    Each trial:
    1. Organoid produces an output pattern (base rates + learned bias).
    2. Compare output with target pattern (cosine similarity).
    3. If match > threshold: UV pulse -> dopamine release -> reinforce matching neurons.
    4. Track pattern match rate over trials.

    Args:
        data: SpikeData for baseline rate estimation.
        n_trials: Number of training trials.
        target_pattern: Optional target pattern (list of 8 floats).
            If None, generates a random structured pattern.

    Returns:
        dict with match_rates, uv_events, pattern_evolution, learning_detected.
    """
    rng = np.random.default_rng(42)
    base_rates = _get_base_rates(data)

    # Target pattern
    if target_pattern is not None:
        target = np.array(target_pattern[:N_ELECTRODES], dtype=float)
        if len(target) < N_ELECTRODES:
            target = np.pad(target, (0, N_ELECTRODES - len(target)), constant_values=0.5)
    else:
        target = _generate_target_pattern(rng)

    # Learned bias (starts at zero, updated by dopamine reinforcement)
    bias = np.zeros(N_ELECTRODES)
    match_rates = []
    uv_events = []
    pattern_evolution = []

    dopamine_boost = 0.15  # how much dopamine reinforces matching neurons
    match_threshold = 0.6  # cosine similarity threshold for UV pulse
    decay_rate = 0.98  # natural forgetting
    noise_std = 0.3

    running_match = 0.0
    ema_alpha = 0.1

    for trial in range(n_trials):
        # Organoid output: base rates + learned bias + noise
        output = base_rates + bias + rng.normal(0, noise_std, N_ELECTRODES)
        output = np.clip(output, 0, None)

        # Compute match score
        match_score = _compute_match_score(output, target)
        running_match = ema_alpha * match_score + (1 - ema_alpha) * running_match

        # UV pulse decision
        uv_fired = match_score >= match_threshold

        if uv_fired:
            uv_events.append({
                "trial": trial,
                "match_score": round(match_score, 4),
                "intensity": round(min(match_score * 1.2, 1.0), 3),
            })

            # Dopamine reinforcement: boost neurons that match the target
            target_norm = _normalize_pattern(target)
            output_norm = _normalize_pattern(output)

            for i in range(N_ELECTRODES):
                # Reinforce proportional to how well this neuron matches target
                alignment = target_norm[i] * output_norm[i]
                bias[i] += dopamine_boost * alignment * match_score

        else:
            # Mild exploration: random perturbation when not matching
            bias += rng.normal(0, 0.02, N_ELECTRODES)

        # Natural decay
        bias *= decay_rate

        # Prevent runaway
        bias = np.clip(bias, -3.0, 3.0)

        match_rates.append(round(match_score, 4))

        # Record pattern snapshot every 10 trials
        if trial % 10 == 0:
            pattern_evolution.append({
                "trial": trial,
                "output": [round(float(o), 3) for o in output],
                "bias": [round(float(b), 3) for b in bias],
                "match_score": round(match_score, 4),
            })

    # Analysis
    match_arr = np.array(match_rates)
    first_quarter = float(np.mean(match_arr[:n_trials // 4])) if n_trials >= 4 else float(match_arr[0])
    last_quarter = float(np.mean(match_arr[-n_trials // 4:])) if n_trials >= 4 else float(match_arr[-1])
    improvement = last_quarter - first_quarter
    learning_detected = improvement > 0.05 and last_quarter > match_threshold * 0.8

    # Smoothed learning curve
    window = min(10, n_trials)
    smoothed = [
        round(float(np.mean(match_arr[max(0, i - window + 1):i + 1])), 4)
        for i in range(n_trials)
    ]

    return {
        "match_rates": smoothed,
        "raw_match_rates": match_rates,
        "uv_events": uv_events,
        "n_uv_pulses": len(uv_events),
        "pattern_evolution": pattern_evolution,
        "target_pattern": [round(float(t), 3) for t in target],
        "final_output": [round(float(o), 3) for o in output],
        "learning_detected": bool(learning_detected),
        "n_trials": n_trials,
        "first_quarter_match": round(first_quarter, 4),
        "last_quarter_match": round(last_quarter, 4),
        "improvement": round(improvement, 4),
        "final_match_score": round(float(match_arr[-1]), 4),
        "mean_match_score": round(float(np.mean(match_arr)), 4),
        "interpretation": (
            f"Dopamine UV training: {n_trials} trials, {len(uv_events)} UV pulses fired. "
            f"Match score: {first_quarter:.3f} -> {last_quarter:.3f} "
            f"(improvement: {improvement:+.3f}). "
            + ("Learning detected -- organoid converges toward target pattern under "
               "dopaminergic reinforcement."
               if learning_detected else
               "No significant convergence -- target may be too far from natural dynamics "
               "or more trials needed.")
        ),
    }
