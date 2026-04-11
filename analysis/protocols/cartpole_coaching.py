"""Cart-Pole Coaching -- balance inverted pendulum with adaptive coaching.

Implements a simplified cart-pole (inverted pendulum) environment where the
organoid must learn to balance a pole by choosing left/right actions. State
(angle, angular velocity) is encoded as stimulation. An adaptive coach
reinforces neurons that are active during successful balance periods.
"""

import numpy as np
from typing import Optional
from ..loader import SpikeData


# Cart-pole physics constants
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_LENGTH = 0.5
FORCE_MAG = 10.0
DT = 0.02  # 20ms timestep
ANGLE_LIMIT = 0.2095  # ~12 degrees (fail threshold)


def _get_base_rates(data: SpikeData, n: int = 8) -> np.ndarray:
    """Extract baseline rates for n electrodes."""
    rates = np.zeros(n)
    duration = max(data.duration, 0.001)
    for e in data.electrode_ids:
        idx = e % n
        rates[idx] += int(np.sum(data.electrodes == e)) / duration
    if np.sum(rates) == 0:
        rates = np.ones(n) * 1.0
    return rates


def _cartpole_step(state: np.ndarray, action: int) -> np.ndarray:
    """Advance cart-pole physics by one timestep.

    State: [x, x_dot, theta, theta_dot]
    Action: 0 = left, 1 = right
    """
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    total_mass = CART_MASS + POLE_MASS
    pole_mass_length = POLE_MASS * POLE_LENGTH

    # Equations of motion
    temp = (force + pole_mass_length * theta_dot ** 2 * sin_t) / total_mass
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_t ** 2 / total_mass)
    )
    x_acc = temp - pole_mass_length * theta_acc * cos_t / total_mass

    # Euler integration
    x_new = x + DT * x_dot
    x_dot_new = x_dot + DT * x_acc
    theta_new = theta + DT * theta_dot
    theta_dot_new = theta_dot + DT * theta_acc

    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])


def _encode_state(theta: float, theta_dot: float, n_electrodes: int = 8) -> np.ndarray:
    """Encode pole angle and angular velocity as stimulation pattern.

    First half of electrodes: angle (Gaussian tuning).
    Second half: angular velocity (Gaussian tuning).
    """
    pattern = np.zeros(n_electrodes)
    half = n_electrodes // 2

    # Angle encoding: -ANGLE_LIMIT to +ANGLE_LIMIT
    angle_norm = (theta + ANGLE_LIMIT) / (2 * ANGLE_LIMIT)
    angle_norm = float(np.clip(angle_norm, 0, 1))
    for i in range(half):
        preferred = i / max(1, half - 1)
        pattern[i] = np.exp(-((angle_norm - preferred) ** 2) / 0.1)

    # Velocity encoding: normalized
    vel_norm = float(np.clip((theta_dot + 2.0) / 4.0, 0, 1))
    for i in range(half, n_electrodes):
        preferred = (i - half) / max(1, n_electrodes - half - 1)
        pattern[i] = np.exp(-((vel_norm - preferred) ** 2) / 0.1)

    return pattern


def _decode_action(output_rates: np.ndarray) -> int:
    """Decode left/right action from output electrode rates.

    Sum of first half vs second half determines direction.
    """
    half = len(output_rates) // 2
    left_sum = float(np.sum(output_rates[:half]))
    right_sum = float(np.sum(output_rates[half:]))
    return 1 if right_sum > left_sum else 0


def simulate_cartpole(data: SpikeData, n_episodes: int = 50, max_steps: int = 200) -> dict:
    """Simulate cart-pole balancing with adaptive coaching.

    Each episode:
    1. Initialize pole near vertical with small random perturbation.
    2. Encode state -> stimulate organoid -> decode action.
    3. Step physics.
    4. Coach: if pole is balanced, reinforce active-during-balance neurons.
    5. Episode ends when pole falls beyond angle limit or max_steps reached.

    Args:
        data: SpikeData for baseline rate estimation.
        n_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.

    Returns:
        dict with episode_durations, learning_curve, best_episode, improvement.
    """
    rng = np.random.default_rng(42)
    base_rates = _get_base_rates(data)
    n_electrodes = 8

    # Weights: state encoding -> action output
    weights = rng.normal(0, 0.1, (n_electrodes, n_electrodes))

    episode_durations = []
    coaching_rate = 0.02
    decay = 0.98

    for ep in range(n_episodes):
        # Initial state: small random angle perturbation
        state = np.array([0.0, 0.0, rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)])
        steps_balanced = 0
        active_log = []

        for step in range(max_steps):
            theta, theta_dot = state[2], state[3]

            # Check failure
            if abs(theta) > ANGLE_LIMIT:
                break

            # Encode state
            stim = _encode_state(theta, theta_dot, n_electrodes)

            # Neural response: base rates modulated by learned weights
            output = base_rates + weights.T @ stim + rng.normal(0, 0.2, n_electrodes)
            output = np.clip(output, 0, None)

            active_log.append(stim.copy())

            # Decode action
            action = _decode_action(output)

            # Step physics
            state = _cartpole_step(state, action)
            steps_balanced += 1

        # Coach: reinforce neurons that were active during balance
        if steps_balanced > 10:
            # Average activity pattern during balance
            avg_active = np.mean(active_log[-min(steps_balanced, 20):], axis=0)

            # Reinforce connections proportional to balance duration
            reward_signal = min(steps_balanced / max_steps, 1.0)
            for i in range(n_electrodes):
                for j in range(n_electrodes):
                    weights[i, j] += coaching_rate * reward_signal * avg_active[i] * avg_active[j]

        # Weight decay to prevent explosion
        weights *= decay
        norm = np.linalg.norm(weights)
        if norm > 5.0:
            weights *= 5.0 / norm

        episode_durations.append(steps_balanced)

    # Analysis
    durations = np.array(episode_durations)
    window = min(5, n_episodes)
    learning_curve = [
        round(float(np.mean(durations[max(0, i - window + 1):i + 1])), 1)
        for i in range(n_episodes)
    ]

    best_episode = int(np.argmax(durations))
    best_duration = int(durations[best_episode])

    first_quarter = float(np.mean(durations[:n_episodes // 4])) if n_episodes >= 4 else float(durations[0])
    last_quarter = float(np.mean(durations[-n_episodes // 4:])) if n_episodes >= 4 else float(durations[-1])
    improvement_pct = ((last_quarter - first_quarter) / max(first_quarter, 1)) * 100

    return {
        "episode_durations": [int(d) for d in durations],
        "learning_curve": learning_curve,
        "best_episode": best_episode,
        "best_duration": best_duration,
        "mean_duration": round(float(np.mean(durations)), 1),
        "max_possible": max_steps,
        "improvement_pct": round(improvement_pct, 1),
        "n_episodes": n_episodes,
        "first_quarter_mean": round(first_quarter, 1),
        "last_quarter_mean": round(last_quarter, 1),
        "solved": bool(last_quarter >= max_steps * 0.8),
        "interpretation": (
            f"Cart-pole: {n_episodes} episodes, best={best_duration}/{max_steps} steps. "
            f"Mean duration: {np.mean(durations):.0f} steps. "
            f"Improvement: {improvement_pct:+.1f}% (first quarter: {first_quarter:.0f}, "
            f"last quarter: {last_quarter:.0f}). "
            + ("Significant learning -- organoid adapts balance strategy."
               if improvement_pct > 20 else
               "Moderate improvement detected."
               if improvement_pct > 5 else
               "Minimal improvement -- coaching strategy may need adjustment.")
        ),
    }
