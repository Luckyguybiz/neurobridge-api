"""DishBrain Pong Protocol -- train organoid to play Pong via free energy principle.

Implements the DishBrain paradigm (Kagan et al., 2022). Ball position is encoded
as frequency on input electrodes [0-3]. Organoid response from output electrodes
[4-7] drives paddle movement. Hit = predictable stimulation (reward). Miss = random
stimulation (punishment). The free energy principle predicts the organoid will learn
to minimize surprise (misses).
"""

import numpy as np
from typing import Optional
from ..loader import SpikeData


INPUT_ELECTRODES = [0, 1, 2, 3]
OUTPUT_ELECTRODES = [4, 5, 6, 7]
FIELD_HEIGHT = 1.0
FIELD_WIDTH = 1.0
PADDLE_HEIGHT = 0.25


def _get_base_rates(data: SpikeData) -> np.ndarray:
    """Extract baseline firing rates for 8 electrodes from SpikeData."""
    rates = np.zeros(8)
    duration = max(data.duration, 0.001)
    for e in data.electrode_ids:
        idx = e % 8
        rates[idx] += int(np.sum(data.electrodes == e)) / duration
    if np.sum(rates) == 0:
        rates = np.ones(8) * 1.0
    return rates


def _encode_ball_position(ball_y: float) -> np.ndarray:
    """Encode ball Y-position as frequency pattern on input electrodes.

    Uses a Gaussian tuning curve: each input electrode has a preferred position.
    """
    pattern = np.zeros(4)
    for i in range(4):
        preferred = i / 3.0  # 0.0, 0.33, 0.67, 1.0
        pattern[i] = np.exp(-((ball_y - preferred) ** 2) / 0.08)
    return pattern


def _decode_paddle_action(output_rates: np.ndarray) -> float:
    """Decode paddle movement from output electrode rates.

    Population vector: weighted average of electrode positions maps to direction.
    """
    positions = np.linspace(-1, 1, 4)  # up to down
    total = np.sum(output_rates)
    if total < 1e-6:
        return 0.0
    return float(np.dot(positions, output_rates) / total)


def simulate_pong_game(data: SpikeData, n_trials: int = 100) -> dict:
    """Simulate a DishBrain Pong game with learning via free energy principle.

    The organoid receives ball position encoded on input electrodes and must
    learn to move a paddle (via output electrodes) to intercept the ball.

    Hit = predictable stimulation (low surprise, reward).
    Miss = random noise stimulation (high surprise, punishment).

    Args:
        data: SpikeData for baseline rate estimation.
        n_trials: Number of ball-serve trials.

    Returns:
        dict with hit_rates, final_hit_rate, learning_detected, positions, scores.
    """
    rng = np.random.default_rng(42)
    base_rates = _get_base_rates(data)

    # Weights connecting input stim to output activity (learned via Hebbian rule)
    weights = rng.normal(0, 0.1, (4, 4))  # input -> output

    hit_rates = []
    ball_positions = []
    paddle_positions = []
    scores = {"hits": 0, "misses": 0}

    learning_rate = 0.05
    window_size = 10
    hits_in_window = 0

    for trial in range(n_trials):
        # Serve ball at random Y position
        ball_y = rng.uniform(0.1, 0.9)

        # Encode ball position
        input_pattern = _encode_ball_position(ball_y)

        # Organoid response: base output rates + learned response to input
        output_modulation = np.dot(input_pattern, weights)
        output_rates = base_rates[4:8] + output_modulation
        output_rates += rng.normal(0, 0.3, 4)  # neural noise
        output_rates = np.clip(output_rates, 0, None)

        # Decode paddle position
        paddle_move = _decode_paddle_action(output_rates)
        paddle_y = 0.5 + paddle_move * 0.4  # center + movement
        paddle_y = float(np.clip(paddle_y, PADDLE_HEIGHT / 2, 1.0 - PADDLE_HEIGHT / 2))

        # Check hit/miss
        hit = abs(ball_y - paddle_y) < PADDLE_HEIGHT / 2

        ball_positions.append(round(float(ball_y), 4))
        paddle_positions.append(round(float(paddle_y), 4))

        if hit:
            scores["hits"] += 1
            hits_in_window += 1

            # Reward: predictable stimulation -- reinforce active pathways (Hebbian)
            for i in range(4):
                for j in range(4):
                    weights[i, j] += learning_rate * input_pattern[i] * output_rates[j]
        else:
            scores["misses"] += 1

            # Punishment: random noise -- decorrelate (anti-Hebbian perturbation)
            weights += rng.normal(0, learning_rate * 0.5, weights.shape)

        # Weight normalization (prevent runaway)
        norm = np.linalg.norm(weights)
        if norm > 5.0:
            weights *= 5.0 / norm

        # Track hit rate in sliding window
        if (trial + 1) % window_size == 0:
            rate = hits_in_window / window_size
            hit_rates.append(round(rate, 3))
            hits_in_window = 0

    # Final stats
    if not hit_rates:
        hit_rates = [scores["hits"] / max(1, n_trials)]

    first_half = np.mean(hit_rates[:len(hit_rates) // 2]) if len(hit_rates) > 1 else hit_rates[0]
    second_half = np.mean(hit_rates[len(hit_rates) // 2:]) if len(hit_rates) > 1 else hit_rates[-1]
    improvement = float(second_half - first_half)
    learning_detected = improvement > 0.08

    return {
        "hit_rates": hit_rates,
        "final_hit_rate": round(float(hit_rates[-1]), 3) if hit_rates else 0.0,
        "learning_detected": bool(learning_detected),
        "improvement": round(improvement, 4),
        "ball_positions": ball_positions,
        "paddle_positions": paddle_positions,
        "scores": scores,
        "n_trials": n_trials,
        "window_size": window_size,
        "learning_curve_summary": {
            "first_half_mean": round(float(first_half), 3),
            "second_half_mean": round(float(second_half), 3),
            "trend": "improving" if improvement > 0.05 else "stable" if improvement > -0.05 else "declining",
        },
        "interpretation": (
            f"DishBrain Pong: {n_trials} trials, {scores['hits']} hits, {scores['misses']} misses. "
            f"Final hit rate: {hit_rates[-1]:.1%}. "
            + ("Learning detected -- organoid minimizes free energy (surprise)."
               if learning_detected else
               "No significant learning -- may need more trials or different encoding.")
        ),
    }
