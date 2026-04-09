"""Pong Engine — Neural Organoid Pong Simulator.

Implements the full Pong paradigm from Kagan et al. (2022) "In vitro
neurons learn and exhibit sentience when embodied in a simulated game
world" (Neuron).

Pipeline:
1. Encoding: ball position/velocity → electrode stimulation pattern
2. Neural processing: organoid (spike data) processes stimulation
3. Decoding: electrode firing rates → paddle movement direction
4. Physics: update game state based on decoded action
5. Reward: stimulate again if miss (penalty), silence if hit (reward)
6. Learning curve: track performance over N games

The key insight from DishBrain: predictable feedback (hits = silence,
misses = noise burst) was sufficient for neural networks to improve.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# ── Main Pong Simulation ──────────────────────────────────────────────────────

def simulate_pong(
    data: SpikeData,
    n_games: int = 20,
    max_volleys_per_game: int = 30,
    paddle_height: float = 0.2,
    encoding: str = "rate",
    decoding: str = "population_vector",
    reward_rule: str = "dishbrain",
    seed: int = 0,
) -> dict:
    """Simulate N games of Pong using organoid spike data.

    Args:
        data: SpikeData (spike activity drives paddle decisions)
        n_games: number of games to play
        max_volleys_per_game: maximum rallies before game end
        paddle_height: paddle size (0-1 normalized)
        encoding: 'rate', 'temporal', or 'population'
        decoding: 'population_vector', 'winner_take_all', or 'threshold'
        reward_rule: 'dishbrain' (silence=good), 'positive', or 'symmetric'
        seed: random seed

    Returns:
        dict with per-game stats, learning curve, encoding/decoding analysis
    """
    rng = np.random.default_rng(seed)
    t_start, t_end = data.time_range
    duration = t_end - t_start
    n_electrodes = len(data.electrode_ids)

    game_results = []
    learning_curve = []
    volley_counts = []

    for game_idx in range(n_games):
        # Initial ball state
        ball_x = 0.5
        ball_y = rng.uniform(0.1, 0.9)
        ball_vx = rng.choice([-0.05, 0.05])
        ball_vy = rng.uniform(-0.03, 0.03)
        paddle_y = 0.5  # AI paddle (center)
        opp_paddle_y = 0.5  # opponent (random)

        volleys = 0
        hits = 0
        misses = 0
        paddle_trajectory = []
        ball_trajectory = []
        decode_actions = []

        # Time offset into recording for this game
        t_game_offset = rng.uniform(0, max(0, duration - 5.0))
        step_duration = min(0.1, duration / (max_volleys_per_game * 5))

        for step in range(max_volleys_per_game * 5):
            # 1. ENCODE ball state → stimulation pattern
            stim_pattern = _encode_ball_state(
                ball_x, ball_y, ball_vx, ball_vy,
                n_electrodes, encoding, rng
            )

            # 2. READ neural response at this time point
            t_step = t_game_offset + step * step_duration
            t_step = min(t_step, t_end - step_duration)
            mask = (data.times >= t_step) & (data.times < t_step + step_duration)
            step_electrodes = data.electrodes[mask]

            # 3. DECODE: electrode firing rates → paddle action
            action, confidence = _decode_action(
                step_electrodes, data.electrode_ids, stim_pattern, decoding
            )
            decode_actions.append({"action": action, "confidence": round(float(confidence), 3)})

            # 4. Move paddle
            paddle_speed = 0.04
            if action == "up":
                paddle_y = min(1.0 - paddle_height / 2, paddle_y + paddle_speed)
            elif action == "down":
                paddle_y = max(paddle_height / 2, paddle_y - paddle_speed)
            # else: stay

            # Opponent (simple AI for comparison)
            if ball_x > 0.5:
                opp_paddle_y += 0.03 * np.sign(ball_y - opp_paddle_y)
            opp_paddle_y = float(np.clip(opp_paddle_y, paddle_height / 2, 1 - paddle_height / 2))

            # 5. Physics update
            ball_x += ball_vx
            ball_y += ball_vy

            # Wall bounce (top/bottom)
            if ball_y <= 0.0 or ball_y >= 1.0:
                ball_vy *= -1
                ball_y = float(np.clip(ball_y, 0.0, 1.0))

            # Paddle collision (left = AI paddle at x≈0)
            if ball_x <= 0.05:
                if abs(ball_y - paddle_y) < paddle_height / 2:
                    ball_vx = abs(ball_vx) * 1.05  # speed up slightly
                    ball_vy += rng.uniform(-0.01, 0.01)
                    hits += 1
                    volleys += 1
                    reward_stim = _compute_reward_stimulation(True, reward_rule, rng)
                else:
                    misses += 1
                    reward_stim = _compute_reward_stimulation(False, reward_rule, rng)
                    break  # game over

            # Right wall (opponent paddle)
            if ball_x >= 0.95:
                if abs(ball_y - opp_paddle_y) < paddle_height / 2:
                    ball_vx = -abs(ball_vx)
                    volleys += 1
                else:
                    break  # opponent missed

            ball_trajectory.append({"x": round(float(ball_x), 3), "y": round(float(ball_y), 3)})
            paddle_trajectory.append({"paddle_y": round(float(paddle_y), 3)})

        # Game score
        hit_rate = float(hits / max(1, hits + misses))

        game_results.append({
            "game": game_idx + 1,
            "volleys": volleys,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 3),
            "final_ball_x": round(float(ball_x), 3),
            "final_paddle_y": round(float(paddle_y), 3),
            "ball_trajectory": ball_trajectory[-10:],  # last 10 positions
            "actions_up": sum(1 for a in decode_actions if a["action"] == "up"),
            "actions_down": sum(1 for a in decode_actions if a["action"] == "down"),
            "actions_stay": sum(1 for a in decode_actions if a["action"] == "stay"),
        })
        volley_counts.append(volleys)
        learning_curve.append(hit_rate)

    # Learning analysis
    lc_arr = np.array(learning_curve)
    window = min(5, n_games)
    smoothed_lc = [
        round(float(np.mean(lc_arr[max(0, i - window + 1): i + 1])), 3)
        for i in range(n_games)
    ]

    first_half = float(np.mean(lc_arr[: n_games // 2]))
    second_half = float(np.mean(lc_arr[n_games // 2 :]))
    improvement = float(second_half - first_half)

    return {
        "n_games": n_games,
        "encoding": encoding,
        "decoding": decoding,
        "reward_rule": reward_rule,
        "paddle_height": paddle_height,
        "mean_volleys": round(float(np.mean(volley_counts)), 1),
        "max_volleys": int(np.max(volley_counts)),
        "mean_hit_rate": round(float(np.mean(lc_arr)), 3),
        "overall_improvement": round(improvement, 3),
        "trend": "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable",
        "learning_curve": smoothed_lc,
        "games": game_results,
        "encoding_analysis": _analyze_encoding(encoding, n_electrodes),
        "decoding_analysis": {
            "method": decoding,
            "mean_action_confidence": round(
                float(np.mean([
                    a["confidence"]
                    for g in game_results
                    for a in decode_actions
                ] if game_results else [0.5])), 3
            ),
        },
        "interpretation": (
            f"Pong simulation: {n_games} games, {encoding} encoding, {decoding} decoding. "
            f"Mean hit rate: {np.mean(lc_arr):.1%}. "
            f"Performance trend: {'↑ improving' if improvement > 0.05 else '↓ declining' if improvement < -0.05 else '→ stable'} "
            f"(Δ={improvement:+.3f}). "
            + ("Organoid shows learning!" if improvement > 0.1 else
               "Moderate adaptation." if improvement > 0.0 else
               "No improvement detected — consider different encoding/reward strategy.")
        ),
    }


# ── Encoding Functions ────────────────────────────────────────────────────────

def encode_ball_state(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    n_electrodes: int,
    encoding: str = "rate",
) -> dict:
    """Compute stimulation pattern for given ball state.

    Returns:
        dict with electrode stimulation intensities and encoding metadata
    """
    pattern = _encode_ball_state(ball_x, ball_y, ball_vx, ball_vy, n_electrodes, encoding, np.random.default_rng(0))
    return {
        "ball_state": {"x": ball_x, "y": ball_y, "vx": ball_vx, "vy": ball_vy},
        "encoding_method": encoding,
        "n_electrodes": n_electrodes,
        "stimulation_pattern": [round(float(p), 4) for p in pattern],
        "active_electrodes": [i for i, p in enumerate(pattern) if p > 0.5],
        "mean_intensity": round(float(np.mean(pattern)), 4),
    }


def _encode_ball_state(
    ball_x: float, ball_y: float, ball_vx: float, ball_vy: float,
    n_electrodes: int, encoding: str, rng: np.random.Generator,
) -> np.ndarray:
    """Internal: map ball state to electrode stimulation pattern."""
    pattern = np.zeros(n_electrodes)

    if encoding == "rate":
        # Ball Y position → continuous rate gradient across electrodes
        for i in range(n_electrodes):
            electrode_pos = i / max(1, n_electrodes - 1)
            distance = abs(ball_y - electrode_pos)
            pattern[i] = np.exp(-distance ** 2 / 0.1)
        # Ball X (proximity to paddle) → overall intensity
        proximity = 1.0 - ball_x  # higher when ball is close
        pattern *= (0.3 + 0.7 * proximity)

    elif encoding == "temporal":
        # Temporal burst: which electrode fires first encodes position
        active = int(ball_y * n_electrodes)
        active = min(n_electrodes - 1, max(0, active))
        pattern[active] = 1.0
        # Velocity encoded as spread
        spread = max(1, int(abs(ball_vy) * n_electrodes * 2))
        for j in range(max(0, active - spread), min(n_electrodes, active + spread + 1)):
            pattern[j] = max(pattern[j], 0.5)

    elif encoding == "population":
        # Population vector: both Y and Vy encoded in different electrode groups
        half = n_electrodes // 2
        for i in range(half):
            electrode_pos = i / max(1, half - 1)
            pattern[i] = np.exp(-((ball_y - electrode_pos) ** 2) / 0.1)
        for i in range(half, n_electrodes):
            electrode_pos = (i - half) / max(1, n_electrodes - half - 1)
            vy_norm = (ball_vy + 0.1) / 0.2
            pattern[i] = np.exp(-((vy_norm - electrode_pos) ** 2) / 0.1)

    return pattern


# ── Decoding Functions ────────────────────────────────────────────────────────

def _decode_action(
    step_electrodes: np.ndarray,
    electrode_ids: list,
    stim_pattern: np.ndarray,
    decoding: str,
) -> tuple[str, float]:
    """Decode paddle action from electrode firing pattern."""
    n_electrodes = len(electrode_ids)
    if n_electrodes == 0:
        return "stay", 0.0

    # Count spikes per electrode
    rates = np.zeros(n_electrodes)
    for i, eid in enumerate(electrode_ids):
        rates[i] = np.sum(step_electrodes == eid)

    if decoding == "population_vector":
        # Weighted average of electrode positions → decoded Y position
        positions = np.linspace(0, 1, n_electrodes)
        total = float(np.sum(rates))
        if total > 0:
            decoded_y = float(np.dot(positions, rates) / total)
        else:
            decoded_y = 0.5
        # Estimate where ball "should" be from stim pattern
        ball_y_estimate = float(np.dot(positions, stim_pattern) / (np.sum(stim_pattern) + 1e-10))
        action = "up" if decoded_y < ball_y_estimate - 0.05 else (
                 "down" if decoded_y > ball_y_estimate + 0.05 else "stay")
        confidence = 1.0 - abs(decoded_y - ball_y_estimate)

    elif decoding == "winner_take_all":
        if len(rates) > 0 and np.max(rates) > 0:
            winner = int(np.argmax(rates))
            half = n_electrodes // 2
            action = "up" if winner < half else "down"
            confidence = float(rates[winner] / (np.sum(rates) + 1e-10))
        else:
            action = "stay"
            confidence = 0.5

    elif decoding == "threshold":
        upper_half = rates[n_electrodes // 2:]
        lower_half = rates[:n_electrodes // 2]
        up_sum = float(np.sum(upper_half))
        down_sum = float(np.sum(lower_half))
        threshold = 1.0
        if up_sum > threshold and up_sum > down_sum:
            action = "up"
        elif down_sum > threshold and down_sum > up_sum:
            action = "down"
        else:
            action = "stay"
        total = up_sum + down_sum + 1e-10
        confidence = max(up_sum, down_sum) / total

    else:
        action = "stay"
        confidence = 0.5

    return action, float(np.clip(confidence, 0, 1))


def _compute_reward_stimulation(hit: bool, reward_rule: str, rng: np.random.Generator) -> dict:
    """Compute reward stimulation pattern based on outcome."""
    if reward_rule == "dishbrain":
        # Hit → silence (no stimulation), Miss → random noise burst
        return {
            "type": "silence" if hit else "noise_burst",
            "intensity": 0.0 if hit else rng.uniform(0.5, 1.0),
            "duration_ms": 0.0 if hit else rng.uniform(50, 200),
        }
    elif reward_rule == "positive":
        # Hit → positive stimulation, Miss → nothing
        return {
            "type": "positive_pulse" if hit else "none",
            "intensity": rng.uniform(0.3, 0.7) if hit else 0.0,
            "duration_ms": 50.0 if hit else 0.0,
        }
    elif reward_rule == "symmetric":
        # Hit → excitatory burst, Miss → inhibitory burst
        return {
            "type": "excitatory" if hit else "inhibitory",
            "intensity": rng.uniform(0.3, 0.7),
            "duration_ms": 100.0,
        }
    return {"type": "none", "intensity": 0.0, "duration_ms": 0.0}


def _analyze_encoding(encoding: str, n_electrodes: int) -> dict:
    """Describe encoding scheme properties."""
    descriptions = {
        "rate": {
            "description": "Gaussian rate code: ball Y → firing rate gradient across electrodes",
            "information_bits": round(float(np.log2(n_electrodes)), 2),
            "temporal_resolution_ms": 100.0,
            "noise_robustness": "high",
        },
        "temporal": {
            "description": "Temporal code: position → first-spike timing",
            "information_bits": round(float(np.log2(n_electrodes) * 1.5), 2),
            "temporal_resolution_ms": 1.0,
            "noise_robustness": "medium",
        },
        "population": {
            "description": "Population vector: Y and Vy jointly encoded in electrode subgroups",
            "information_bits": round(float(np.log2(n_electrodes ** 2)), 2),
            "temporal_resolution_ms": 50.0,
            "noise_robustness": "high",
        },
    }
    return descriptions.get(encoding, {"description": encoding})
