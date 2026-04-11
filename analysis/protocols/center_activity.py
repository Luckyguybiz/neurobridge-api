"""Center of Activity Protocol -- shift neural activity center via distant electrode stimulation.

Implements the Center of Activity (CA) paradigm: compute the spatial centroid of
neural firing across a 2x4 MEA grid, then iteratively stimulate the electrode
farthest from the CA to shift it across the array.

MEA layout (2x4 grid):
    E0(0,0)  E1(1,0)  E2(2,0)  E3(3,0)
    E4(0,1)  E5(1,1)  E6(2,1)  E7(3,1)
"""

import numpy as np
from typing import Optional
from ..loader import SpikeData


# Electrode positions in the 2x4 grid
ELECTRODE_POSITIONS = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (2.0, 0.0),
    3: (3.0, 0.0),
    4: (0.0, 1.0),
    5: (1.0, 1.0),
    6: (2.0, 1.0),
    7: (3.0, 1.0),
}

N_ELECTRODES = 8


def _get_electrode_rates(data: SpikeData) -> np.ndarray:
    """Compute firing rates for electrodes 0-7 from SpikeData."""
    rates = np.zeros(N_ELECTRODES)
    duration = max(data.duration, 0.001)
    for e in data.electrode_ids:
        idx = e % N_ELECTRODES
        n_spikes = int(np.sum(data.electrodes == e))
        rates[idx] += n_spikes / duration
    # If no spikes at all, use small uniform baseline
    if np.sum(rates) == 0:
        rates = np.ones(N_ELECTRODES) * 0.1
    return rates


def compute_center_of_activity(data: SpikeData) -> dict:
    """Compute the Center of Activity (CA) from spike data.

    CA = sum(Fk * (Xk, Yk)) / sum(Fk)

    where Fk is the firing rate of electrode k, and (Xk, Yk) is its position
    in the 2x4 MEA grid.

    Args:
        data: SpikeData object with spike times, electrodes, amplitudes.

    Returns:
        dict with CA coordinates, per-electrode rates, and grid info.
    """
    rates = _get_electrode_rates(data)
    total_rate = float(np.sum(rates))

    if total_rate == 0:
        ca_x, ca_y = 1.5, 0.5  # grid center
    else:
        ca_x = sum(rates[k] * ELECTRODE_POSITIONS[k][0] for k in range(N_ELECTRODES)) / total_rate
        ca_y = sum(rates[k] * ELECTRODE_POSITIONS[k][1] for k in range(N_ELECTRODES)) / total_rate

    per_electrode = {}
    for k in range(N_ELECTRODES):
        pos = ELECTRODE_POSITIONS[k]
        dist = float(np.sqrt((ca_x - pos[0]) ** 2 + (ca_y - pos[1]) ** 2))
        per_electrode[k] = {
            "position": list(pos),
            "rate_hz": round(float(rates[k]), 3),
            "distance_to_ca": round(dist, 4),
        }

    return {
        "center_of_activity": {"x": round(float(ca_x), 4), "y": round(float(ca_y), 4)},
        "total_firing_rate_hz": round(total_rate, 3),
        "electrode_rates": per_electrode,
        "grid": "2x4",
        "n_electrodes": N_ELECTRODES,
    }


def simulate_ca_shift(data: SpikeData, n_steps: int = 20) -> dict:
    """Simulate shifting the Center of Activity by stimulating the farthest electrode.

    At each step:
    1. Compute current CA from electrode rates.
    2. Find the electrode farthest from CA.
    3. Stimulate that electrode (boost its firing rate).
    4. Apply decay to all rates (natural adaptation).
    5. Record new CA position.

    Args:
        data: SpikeData for initial rate estimation.
        n_steps: Number of stimulation steps to simulate.

    Returns:
        dict with CA trajectory, per-step electrode rates, target electrodes,
        total shift distance, and learning metrics.
    """
    rng = np.random.default_rng(42)
    rates = _get_electrode_rates(data)

    trajectory = []
    rate_history = []
    target_electrodes = []

    stim_boost = 0.3  # fractional boost per stimulation
    decay_rate = 0.95  # natural rate decay per step
    noise_std = 0.02  # stochastic noise

    for step in range(n_steps):
        # Compute current CA
        total = float(np.sum(rates))
        if total == 0:
            ca_x, ca_y = 1.5, 0.5
        else:
            ca_x = sum(rates[k] * ELECTRODE_POSITIONS[k][0] for k in range(N_ELECTRODES)) / total
            ca_y = sum(rates[k] * ELECTRODE_POSITIONS[k][1] for k in range(N_ELECTRODES)) / total

        trajectory.append({"step": step, "x": round(float(ca_x), 4), "y": round(float(ca_y), 4)})
        rate_history.append({k: round(float(rates[k]), 3) for k in range(N_ELECTRODES)})

        # Find farthest electrode from CA
        distances = np.array([
            np.sqrt((ca_x - ELECTRODE_POSITIONS[k][0]) ** 2 + (ca_y - ELECTRODE_POSITIONS[k][1]) ** 2)
            for k in range(N_ELECTRODES)
        ])
        target = int(np.argmax(distances))
        target_electrodes.append(target)

        # Stimulate target: boost its rate
        rates[target] *= (1.0 + stim_boost)
        rates[target] += rng.normal(0, noise_std) * rates[target]

        # Natural decay + noise for all electrodes
        rates *= decay_rate
        rates += rng.normal(0, noise_std, N_ELECTRODES) * rates
        rates = np.clip(rates, 0.01, None)

    # Final CA
    total_final = float(np.sum(rates))
    if total_final > 0:
        final_ca_x = sum(rates[k] * ELECTRODE_POSITIONS[k][0] for k in range(N_ELECTRODES)) / total_final
        final_ca_y = sum(rates[k] * ELECTRODE_POSITIONS[k][1] for k in range(N_ELECTRODES)) / total_final
    else:
        final_ca_x, final_ca_y = 1.5, 0.5

    # Compute total shift
    if len(trajectory) >= 2:
        dx = trajectory[-1]["x"] - trajectory[0]["x"]
        dy = trajectory[-1]["y"] - trajectory[0]["y"]
        total_shift = float(np.sqrt(dx ** 2 + dy ** 2))
    else:
        total_shift = 0.0

    # Path length (cumulative)
    path_length = 0.0
    for i in range(1, len(trajectory)):
        ddx = trajectory[i]["x"] - trajectory[i - 1]["x"]
        ddy = trajectory[i]["y"] - trajectory[i - 1]["y"]
        path_length += float(np.sqrt(ddx ** 2 + ddy ** 2))

    return {
        "trajectory": trajectory,
        "rate_history": rate_history,
        "target_electrodes": target_electrodes,
        "initial_ca": {"x": trajectory[0]["x"], "y": trajectory[0]["y"]},
        "final_ca": {"x": round(float(final_ca_x), 4), "y": round(float(final_ca_y), 4)},
        "total_shift": round(total_shift, 4),
        "path_length": round(path_length, 4),
        "n_steps": n_steps,
        "shift_detected": total_shift > 0.2,
        "interpretation": (
            f"CA shifted {total_shift:.3f} units over {n_steps} steps "
            f"(path length {path_length:.3f}). "
            + ("Significant shift detected -- stimulation effectively guided activity."
               if total_shift > 0.2 else
               "Minimal shift -- organoid may resist spatial reorganization.")
        ),
    }
