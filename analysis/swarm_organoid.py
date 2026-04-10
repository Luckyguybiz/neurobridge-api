"""Multi-organoid swarm intelligence simulation.

Scientific basis:
    If individual organoids have limited computation, could multiple
    organoids working together achieve more? This module simulates
    information transfer between organoids via shared channels,
    measuring whether collective computation exceeds individual.
"""
import numpy as np
from .loader import SpikeData


def simulate_swarm(data: SpikeData, n_organoids: int = 4, coupling_strength: float = 0.3) -> dict:
    """Simulate a swarm of organoids sharing information."""
    # Split data into n_organoids virtual sub-networks
    electrodes_per = max(1, data.n_electrodes // n_organoids)
    organoid_rates = []

    for org_idx in range(n_organoids):
        start_e = org_idx * electrodes_per
        end_e = min(start_e + electrodes_per, data.n_electrodes)
        eids = data.electrode_ids[start_e:end_e]
        mask = np.isin(data.electrodes, eids)
        rate = float(np.sum(mask)) / max(data.duration, 0.001)
        organoid_rates.append(rate)

    # Simulate coupling: each organoid's activity is influenced by neighbors
    coupled_rates = np.array(organoid_rates, dtype=float)
    for _ in range(10):  # 10 iterations of coupling
        new_rates = coupled_rates.copy()
        for i in range(n_organoids):
            neighbor_mean = np.mean([coupled_rates[j] for j in range(n_organoids) if j != i])
            new_rates[i] = coupled_rates[i] * (1 - coupling_strength) + neighbor_mean * coupling_strength
        coupled_rates = new_rates

    # Measure collective vs individual performance
    individual_entropy = float(-np.sum(
        [r / max(sum(organoid_rates), 1e-10) * np.log2(max(r / max(sum(organoid_rates), 1e-10), 1e-10))
         for r in organoid_rates if r > 0]
    ))
    coupled_entropy = float(-np.sum(
        [r / max(sum(coupled_rates), 1e-10) * np.log2(max(r / max(sum(coupled_rates), 1e-10), 1e-10))
         for r in coupled_rates if r > 0]
    ))

    synergy = coupled_entropy - individual_entropy

    return {
        "n_organoids": n_organoids,
        "coupling_strength": coupling_strength,
        "individual_rates": [float(r) for r in organoid_rates],
        "coupled_rates": coupled_rates.tolist(),
        "individual_entropy": individual_entropy,
        "coupled_entropy": coupled_entropy,
        "synergy": float(synergy),
        "collective_benefit": synergy > 0,
        "verdict": (
            "Positive synergy — swarm outperforms individuals" if synergy > 0.1
            else "Marginal benefit from coupling" if synergy > 0
            else "No benefit — individuals sufficient"
        ),
    }
