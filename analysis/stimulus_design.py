"""Automatic stimulus protocol design using evolutionary optimization.

Generates optimal stimulation protocols by testing on digital twin
and evolving the best candidates over multiple generations.
"""
import numpy as np
from .loader import SpikeData
from .digital_twin import fit_lif_parameters, simulate_lif_network

def _create_random_protocol(n_electrodes: int, duration_ms: float = 1000.0) -> dict:
    """Generate a random stimulation protocol."""
    n_pulses = np.random.randint(5, 30)
    return {
        "electrode": int(np.random.randint(0, n_electrodes)),
        "pulses": sorted([float(np.random.uniform(0, duration_ms)) for _ in range(n_pulses)]),
        "amplitude_uv": float(np.random.uniform(10, 100)),
        "pulse_width_ms": float(np.random.uniform(0.1, 2.0)),
        "frequency_hz": float(np.random.uniform(1, 40)),
        "duration_ms": duration_ms,
    }

def _evaluate_protocol(protocol: dict, twin_params: dict) -> float:
    """Evaluate a protocol on digital twin — return fitness score."""
    try:
        sim = simulate_lif_network(twin_params, duration_ms=protocol["duration_ms"])
        spikes = sim.get("spike_times", sim.get("spikes", {}))
        if isinstance(spikes, dict):
            total = sum(len(v) if isinstance(v, list) else 0 for v in spikes.values())
        elif isinstance(spikes, list):
            total = len(spikes)
        else:
            total = 0
        # Fitness: moderate activity (not too much, not too little)
        target = 50
        fitness = -abs(total - target) + 10 * min(total, target)
        return float(fitness)
    except Exception:
        return 0.0

def evolve_protocol(data: SpikeData, generations: int = 50, population_size: int = 20) -> dict:
    """Evolutionary optimization of stimulation protocol."""
    twin_params = fit_lif_parameters(data)

    # Initialize population
    population = [_create_random_protocol(data.n_electrodes) for _ in range(population_size)]

    best_fitness_history = []
    mean_fitness_history = []

    for gen in range(generations):
        # Evaluate
        fitnesses = [_evaluate_protocol(p, twin_params) for p in population]

        best_idx = int(np.argmax(fitnesses))
        best_fitness_history.append(float(fitnesses[best_idx]))
        mean_fitness_history.append(float(np.mean(fitnesses)))

        # Select top 50%
        sorted_idx = np.argsort(fitnesses)[::-1]
        survivors = [population[i] for i in sorted_idx[:population_size // 2]]

        # Mutate and create offspring
        offspring = []
        for parent in survivors:
            child = parent.copy()
            child["amplitude_uv"] = max(1, child["amplitude_uv"] + np.random.normal(0, 10))
            child["frequency_hz"] = max(0.1, child["frequency_hz"] + np.random.normal(0, 5))
            child["electrode"] = int(np.random.randint(0, data.n_electrodes))
            offspring.append(child)

        population = survivors + offspring

    # Return best
    fitnesses = [_evaluate_protocol(p, twin_params) for p in population]
    best_idx = int(np.argmax(fitnesses))

    return {
        "best_protocol": population[best_idx],
        "best_fitness": float(fitnesses[best_idx]),
        "generations": generations,
        "population_size": population_size,
        "fitness_history": best_fitness_history,
        "mean_fitness_history": mean_fitness_history,
        "improvement_pct": float((best_fitness_history[-1] - best_fitness_history[0]) / max(abs(best_fitness_history[0]), 1) * 100) if best_fitness_history else 0.0,
    }
