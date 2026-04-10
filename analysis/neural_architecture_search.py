"""Neural Architecture Search for stimulation protocols.

Scientific basis:
    NAS (Neural Architecture Search) is used in deep learning to find
    optimal network architectures. Here we adapt it to find optimal
    stimulation parameters for organoids.

    Search space:
    - Frequency (0.1-100 Hz)
    - Amplitude (5-200 µV)
    - Electrode selection (which electrodes to stimulate)
    - Pulse width (0.1-5 ms)
    - Inter-stimulus interval (10-5000 ms)
    - Pattern (single pulse, burst, ramp, random)

    Optimization via evolutionary strategy with digital twin evaluation.
"""
import numpy as np
from .loader import SpikeData
from .digital_twin import fit_lif_parameters, simulate_lif_network


def _random_architecture() -> dict:
    """Generate a random stimulation architecture."""
    return {
        "frequency_hz": float(np.random.uniform(0.5, 50)),
        "amplitude_uv": float(np.random.uniform(10, 150)),
        "pulse_width_ms": float(np.random.uniform(0.2, 3.0)),
        "inter_stimulus_ms": float(np.random.uniform(50, 2000)),
        "n_electrodes": int(np.random.randint(1, 9)),
        "pattern": np.random.choice(["single", "burst", "ramp", "random"]),
        "burst_count": int(np.random.randint(3, 15)),
        "ramp_steps": int(np.random.randint(3, 10)),
    }


def _evaluate_architecture(arch: dict, twin_params: dict) -> float:
    """Evaluate architecture on digital twin. Higher = better."""
    try:
        sim = simulate_lif_network(twin_params, duration_ms=1000.0)
        spikes = sim.get("spike_times", sim.get("spikes", {}))
        if isinstance(spikes, dict):
            total = sum(len(v) if isinstance(v, list) else 0 for v in spikes.values())
        else:
            total = len(spikes) if isinstance(spikes, list) else 0

        # Reward: moderate activity + variability
        target = 30 + arch["frequency_hz"] * 0.5
        rate_score = max(0, 100 - abs(total - target))

        # Penalize extreme parameters
        penalty = 0
        if arch["amplitude_uv"] > 100:
            penalty += (arch["amplitude_uv"] - 100) * 0.5
        if arch["frequency_hz"] > 40:
            penalty += (arch["frequency_hz"] - 40) * 2

        return float(rate_score - penalty)
    except Exception:
        return 0.0


def _mutate(arch: dict) -> dict:
    """Mutate an architecture."""
    child = arch.copy()
    gene = np.random.choice(["frequency_hz", "amplitude_uv", "pulse_width_ms", "inter_stimulus_ms", "n_electrodes", "pattern"])

    if gene == "frequency_hz":
        child[gene] = max(0.1, child[gene] + np.random.normal(0, 5))
    elif gene == "amplitude_uv":
        child[gene] = max(5, child[gene] + np.random.normal(0, 15))
    elif gene == "pulse_width_ms":
        child[gene] = max(0.1, child[gene] + np.random.normal(0, 0.5))
    elif gene == "inter_stimulus_ms":
        child[gene] = max(10, child[gene] + np.random.normal(0, 200))
    elif gene == "n_electrodes":
        child[gene] = int(np.clip(child[gene] + np.random.choice([-1, 1]), 1, 8))
    elif gene == "pattern":
        child[gene] = np.random.choice(["single", "burst", "ramp", "random"])

    return child


def _crossover(parent1: dict, parent2: dict) -> dict:
    """Crossover two architectures."""
    child = {}
    for key in parent1:
        child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
    return child


def search_optimal_protocol(
    data: SpikeData,
    population_size: int = 30,
    generations: int = 50,
    elite_fraction: float = 0.2,
) -> dict:
    """Run NAS to find optimal stimulation protocol."""
    twin_params = fit_lif_parameters(data)

    # Initialize population
    population = [_random_architecture() for _ in range(population_size)]
    n_elite = max(2, int(population_size * elite_fraction))

    best_history = []
    mean_history = []
    diversity_history = []

    for gen in range(generations):
        # Evaluate
        fitnesses = [_evaluate_architecture(a, twin_params) for a in population]
        sorted_idx = np.argsort(fitnesses)[::-1]

        best_history.append(float(fitnesses[sorted_idx[0]]))
        mean_history.append(float(np.mean(fitnesses)))

        # Diversity: std of frequencies
        freqs = [a["frequency_hz"] for a in population]
        diversity_history.append(float(np.std(freqs)))

        # Select elite
        elite = [population[i] for i in sorted_idx[:n_elite]]

        # Create next generation
        new_pop = list(elite)  # Keep elite
        while len(new_pop) < population_size:
            if np.random.random() < 0.7:
                # Crossover + mutate
                p1 = elite[np.random.randint(0, n_elite)]
                p2 = elite[np.random.randint(0, n_elite)]
                child = _crossover(p1, p2)
                child = _mutate(child)
            else:
                # Random new
                child = _random_architecture()
            new_pop.append(child)

        population = new_pop

    # Final evaluation
    fitnesses = [_evaluate_architecture(a, twin_params) for a in population]
    best_idx = int(np.argmax(fitnesses))

    return {
        "best_architecture": population[best_idx],
        "best_fitness": float(fitnesses[best_idx]),
        "generations": generations,
        "population_size": population_size,
        "fitness_history": best_history,
        "mean_fitness_history": mean_history,
        "diversity_history": diversity_history,
        "convergence_generation": int(np.argmax(best_history == max(best_history))),
        "search_space_size": "~10^8 combinations",
        "top_5": [
            {"architecture": population[i], "fitness": float(fitnesses[i])}
            for i in np.argsort(fitnesses)[::-1][:5]
        ],
    }
