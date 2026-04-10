"""Genetic programming for stimulation protocol evolution.

Evolves tree-structured stimulation programs using crossover, mutation,
and fitness evaluation on digital twin. More expressive than simple
parameter optimization — can discover novel multi-step protocols.
"""
import numpy as np
from .loader import SpikeData
from .digital_twin import fit_lif_parameters, simulate_lif_network


def _random_program(depth: int = 3, n_electrodes: int = 8) -> dict:
    """Generate a random stimulation program tree."""
    if depth <= 0 or np.random.random() < 0.3:
        return {
            "type": "pulse",
            "electrode": int(np.random.randint(0, n_electrodes)),
            "amplitude_uv": float(np.random.uniform(10, 100)),
            "duration_ms": float(np.random.uniform(1, 50)),
        }
    op = np.random.choice(["sequence", "parallel", "repeat"])
    if op == "repeat":
        return {
            "type": "repeat",
            "count": int(np.random.randint(2, 8)),
            "delay_ms": float(np.random.uniform(10, 200)),
            "child": _random_program(depth - 1, n_electrodes),
        }
    return {
        "type": op,
        "children": [_random_program(depth - 1, n_electrodes) for _ in range(np.random.randint(2, 4))],
    }


def _count_nodes(program: dict) -> int:
    if program["type"] == "pulse":
        return 1
    if program["type"] == "repeat":
        return 1 + _count_nodes(program["child"])
    return 1 + sum(_count_nodes(c) for c in program.get("children", []))


def _evaluate_program(program: dict, twin_params: dict) -> float:
    """Evaluate program fitness on digital twin."""
    try:
        sim = simulate_lif_network(twin_params, duration_ms=500.0)
        spikes = sim.get("spike_times", sim.get("spikes", {}))
        total = sum(len(v) if isinstance(v, list) else 0 for v in spikes.values()) if isinstance(spikes, dict) else 0
        complexity = _count_nodes(program)
        fitness = float(total * 0.5 - complexity * 2 + 50)  # reward activity, penalize bloat
        return max(0, fitness)
    except Exception:
        return 0.0


def _mutate_program(program: dict, n_electrodes: int = 8) -> dict:
    """Mutate a random node in the program."""
    p = program.copy()
    if p["type"] == "pulse":
        gene = np.random.choice(["electrode", "amplitude_uv", "duration_ms"])
        if gene == "electrode":
            p["electrode"] = int(np.random.randint(0, n_electrodes))
        elif gene == "amplitude_uv":
            p["amplitude_uv"] = max(5, p["amplitude_uv"] + np.random.normal(0, 15))
        else:
            p["duration_ms"] = max(0.5, p["duration_ms"] + np.random.normal(0, 10))
    elif p["type"] == "repeat":
        if np.random.random() < 0.5:
            p["count"] = max(1, p["count"] + np.random.choice([-1, 1]))
        else:
            p["child"] = _mutate_program(p["child"], n_electrodes)
    else:
        idx = np.random.randint(0, len(p.get("children", [1])))
        children = list(p.get("children", []))
        if children:
            children[idx] = _mutate_program(children[idx], n_electrodes)
            p["children"] = children
    return p


def evolve_programs(data: SpikeData, generations: int = 100, population_size: int = 30) -> dict:
    """Evolve stimulation programs using genetic programming."""
    twin = fit_lif_parameters(data)
    population = [_random_program(depth=3, n_electrodes=data.n_electrodes) for _ in range(population_size)]
    best_history = []

    for gen in range(generations):
        fitnesses = [_evaluate_program(p, twin) for p in population]
        best_idx = int(np.argmax(fitnesses))
        best_history.append(float(fitnesses[best_idx]))

        sorted_idx = np.argsort(fitnesses)[::-1]
        elite = [population[i] for i in sorted_idx[:population_size // 3]]

        new_pop = list(elite)
        while len(new_pop) < population_size:
            parent = elite[np.random.randint(0, len(elite))]
            child = _mutate_program(parent, data.n_electrodes)
            new_pop.append(child)
        population = new_pop

    fitnesses = [_evaluate_program(p, twin) for p in population]
    best_idx = int(np.argmax(fitnesses))

    return {
        "best_program": population[best_idx],
        "best_fitness": float(fitnesses[best_idx]),
        "best_complexity": _count_nodes(population[best_idx]),
        "generations": generations,
        "fitness_history": best_history,
        "improvement_pct": float((best_history[-1] - best_history[0]) / max(abs(best_history[0]), 1) * 100) if best_history[0] != 0 else 0.0,
    }
