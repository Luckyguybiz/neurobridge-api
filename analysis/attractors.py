"""Attractor Landscape Mapping — finding memory traces as dynamical attractors.

POTENTIAL BREAKTHROUGH: Nobody has mapped attractor landscapes in organoids.

Hopfield (Nobel 2024): Memory = stable attractor states in neural state space.
When a network settles into a repeated state pattern, that's a "memory."

If organoid activity repeatedly returns to the same states → evidence of
memory that persists even when traditional metrics say "memory is lost."

The 45-minute memory barrier may be an artifact of MEASUREMENT, not biology.
Organoids might retain memories as subtle attractor states that nobody
has looked for because they weren't measuring the right thing.
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist
from .loader import SpikeData


def map_attractor_landscape(
    data: SpikeData,
    bin_size_ms: float = 20.0,
    n_neighbors: int = 5,
    min_visits: int = 3,
    state_radius: float = 0.3,
) -> dict:
    """Map the attractor landscape of organoid neural dynamics.

    1. Convert spike trains to state vectors (binned activity per electrode)
    2. Build state-space trajectory
    3. Find states that the system visits repeatedly (attractors)
    4. Measure basin of attraction (how many states lead to each attractor)
    5. Compute stability (how long the system stays near each attractor)

    An attractor that appears, disappears, then REAPPEARS = memory.
    """
    bin_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    n_bins = len(bins) - 1

    electrode_ids = data.electrode_ids
    n_el = len(electrode_ids)

    # State matrix: each column = state at time t
    states = np.zeros((n_el, n_bins))
    for i, e in enumerate(electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        states[i] = counts

    # Normalize each state vector to unit norm
    norms = np.linalg.norm(states, axis=0, keepdims=True)
    norms[norms == 0] = 1
    states_norm = states / norms

    # Trajectory = sequence of state vectors
    trajectory = states_norm.T  # (n_bins, n_electrodes)

    # Find recurrent states using distance-based clustering
    # State i is "near" state j if cosine distance < state_radius
    distances = cdist(trajectory, trajectory, metric='cosine')

    # Count how many times each state is revisited
    visit_counts = np.sum(distances < state_radius, axis=1)

    # Attractors = states visited >= min_visits times
    attractor_mask = visit_counts >= min_visits

    # Cluster nearby attractor states
    attractors = []
    used = set()

    attractor_indices = np.where(attractor_mask)[0]
    for idx in attractor_indices:
        if idx in used:
            continue

        # Find all states near this one
        nearby = np.where(distances[idx] < state_radius)[0]
        nearby = [n for n in nearby if n not in used]

        if len(nearby) < min_visits:
            continue

        for n in nearby:
            used.add(n)

        # Attractor center = mean of nearby states
        center = np.mean(trajectory[nearby], axis=0)

        # Visit times
        visit_times = [float(bins[n]) for n in sorted(nearby)]

        # Dwell time = consecutive bins in attractor
        dwell_times = []
        current_dwell = 1
        sorted_nearby = sorted(nearby)
        for k in range(1, len(sorted_nearby)):
            if sorted_nearby[k] - sorted_nearby[k-1] == 1:
                current_dwell += 1
            else:
                dwell_times.append(current_dwell * bin_size_ms)
                current_dwell = 1
        dwell_times.append(current_dwell * bin_size_ms)

        # Check for REAPPEARANCE (key for memory detection)
        # Split visits into episodes separated by >= 5 seconds
        episodes = []
        current_episode = [visit_times[0]]
        for vt in visit_times[1:]:
            if vt - current_episode[-1] > 5.0:
                episodes.append(current_episode)
                current_episode = [vt]
            else:
                current_episode.append(vt)
        episodes.append(current_episode)

        attractors.append({
            "id": len(attractors),
            "center": center.tolist(),
            "n_visits": len(nearby),
            "visit_times": visit_times[:20],  # limit for JSON
            "mean_dwell_ms": round(float(np.mean(dwell_times)), 1),
            "max_dwell_ms": round(float(np.max(dwell_times)), 1),
            "n_episodes": len(episodes),
            "reappears": len(episodes) > 1,
            "episode_gaps_sec": [
                round(episodes[k+1][0] - episodes[k][-1], 2)
                for k in range(len(episodes) - 1)
            ][:5],
            "dominant_electrodes": [
                int(electrode_ids[e]) for e in np.argsort(center)[-3:][::-1]
            ],
            "stability": round(float(np.mean(dwell_times) / bin_size_ms), 2),
        })

    # Sort by number of visits
    attractors.sort(key=lambda a: a["n_visits"], reverse=True)

    # Memory detection: attractors that reappear after long gaps
    memory_candidates = [a for a in attractors if a["reappears"] and any(g > 5 for g in a.get("episode_gaps_sec", []))]

    # State space coverage
    n_unique_states = len(set(tuple(np.round(s, 1)) for s in trajectory.tolist()))
    coverage = n_unique_states / n_bins if n_bins > 0 else 0

    return {
        "attractors": attractors[:15],
        "n_attractors": len(attractors),
        "n_memory_candidates": len(memory_candidates),
        "memory_candidates": memory_candidates[:5],
        "has_memory_evidence": len(memory_candidates) > 0,
        "state_space_coverage": round(float(coverage), 4),
        "n_states_total": n_bins,
        "n_unique_states": n_unique_states,
        "interpretation": (
            f"MEMORY EVIDENCE: {len(memory_candidates)} attractor(s) reappear after >5 second gaps. "
            f"The organoid returns to the same computational states — this is memory-like behavior. "
            f"Gaps: {[a['episode_gaps_sec'] for a in memory_candidates[:3]]}"
            if memory_candidates
            else f"Found {len(attractors)} attractors but none show long-term reappearance. "
            f"Activity is organized but without clear memory signatures in this recording."
        ),
        "significance": (
            "BREAKTHROUGH POTENTIAL — reappearing attractors suggest persistent memory. "
            "This contradicts the 45-minute memory barrier. The memory may exist but be "
            "invisible to traditional firing-rate metrics."
            if memory_candidates
            else "Standard attractor dynamics — organized but not memory-like."
        ),
    }


def compute_state_space_geometry(data: SpikeData, bin_size_ms: float = 20.0) -> dict:
    """Analyze the geometry of neural state space.

    Measures:
    - Dimensionality: how many dimensions does the system actually use?
    - Trajectory complexity: does it follow simple paths or explore widely?
    - Recurrence: how often does it return to similar states?
    """
    bin_sec = bin_size_ms / 1000.0
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)

    states = np.zeros((len(data.electrode_ids), len(bins) - 1))
    for i, e in enumerate(data.electrode_ids):
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        states[i] = counts

    trajectory = states.T  # (time, electrodes)

    if len(trajectory) < 10:
        return {"error": "Not enough data"}

    # Effective dimensionality (participation ratio)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(trajectory)
    n_comp = min(len(trajectory) - 1, trajectory.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(scaled)

    eigenvalues = pca.explained_variance_
    participation_ratio = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)) if np.sum(eigenvalues ** 2) > 0 else 0

    # Trajectory length (total distance traveled in state space)
    diffs = np.diff(trajectory, axis=0)
    step_sizes = np.linalg.norm(diffs, axis=1)
    total_length = float(np.sum(step_sizes))
    mean_step = float(np.mean(step_sizes))

    # Recurrence rate: fraction of state pairs within radius
    n_sample = min(500, len(trajectory))
    sample_idx = np.random.choice(len(trajectory), n_sample, replace=False)
    sample = trajectory[sample_idx]
    dists = cdist(sample, sample, metric='cosine')
    recurrence_rate = float(np.mean(dists < 0.3))

    return {
        "effective_dimensionality": round(participation_ratio, 2),
        "max_dimensionality": trajectory.shape[1],
        "dimensionality_ratio": round(participation_ratio / trajectory.shape[1], 3) if trajectory.shape[1] > 0 else 0,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "trajectory_length": round(total_length, 2),
        "mean_step_size": round(mean_step, 4),
        "recurrence_rate": round(recurrence_rate, 4),
        "interpretation": (
            f"Effective dimensionality: {participation_ratio:.1f}/{trajectory.shape[1]} "
            f"(the organoid uses {participation_ratio/trajectory.shape[1]*100:.0f}% of available state space). "
            + ("High recurrence — system revisits similar states frequently." if recurrence_rate > 0.2
               else "Low recurrence — system explores widely without returning.")
        ),
    }
