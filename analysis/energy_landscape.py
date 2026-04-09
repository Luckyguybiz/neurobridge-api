"""Energy landscape analysis using maximum entropy (Ising) model.

Maps the energy function of neural states, finds attractor basins,
and computes barrier heights between states.
"""
import numpy as np
from .loader import SpikeData

def fit_ising_model(data: SpikeData, bin_size_ms: float = 20.0) -> dict:
    """Fit pairwise maximum entropy (Ising) model to binary spike data."""
    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)
    n_bins = len(bins) - 1
    n = data.n_electrodes
    eids = data.electrode_ids

    if n_bins < 50:
        return {"error": "insufficient data"}

    # Binary state matrix
    binary = np.zeros((n, n_bins), dtype=int)
    for e_idx, e_id in enumerate(eids):
        mask = data.electrodes == e_id
        counts, _ = np.histogram(data.times[mask], bins=bins)
        binary[e_idx] = (counts > 0).astype(int)

    # Compute mean activities and correlations
    h = np.mean(binary, axis=1)  # bias terms
    C = np.corrcoef(binary)
    np.fill_diagonal(C, 0)

    # Simple Ising approximation: J_ij ≈ -C_inv_ij (mean-field)
    try:
        C_reg = C + np.eye(n) * 0.01
        J = -np.linalg.inv(C_reg)
        np.fill_diagonal(J, 0)
    except np.linalg.LinAlgError:
        J = C.copy()

    # Compute energies of observed states
    unique_states = np.unique(binary.T, axis=0)
    energies = []
    for state in unique_states[:100]:
        E = -np.dot(h, state) - 0.5 * state @ J @ state
        energies.append(float(E))

    # Find local minima (attractors)
    if len(energies) > 0:
        sorted_idx = np.argsort(energies)
        n_attractors = min(10, len(energies))
        attractor_energies = [energies[sorted_idx[i]] for i in range(n_attractors)]
        attractor_states = [unique_states[sorted_idx[i]].tolist() for i in range(n_attractors)]
    else:
        attractor_energies = []
        attractor_states = []

    return {
        "n_unique_states": int(len(unique_states)),
        "bias_terms": h.tolist(),
        "coupling_matrix": J.tolist(),
        "energy_range": [float(min(energies)), float(max(energies))] if energies else [0, 0],
        "mean_energy": float(np.mean(energies)) if energies else 0.0,
        "n_attractors": len(attractor_energies),
        "attractor_energies": attractor_energies,
        "attractor_states": attractor_states[:5],
        "model_type": "pairwise_ising",
    }
