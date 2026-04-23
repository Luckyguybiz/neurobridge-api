"""Extended consciousness metrics for organoid assessment.

Beyond Phi (in emergence.py): PCI, recurrent processing,
global workspace capacity, and a composite consciousness score.
"""
import numpy as np
from .loader import SpikeData
from .emergence import compute_integrated_information
from .connectivity import compute_connectivity_graph

def compute_perturbational_complexity(data: SpikeData, bin_size_ms: float = 10.0) -> dict:
    """Compute Perturbational Complexity Index (PCI) approximation."""
    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)
    n_bins = len(bins) - 1

    if n_bins < 20:
        return {"pci": 0.0, "reason": "insufficient data"}

    binary = np.zeros((data.n_electrodes, n_bins), dtype=int)
    for e_idx, e_id in enumerate(data.electrode_ids):
        mask = data.electrodes == e_id
        counts, _ = np.histogram(data.times[mask], bins=bins)
        binary[e_idx] = (counts > 0).astype(int)

    # PCI = Lempel-Ziv complexity of the binary response matrix
    # Flatten binary matrix row by row
    flat = binary.flatten()

    # LZ complexity
    n = len(flat)
    if n == 0:
        return {"pci": 0.0}

    i, k, l = 0, 1, 1
    c = 1
    while k < n:
        if flat[k] != flat[k - l]:
            k += 1
            l += 1
        else:
            i += 1
            k += 1
            if k - i > l:
                c += 1
                i = k - 1
                l = 1
    c += 1

    # Normalize
    pci = float(c * np.log2(max(n, 2)) / max(n, 1))

    return {
        "pci": pci,
        "n_bins": n_bins,
        "n_electrodes": data.n_electrodes,
        "interpretation": "high" if pci > 0.5 else "moderate" if pci > 0.2 else "low",
    }

def compute_recurrent_processing(data: SpikeData) -> dict:
    """Measure recurrent (feedback) vs feedforward processing.

    Passes n_surrogates=30 because this is a sub-call inside the
    consciousness composite. Full significance testing isn't needed —
    we only care about the TE matrix values for recurrence, not p-values.
    30 surrogates × 1024 pairs = 30K calls, completes in ~25s on 300s data.
    """
    from .connectivity import compute_transfer_entropy
    te = compute_transfer_entropy(data, n_surrogates=30)

    matrix = np.array(getattr(te, 'te_matrix', None) if not isinstance(te, dict) else te.get("te_matrix", te.get("matrix", [[0]])))
    if matrix is None:
        matrix = np.array([[0]])
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"recurrence_index": 0.0}

    # Recurrence = symmetric TE (both directions)
    forward = np.triu(matrix, k=1)
    backward = np.tril(matrix, k=-1)

    total_flow = float(np.sum(matrix))
    recurrent_flow = float(np.sum(np.minimum(forward, backward.T)))

    return {
        "recurrence_index": recurrent_flow / max(total_flow, 1e-10),
        "forward_flow": float(np.sum(forward)),
        "backward_flow": float(np.sum(backward)),
        "total_flow": total_flow,
        "is_recurrent": recurrent_flow / max(total_flow, 1e-10) > 0.3,
    }

def compute_consciousness_score(data: SpikeData) -> dict:
    """Composite consciousness assessment score (0-1)."""
    phi_result = compute_integrated_information(data)
    pci_result = compute_perturbational_complexity(data)
    recurrent = compute_recurrent_processing(data)

    phi = float(phi_result.get("phi", phi_result.get("phi_value", 0)))
    pci = float(pci_result.get("pci", 0))
    rec = float(recurrent.get("recurrence_index", 0))

    # Normalize each to 0-1
    phi_norm = min(phi / 1.0, 1.0)
    pci_norm = min(pci / 1.0, 1.0)
    rec_norm = min(rec / 0.5, 1.0)

    composite = (phi_norm * 0.4 + pci_norm * 0.3 + rec_norm * 0.3)

    return {
        "consciousness_score": float(composite),
        "phi": phi,
        "phi_normalized": float(phi_norm),
        "pci": pci,
        "pci_normalized": float(pci_norm),
        "recurrence_index": rec,
        "recurrence_normalized": float(rec_norm),
        "interpretation": "high" if composite > 0.6 else "moderate" if composite > 0.3 else "minimal",
        "ethical_flag": composite > 0.5,
    }
