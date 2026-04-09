"""Directed information flow mapping.

Maps how information flows through the organoid network using
Granger causality-like measures and hub detection.
"""
import numpy as np
from .loader import SpikeData

def compute_granger_causality(data: SpikeData, bin_size_ms: float = 10.0, max_lag: int = 5) -> dict:
    """Compute pairwise Granger causality between electrodes."""
    from sklearn.linear_model import Ridge

    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)
    n_bins = len(bins) - 1

    if n_bins < max_lag * 3:
        return {"pairs": [], "reason": "insufficient data"}

    # Compute rates
    rates = np.zeros((data.n_electrodes, n_bins))
    eids = data.electrode_ids
    for e_idx, e_id in enumerate(eids):
        mask = data.electrodes == e_id
        counts, _ = np.histogram(data.times[mask], bins=bins)
        rates[e_idx] = counts

    # Granger causality: does past of X help predict Y beyond past of Y alone?
    gc_matrix = np.zeros((data.n_electrodes, data.n_electrodes))

    for i in range(data.n_electrodes):
        for j in range(data.n_electrodes):
            if i == j:
                continue

            # Build lagged features
            Y = rates[j, max_lag:]
            X_self = np.column_stack([rates[j, max_lag-k-1:-k-1 if -k-1 != 0 else n_bins] for k in range(max_lag)])
            X_both = np.column_stack([X_self] + [rates[i, max_lag-k-1:-k-1 if -k-1 != 0 else n_bins] for k in range(max_lag)])

            if len(Y) < 10:
                continue

            # Fit models
            model_self = Ridge(alpha=1.0).fit(X_self[:len(Y)], Y)
            model_both = Ridge(alpha=1.0).fit(X_both[:len(Y)], Y)

            err_self = np.var(Y - model_self.predict(X_self[:len(Y)]))
            err_both = np.var(Y - model_both.predict(X_both[:len(Y)]))

            gc = float(np.log(max(err_self, 1e-10) / max(err_both, 1e-10)))
            gc_matrix[i, j] = max(gc, 0.0)

    # Find hubs
    out_strength = gc_matrix.sum(axis=1)
    in_strength = gc_matrix.sum(axis=0)
    hub_idx = int(np.argmax(out_strength))

    # Top pairs
    pairs = []
    for i in range(data.n_electrodes):
        for j in range(data.n_electrodes):
            if i != j and gc_matrix[i, j] > 0.01:
                pairs.append({
                    "source": int(eids[i]),
                    "target": int(eids[j]),
                    "gc_value": float(gc_matrix[i, j]),
                })
    pairs.sort(key=lambda x: x["gc_value"], reverse=True)

    return {
        "gc_matrix": gc_matrix.tolist(),
        "electrode_ids": [int(e) for e in eids],
        "hub_electrode": int(eids[hub_idx]),
        "hub_out_strength": float(out_strength[hub_idx]),
        "top_pairs": pairs[:20],
        "mean_gc": float(np.mean(gc_matrix[gc_matrix > 0])) if np.any(gc_matrix > 0) else 0.0,
        "flow_asymmetry": float(np.mean(np.abs(gc_matrix - gc_matrix.T))),
    }
