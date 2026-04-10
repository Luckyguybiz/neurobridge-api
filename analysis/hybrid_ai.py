"""Hybrid bio-digital AI architecture.

Scientific basis:
    Organoids can serve as biological feature extractors while digital
    neural networks handle classification. This hybrid approach combines:
    - Biological temporal dynamics (reservoir computing)
    - Digital precision (trained readout layer)

    Benchmark: compare hybrid (organoid+digital) vs pure digital (ESN)
    vs pure biological (organoid-only readout) on classification tasks.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from .loader import SpikeData


def _bin_spikes(data: SpikeData, bin_size_ms: float = 20.0) -> np.ndarray:
    """Convert spikes to binned rate matrix (n_electrodes x n_bins)."""
    bin_size = bin_size_ms / 1000.0
    bins = np.arange(0, data.duration + bin_size, bin_size)
    n_bins = len(bins) - 1
    rates = np.zeros((data.n_electrodes, n_bins))
    for idx, eid in enumerate(data.electrode_ids):
        mask = data.electrodes == eid
        counts, _ = np.histogram(data.times[mask], bins=bins)
        rates[idx] = counts
    return rates


def _generate_synthetic_task(n_samples: int = 100, n_classes: int = 4) -> tuple:
    """Generate a synthetic classification task."""
    X = np.random.randn(n_samples, 8)  # 8-dimensional input
    y = np.random.randint(0, n_classes, n_samples)
    # Add class-dependent structure
    for c in range(n_classes):
        mask = y == c
        X[mask, c % 8] += 2.0  # Make class c have higher values in feature c
    return X, y


def benchmark_hybrid(data: SpikeData, n_samples: int = 80, n_classes: int = 4) -> dict:
    """Benchmark hybrid bio-digital architecture against baselines."""
    rates = _bin_spikes(data)
    n_electrodes, n_bins = rates.shape

    if n_bins < 20:
        return {"error": "Recording too short for benchmark"}

    X_task, y_task = _generate_synthetic_task(n_samples, n_classes)

    # Split train/test
    split = int(n_samples * 0.7)
    X_train, X_test = X_task[:split], X_task[split:]
    y_train, y_test = y_task[:split], y_task[split:]

    # === Model 1: Pure Digital (Ridge on raw features) ===
    digital_model = Ridge(alpha=1.0)
    digital_model.fit(X_train, y_train)
    digital_preds = np.clip(np.round(digital_model.predict(X_test)), 0, n_classes - 1).astype(int)
    digital_acc = float(accuracy_score(y_test, digital_preds))

    # === Model 2: Organoid Reservoir (random projection through organoid dynamics) ===
    # Use organoid's firing rate correlations as a nonlinear transformation
    corr_matrix = np.corrcoef(rates)
    np.fill_diagonal(corr_matrix, 0)
    # Random projection through organoid "filter"
    W_reservoir = corr_matrix[:min(8, n_electrodes), :min(8, n_electrodes)]
    if W_reservoir.shape[0] < 8:
        W_reservoir = np.pad(W_reservoir, ((0, 8 - W_reservoir.shape[0]), (0, 8 - W_reservoir.shape[1])))

    X_reservoir_train = np.tanh(X_train @ W_reservoir)
    X_reservoir_test = np.tanh(X_test @ W_reservoir)

    reservoir_model = Ridge(alpha=1.0)
    reservoir_model.fit(X_reservoir_train, y_train)
    reservoir_preds = np.clip(np.round(reservoir_model.predict(X_reservoir_test)), 0, n_classes - 1).astype(int)
    reservoir_acc = float(accuracy_score(y_test, reservoir_preds))

    # === Model 3: Hybrid (organoid features + digital features) ===
    X_hybrid_train = np.hstack([X_train, X_reservoir_train])
    X_hybrid_test = np.hstack([X_test, X_reservoir_test])

    hybrid_model = Ridge(alpha=1.0)
    hybrid_model.fit(X_hybrid_train, y_train)
    hybrid_preds = np.clip(np.round(hybrid_model.predict(X_hybrid_test)), 0, n_classes - 1).astype(int)
    hybrid_acc = float(accuracy_score(y_test, hybrid_preds))

    # === Model 4: Random baseline ===
    random_acc = float(1.0 / n_classes)

    # Winner
    accs = {"digital": digital_acc, "reservoir": reservoir_acc, "hybrid": hybrid_acc, "random": random_acc}
    winner = max(accs, key=accs.get)

    return {
        "accuracies": accs,
        "winner": winner,
        "hybrid_advantage": hybrid_acc - digital_acc,
        "reservoir_advantage": reservoir_acc - random_acc,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "n_train": split,
        "n_test": n_samples - split,
        "biological_contribution": reservoir_acc - random_acc,
        "verdict": (
            "Hybrid outperforms digital — biological features add value"
            if hybrid_acc > digital_acc + 0.05
            else "Digital sufficient — organoid doesn't improve classification"
            if digital_acc >= hybrid_acc
            else "Comparable performance — marginal biological contribution"
        ),
    }
