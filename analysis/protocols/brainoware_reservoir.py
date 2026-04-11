"""Brainoware Reservoir Computing -- use organoid as fixed nonlinear reservoir for classification.

Implements the reservoir computing paradigm from Cai et al. (2023). The organoid's
intrinsic dynamics serve as a fixed nonlinear transformation layer. Input patterns
are projected through the organoid's correlation structure, and a simple linear
readout (Ridge regression) is trained to classify the transformed representations.
"""

import numpy as np
from typing import Optional
from ..loader import SpikeData


def _build_reservoir_matrix(data: SpikeData, n_electrodes: int = 8) -> np.ndarray:
    """Build a reservoir transformation matrix from organoid correlation structure.

    Uses the pairwise correlation between electrode spike trains as the
    nonlinear mixing matrix that defines the reservoir's computational properties.
    """
    duration = max(data.duration, 0.001)
    n_bins = max(50, int(duration * 10))
    t_start, t_end = data.time_range if data.n_spikes > 0 else (0.0, 1.0)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)

    # Build binned spike count matrix (n_electrodes x n_bins)
    spike_matrix = np.zeros((n_electrodes, n_bins))
    for e in data.electrode_ids:
        idx = e % n_electrodes
        e_times = data.times[data.electrodes == e]
        if len(e_times) > 0:
            counts, _ = np.histogram(e_times, bins=bin_edges)
            spike_matrix[idx] += counts

    # Correlation matrix as reservoir weights
    if np.std(spike_matrix) < 1e-10:
        # Fallback: random orthogonal matrix if data is too sparse
        rng = np.random.default_rng(42)
        W = rng.normal(0, 0.5, (n_electrodes, n_electrodes))
        W = (W + W.T) / 2  # symmetrize
    else:
        corr = np.corrcoef(spike_matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        W = corr

    # Spectral normalization for echo state property
    eigenvalues = np.linalg.eigvalsh(W)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius > 0:
        W = W * 0.9 / spectral_radius  # target spectral radius 0.9

    return W


def _generate_synthetic_inputs(n_classes: int, n_samples: int, input_dim: int, rng: np.random.Generator) -> tuple:
    """Generate synthetic classification inputs with distinct patterns per class."""
    X = np.zeros((n_samples, input_dim))
    y = np.zeros(n_samples, dtype=int)

    samples_per_class = n_samples // n_classes

    for c in range(n_classes):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class
        if c == n_classes - 1:
            end_idx = n_samples

        n_c = end_idx - start_idx
        y[start_idx:end_idx] = c

        # Each class has a distinct center and pattern
        center = rng.normal(0, 1.0, input_dim)
        center[c % input_dim] += 2.0  # make classes separable
        X[start_idx:end_idx] = center + rng.normal(0, 0.5, (n_c, input_dim))

    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


def _reservoir_transform(X: np.ndarray, W: np.ndarray, n_steps: int = 5) -> np.ndarray:
    """Transform inputs through the reservoir (nonlinear expansion).

    Applies the reservoir matrix iteratively with tanh nonlinearity,
    simulating the organoid's nonlinear temporal dynamics.
    """
    n_samples, input_dim = X.shape
    res_dim = W.shape[0]

    # Project input to reservoir dimension
    rng = np.random.default_rng(7)
    W_in = rng.normal(0, 0.3, (input_dim, res_dim))

    states = np.zeros((n_samples, res_dim))

    for i in range(n_samples):
        state = np.tanh(X[i] @ W_in)
        for _ in range(n_steps):
            state = np.tanh(W @ state + X[i] @ W_in * 0.1)
        states[i] = state

    return states


def _ridge_classify(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    n_classes: int, alpha: float = 1.0) -> dict:
    """Train and evaluate Ridge regression classifier."""
    n_train = X_train.shape[0]

    # One-hot encode targets
    Y_train = np.zeros((n_train, n_classes))
    for i, c in enumerate(y_train):
        Y_train[i, c] = 1.0

    # Ridge regression: W = (X^T X + alpha I)^{-1} X^T Y
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    XtY = X_train.T @ Y_train
    W_readout = np.linalg.solve(XtX, XtY)

    # Predict
    y_pred = np.argmax(X_test @ W_readout, axis=1)
    accuracy = float(np.mean(y_pred == y_test))

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        confusion[true, pred] += 1

    # Per-class scores
    class_scores = {}
    for c in range(n_classes):
        tp = confusion[c, c]
        total_true = int(np.sum(confusion[c, :]))
        total_pred = int(np.sum(confusion[:, c]))
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_scores[c] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": confusion.tolist(),
        "class_scores": class_scores,
        "y_pred": y_pred.tolist(),
    }


def simulate_reservoir_classification(data: SpikeData, n_classes: int = 4, n_samples: int = 100) -> dict:
    """Simulate reservoir computing classification using organoid dynamics.

    1. Build reservoir matrix from organoid correlation structure.
    2. Generate synthetic input patterns (n_classes classes).
    3. Transform inputs through the organoid reservoir.
    4. Train Ridge readout on transformed representations.
    5. Compare with random baseline reservoir.

    Args:
        data: SpikeData for building the reservoir matrix.
        n_classes: Number of classification classes.
        n_samples: Total number of samples to generate.

    Returns:
        dict with accuracy, confusion matrix, comparison vs random, class scores.
    """
    rng = np.random.default_rng(42)
    n_electrodes = min(8, max(len(data.electrode_ids), 4))
    input_dim = n_electrodes

    # Build reservoir from organoid data
    W_organoid = _build_reservoir_matrix(data, n_electrodes)

    # Generate synthetic data
    X, y = _generate_synthetic_inputs(n_classes, n_samples, input_dim, rng)

    # Train/test split (70/30)
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Transform through organoid reservoir
    X_train_res = _reservoir_transform(X_train, W_organoid)
    X_test_res = _reservoir_transform(X_test, W_organoid)

    # Classify with Ridge readout
    organoid_result = _ridge_classify(X_train_res, y_train, X_test_res, y_test, n_classes)

    # Random reservoir baseline
    W_random = rng.normal(0, 0.3, (n_electrodes, n_electrodes))
    eigenvalues_r = np.linalg.eigvalsh(W_random)
    sr = float(np.max(np.abs(eigenvalues_r)))
    if sr > 0:
        W_random = W_random * 0.9 / sr

    X_train_rand = _reservoir_transform(X_train, W_random)
    X_test_rand = _reservoir_transform(X_test, W_random)
    random_result = _ridge_classify(X_train_rand, y_train, X_test_rand, y_test, n_classes)

    # Direct linear baseline (no reservoir)
    linear_result = _ridge_classify(X_train, y_train, X_test, y_test, n_classes)

    advantage = organoid_result["accuracy"] - random_result["accuracy"]
    chance_level = 1.0 / n_classes

    return {
        "accuracy": organoid_result["accuracy"],
        "confusion_matrix": organoid_result["confusion_matrix"],
        "class_scores": organoid_result["class_scores"],
        "comparison_vs_random": {
            "organoid_accuracy": organoid_result["accuracy"],
            "random_reservoir_accuracy": random_result["accuracy"],
            "linear_baseline_accuracy": linear_result["accuracy"],
            "chance_level": round(chance_level, 3),
            "organoid_advantage": round(advantage, 4),
        },
        "n_classes": n_classes,
        "n_samples": n_samples,
        "reservoir_size": n_electrodes,
        "reservoir_spectral_radius": round(float(np.max(np.abs(np.linalg.eigvalsh(W_organoid)))), 4),
        "interpretation": (
            f"Reservoir classification: {n_classes} classes, {n_samples} samples. "
            f"Organoid accuracy: {organoid_result['accuracy']:.1%} vs "
            f"random: {random_result['accuracy']:.1%} vs "
            f"linear: {linear_result['accuracy']:.1%} (chance: {chance_level:.1%}). "
            + (f"Organoid reservoir outperforms random by {advantage:+.1%} -- "
               "meaningful computational structure detected."
               if advantage > 0.05 else
               "Organoid reservoir comparable to random -- "
               "computational structure may be weak or data too sparse.")
        ),
    }
