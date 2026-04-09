"""Brainoware-inspired Japanese Vowels classification via reservoir computing.

Scientific basis:
    Cai et al. (2023) "Brainoware" demonstrated that brain organoids can
    perform voice recognition using reservoir computing. The organoid acts
    as a nonlinear dynamical reservoir: input signals are projected into
    the organoid's high-dimensional state space, and a simple linear
    readout is trained on the resulting neural activity.

    The Japanese Vowels dataset (UCI ML Repository) contains 640 samples
    of 9 speakers uttering Japanese vowels, represented as 12-dimensional
    LPC cepstral coefficients across variable-length time frames.

    This module simulates the Brainoware approach:
    1. Generate 240 synthetic vowel feature vectors (8 classes: /a/, /i/,
       /u/, /e/, /o/, /ka/, /ki/, /ku/) with realistic spectral properties.
    2. Use a random reservoir (simulating the organoid's fixed dynamics)
       to project inputs into a higher-dimensional space.
    3. Train a linear readout layer (ridge regression) for classification.
    4. Evaluate accuracy and per-class performance.

    The reservoir's random projection matrix simulates the organoid's
    inherent nonlinear transformation of input signals. The key insight
    is that biological neural networks provide rich, nonlinear dynamics
    that enable computation without explicit training of the network itself.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# Vowel class definitions with formant-inspired spectral centroids (Hz)
_VOWEL_CLASSES = {
    0: {"label": "/a/", "f1": 730, "f2": 1090, "f3": 2440},
    1: {"label": "/i/", "f1": 270, "f2": 2290, "f3": 3010},
    2: {"label": "/u/", "f1": 300, "f2": 870,  "f3": 2240},
    3: {"label": "/e/", "f1": 530, "f2": 1840, "f3": 2480},
    4: {"label": "/o/", "f1": 570, "f2": 840,  "f3": 2410},
    5: {"label": "/ka/", "f1": 680, "f2": 1150, "f3": 2500},
    6: {"label": "/ki/", "f1": 310, "f2": 2200, "f3": 2960},
    7: {"label": "/ku/", "f1": 340, "f2": 920,  "f3": 2300},
}


def generate_synthetic_vowels(
    n_samples: int = 240,
    n_features: int = 12,
    n_classes: int = 8,
    noise_std: float = 0.15,
    seed: Optional[int] = None,
) -> dict:
    """Generate synthetic vowel feature vectors mimicking LPC cepstral coefficients.

    Each vowel class has a distinct centroid in feature space derived from
    formant frequencies. Features simulate 12-dimensional LPC cepstral
    coefficients with class-specific means and inter-speaker variability.

    Args:
        n_samples: Total number of samples across all classes.
        n_features: Dimensionality of feature vectors (LPC order).
        n_classes: Number of vowel classes (max 8).
        noise_std: Standard deviation of Gaussian noise for variability.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'features' (n_samples x n_features), 'labels' (n_samples,),
        'class_names', and generation metadata.
    """
    rng = np.random.default_rng(seed)
    n_classes = min(n_classes, len(_VOWEL_CLASSES))
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    features = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples, dtype=np.int32)
    class_names = []

    idx = 0
    for cls_id in range(n_classes):
        cls_info = _VOWEL_CLASSES[cls_id]
        class_names.append(cls_info["label"])

        n_cls = samples_per_class + (1 if cls_id < remainder else 0)

        # Build centroid from formant frequencies normalized to [0, 1]
        f1_norm = cls_info["f1"] / 1000.0
        f2_norm = cls_info["f2"] / 3000.0
        f3_norm = cls_info["f3"] / 4000.0

        centroid = np.zeros(n_features)
        centroid[0] = f1_norm
        centroid[1] = f2_norm
        centroid[2] = f3_norm
        # Higher cepstral coefficients decay with formant-derived modulation
        for k in range(3, n_features):
            centroid[k] = (
                0.3 * f1_norm * np.cos(np.pi * k * f1_norm)
                + 0.3 * f2_norm * np.cos(np.pi * k * f2_norm)
                + 0.2 * f3_norm * np.cos(np.pi * k * f3_norm)
            ) * np.exp(-0.15 * k)

        # Generate samples with speaker variability
        for j in range(n_cls):
            speaker_shift = rng.normal(0, 0.05, n_features)
            sample = centroid + speaker_shift + rng.normal(0, noise_std, n_features)
            features[idx] = sample
            labels[idx] = cls_id
            idx += 1

    # Shuffle
    perm = rng.permutation(n_samples)
    features = features[perm]
    labels = labels[perm]

    return {
        "features": features,
        "labels": labels,
        "class_names": class_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "noise_std": noise_std,
    }


def build_reservoir(
    input_dim: int,
    reservoir_size: int = 256,
    spectral_radius: float = 0.9,
    sparsity: float = 0.9,
    seed: Optional[int] = None,
) -> dict:
    """Build a random reservoir (echo state network style).

    The reservoir simulates the organoid's fixed nonlinear dynamics.
    A random recurrent weight matrix W with controlled spectral radius
    provides the echo state property, and a random input weight matrix
    W_in maps inputs to reservoir space.

    Args:
        input_dim: Dimensionality of input features.
        reservoir_size: Number of reservoir neurons (hidden units).
        spectral_radius: Spectral radius of recurrent matrix (controls
            memory/stability tradeoff; <1.0 for echo state property).
        sparsity: Fraction of zero weights in recurrent matrix.
        seed: Random seed for reproducibility.

    Returns:
        Dict with weight matrices and reservoir parameters.
    """
    rng = np.random.default_rng(seed)

    # Input weight matrix
    W_in = rng.uniform(-1, 1, (reservoir_size, input_dim))

    # Sparse recurrent weight matrix
    W = rng.normal(0, 1, (reservoir_size, reservoir_size))
    mask = rng.random((reservoir_size, reservoir_size)) < sparsity
    W[mask] = 0.0

    # Scale to desired spectral radius
    eigenvalues = np.linalg.eigvals(W)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if max_eigenvalue > 0:
        W = W * (spectral_radius / max_eigenvalue)

    # Bias vector
    bias = rng.uniform(-0.1, 0.1, reservoir_size)

    return {
        "W_in": W_in,
        "W": W,
        "bias": bias,
        "reservoir_size": reservoir_size,
        "spectral_radius": spectral_radius,
        "sparsity": sparsity,
        "input_dim": input_dim,
    }


def reservoir_transform(
    features: np.ndarray,
    reservoir: dict,
    leak_rate: float = 0.3,
) -> np.ndarray:
    """Transform input features through the reservoir.

    Each input vector is projected into the reservoir's state space.
    The reservoir state evolves with leaky integration, simulating
    how neural tissue integrates incoming signals over time.

    Args:
        features: Input feature matrix (n_samples x input_dim).
        reservoir: Reservoir dict from build_reservoir().
        leak_rate: Leaky integrator rate (0=no update, 1=full update).

    Returns:
        Reservoir states (n_samples x reservoir_size).
    """
    W_in = reservoir["W_in"]
    W = reservoir["W"]
    bias = reservoir["bias"]
    reservoir_size = reservoir["reservoir_size"]
    n_samples = features.shape[0]

    states = np.zeros((n_samples, reservoir_size))
    state = np.zeros(reservoir_size)

    for t in range(n_samples):
        pre_activation = W_in @ features[t] + W @ state + bias
        new_state = np.tanh(pre_activation)
        state = (1 - leak_rate) * state + leak_rate * new_state
        states[t] = state

    return states


def train_linear_readout(
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    alpha: float = 1.0,
    train_ratio: float = 0.7,
) -> dict:
    """Train a linear readout layer via ridge regression.

    This is the only trained component in reservoir computing.
    The readout maps reservoir states to class predictions.
    One-vs-all encoding is used for multi-class classification.

    Args:
        states: Reservoir state matrix (n_samples x reservoir_size).
        labels: Class labels (n_samples,).
        n_classes: Number of output classes.
        alpha: Ridge regularization parameter.
        train_ratio: Fraction of data for training.

    Returns:
        Dict with readout weights, train/test accuracy, and per-class metrics.
    """
    n_samples = states.shape[0]
    n_train = int(n_samples * train_ratio)

    X_train = states[:n_train]
    X_test = states[n_train:]
    y_train = labels[:n_train]
    y_test = labels[n_train:]

    # One-hot encode targets
    Y_train = np.zeros((n_train, n_classes))
    for i, lbl in enumerate(y_train):
        Y_train[i, lbl] = 1.0

    # Ridge regression: W_out = (X^T X + alpha I)^{-1} X^T Y
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    XtY = X_train.T @ Y_train
    W_out = np.linalg.solve(XtX, XtY)

    # Predictions
    train_pred = np.argmax(X_train @ W_out, axis=1)
    test_pred = np.argmax(X_test @ W_out, axis=1)

    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    # Per-class metrics
    per_class = {}
    for cls in range(n_classes):
        cls_mask_test = y_test == cls
        if np.sum(cls_mask_test) > 0:
            cls_acc = float(np.mean(test_pred[cls_mask_test] == cls))
        else:
            cls_acc = 0.0
        tp = float(np.sum((test_pred == cls) & (y_test == cls)))
        fp = float(np.sum((test_pred == cls) & (y_test != cls)))
        fn = float(np.sum((test_pred != cls) & (y_test == cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "accuracy": round(cls_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_test_samples": int(np.sum(cls_mask_test)),
        }

    return {
        "W_out": W_out,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class": per_class,
        "n_train": n_train,
        "n_test": n_samples - n_train,
        "alpha": alpha,
    }


def run_vowel_classification(
    data: SpikeData,
    n_samples: int = 240,
    reservoir_size: int = 256,
    spectral_radius: float = 0.9,
    seed: Optional[int] = 42,
) -> dict:
    """Run full Brainoware-style vowel classification pipeline.

    End-to-end pipeline:
    1. Generate synthetic vowel data.
    2. Augment reservoir with organoid spike statistics (firing rates,
       ISI distributions) to condition the reservoir dynamics on real
       biological data.
    3. Build and run reservoir computing.
    4. Train linear readout and evaluate.

    The organoid's spike data influences reservoir parameters: higher
    firing rates increase the effective spectral radius, and ISI
    variability modulates the leak rate, connecting the biological
    substrate's properties to the computational model.

    Args:
        data: SpikeData from the organoid recording.
        n_samples: Number of synthetic vowel samples.
        reservoir_size: Number of reservoir units.
        spectral_radius: Base spectral radius (adjusted by organoid stats).
        seed: Random seed.

    Returns:
        Dict with classification results, reservoir info, and organoid influence.
    """
    # Generate vowels
    vowels = generate_synthetic_vowels(n_samples=n_samples, seed=seed)
    features = vowels["features"]
    labels = vowels["labels"]
    n_classes = vowels["n_classes"]
    class_names = vowels["class_names"]

    # Extract organoid statistics to condition the reservoir
    firing_rates = []
    isis = []
    for eid in data.electrode_ids:
        e_times = data.times[data.electrodes == eid]
        if len(e_times) > 1:
            rate = len(e_times) / max(data.duration, 1e-6)
            firing_rates.append(rate)
            isi = np.diff(e_times)
            isis.extend(isi.tolist())

    mean_firing_rate = float(np.mean(firing_rates)) if firing_rates else 1.0
    isi_cv = float(np.std(isis) / np.mean(isis)) if isis and np.mean(isis) > 0 else 1.0

    # Adjust reservoir parameters based on organoid properties
    # Higher firing rate -> slightly larger spectral radius (more active dynamics)
    bio_spectral_radius = min(0.99, spectral_radius * (1 + 0.01 * np.log1p(mean_firing_rate)))
    # Higher ISI variability -> lower leak rate (more memory retention)
    bio_leak_rate = max(0.05, min(0.8, 0.3 / (1 + 0.1 * isi_cv)))

    # Build reservoir
    reservoir = build_reservoir(
        input_dim=vowels["n_features"],
        reservoir_size=reservoir_size,
        spectral_radius=bio_spectral_radius,
        seed=seed,
    )

    # Transform through reservoir
    states = reservoir_transform(features, reservoir, leak_rate=bio_leak_rate)

    # Train readout
    readout = train_linear_readout(states, labels, n_classes)

    # Confusion matrix
    n_train = readout["n_train"]
    X_test = states[n_train:]
    y_test = labels[n_train:]
    test_pred = np.argmax(X_test @ readout["W_out"], axis=1)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_lbl, pred_lbl in zip(y_test, test_pred):
        confusion[true_lbl, pred_lbl] += 1

    # Brainoware comparison context
    brainoware_reported_accuracy = 0.78  # From Cai et al. 2023

    return {
        "test_accuracy": readout["test_accuracy"],
        "train_accuracy": readout["train_accuracy"],
        "per_class_metrics": {
            class_names[k]: v for k, v in readout["per_class"].items()
        },
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
        "n_samples": n_samples,
        "n_train": readout["n_train"],
        "n_test": readout["n_test"],
        "reservoir": {
            "size": reservoir_size,
            "spectral_radius_base": spectral_radius,
            "spectral_radius_bio": round(bio_spectral_radius, 4),
            "leak_rate": round(bio_leak_rate, 4),
        },
        "organoid_influence": {
            "mean_firing_rate_hz": round(mean_firing_rate, 2),
            "isi_cv": round(isi_cv, 4),
            "n_active_electrodes": len(firing_rates),
        },
        "brainoware_comparison": {
            "brainoware_accuracy": brainoware_reported_accuracy,
            "our_accuracy": readout["test_accuracy"],
            "delta": round(readout["test_accuracy"] - brainoware_reported_accuracy, 4),
        },
        "interpretation": (
            f"Reservoir computing vowel classification: {readout['test_accuracy']:.1%} accuracy "
            f"({n_classes} classes, {n_samples} samples). "
            f"Organoid-conditioned reservoir (spectral_radius={bio_spectral_radius:.3f}, "
            f"leak_rate={bio_leak_rate:.3f}). "
            f"Brainoware reported {brainoware_reported_accuracy:.0%} on similar task."
        ),
    }
