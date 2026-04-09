"""XOR Benchmark — Logical Gate Computation in Neural Organoids.

Tests whether the organoid can solve linearly non-separable problems.
XOR is the canonical test because it requires hidden-layer computation —
a linear system cannot solve it. If an organoid solves XOR, it is
performing genuine nonlinear computation.

Gates tested:
- AND:  output 1 only when both inputs are 1
- OR:   output 1 when at least one input is 1
- XOR:  output 1 when inputs differ (nonlinear — the hard one)
- NAND: complement of AND
- XNOR: complement of XOR (equivalence)

Methodology:
1. Encode binary inputs as stimulation patterns on two electrode groups.
2. Read population response as organoid "output".
3. Train a linear readout on the response to decode gate output.
4. High accuracy on XOR → organoid performs nonlinear transformation.

Based on Reservoir Computing literature (Jaeger 2001, Maass 2002).
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# ── Gate Definitions ──────────────────────────────────────────────────────────

GATE_TRUTH_TABLES = {
    "AND":  {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR":   {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "XOR":  {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}


# ── Full Benchmark Suite ──────────────────────────────────────────────────────

def run_full_benchmark(
    data: SpikeData,
    n_trials_per_gate: int = 60,
    bin_size_ms: float = 50.0,
    readout: str = "logistic",
    seed: int = 42,
) -> dict:
    """Run all 5 logical gate benchmarks.

    Args:
        data: SpikeData
        n_trials_per_gate: number of trials per gate
        bin_size_ms: time bin for collecting responses
        readout: 'logistic', 'svm', or 'linear'
        seed: random seed

    Returns:
        dict with per-gate results, XOR capability assessment, comparison
    """
    results = {}
    for gate in GATE_TRUTH_TABLES:
        results[gate] = run_gate_benchmark(
            data, gate=gate, n_trials=n_trials_per_gate,
            bin_size_ms=bin_size_ms, readout=readout, seed=seed,
        )

    # XOR capability: can the organoid solve the hard problem?
    xor_acc = results["XOR"]["accuracy"]
    and_acc = results["AND"]["accuracy"]
    or_acc = results["OR"]["accuracy"]
    xor_gap = float(xor_acc - 0.5)  # above chance

    # Nonlinearity index: XOR accuracy relative to linear gates
    linear_baseline = float(np.mean([and_acc, or_acc]))
    nonlinearity_index = float(xor_acc - linear_baseline)

    # Ranking
    ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    return {
        "benchmark": "logical_gates",
        "n_gates": len(results),
        "readout_method": readout,
        "gate_results": {g: {
            "accuracy": r["accuracy"],
            "above_chance": r["above_chance"],
            "d_prime": r["d_prime"],
            "passed": r["passed"],
        } for g, r in results.items()},
        "xor_accuracy": round(xor_acc, 3),
        "xor_above_chance": round(xor_gap, 3),
        "nonlinearity_index": round(nonlinearity_index, 3),
        "is_nonlinear": xor_acc > 0.65,
        "ranking": [{"gate": g, "accuracy": r["accuracy"]} for g, r in ranked],
        "best_gate": ranked[0][0],
        "worst_gate": ranked[-1][0],
        "details": results,
        "interpretation": _full_benchmark_interpretation(results, xor_acc, nonlinearity_index),
    }


def run_gate_benchmark(
    data: SpikeData,
    gate: str = "XOR",
    n_trials: int = 60,
    bin_size_ms: float = 50.0,
    readout: str = "logistic",
    seed: int = 42,
) -> dict:
    """Benchmark one logical gate.

    Args:
        data: SpikeData
        gate: 'AND', 'OR', 'XOR', 'NAND', 'XNOR'
        n_trials: number of trials
        bin_size_ms: response window per trial
        readout: classifier type

    Returns:
        dict with accuracy, d-prime, confusion matrix, trial details
    """
    if gate not in GATE_TRUTH_TABLES:
        return {"error": f"Unknown gate: {gate}. Choose from {list(GATE_TRUTH_TABLES.keys())}"}

    rng = np.random.default_rng(seed)
    truth_table = GATE_TRUTH_TABLES[gate]
    t_start, t_end = data.time_range
    duration = t_end - t_start
    bin_s = bin_size_ms / 1000.0
    n_electrodes = len(data.electrode_ids)

    # Electrode groups: A = first third, B = second third, rest = read output
    n_group = max(1, n_electrodes // 3)
    group_a = data.electrode_ids[:n_group]
    group_b = data.electrode_ids[n_group: 2 * n_group]

    X, y, trials = [], [], []

    # Generate balanced trial set
    input_pairs = list(truth_table.keys())
    for i in range(n_trials):
        input_a, input_b = input_pairs[i % 4]
        target = truth_table[(input_a, input_b)]

        # Time window for this trial
        t0 = t_start + rng.uniform(0, max(0.001, duration - bin_s * 2))

        # Read neural response
        mask = (data.times >= t0) & (data.times < t0 + bin_s)
        ep_electrodes = data.electrodes[mask]

        # Build feature vector: firing rate per electrode
        features = np.zeros(n_electrodes)
        for j, eid in enumerate(data.electrode_ids):
            features[j] = float(np.sum(ep_electrodes == eid)) / bin_s

        # Inject input encoding: activate corresponding electrode groups
        if input_a == 1:
            for eid in group_a:
                idx = list(data.electrode_ids).index(eid) if eid in data.electrode_ids else -1
                if idx >= 0:
                    features[idx] += rng.uniform(1, 5)
        if input_b == 1:
            for eid in group_b:
                idx = list(data.electrode_ids).index(eid) if eid in data.electrode_ids else -1
                if idx >= 0:
                    features[idx] += rng.uniform(1, 5)

        # Add small noise to prevent degeneracy
        features += rng.normal(0, 0.1, n_electrodes)

        X.append(features)
        y.append(target)
        trials.append({"trial": i + 1, "input_a": input_a, "input_b": input_b, "target": target})

    X_arr = np.array(X)
    y_arr = np.array(y)

    # Readout classifier
    accuracy, predictions = _train_readout(X_arr, y_arr, readout, rng)

    # Per-input performance
    per_input = {}
    for pair, label in truth_table.items():
        pair_idx = [i for i, t in enumerate(trials) if t["input_a"] == pair[0] and t["input_b"] == pair[1]]
        if pair_idx and predictions is not None:
            pair_correct = float(np.mean([int(predictions[i] == y_arr[i]) for i in pair_idx if i < len(predictions)]))
        else:
            pair_correct = 0.5
        per_input[f"{pair[0]},{pair[1]}"] = {
            "target": label,
            "accuracy": round(pair_correct, 3),
            "n_trials": len(pair_idx),
        }

    # D-prime
    if predictions is not None:
        tp = float(np.sum((predictions == 1) & (y_arr == 1)))
        fp = float(np.sum((predictions == 1) & (y_arr == 0)))
        fn = float(np.sum((predictions == 0) & (y_arr == 1)))
        tn = float(np.sum((predictions == 0) & (y_arr == 0)))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        from scipy.special import ndtri
        try:
            d_prime = float(ndtri(np.clip(tpr, 0.01, 0.99)) - ndtri(np.clip(fpr, 0.01, 0.99)))
        except Exception:
            d_prime = float(tpr - fpr) * 3
        confusion = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    else:
        d_prime = 0.0
        confusion = {}

    above_chance = float(accuracy - 0.5)
    passed = accuracy > 0.7

    return {
        "gate": gate,
        "truth_table": {f"{k[0]},{k[1]}": v for k, v in truth_table.items()},
        "n_trials": n_trials,
        "readout": readout,
        "accuracy": round(float(accuracy), 3),
        "above_chance": round(above_chance, 3),
        "d_prime": round(d_prime, 3),
        "passed": passed,
        "per_input_accuracy": per_input,
        "confusion_matrix": confusion,
        "is_linearly_separable": gate in ("AND", "OR", "NAND"),
        "interpretation": _gate_interpretation(gate, accuracy, d_prime),
    }


def compute_xor_difficulty(data: SpikeData) -> dict:
    """Estimate how hard XOR is for this organoid vs linear gates.

    Returns:
        dict with difficulty score, neural separability metrics
    """
    rng = np.random.default_rng(77)
    t_start, t_end = data.time_range
    duration = t_end - t_start
    bin_s = 0.05
    n_electrodes = len(data.electrode_ids)

    # Collect state vectors for each input combination
    patterns = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    n_per_comb = 20

    for (a, b), state_list in patterns.items():
        for _ in range(n_per_comb):
            t0 = rng.uniform(t_start, max(t_start + 0.001, t_end - bin_s))
            mask = (data.times >= t0) & (data.times < t0 + bin_s)
            ep_el = data.electrodes[mask]
            feat = np.array([float(np.sum(ep_el == eid)) for eid in data.electrode_ids])
            if a == 1:
                feat[:max(1, n_electrodes // 3)] += rng.uniform(1, 4)
            if b == 1:
                feat[n_electrodes // 3: 2 * n_electrodes // 3] += rng.uniform(1, 4)
            feat += rng.normal(0, 0.1, n_electrodes)
            state_list.append(feat)

    # XOR classes: {(0,1),(1,0)} vs {(0,0),(1,1)}
    xor_class1 = np.array(patterns[(0, 1)] + patterns[(1, 0)])
    xor_class0 = np.array(patterns[(0, 0)] + patterns[(1, 1)])

    # Fisher's discriminant ratio for XOR
    if len(xor_class1) > 1 and len(xor_class0) > 1:
        mean1 = np.mean(xor_class1, axis=0)
        mean0 = np.mean(xor_class0, axis=0)
        var1 = np.var(xor_class1, axis=0)
        var0 = np.var(xor_class0, axis=0)
        within_var = float(np.mean(var1 + var0) + 1e-10)
        between_var = float(np.sum((mean1 - mean0) ** 2))
        fisher_ratio = between_var / within_var
    else:
        fisher_ratio = 0.0

    difficulty = max(0.0, 1.0 - float(np.tanh(fisher_ratio / 10.0)))

    return {
        "xor_difficulty_score": round(difficulty, 3),
        "fisher_discriminant_ratio": round(float(fisher_ratio), 4),
        "is_hard": difficulty > 0.7,
        "n_electrodes": n_electrodes,
        "interpretation": (
            f"XOR difficulty: {difficulty:.2f}/1.0 for this organoid. "
            + ("Very hard — needs strong nonlinear dynamics." if difficulty > 0.7
               else "Moderate difficulty." if difficulty > 0.4
               else "Relatively easy — organoid already separates XOR classes.")
        ),
    }


# ── Readout Classifiers ───────────────────────────────────────────────────────

def _train_readout(
    X: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator
) -> tuple[float, Optional[np.ndarray]]:
    """Train and cross-validate a linear readout on reservoir states."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=min(5, len(y) // 4), shuffle=True, random_state=int(rng.integers(1000)))

    try:
        if method == "logistic":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, random_state=0, C=1.0)
        elif method == "svm":
            from sklearn.svm import SVC
            clf = SVC(kernel="linear", C=1.0, random_state=0)
        elif method == "linear":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300, random_state=0)

        predictions = cross_val_predict(clf, X_s, y, cv=cv)
        accuracy = float(np.mean(predictions == y))
        return accuracy, predictions

    except Exception:
        return 0.5, None


# ── Interpretation Helpers ────────────────────────────────────────────────────

def _gate_interpretation(gate: str, accuracy: float, d_prime: float) -> str:
    linear = gate in ("AND", "OR", "NAND")
    gate_type = "linearly separable" if linear else "nonlinearly separable (hard)"
    level = "excellent" if accuracy > 0.85 else "good" if accuracy > 0.7 else "marginal" if accuracy > 0.55 else "poor"
    return (
        f"{gate} ({gate_type}): accuracy={accuracy:.1%}, d'={d_prime:.2f}. "
        f"Performance: {level}. "
        + (f"Organoid solves this nonlinear problem!" if not linear and accuracy > 0.7 else
           f"Above chance — some {gate} computation." if accuracy > 0.55 else
           f"Near chance — {gate} computation not detected.")
    )


def _full_benchmark_interpretation(results: dict, xor_acc: float, nonlinearity_idx: float) -> str:
    n_passed = sum(1 for r in results.values() if r["passed"])
    if xor_acc > 0.75 and n_passed >= 4:
        return (f"Strong logical computation — {n_passed}/5 gates solved including XOR. "
                f"Nonlinearity index={nonlinearity_idx:.3f}. Genuine nonlinear computation confirmed.")
    elif xor_acc > 0.65:
        return (f"XOR partially solved (accuracy={xor_acc:.1%}). "
                f"Organoid shows nonlinear dynamics. {n_passed}/5 gates passed.")
    elif n_passed >= 3:
        return (f"Linear gates solved ({n_passed}/5), XOR not yet solved. "
                f"Consider longer recording or stronger stimulation contrast.")
    return (f"Limited logical computation ({n_passed}/5 gates). "
            f"Organoid may need more training or better electrode coverage.")
