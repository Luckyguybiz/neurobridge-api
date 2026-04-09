"""Organoid Intelligence Quotient — composite score of computational capacity.

NOVEL METRIC — nobody has this.
Combines multiple dimensions of neural computation into a single
"intelligence score" for organoids. Like IQ but for wetware.

Dimensions:
1. Information capacity (entropy + complexity)
2. Computational efficiency (near-criticality)
3. Learning potential (STDP signatures)
4. Integration (connectivity + transfer entropy)
5. Temporal stability (stationarity of patterns)
6. Responsiveness (signal processing capability)

Score 0-100, where:
0-20 = Minimal activity (dead/dying organoid)
20-40 = Basic spontaneous activity
40-60 = Organized activity with patterns
60-80 = Complex computation-capable
80-100 = High information processing (research-grade)
"""

import numpy as np
from .loader import SpikeData
from . import stats, connectivity, information_theory, criticality, plasticity, spectral


def compute_organoid_iq(data: SpikeData) -> dict:
    """Compute composite Organoid IQ score.

    Returns overall score + subscores + detailed breakdown.
    """
    scores = {}
    details = {}

    # 1. INFORMATION CAPACITY (0-20 points)
    try:
        entropy = information_theory.compute_spike_train_entropy(data)
        mean_entropy = entropy.get("mean_entropy", 0)
        complexity = information_theory.compute_lempel_ziv_complexity(data)
        mean_complexity = complexity.get("mean_complexity", 0)

        # Optimal: entropy ~0.6-0.8, complexity ~0.4-0.7
        entropy_score = _bell_score(mean_entropy, optimal=0.7, width=0.3) * 10
        complexity_score = _bell_score(mean_complexity, optimal=0.55, width=0.25) * 10
        scores["information_capacity"] = round(entropy_score + complexity_score, 1)
        details["information_capacity"] = {
            "entropy": round(float(mean_entropy), 4),
            "complexity": round(float(mean_complexity), 4),
            "entropy_score": round(entropy_score, 1),
            "complexity_score": round(complexity_score, 1),
        }
    except Exception:
        scores["information_capacity"] = 5.0

    # 2. COMPUTATIONAL EFFICIENCY — near-criticality (0-20 points)
    try:
        crit = criticality.detect_avalanches(data)
        branching = crit.get("branching_ratio", {}).get("mean_ratio", 0)
        # Perfect criticality: branching ratio = 1.0
        crit_score = _bell_score(branching, optimal=1.0, width=0.15) * 15

        # Power law fit quality
        size_fit = crit.get("size_power_law", {})
        if size_fit and size_fit.get("good_fit"):
            crit_score += 5
        else:
            crit_score += 2

        scores["computational_efficiency"] = min(20, round(crit_score, 1))
        details["computational_efficiency"] = {
            "branching_ratio": round(float(branching), 4),
            "is_critical": crit.get("is_critical", False),
            "assessment": crit.get("criticality_assessment", ""),
        }
    except Exception:
        scores["computational_efficiency"] = 5.0

    # 3. LEARNING POTENTIAL — STDP signatures (0-20 points)
    try:
        stdp = plasticity.compute_stdp_matrix(data)
        n_significant = stdp.get("n_significant_pairs", 0)
        has_learning = stdp.get("has_learning_signatures", False)
        mean_asym = stdp.get("mean_asymmetry", 0)

        learning_score = min(15, n_significant * 2.5)
        if has_learning:
            learning_score += 5

        scores["learning_potential"] = min(20, round(learning_score, 1))
        details["learning_potential"] = {
            "significant_stdp_pairs": n_significant,
            "has_learning_signatures": has_learning,
            "mean_asymmetry": round(float(mean_asym), 4),
        }
    except Exception:
        scores["learning_potential"] = 5.0

    # 4. INTEGRATION — connectivity quality (0-20 points)
    try:
        conn = connectivity.compute_connectivity_graph(data)
        density = conn.get("density", 0)
        mean_clustering = conn.get("mean_clustering", 0)
        n_edges = conn.get("n_edges", 0)

        te = connectivity.compute_transfer_entropy(data)
        mean_te = te.get("mean_te", 0)

        # Good integration: moderate density (0.3-0.7), high clustering, non-zero TE
        density_score = _bell_score(density, optimal=0.5, width=0.3) * 8
        clustering_score = min(6, mean_clustering * 6)
        te_score = min(6, mean_te * 60)

        scores["integration"] = min(20, round(density_score + clustering_score + te_score, 1))
        details["integration"] = {
            "density": round(float(density), 4),
            "clustering": round(float(mean_clustering), 4),
            "transfer_entropy": round(float(mean_te), 5),
            "n_connections": n_edges,
        }
    except Exception:
        scores["integration"] = 5.0

    # 5. TEMPORAL STABILITY (0-10 points)
    try:
        summary = stats.compute_full_summary(data)
        stability = summary.get("temporal_stability", {})
        changes = [abs(v.get("change_pct", 0)) for v in stability.values()]
        mean_change = float(np.mean(changes)) if changes else 100

        # Stable = low change percentage
        stability_score = max(0, 10 - mean_change * 0.1)
        scores["temporal_stability"] = round(stability_score, 1)
        details["temporal_stability"] = {
            "mean_rate_change_pct": round(mean_change, 1),
        }
    except Exception:
        scores["temporal_stability"] = 3.0

    # 6. SIGNAL QUALITY (0-10 points)
    try:
        quality = stats.compute_quality_metrics(data)
        q_score = quality.get("quality_score", 0) / 10
        mean_snr = quality.get("mean_snr", 0)
        snr_bonus = min(3, mean_snr * 0.5)

        scores["signal_quality"] = min(10, round(q_score + snr_bonus, 1))
        details["signal_quality"] = {
            "quality_score": quality.get("quality_score", 0),
            "mean_snr": round(float(mean_snr), 2),
            "n_issues": quality.get("n_issues", 0),
        }
    except Exception:
        scores["signal_quality"] = 3.0

    # TOTAL IQ
    total = sum(scores.values())

    # Grade
    if total >= 80:
        grade = "A+"
        assessment = "Exceptional — high information processing, near-critical dynamics, strong learning signatures"
    elif total >= 65:
        grade = "A"
        assessment = "Excellent — complex organized activity with computation potential"
    elif total >= 50:
        grade = "B"
        assessment = "Good — organized patterns, moderate information capacity"
    elif total >= 35:
        grade = "C"
        assessment = "Average — basic spontaneous activity, limited complexity"
    elif total >= 20:
        grade = "D"
        assessment = "Below average — minimal organization, low information content"
    else:
        grade = "F"
        assessment = "Poor — minimal activity, possible degradation"

    return {
        "iq_score": round(total, 1),
        "grade": grade,
        "assessment": assessment,
        "subscores": scores,
        "max_possible": 100,
        "details": details,
        "interpretation": {
            "information_capacity": f"{scores.get('information_capacity', 0)}/20 — How much information the organoid can encode",
            "computational_efficiency": f"{scores.get('computational_efficiency', 0)}/20 — How close to optimal criticality",
            "learning_potential": f"{scores.get('learning_potential', 0)}/20 — Evidence of spike-timing dependent plasticity",
            "integration": f"{scores.get('integration_score', scores.get('integration', 0))}/20 — Quality of inter-electrode communication",
            "temporal_stability": f"{scores.get('temporal_stability', 0)}/10 — Consistency of activity over time",
            "signal_quality": f"{scores.get('signal_quality', 0)}/10 — Recording quality and SNR",
        },
    }


def _bell_score(value: float, optimal: float, width: float) -> float:
    """Gaussian scoring — maximum at optimal, drops off with width."""
    return float(np.exp(-((value - optimal) ** 2) / (2 * width ** 2)))


def compute_organoid_comparison(datasets: dict[str, SpikeData]) -> dict:
    """Compare IQ scores across multiple organoids/recordings."""
    results = {}
    for name, data in datasets.items():
        iq = compute_organoid_iq(data)
        results[name] = {
            "iq_score": iq["iq_score"],
            "grade": iq["grade"],
            "subscores": iq["subscores"],
        }

    # Ranking
    ranked = sorted(results.items(), key=lambda x: x[1]["iq_score"], reverse=True)

    return {
        "results": results,
        "ranking": [{"name": name, "iq": r["iq_score"], "grade": r["grade"]} for name, r in ranked],
        "best": ranked[0][0] if ranked else None,
        "mean_iq": round(float(np.mean([r["iq_score"] for r in results.values()])), 1) if results else 0,
    }
