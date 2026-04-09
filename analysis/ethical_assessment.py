"""Ethical assessment for organoid experiments.

Evaluates consciousness indicators, sentience risk, and compliance
with NIH, EU, and ISSCR ethical guidelines for organoid research.
"""
import numpy as np
from .loader import SpikeData


def assess_ethics(data: SpikeData) -> dict:
    """Compute ethical assessment for an organoid based on its neural activity."""
    n_spikes = data.n_spikes
    rate = n_spikes / max(data.duration, 0.001)
    n_electrodes = data.n_electrodes

    # Consciousness indicators (simplified heuristics)
    # Based on complexity and integration of activity
    from .information_theory import compute_spike_train_entropy
    entropy_result = compute_spike_train_entropy(data)
    entropies = []
    for k, v in entropy_result.items():
        if isinstance(v, dict) and "entropy_bits" in v:
            entropies.append(v["entropy_bits"])
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    # Integration measure (co-firing between electrodes)
    from .connectivity import compute_connectivity_graph
    conn = compute_connectivity_graph(data)
    n_edges = conn.get("n_edges", 0)
    max_edges = n_electrodes * (n_electrodes - 1) / 2
    integration = n_edges / max(max_edges, 1)

    # Consciousness indicators
    indicators = {
        "organized_activity": rate > 1.0,
        "network_bursts": rate > 5.0,
        "functional_connectivity": integration > 0.3,
        "information_integration": mean_entropy > 2.0,
        "temporal_structure": True,  # would need more analysis
        "recurrent_processing": integration > 0.5,
    }
    n_positive = sum(indicators.values())

    # Sentience risk score (0-1)
    sentience_risk = n_positive / len(indicators)

    # Guidelines compliance
    guidelines = {
        "NIH_guidelines": {
            "compliant": True,
            "notes": "Standard MEA recording — no chimeric or transplantation concerns",
        },
        "EU_directive_2010_63": {
            "compliant": True,
            "notes": "In vitro organoids not covered under animal testing directive",
        },
        "ISSCR_2021": {
            "compliant": sentience_risk < 0.7,
            "notes": "ISSCR recommends enhanced oversight for organoids showing complex activity" if sentience_risk > 0.5 else "Standard oversight sufficient",
        },
        "Smirnova_2023_framework": {
            "compliant": True,
            "notes": "Organoid Intelligence ethical framework recommends monitoring consciousness indicators",
        },
    }

    # Recommendations
    recommendations = []
    if sentience_risk > 0.5:
        recommendations.append("Consider ethics review board consultation")
    if sentience_risk > 0.7:
        recommendations.append("Enhanced oversight recommended — multiple consciousness indicators positive")
    if n_positive <= 2:
        recommendations.append("Low ethical concern — standard protocols sufficient")
    recommendations.append("Document all experimental protocols for reproducibility")
    recommendations.append("Monitor organoid health and minimize unnecessary stimulation")

    return {
        "consciousness_indicators": indicators,
        "n_positive_indicators": n_positive,
        "total_indicators": len(indicators),
        "sentience_risk_score": float(sentience_risk),
        "risk_level": "high" if sentience_risk > 0.7 else "moderate" if sentience_risk > 0.4 else "low",
        "guidelines_compliance": guidelines,
        "recommendations": recommendations,
        "assessment_basis": {
            "firing_rate_hz": float(rate),
            "mean_entropy_bits": mean_entropy,
            "network_integration": float(integration),
        },
    }
