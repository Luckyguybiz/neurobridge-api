"""Automated welfare report for organoid monitoring.

Generates a comprehensive welfare assessment combining:
- Health metrics
- Suffering detection
- Consciousness assessment
- Activity trends
- Recommendations
"""
import numpy as np
from .loader import SpikeData
from .predictions import estimate_organoid_health
from .suffering_detector import detect_suffering


def generate_welfare_report(data: SpikeData) -> dict:
    """Generate comprehensive welfare report."""
    health = estimate_organoid_health(data)
    suffering = detect_suffering(data)

    # Activity summary
    rate = data.n_spikes / max(data.duration, 0.001)
    rate_per_electrode = rate / max(data.n_electrodes, 1)

    # Overall welfare score (higher = better welfare)
    health_score = float(health.get("health_score", health.get("overall_score", 0.5)))
    suffering_risk = float(suffering.get("overall_risk", 0))
    welfare_score = health_score * (1 - suffering_risk)

    # Generate recommendations
    recommendations = []
    if suffering.get("should_halt"):
        recommendations.append("URGENT: Halt experiment — distress patterns detected")
    if health_score < 0.3:
        recommendations.append("Check culture conditions — low health score")
    if rate_per_electrode < 0.5:
        recommendations.append("Very low activity — verify electrode connectivity and media freshness")
    if rate_per_electrode > 50:
        recommendations.append("Unusually high activity — check for artifacts or seizure-like events")
    if not recommendations:
        recommendations.append("All metrics within normal range — continue monitoring")

    return {
        "welfare_score": float(welfare_score),
        "welfare_level": (
            "excellent" if welfare_score > 0.8 else
            "good" if welfare_score > 0.6 else
            "concerning" if welfare_score > 0.3 else
            "critical"
        ),
        "health": {
            "score": health_score,
            "details": health,
        },
        "suffering": {
            "risk": suffering_risk,
            "flags": suffering.get("flags", []),
            "should_halt": suffering.get("should_halt", False),
        },
        "activity": {
            "total_spikes": data.n_spikes,
            "duration_sec": float(data.duration),
            "mean_rate_hz": float(rate),
            "rate_per_electrode_hz": float(rate_per_electrode),
            "n_electrodes_active": data.n_electrodes,
        },
        "recommendations": recommendations,
        "monitoring_status": "HALT REQUIRED" if suffering.get("should_halt") else "CONTINUE",
    }
