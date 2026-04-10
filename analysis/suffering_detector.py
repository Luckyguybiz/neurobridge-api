"""Suffering/stress pattern detection for ethical monitoring.

Monitors for neural patterns associated with distress:
- Persistent high-frequency bursting (seizure-like)
- Response suppression (shutdown)
- Abnormal synchronization
- Dramatic activity changes

Provides automatic alerts and recommends intervention.
"""
import numpy as np
from .loader import SpikeData


def detect_suffering(data: SpikeData, window_sec: float = 5.0) -> dict:
    """Detect potential suffering/stress patterns in organoid activity."""
    bins = np.arange(0, data.duration, window_sec)
    n_windows = len(bins) - 1
    if n_windows < 2:
        return {"distress_detected": False, "flags": [], "overall_risk": 0.0}

    # Population rate per window
    pop_rates = np.zeros(n_windows)
    for w in range(n_windows):
        mask = (data.times >= bins[w]) & (data.times < bins[w + 1])
        pop_rates[w] = float(np.sum(mask)) / window_sec

    mean_rate = float(np.mean(pop_rates))
    flags = []
    risk_score = 0.0

    # Flag 1: Seizure-like activity (>5x mean sustained over 3+ windows)
    seizure_windows = np.sum(pop_rates > mean_rate * 5)
    if seizure_windows >= 3:
        flags.append({
            "type": "seizure_like",
            "severity": "high",
            "description": f"Sustained high-frequency bursting in {seizure_windows} windows (>5x mean rate)",
            "recommendation": "Consider reducing stimulation intensity",
        })
        risk_score += 0.4

    # Flag 2: Response suppression (activity drops to near zero)
    silent_windows = np.sum(pop_rates < mean_rate * 0.1)
    if silent_windows >= 2:
        flags.append({
            "type": "suppression",
            "severity": "medium",
            "description": f"Activity suppression in {silent_windows} windows (<10% of mean)",
            "recommendation": "Allow recovery period, check organoid viability",
        })
        risk_score += 0.25

    # Flag 3: Extreme variability (CV > 2)
    cv = float(np.std(pop_rates) / max(mean_rate, 0.01))
    if cv > 2.0:
        flags.append({
            "type": "instability",
            "severity": "medium",
            "description": f"Extreme rate variability (CV={cv:.2f})",
            "recommendation": "Network may be in pathological state",
        })
        risk_score += 0.2

    # Flag 4: Monotonic decline (dying organoid)
    if n_windows >= 4:
        half = n_windows // 2
        first_half_rate = float(np.mean(pop_rates[:half]))
        second_half_rate = float(np.mean(pop_rates[half:]))
        if first_half_rate > 0 and second_half_rate < first_half_rate * 0.3:
            flags.append({
                "type": "declining",
                "severity": "high",
                "description": f"Activity declined {((1 - second_half_rate/first_half_rate)*100):.0f}% during recording",
                "recommendation": "Organoid may be degrading — check media and environment",
            })
            risk_score += 0.3

    risk_score = min(1.0, risk_score)

    return {
        "distress_detected": len(flags) > 0,
        "flags": flags,
        "n_flags": len(flags),
        "overall_risk": float(risk_score),
        "risk_level": "critical" if risk_score > 0.6 else "high" if risk_score > 0.4 else "low" if risk_score < 0.15 else "medium",
        "should_halt": risk_score > 0.6,
        "mean_rate": mean_rate,
        "rate_cv": cv,
        "population_rate_timeline": pop_rates.tolist()[:100],
    }
