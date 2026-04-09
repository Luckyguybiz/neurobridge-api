"""Prediction module — forecast organoid behavior.

Uses recorded history to predict:
- Future firing rates (trend extrapolation)
- Burst probability in next N seconds
- Organoid health/viability trajectory
- Optimal stimulation timing windows
"""

import numpy as np
from .loader import SpikeData


def predict_firing_rates(
    data: SpikeData,
    forecast_sec: float = 300.0,
    bin_size_sec: float = 60.0,
) -> dict:
    """Predict future firing rates using linear regression + confidence intervals."""
    from scipy import stats as scipy_stats

    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    predictions = {}
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        rates = counts / bin_size_sec

        if len(rates) < 3:
            continue

        # Linear regression
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(bin_centers, rates)

        # Forecast
        future_bins = np.arange(t_end, t_end + forecast_sec, bin_size_sec)
        forecast = slope * future_bins + intercept
        forecast = np.maximum(0, forecast)  # Can't have negative rates

        # Confidence interval (95%)
        n = len(rates)
        se = std_err * np.sqrt(1 + 1/n + (future_bins - np.mean(bin_centers))**2 / np.sum((bin_centers - np.mean(bin_centers))**2))
        ci_upper = forecast + 1.96 * se
        ci_lower = np.maximum(0, forecast - 1.96 * se)

        predictions[int(e)] = {
            "current_rate_hz": round(float(rates[-1]), 3),
            "trend_slope": round(float(slope), 5),
            "trend_direction": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
            "r_squared": round(float(r_val**2), 4),
            "forecast_times": future_bins.tolist(),
            "forecast_rates": forecast.tolist(),
            "ci_upper": ci_upper.tolist(),
            "ci_lower": ci_lower.tolist(),
        }

    return {
        "predictions": predictions,
        "forecast_duration_sec": forecast_sec,
        "bin_size_sec": bin_size_sec,
    }


def predict_burst_probability(data: SpikeData, window_sec: float = 10.0) -> dict:
    """Estimate probability of burst in next window based on recent activity."""
    from .bursts import detect_bursts

    burst_result = detect_bursts(data)
    bursts = burst_result.get("bursts", [])

    if len(bursts) < 2:
        return {"burst_probability": 0.1, "confidence": "low", "n_historical_bursts": len(bursts)}

    # Inter-burst intervals
    ibis = []
    for i in range(1, len(bursts)):
        ibis.append(bursts[i]["start"] - bursts[i-1]["end"])

    mean_ibi = float(np.mean(ibis))
    time_since_last = data.time_range[1] - bursts[-1]["end"]

    # Exponential distribution estimate
    rate_param = 1.0 / mean_ibi if mean_ibi > 0 else 0
    prob = 1.0 - np.exp(-rate_param * window_sec) if rate_param > 0 else 0

    # Adjust based on time since last burst
    if time_since_last > mean_ibi * 2:
        prob = min(0.95, prob * 1.5)  # Overdue

    return {
        "burst_probability": round(float(prob), 4),
        "window_sec": window_sec,
        "mean_inter_burst_interval_sec": round(mean_ibi, 2),
        "time_since_last_burst_sec": round(float(time_since_last), 2),
        "n_historical_bursts": len(bursts),
        "confidence": "high" if len(bursts) > 10 else "medium" if len(bursts) > 3 else "low",
    }


def estimate_organoid_health(data: SpikeData) -> dict:
    """Estimate organoid health/viability based on activity patterns.

    Healthy indicators:
    - Consistent firing rates (low CV)
    - Active on most electrodes
    - Regular burst activity
    - Good SNR
    - No prolonged silence

    Degradation indicators:
    - Decreasing firing rates over time
    - Silent electrodes appearing
    - Loss of burst activity
    - Increasing noise / decreasing SNR
    """
    from . import stats as stats_mod

    summary = stats_mod.compute_full_summary(data)
    quality = stats_mod.compute_quality_metrics(data)
    temporal = stats_mod.compute_temporal_dynamics(data, bin_size_sec=max(1.0, data.duration / 20))

    health_score = 100.0
    issues = []

    # Check firing rate trend
    trend = temporal.get("trend", {})
    if trend.get("significant") and trend.get("direction") == "decreasing":
        health_score -= 20
        issues.append("Decreasing firing rate trend (possible degradation)")

    # Check electrode silence
    missing = quality.get("missing_electrodes", [])
    if missing:
        health_score -= len(missing) * 5
        issues.append(f"{len(missing)} silent electrodes: {missing}")

    # Check firing rate consistency
    cv = temporal.get("cv_population_rate", 0)
    if cv > 1.0:
        health_score -= 15
        issues.append(f"Highly variable firing rate (CV={cv:.2f})")

    # Check SNR
    mean_snr = quality.get("mean_snr", 0)
    if mean_snr < 2.0:
        health_score -= 10
        issues.append(f"Low signal-to-noise ratio ({mean_snr:.1f})")

    # Check Fano factor (should be ~1 for Poisson, >1 for bursty, <1 for regular)
    mean_fano = temporal.get("mean_fano_factor", 1.0)
    if mean_fano < 0.3:
        health_score -= 10
        issues.append("Abnormally regular activity (Fano < 0.3)")
    elif mean_fano > 5:
        health_score -= 10
        issues.append(f"Extremely bursty activity (Fano = {mean_fano:.1f})")

    health_score = max(0, min(100, health_score))

    status = (
        "Excellent" if health_score >= 80
        else "Good" if health_score >= 60
        else "Fair" if health_score >= 40
        else "Poor" if health_score >= 20
        else "Critical"
    )

    return {
        "health_score": round(health_score, 1),
        "status": status,
        "issues": issues,
        "n_issues": len(issues),
        "metrics": {
            "firing_rate_trend": trend.get("direction", "unknown"),
            "cv_population_rate": round(float(cv), 3),
            "mean_snr": round(float(mean_snr), 2),
            "mean_fano_factor": round(float(mean_fano), 3),
            "n_active_electrodes": quality.get("n_active_electrodes", 0),
            "n_silent_electrodes": len(missing),
        },
        "recommendation": (
            "Organoid is healthy and producing high-quality data"
            if health_score >= 80
            else "Monitor closely — some signs of instability"
            if health_score >= 60
            else "Consider adjusting conditions (temperature, medium, CO2)"
            if health_score >= 40
            else "Significant degradation — may need organoid replacement"
        ),
    }
