"""Habituation analysis — simplest form of learning.

Measures whether response to repeated patterns decreases over time.
Habituation = evidence of learning even without external stimulation.
"""
import numpy as np
from scipy.optimize import curve_fit
from .loader import SpikeData

def _exponential_decay(n, r0, tau, r_inf):
    return r0 * np.exp(-n / max(tau, 0.01)) + r_inf

def detect_repeated_patterns(data: SpikeData, window_ms: float = 100.0) -> dict:
    """Find repeated activity patterns and measure response decay."""
    bin_size = window_ms / 1000.0
    bins = np.arange(0, data.duration, bin_size)

    # Compute population rate per bin
    counts, _ = np.histogram(data.times, bins=bins)

    # Find burst-like events (above mean + 1 std)
    threshold = np.mean(counts) + np.std(counts)
    event_bins = np.where(counts > threshold)[0]

    if len(event_bins) < 3:
        return {
            "habituation_detected": False,
            "n_events": int(len(event_bins)),
            "reason": "too few events for analysis"
        }

    # Measure amplitude of each event
    event_amplitudes = [float(counts[i]) for i in event_bins]
    event_times = [float(bins[i]) for i in event_bins]

    # Try exponential decay fit
    n_events = np.arange(len(event_amplitudes), dtype=float)
    try:
        popt, pcov = curve_fit(
            _exponential_decay, n_events, event_amplitudes,
            p0=[event_amplitudes[0], len(event_amplitudes)/2, event_amplitudes[-1]],
            maxfev=5000
        )
        r0, tau, r_inf = popt
        fitted = _exponential_decay(n_events, *popt)
        residuals = np.array(event_amplitudes) - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((np.array(event_amplitudes) - np.mean(event_amplitudes))**2))
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
        habituation = r_squared > 0.3 and r0 > r_inf * 1.5
    except Exception:
        r0, tau, r_inf, r_squared = 0.0, 0.0, 0.0, 0.0
        habituation = False

    return {
        "habituation_detected": bool(habituation),
        "n_events": int(len(event_amplitudes)),
        "event_amplitudes": event_amplitudes[:100],
        "event_times": event_times[:100],
        "decay_fit": {
            "r0": float(r0),
            "tau": float(tau),
            "r_inf": float(r_inf),
            "r_squared": float(r_squared),
        },
        "amplitude_decrease_pct": float((1 - event_amplitudes[-1] / max(event_amplitudes[0], 1e-10)) * 100),
    }
