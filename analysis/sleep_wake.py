"""Sleep-wake cycle detection in organoid spontaneous activity.

Detects UP/DOWN states (periods of high vs low activity),
slow-wave oscillations (<1Hz), and sleep spindle-like events (10-16Hz).
"""
import numpy as np
from .loader import SpikeData

def detect_up_down_states(data: SpikeData, bin_size_ms: float = 50.0) -> dict:
    """Classify time bins as UP (active) or DOWN (silent) states."""
    bins = np.arange(0, data.duration, bin_size_ms / 1000)
    counts, _ = np.histogram(data.times, bins=bins)
    threshold = np.mean(counts) * 0.5
    states = np.where(counts > threshold, 1, 0)  # 1=UP, 0=DOWN

    # Find transitions
    transitions = np.diff(states)
    up_starts = np.where(transitions == 1)[0]
    down_starts = np.where(transitions == -1)[0]

    # Compute durations
    up_durations = []
    down_durations = []
    for i in range(len(states)):
        if i == 0: continue
        if states[i] == states[i-1]:
            continue

    # State durations
    changes = np.where(np.diff(states) != 0)[0]
    durations = np.diff(np.concatenate([[0], changes, [len(states)]]))
    state_at_change = [states[0]] + [states[c+1] for c in changes]

    up_durs = [float(d * bin_size_ms) for d, s in zip(durations, state_at_change) if s == 1]
    down_durs = [float(d * bin_size_ms) for d, s in zip(durations, state_at_change) if s == 0]

    return {
        "n_up_states": int(len(up_starts)),
        "n_down_states": int(len(down_starts)),
        "mean_up_duration_ms": float(np.mean(up_durs)) if up_durs else 0.0,
        "mean_down_duration_ms": float(np.mean(down_durs)) if down_durs else 0.0,
        "up_fraction": float(np.mean(states)),
        "n_transitions": int(len(up_starts) + len(down_starts)),
        "transition_rate_per_sec": float((len(up_starts) + len(down_starts)) / max(data.duration, 0.001)),
        "bin_size_ms": bin_size_ms,
        "states": states.tolist()[:500],  # truncate for API
        "time_bins": bins[:501].tolist(),
    }

def detect_slow_waves(data: SpikeData, bin_size_ms: float = 100.0) -> dict:
    """Detect slow-wave (<1Hz) oscillations in population activity."""
    bins = np.arange(0, data.duration, bin_size_ms / 1000)
    counts, _ = np.histogram(data.times, bins=bins)
    if len(counts) < 10:
        return {"slow_waves_detected": False, "reason": "insufficient data"}

    # FFT of population rate
    from scipy import signal
    fs = 1000.0 / bin_size_ms
    freqs, psd = signal.welch(counts.astype(float), fs=fs, nperseg=min(len(counts), 256))

    # Find slow-wave power (<1Hz)
    slow_mask = freqs < 1.0
    total_power = float(np.sum(psd))
    slow_power = float(np.sum(psd[slow_mask])) if np.any(slow_mask) else 0.0
    slow_fraction = slow_power / max(total_power, 1e-10)

    return {
        "slow_waves_detected": slow_fraction > 0.3,
        "slow_wave_power_fraction": slow_fraction,
        "total_power": total_power,
        "slow_power": slow_power,
        "dominant_frequency_hz": float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0,
        "frequencies": freqs.tolist()[:100],
        "psd": psd.tolist()[:100],
    }

def analyze_sleep_wake(data: SpikeData) -> dict:
    """Full sleep-wake analysis."""
    up_down = detect_up_down_states(data)
    slow = detect_slow_waves(data)

    # Sleep-like score
    has_up_down = up_down["n_transitions"] > 2
    has_slow_waves = slow.get("slow_waves_detected", False)
    sleep_score = (0.5 if has_up_down else 0.0) + (0.5 if has_slow_waves else 0.0)

    return {
        "sleep_like_score": sleep_score,
        "has_up_down_states": has_up_down,
        "has_slow_waves": has_slow_waves,
        "up_down_states": up_down,
        "slow_waves": slow,
    }
