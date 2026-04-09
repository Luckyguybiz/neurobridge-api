"""Spectral analysis module — frequency domain analysis of neural activity.

- Power spectral density of firing rates
- Coherence between electrode pairs
- Oscillation detection (theta, alpha, beta, gamma bands)
- Spectral entropy
"""

import numpy as np
from scipy import signal as sig
from typing import Optional
from .loader import SpikeData


# Neural oscillation frequency bands
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 100),
}


def compute_power_spectrum(
    data: SpikeData,
    bin_size_ms: float = 1.0,
    max_freq_hz: float = 100.0,
) -> dict:
    """Compute power spectral density of firing rate time series.

    Converts spike trains to continuous rate signals, then FFT.
    """
    bin_size_sec = bin_size_ms / 1000.0
    fs = 1.0 / bin_size_sec  # sampling frequency of binned signal
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    results = {}
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        rate = counts / bin_size_sec

        # Welch's method for PSD
        nperseg = min(len(rate), int(fs * 2))  # 2-second windows
        if nperseg < 8:
            continue
        freqs, psd = sig.welch(rate, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Limit to max_freq
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        psd = psd[mask]

        # Band power
        band_power = {}
        total_power = float(np.sum(psd))
        for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            bp = float(np.sum(psd[band_mask]))
            band_power[band_name] = {
                "absolute_power": round(bp, 4),
                "relative_power": round(bp / total_power, 4) if total_power > 0 else 0,
            }

        # Peak frequency
        peak_idx = int(np.argmax(psd[1:])) + 1  # skip DC
        peak_freq = float(freqs[peak_idx]) if len(freqs) > 1 else 0

        # Spectral entropy
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0
        max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
        normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0

        results[int(e)] = {
            "frequencies": freqs.tolist(),
            "psd": psd.tolist(),
            "peak_frequency_hz": round(peak_freq, 2),
            "total_power": round(total_power, 4),
            "band_power": band_power,
            "spectral_entropy": round(spectral_entropy, 4),
            "normalized_spectral_entropy": round(float(normalized_entropy), 4),
        }

    # Dominant band across population
    dominant_bands = {}
    for e, r in results.items():
        max_band = max(r["band_power"].items(), key=lambda x: x[1]["relative_power"])
        dominant_bands[e] = max_band[0]

    return {
        "per_electrode": results,
        "dominant_bands": dominant_bands,
        "bin_size_ms": bin_size_ms,
        "frequency_bands": FREQUENCY_BANDS,
    }


def compute_coherence(
    data: SpikeData,
    bin_size_ms: float = 1.0,
    max_freq_hz: float = 100.0,
) -> dict:
    """Compute spectral coherence between all electrode pairs.

    Coherence measures frequency-specific correlation (0-1 per frequency).
    """
    bin_size_sec = bin_size_ms / 1000.0
    fs = 1.0 / bin_size_sec
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)

    # Create rate signals
    rates = {}
    for e in data.electrode_ids:
        counts, _ = np.histogram(data.times[data.electrodes == e], bins=bins)
        rates[e] = counts / bin_size_sec

    electrode_ids = data.electrode_ids
    n = len(electrode_ids)
    coherence_results = {}

    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = electrode_ids[i], electrode_ids[j]
            r1, r2 = rates[e1], rates[e2]

            nperseg = min(len(r1), int(fs * 2))
            if nperseg < 8:
                continue

            freqs, coh = sig.coherence(r1, r2, fs=fs, nperseg=nperseg)
            mask = freqs <= max_freq_hz
            freqs = freqs[mask]
            coh = coh[mask]

            # Mean coherence per band
            band_coherence = {}
            for band, (f_low, f_high) in FREQUENCY_BANDS.items():
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                if np.any(band_mask):
                    band_coherence[band] = round(float(np.mean(coh[band_mask])), 4)

            peak_idx = int(np.argmax(coh[1:])) + 1 if len(coh) > 1 else 0

            coherence_results[f"{e1}-{e2}"] = {
                "mean_coherence": round(float(np.mean(coh)), 4),
                "peak_coherence": round(float(np.max(coh)), 4),
                "peak_frequency_hz": round(float(freqs[peak_idx]), 2) if peak_idx < len(freqs) else 0,
                "band_coherence": band_coherence,
            }

    return {
        "pairs": coherence_results,
        "n_pairs": len(coherence_results),
        "mean_coherence": round(float(np.mean([r["mean_coherence"] for r in coherence_results.values()])), 4) if coherence_results else 0,
    }
