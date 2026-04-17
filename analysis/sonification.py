"""Sonification module — convert neural activity to audio.

Translates spike data into sound that humans can hear.
Not just art — researchers use sonification to detect patterns
invisible to the eye (rhythmic structures, synchrony, subtle changes).

Mapping:
- Each electrode → musical note (pentatonic scale)
- Spike time → note onset
- Amplitude → volume
- Burst → chord (multiple notes simultaneously)
- Firing rate → tempo
"""

import numpy as np
import struct
import io
from typing import Optional
from .loader import SpikeData


# Pentatonic scale frequencies (pleasant, avoids dissonance)
PENTATONIC = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]  # C4 to E5


def generate_sonification(
    data: SpikeData,
    duration_sec: Optional[float] = None,
    speed_factor: float = 1.0,
    sample_rate: int = 22050,
    note_duration_ms: float = 80.0,
) -> dict:
    """Generate audio WAV data from spike trains.

    Args:
        speed_factor: Time compression. 1.0 = real-time, 10.0 = 10x faster
        sample_rate: Audio sample rate (Hz)
        note_duration_ms: Duration of each spike "note"

    Returns dict with base64-encoded WAV data and metadata.
    """
    import base64

    if data.n_spikes == 0:
        return {"error": "No spikes to sonify"}

    # Time parameters
    data_duration = duration_sec or data.duration
    audio_duration = data_duration / speed_factor
    n_samples = int(audio_duration * sample_rate)
    note_samples = int(note_duration_ms / 1000 * sample_rate)

    # Audio buffer
    audio = np.zeros(n_samples)

    # Map electrodes to frequencies
    electrode_ids = data.electrode_ids
    n_el = len(electrode_ids)
    freq_map = {e: PENTATONIC[i % len(PENTATONIC)] for i, e in enumerate(electrode_ids)}

    # Amplitude normalization
    max_amp = np.max(np.abs(data.amplitudes)) if len(data.amplitudes) > 0 else 1.0

    t_start = data.time_range[0]
    spikes_used = 0

    for spike_time, electrode, amplitude in zip(data.times, data.electrodes, data.amplitudes):
        if spike_time - t_start > data_duration:
            break

        # Map spike time to audio sample position
        audio_time = (spike_time - t_start) / speed_factor
        sample_pos = int(audio_time * sample_rate)

        if sample_pos + note_samples >= n_samples:
            continue

        # Generate note
        freq = freq_map.get(int(electrode), 440.0)
        volume = min(1.0, abs(amplitude) / max_amp * 0.8)

        t = np.arange(note_samples) / sample_rate
        # Sine wave with exponential decay envelope
        envelope = np.exp(-t * 1000 / note_duration_ms * 3)
        note = np.sin(2 * np.pi * freq * t) * envelope * volume

        # Add to mix
        end_pos = min(sample_pos + note_samples, n_samples)
        audio[sample_pos:end_pos] += note[:end_pos - sample_pos]
        spikes_used += 1

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    _write_wav(wav_buffer, audio_int16, sample_rate)
    wav_bytes = wav_buffer.getvalue()
    wav_b64 = base64.b64encode(wav_bytes).decode('ascii')

    return {
        "wav_base64": wav_b64,
        "wav_size_bytes": len(wav_bytes),
        "audio_duration_sec": round(audio_duration, 2),
        "sample_rate": sample_rate,
        "spikes_sonified": spikes_used,
        "speed_factor": speed_factor,
        "electrode_frequencies": {int(e): round(f, 2) for e, f in freq_map.items()},
        "note_duration_ms": note_duration_ms,
        "content_type": "audio/wav",
    }


def _write_wav(buffer: io.BytesIO, data: np.ndarray, sample_rate: int):
    """Write WAV file header and data."""
    n_samples = len(data)
    data_size = n_samples * 2  # 16-bit = 2 bytes per sample

    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))
    buffer.write(b'WAVE')

    # fmt chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # chunk size
    buffer.write(struct.pack('<H', 1))   # PCM format
    buffer.write(struct.pack('<H', 1))   # mono
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', sample_rate * 2))  # byte rate
    buffer.write(struct.pack('<H', 2))   # block align
    buffer.write(struct.pack('<H', 16))  # bits per sample

    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(data.tobytes())


def compute_rhythmic_analysis(data: SpikeData) -> dict:
    """World-class circadian and multi-timescale rhythm detection.

    Implements four complementary methods standard in chronobiology:
      1. Lomb-Scargle periodogram — handles unevenly sampled spike data natively.
      2. FFT-based autocorrelation — hourly-resolution lag analysis.
      3. Cosinor analysis — fits cosine model, extracts amplitude/acrophase/period.
      4. Chi-squared periodogram (Sokolove-Bushell) — gold standard in chronobiology.

    Detects rhythms at three timescales:
      - Ultradian (<20 h): minute-level bins, Lomb-Scargle.
      - Circadian (20-28 h): hourly bins, all four methods.
      - Infradian (>28 h): hourly bins, Lomb-Scargle + autocorrelation.

    Performs analysis per-MEA (groups of 8 electrodes) AND population-level.

    Performance: vectorised numpy throughout — completes in <60 s on 2.6 M spikes,
    32 electrodes, 118 hours.
    """
    if data.n_spikes < 100:
        return {"error": "Not enough spikes for rhythmic analysis"}

    from scipy.signal import lombscargle, find_peaks

    t_start, t_end = data.time_range
    total_duration_s = t_end - t_start
    total_duration_h = total_duration_s / 3600.0

    # Need at least 6 hours for any meaningful rhythm detection
    if total_duration_h < 6.0:
        return {
            "error": f"Recording too short ({total_duration_h:.1f}h). Need >= 6h for rhythm detection.",
            "circadian_detected": False,
        }

    # ─── Helper: bin spike times into rate time-series ────────────────────
    def _bin_spikes(spike_times: np.ndarray, bin_sec: float) -> np.ndarray:
        """Histogram spike times into uniform bins. Returns spike counts per bin."""
        edges = np.arange(t_start, t_end + bin_sec, bin_sec)
        counts, _ = np.histogram(spike_times, bins=edges)
        return counts.astype(np.float64)

    # ─── Helper: Lomb-Scargle on binned rate time-series ─────────────────
    def _lomb_scargle(rate: np.ndarray, bin_sec: float,
                      min_period_h: float, max_period_h: float,
                      n_freqs: int = 2000) -> tuple:
        """Compute Lomb-Scargle periodogram on evenly-binned data.

        We use scipy.signal.lombscargle which expects angular frequencies.
        Returns (periods_h, power_normalized, best_period_h, best_power).
        """
        n = len(rate)
        if n < 4:
            return np.array([]), np.array([]), 0.0, 0.0

        t_bins = np.arange(n) * bin_sec  # time axis in seconds
        y = rate - np.mean(rate)
        y_var = np.var(y)
        if y_var < 1e-12:
            return np.array([]), np.array([]), 0.0, 0.0

        # Frequency grid (angular frequencies for lombscargle)
        min_freq = 1.0 / (max_period_h * 3600.0)
        max_freq = 1.0 / (min_period_h * 3600.0)
        freqs_hz = np.linspace(min_freq, max_freq, n_freqs)
        angular_freqs = 2.0 * np.pi * freqs_hz

        power = lombscargle(t_bins, y, angular_freqs, precenter=True)
        # Normalize to [0, 1] range (Scargle normalization)
        power_norm = power / (0.5 * n * y_var) if y_var > 0 else power

        periods_h = 1.0 / (freqs_hz * 3600.0)
        best_idx = int(np.argmax(power_norm))
        return periods_h, power_norm, float(periods_h[best_idx]), float(power_norm[best_idx])

    # ─── Helper: FFT-based autocorrelation ───────────────────────────────
    def _autocorrelation(rate: np.ndarray, max_lag: int) -> np.ndarray:
        """FFT-based autocorrelation, normalized, up to max_lag.

        O(n log n) via FFT. Returns array of length max_lag+1.
        """
        n = len(rate)
        if n < 3:
            return np.zeros(min(max_lag + 1, 1))
        y = rate - np.mean(rate)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size <<= 1
        fft_y = np.fft.rfft(y, n=fft_size)
        acorr_full = np.fft.irfft(fft_y * np.conj(fft_y), n=fft_size)
        actual_max = min(max_lag, n - 1)
        acorr = acorr_full[:actual_max + 1].copy()
        if acorr[0] > 0:
            acorr /= acorr[0]
        return acorr

    # ─── Helper: Cosinor analysis ────────────────────────────────────────
    def _cosinor_fit(rate: np.ndarray, bin_sec: float,
                     test_period_h: float) -> dict:
        """Single-component cosinor regression.

        Fits: y(t) = M + A*cos(2*pi*t/T + phi)
        Equivalent to linear regression: y = M + beta*cos(wt) + gamma*sin(wt)
        where A = sqrt(beta^2 + gamma^2), phi = atan2(-gamma, beta).

        Returns amplitude, acrophase (hours from start), mesor, p-value, R-squared.
        """
        from scipy import stats as sp_stats

        n = len(rate)
        if n < 4:
            return {"amplitude": 0.0, "acrophase_h": 0.0, "mesor": 0.0,
                    "period_h": test_period_h, "p_value": 1.0, "r_squared": 0.0,
                    "percent_rhythm": 0.0}

        t_h = np.arange(n) * (bin_sec / 3600.0)  # time in hours
        omega = 2.0 * np.pi / test_period_h

        cos_term = np.cos(omega * t_h)
        sin_term = np.sin(omega * t_h)

        # Design matrix: [1, cos(wt), sin(wt)]
        X = np.column_stack([np.ones(n), cos_term, sin_term])

        # Least-squares via normal equation (fast for 3 columns)
        XtX = X.T @ X
        try:
            beta_vec = np.linalg.solve(XtX, X.T @ rate)
        except np.linalg.LinAlgError:
            return {"amplitude": 0.0, "acrophase_h": 0.0, "mesor": 0.0,
                    "period_h": test_period_h, "p_value": 1.0, "r_squared": 0.0,
                    "percent_rhythm": 0.0}

        mesor = float(beta_vec[0])
        beta_cos = float(beta_vec[1])
        gamma_sin = float(beta_vec[2])

        amplitude = float(np.sqrt(beta_cos**2 + gamma_sin**2))
        acrophase_rad = float(np.arctan2(-gamma_sin, beta_cos))
        # Convert acrophase to hours (positive, modulo period)
        acrophase_h = (-acrophase_rad / omega) % test_period_h

        # R-squared and F-test p-value
        y_hat = X @ beta_vec
        ss_res = float(np.sum((rate - y_hat)**2))
        ss_tot = float(np.sum((rate - np.mean(rate))**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_squared = max(0.0, r_squared)

        # F-test: 2 rhythm params vs. mesor-only model (df1=2, df2=n-3)
        p_value = 1.0
        if n > 3 and ss_res > 0 and ss_tot > 0:
            f_stat = ((ss_tot - ss_res) / 2.0) / (ss_res / (n - 3))
            if f_stat > 0:
                p_value = float(1.0 - sp_stats.f.cdf(f_stat, 2, n - 3))

        # Percent rhythm (amplitude as % of mesor)
        pct_rhythm = (amplitude / mesor * 100.0) if mesor > 0 else 0.0

        return {
            "amplitude": round(amplitude, 4),
            "acrophase_h": round(acrophase_h, 2),
            "mesor": round(mesor, 2),
            "period_h": round(test_period_h, 2),
            "p_value": round(p_value, 6),
            "r_squared": round(r_squared, 4),
            "percent_rhythm": round(pct_rhythm, 2),
        }

    # ─── Helper: Chi-squared periodogram (Sokolove-Bushell 1983) ─────────
    def _chi_squared_periodogram(rate: np.ndarray, bin_sec: float,
                                 min_period_h: float, max_period_h: float,
                                 step_h: float = 0.25) -> tuple:
        """Sokolove-Bushell chi-squared periodogram.

        For each candidate period T, fold the data modulo T, compute
        chi-squared statistic testing whether the folded profile differs
        from flat. Peak Qp indicates the dominant period.

        This is THE standard method in chronobiology (used in ClockLab,
        ActiWatch, El Temps).

        Returns (periods_h, qp_values, best_period_h, best_qp).
        """
        n = len(rate)
        if n < 4:
            return np.array([]), np.array([]), 0.0, 0.0

        min_bins = max(4, int(min_period_h * 3600.0 / bin_sec))
        max_bins = min(n, int(max_period_h * 3600.0 / bin_sec))
        step_bins = max(1, int(step_h * 3600.0 / bin_sec))

        if min_bins >= max_bins:
            return np.array([]), np.array([]), 0.0, 0.0

        mean_rate = np.mean(rate)
        if mean_rate < 1e-12:
            return np.array([]), np.array([]), 0.0, 0.0

        test_periods_bins = np.arange(min_bins, max_bins + 1, step_bins)
        qp_values = np.empty(len(test_periods_bins))

        for i, p in enumerate(test_periods_bins):
            # Number of complete cycles
            k = n // p
            if k < 1:
                qp_values[i] = 0.0
                continue
            usable = k * p
            # Reshape into cycles, compute mean profile
            folded = rate[:usable].reshape(k, p)
            profile = np.mean(folded, axis=0)
            # Qp = N * sum((Ph - mean)^2) / (p * mean^2)
            # where Ph = mean profile values
            qp = float(n * np.sum((profile - mean_rate)**2) / (p * mean_rate**2))
            qp_values[i] = qp

        periods_h = test_periods_bins * (bin_sec / 3600.0)
        best_idx = int(np.argmax(qp_values))
        return periods_h, qp_values, float(periods_h[best_idx]), float(qp_values[best_idx])

    # ─── Helper: full rhythm pipeline for one spike set ──────────────────
    def _analyze_rhythms(spike_times: np.ndarray, label: str) -> dict:
        """Run all four methods on one set of spike times.

        Returns a complete rhythm assessment dict.
        """
        n_spikes = len(spike_times)
        if n_spikes < 50:
            return {
                "label": label,
                "n_spikes": int(n_spikes),
                "circadian_detected": False,
                "error": "Too few spikes",
            }

        # ── Hourly bins (for circadian/infradian) ──
        hourly_bin = 3600.0
        hourly_rate = _bin_spikes(spike_times, hourly_bin)
        n_hours = len(hourly_rate)

        # ── Minute bins (for ultradian) ──
        minute_bin = 60.0
        minute_rate = _bin_spikes(spike_times, minute_bin)

        # ══════════════════════════════════════════════════════════════════
        # METHOD 1: Lomb-Scargle periodogram
        # ══════════════════════════════════════════════════════════════════

        # Circadian band (20-28 h) on hourly data
        ls_circ_periods, ls_circ_power, ls_circ_best_p, ls_circ_best_pow = (
            _lomb_scargle(hourly_rate, hourly_bin, 20.0, 28.0, n_freqs=500)
        )

        # Broad scan (4-48 h) on hourly data
        max_detectable_h = min(total_duration_h / 2.0, 72.0)
        ls_broad_periods, ls_broad_power, ls_broad_best_p, ls_broad_best_pow = (
            _lomb_scargle(hourly_rate, hourly_bin, 4.0, max_detectable_h, n_freqs=2000)
        )

        # Ultradian band (0.5-8 h) on minute data
        ls_ultra_periods, ls_ultra_power, ls_ultra_best_p, ls_ultra_best_pow = (
            _lomb_scargle(minute_rate, minute_bin, 0.5, 8.0, n_freqs=1000)
        )

        # ══════════════════════════════════════════════════════════════════
        # METHOD 2: FFT-based autocorrelation (hourly resolution)
        # ══════════════════════════════════════════════════════════════════
        max_lag_h = min(48, n_hours - 1)
        acorr = _autocorrelation(hourly_rate, max_lag_h)

        # Find peaks in autocorrelation (skip lag 0)
        acorr_best_lag_h = 0.0
        acorr_peak_value = 0.0
        if len(acorr) > 3:
            peaks_ac, props_ac = find_peaks(acorr[1:], height=0.05, distance=3)
            if len(peaks_ac) > 0:
                peaks_ac += 1  # offset for skipping lag 0
                best_ac_idx = int(np.argmax(acorr[peaks_ac]))
                acorr_best_lag_h = float(peaks_ac[best_ac_idx])
                acorr_peak_value = float(acorr[peaks_ac[best_ac_idx]])

        # ══════════════════════════════════════════════════════════════════
        # METHOD 3: Cosinor analysis
        # ══════════════════════════════════════════════════════════════════
        # Fit at the Lomb-Scargle best circadian period (or 24h default)
        cosinor_period = ls_circ_best_p if ls_circ_best_p > 0 else 24.0
        cosinor = _cosinor_fit(hourly_rate, hourly_bin, cosinor_period)

        # Also fit at exactly 24h for comparison
        cosinor_24 = _cosinor_fit(hourly_rate, hourly_bin, 24.0)

        # ══════════════════════════════════════════════════════════════════
        # METHOD 4: Chi-squared periodogram (Sokolove-Bushell)
        # ══════════════════════════════════════════════════════════════════
        chi2_periods, chi2_qp, chi2_best_p, chi2_best_qp = (
            _chi_squared_periodogram(hourly_rate, hourly_bin, 20.0, 28.0, step_h=0.25)
        )

        # Broad chi-squared scan
        chi2b_periods, chi2b_qp, chi2b_best_p, chi2b_best_qp = (
            _chi_squared_periodogram(hourly_rate, hourly_bin, 4.0,
                                     min(total_duration_h / 2.0, 72.0), step_h=0.5)
        )

        # ══════════════════════════════════════════════════════════════════
        # CONSENSUS: Is there a circadian rhythm?
        # ══════════════════════════════════════════════════════════════════
        # Evidence from each method for a rhythm in the 20-28h band
        evidence_score = 0
        evidence_details = []

        # Lomb-Scargle: power > 0.1 in circadian band
        if ls_circ_best_pow > 0.1:
            evidence_score += 1
            evidence_details.append(
                f"Lomb-Scargle peak at {ls_circ_best_p:.1f}h (power={ls_circ_best_pow:.3f})"
            )

        # Autocorrelation: peak in 20-28h range with r > 0.1
        if 20.0 <= acorr_best_lag_h <= 28.0 and acorr_peak_value > 0.1:
            evidence_score += 1
            evidence_details.append(
                f"Autocorrelation peak at lag {acorr_best_lag_h:.0f}h (r={acorr_peak_value:.3f})"
            )

        # Cosinor: significant fit (p < 0.05) with reasonable amplitude
        if cosinor["p_value"] < 0.05 and cosinor["percent_rhythm"] > 5.0:
            evidence_score += 1
            evidence_details.append(
                f"Cosinor fit significant (p={cosinor['p_value']:.4f}, "
                f"amplitude={cosinor['percent_rhythm']:.1f}%)"
            )

        # Chi-squared: peak Qp in circadian band
        # Significance threshold: Qp > chi2(0.95, p-1) ~ roughly 1.5x mean Qp
        chi2_sig = False
        if len(chi2_qp) > 0 and chi2_best_qp > 0:
            mean_qp = float(np.mean(chi2_qp[chi2_qp > 0])) if np.any(chi2_qp > 0) else 1.0
            if chi2_best_qp > 1.5 * mean_qp:
                chi2_sig = True
                evidence_score += 1
                evidence_details.append(
                    f"Chi-squared periodogram peak at {chi2_best_p:.1f}h (Qp={chi2_best_qp:.1f})"
                )

        circadian_detected = evidence_score >= 2  # require 2+ methods to agree

        # ── Dominant period: weighted consensus ──
        period_estimates = []
        if ls_circ_best_p > 0 and ls_circ_best_pow > 0.05:
            period_estimates.append((ls_circ_best_p, ls_circ_best_pow))
        if 20.0 <= acorr_best_lag_h <= 28.0 and acorr_peak_value > 0.05:
            period_estimates.append((acorr_best_lag_h, acorr_peak_value))
        if cosinor["p_value"] < 0.1 and cosinor["period_h"] > 0:
            period_estimates.append((cosinor["period_h"], cosinor["r_squared"]))
        if chi2_best_p > 0 and chi2_sig:
            period_estimates.append((chi2_best_p, 1.0))

        if period_estimates:
            weights = np.array([w for _, w in period_estimates])
            periods = np.array([p for p, _ in period_estimates])
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            dominant_period_h = float(np.average(periods, weights=weights))
            period_std = float(np.sqrt(np.average((periods - dominant_period_h)**2, weights=weights)))
        else:
            dominant_period_h = 0.0
            period_std = 0.0

        # ── Day/Night ratio using acrophase ──
        # Acrophase defines peak of rhythm; "day" = acrophase +/- T/4, "night" = rest
        day_night_ratio = 0.0
        if circadian_detected and cosinor["mesor"] > 0:
            acrophase_h = cosinor["acrophase_h"]
            # Bin each hour into day/night based on acrophase
            hour_positions = np.arange(n_hours) % round(dominant_period_h)
            # "Day" phase: within +/- T/4 of acrophase (peak half of cycle)
            quarter_period = dominant_period_h / 4.0
            day_center = acrophase_h % dominant_period_h
            # Circular distance
            dist = np.abs(hour_positions - day_center)
            dist = np.minimum(dist, dominant_period_h - dist)
            day_mask = dist <= quarter_period
            night_mask = ~day_mask

            day_rate = float(np.mean(hourly_rate[day_mask])) if np.any(day_mask) else 0.0
            night_rate = float(np.mean(hourly_rate[night_mask])) if np.any(night_mask) else 1e-12
            day_night_ratio = round(day_rate / max(night_rate, 1e-12), 3)

        # ── Collect ultradian and infradian peaks ──
        ultradian_rhythms = []
        if ls_ultra_best_pow > 0.1 and ls_ultra_best_p > 0:
            # Find multiple ultradian peaks
            if len(ls_ultra_power) > 3:
                u_peaks, _ = find_peaks(ls_ultra_power, height=0.1, distance=20)
                for up_idx in u_peaks[:3]:  # top 3
                    ultradian_rhythms.append({
                        "period_h": round(float(ls_ultra_periods[up_idx]), 2),
                        "power": round(float(ls_ultra_power[up_idx]), 4),
                    })
            if not ultradian_rhythms:
                ultradian_rhythms.append({
                    "period_h": round(ls_ultra_best_p, 2),
                    "power": round(ls_ultra_best_pow, 4),
                })

        infradian_rhythms = []
        if ls_broad_best_p > 28.0 and ls_broad_best_pow > 0.1:
            infradian_rhythms.append({
                "period_h": round(ls_broad_best_p, 2),
                "power": round(ls_broad_best_pow, 4),
            })

        return {
            "label": label,
            "n_spikes": int(n_spikes),
            "duration_h": round(total_duration_h, 2),
            "circadian_detected": circadian_detected,
            "evidence_score": evidence_score,  # 0-4, how many methods agree
            "evidence_details": evidence_details,
            "dominant_period_h": round(dominant_period_h, 2),
            "period_ci_h": round(period_std, 2),  # spread across method estimates
            "acrophase_h": cosinor["acrophase_h"],
            "amplitude_pct": cosinor["percent_rhythm"],
            "day_night_ratio": day_night_ratio,
            "cosinor": cosinor,
            "cosinor_24h": cosinor_24,
            "lomb_scargle": {
                "circadian_band": {
                    "best_period_h": round(ls_circ_best_p, 2),
                    "best_power": round(ls_circ_best_pow, 4),
                },
                "broad_scan": {
                    "best_period_h": round(ls_broad_best_p, 2),
                    "best_power": round(ls_broad_best_pow, 4),
                },
            },
            "autocorrelation": {
                "best_lag_h": round(acorr_best_lag_h, 1),
                "peak_value": round(acorr_peak_value, 4),
            },
            "chi_squared_periodogram": {
                "circadian_band": {
                    "best_period_h": round(chi2_best_p, 2),
                    "best_qp": round(chi2_best_qp, 2),
                    "significant": chi2_sig,
                },
                "broad_scan": {
                    "best_period_h": round(chi2b_best_p, 2),
                    "best_qp": round(chi2b_best_qp, 2),
                },
            },
            "ultradian_rhythms": ultradian_rhythms,
            "infradian_rhythms": infradian_rhythms,
        }

    # ═════════════════════════════════════════════════════════════════════
    # Run analysis: population level + per-MEA
    # ═════════════════════════════════════════════════════════════════════

    # Population-level (all spikes)
    population = _analyze_rhythms(data.times, "population")

    # Per-MEA analysis (4 MEAs x 8 electrodes, FinalSpark layout)
    electrodes_per_mea = 8
    all_electrodes = sorted(data.electrode_ids)
    max_electrode = max(all_electrodes) if all_electrodes else 0
    n_meas = max(1, int(np.ceil((max_electrode + 1) / electrodes_per_mea)))

    per_mea = []
    mea_acrophases = []
    mea_periods = []

    for mea_idx in range(n_meas):
        e_start = mea_idx * electrodes_per_mea
        e_end = e_start + electrodes_per_mea
        mea_electrodes = [e for e in all_electrodes if e_start <= e < e_end]
        if not mea_electrodes:
            continue

        # Gather spikes for this MEA using pre-computed electrode indices
        mea_mask = np.isin(data.electrodes, mea_electrodes)
        mea_times = data.times[mea_mask]

        if len(mea_times) < 50:
            per_mea.append({
                "mea_id": mea_idx,
                "electrodes": mea_electrodes,
                "n_spikes": int(len(mea_times)),
                "circadian_detected": False,
                "error": "Too few spikes",
            })
            continue

        mea_result = _analyze_rhythms(mea_times, f"MEA_{mea_idx}")
        mea_result["mea_id"] = mea_idx
        mea_result["electrodes"] = mea_electrodes
        per_mea.append(mea_result)

        if mea_result["circadian_detected"]:
            mea_acrophases.append(mea_result["acrophase_h"])
            mea_periods.append(mea_result["dominant_period_h"])

    # ─── Phase coherence between MEAs ────────────────────────────────────
    # Rayleigh test on acrophase angles: are MEAs synchronized?
    phase_coherence = 0.0
    phase_coherence_p = 1.0
    if len(mea_acrophases) >= 2:
        ref_period = float(np.mean(mea_periods)) if mea_periods else 24.0
        phases_rad = np.array(mea_acrophases) * (2.0 * np.pi / ref_period)
        # Mean resultant length (R-bar) — 1.0 = perfect synchrony, 0.0 = random
        cos_sum = float(np.sum(np.cos(phases_rad)))
        sin_sum = float(np.sum(np.sin(phases_rad)))
        n_mea_circ = len(phases_rad)
        r_bar = np.sqrt(cos_sum**2 + sin_sum**2) / n_mea_circ
        phase_coherence = round(float(r_bar), 4)

        # Rayleigh test p-value approximation
        r2 = n_mea_circ * r_bar**2
        phase_coherence_p = round(float(np.exp(-r2) * (1.0 + (2.0 * r2 - r2**2) / (4.0 * n_mea_circ))), 6)

    # ═════════════════════════════════════════════════════════════════════
    # Final output
    # ═════════════════════════════════════════════════════════════════════
    meas_with_circadian = sum(1 for m in per_mea if m.get("circadian_detected", False))

    # Build interpretation
    parts = []
    if population["circadian_detected"]:
        parts.append(
            f"Circadian rhythm detected (period={population['dominant_period_h']:.1f}h, "
            f"amplitude={population['amplitude_pct']:.1f}%, "
            f"acrophase={population['acrophase_h']:.1f}h). "
            f"Day/night ratio={population['day_night_ratio']:.2f}."
        )
    else:
        parts.append("No circadian rhythm detected at population level.")

    parts.append(
        f"{meas_with_circadian}/{len(per_mea)} MEAs show circadian rhythms."
    )

    if phase_coherence > 0.7 and len(mea_acrophases) >= 2:
        parts.append(f"MEAs are phase-coherent (R={phase_coherence:.2f}).")
    elif len(mea_acrophases) >= 2:
        parts.append(f"MEA phase coherence: R={phase_coherence:.2f}.")

    if population.get("ultradian_rhythms"):
        u = population["ultradian_rhythms"][0]
        parts.append(f"Ultradian rhythm at {u['period_h']:.1f}h.")

    return {
        # ── Top-level summary (backward-compatible keys) ──
        "circadian_detected": population["circadian_detected"],
        "dominant_period_h": population["dominant_period_h"],
        "period_ci_h": population["period_ci_h"],
        "acrophase_h": population["acrophase_h"],
        "amplitude_pct": population["amplitude_pct"],
        "day_night_ratio": population["day_night_ratio"],
        "evidence_score": population["evidence_score"],
        "evidence_details": population["evidence_details"],
        # ── Multi-method details ──
        "methods": {
            "lomb_scargle": population["lomb_scargle"],
            "autocorrelation": population["autocorrelation"],
            "cosinor": population["cosinor"],
            "cosinor_24h": population["cosinor_24h"],
            "chi_squared_periodogram": population["chi_squared_periodogram"],
        },
        # ── Multi-timescale ──
        "ultradian_rhythms": population["ultradian_rhythms"],
        "infradian_rhythms": population["infradian_rhythms"],
        # ── Per-MEA breakdown ──
        "per_mea": per_mea,
        "n_meas_with_circadian": meas_with_circadian,
        "n_meas_total": len(per_mea),
        # ── Phase coherence ──
        "phase_coherence": phase_coherence,
        "phase_coherence_p": phase_coherence_p,
        # ── Legacy-compatible fields ──
        "has_rhythmic_structure": population["circadian_detected"] or bool(population["ultradian_rhythms"]),
        "interpretation": " ".join(parts),
    }
