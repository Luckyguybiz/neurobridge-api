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
    """Analyze rhythmic structure of spike activity.

    Finds periodicity and rhythmic patterns that may be
    easier to detect through sonification.
    """
    if data.n_spikes < 50:
        return {"error": "Not enough spikes"}

    # Autocorrelation of population firing rate
    bin_sec = 0.01  # 10ms
    t_start, t_end = data.time_range
    bins = np.arange(t_start, t_end + bin_sec, bin_sec)
    counts, _ = np.histogram(data.times, bins=bins)

    # Autocorrelation
    rate = counts.astype(float)
    rate -= np.mean(rate)
    n = len(rate)
    autocorr = np.correlate(rate, rate, mode='full')
    autocorr = autocorr[n - 1:]  # positive lags only
    autocorr /= autocorr[0] if autocorr[0] > 0 else 1

    # Find peaks in autocorrelation (periodicities)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(autocorr[1:], height=0.1, distance=5)
    peaks += 1  # offset from skip

    periodicities = []
    for p in peaks[:5]:
        period_ms = p * bin_sec * 1000
        frequency_hz = 1000 / period_ms if period_ms > 0 else 0
        periodicities.append({
            "period_ms": round(float(period_ms), 1),
            "frequency_hz": round(float(frequency_hz), 2),
            "strength": round(float(autocorr[p]), 4),
        })

    return {
        "periodicities": periodicities,
        "n_periodicities": len(periodicities),
        "dominant_rhythm_ms": periodicities[0]["period_ms"] if periodicities else 0,
        "has_rhythmic_structure": len(periodicities) > 0,
        "interpretation": (
            f"Dominant rhythm at {periodicities[0]['period_ms']:.0f}ms ({periodicities[0]['frequency_hz']:.1f}Hz). "
            f"Found {len(periodicities)} periodic components."
            if periodicities
            else "No clear rhythmic structure detected."
        ),
    }
