"""Shared test fixtures for NeuroBridge API tests."""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.loader import SpikeData


@pytest.fixture
def synthetic_data():
    """Generate synthetic spike data for testing (30s, 8 electrodes)."""
    np.random.seed(42)
    n_spikes = 3000
    duration = 30.0
    n_electrodes = 8

    times = np.sort(np.random.uniform(0, duration, n_spikes))
    electrodes = np.random.randint(0, n_electrodes, n_spikes)
    amplitudes = np.random.normal(-50, 20, n_spikes)

    # Generate waveforms: 90 samples per spike (3ms at 30kHz)
    waveform_samples = 90
    waveforms = np.random.randn(n_spikes, waveform_samples) * 20

    return SpikeData(
        times=times,
        electrodes=electrodes,
        amplitudes=amplitudes,
        waveforms=waveforms,
        sampling_rate=30000.0,
    )


@pytest.fixture
def small_data():
    """Minimal dataset for fast tests (5s, 4 electrodes)."""
    np.random.seed(123)
    n_spikes = 500
    duration = 5.0
    n_electrodes = 4

    times = np.sort(np.random.uniform(0, duration, n_spikes))
    electrodes = np.random.randint(0, n_electrodes, n_spikes)
    amplitudes = np.random.normal(-40, 15, n_spikes)
    waveforms = np.random.randn(n_spikes, 90) * 15

    return SpikeData(
        times=times,
        electrodes=electrodes,
        amplitudes=amplitudes,
        waveforms=waveforms,
        sampling_rate=30000.0,
    )


@pytest.fixture
def bursty_data():
    """Data with clear burst patterns for burst detection tests."""
    np.random.seed(99)
    n_electrodes = 8
    duration = 30.0

    # Background: random spikes
    bg_times = np.random.uniform(0, duration, 1000)
    bg_electrodes = np.random.randint(0, n_electrodes, 1000)

    # Bursts: clusters of spikes across multiple electrodes
    burst_times = []
    burst_electrodes = []
    for burst_center in [5.0, 10.0, 15.0, 20.0, 25.0]:
        for _ in range(50):
            burst_times.append(burst_center + np.random.uniform(-0.02, 0.02))
            burst_electrodes.append(np.random.randint(0, n_electrodes))

    times = np.sort(np.concatenate([bg_times, burst_times]))
    electrodes = np.concatenate([bg_electrodes, burst_electrodes]).astype(int)
    amplitudes = np.random.normal(-50, 20, len(times))
    waveforms = np.random.randn(len(times), 90) * 20

    return SpikeData(
        times=times,
        electrodes=electrodes,
        amplitudes=amplitudes,
        waveforms=waveforms,
        sampling_rate=30000.0,
    )
