"""Tests for data loader module."""

import tempfile
import os
import numpy as np
from analysis.loader import SpikeData, load_file


def test_spike_data_creation():
    data = SpikeData(
        times=np.array([0.1, 0.2, 0.3]),
        electrodes=np.array([0, 1, 0]),
        amplitudes=np.array([-50, -45, -55]),
        waveforms=np.random.randn(3, 90),
        sampling_rate=30000.0,
    )
    assert len(data.times) == 3
    assert data.n_electrodes == 2


def test_load_csv():
    """Test loading from a CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("time,electrode,amplitude\n")
        for i in range(100):
            f.write(f"{i * 0.01},{i % 4},{-50 + np.random.randn() * 10}\n")
        fname = f.name

    try:
        data = load_file(fname)
        assert isinstance(data, SpikeData)
        assert len(data.times) == 100
        assert data.n_electrodes == 4
    finally:
        os.unlink(fname)


def test_spike_data_electrode_filter(synthetic_data):
    """Test filtering by electrode."""
    mask = synthetic_data.electrodes == 0
    filtered_times = synthetic_data.times[mask]
    assert len(filtered_times) > 0
    assert len(filtered_times) < len(synthetic_data.times)
