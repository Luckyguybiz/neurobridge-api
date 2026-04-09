"""Tests for burst detection module."""

from analysis.bursts import detect_bursts, compute_burst_profiles, detect_single_channel_bursts


def test_detect_bursts(synthetic_data):
    result = detect_bursts(synthetic_data)
    assert "bursts" in result
    assert "n_bursts" in result
    assert isinstance(result["bursts"], list)


def test_detect_bursts_bursty_data(bursty_data):
    result = detect_bursts(bursty_data, min_electrodes=3, window_ms=50.0)
    assert result["n_bursts"] > 0, "Should detect bursts in bursty data"


def test_burst_profiles(bursty_data):
    bursts_result = detect_bursts(bursty_data)
    if bursts_result["n_bursts"] > 0:
        profiles = compute_burst_profiles(bursty_data, bursts_result["bursts"])
        assert "profiles" in profiles
        assert len(profiles["profiles"]) > 0


def test_single_channel_bursts(synthetic_data):
    # Get spike times for electrode 0
    mask = synthetic_data.electrodes == 0
    spike_times = synthetic_data.times[mask]
    result = detect_single_channel_bursts(spike_times)
    assert isinstance(result, list)
