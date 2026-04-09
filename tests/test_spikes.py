"""Tests for spike analysis module."""

from analysis.spikes import compute_isi, compute_firing_rates, compute_amplitude_stats, sort_spikes


def test_compute_isi(synthetic_data):
    result = compute_isi(synthetic_data)
    assert isinstance(result, dict)
    # Results keyed by electrode ID (int) or contains population stats
    assert len(result) > 0


def test_compute_isi_single_electrode(synthetic_data):
    result = compute_isi(synthetic_data, electrode=0)
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_compute_firing_rates(synthetic_data):
    result = compute_firing_rates(synthetic_data, bin_size_sec=1.0)
    assert isinstance(result, dict)
    # Should have rates and bins
    assert "rates" in result or "bins" in result or "mean_rates" in result


def test_compute_firing_rates_custom_bin(synthetic_data):
    result = compute_firing_rates(synthetic_data, bin_size_sec=5.0)
    assert isinstance(result, dict)
    if "bins" in result:
        assert len(result["bins"]) <= 10  # 30s / 5s = 6 bins


def test_compute_amplitude_stats(synthetic_data):
    result = compute_amplitude_stats(synthetic_data)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Results keyed by electrode ID
    first_key = list(result.keys())[0]
    electrode_stats = result[first_key]
    assert "mean_uv" in electrode_stats or "mean" in electrode_stats


def test_sort_spikes(synthetic_data):
    result = sort_spikes(
        synthetic_data.waveforms,
        method="pca_kmeans",
        n_components=3,
        n_clusters=3,
        min_cluster_size=10,
    )
    assert "labels" in result
    assert "n_clusters" in result
    assert result["n_clusters"] >= 1
    assert len(result["labels"]) == len(synthetic_data.waveforms)
