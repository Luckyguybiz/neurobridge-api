"""Tests for statistics module."""

from analysis.stats import compute_full_summary, compute_temporal_dynamics, compute_quality_metrics


def test_full_summary(synthetic_data):
    result = compute_full_summary(synthetic_data)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_temporal_dynamics(synthetic_data):
    result = compute_temporal_dynamics(synthetic_data)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Should have bin_centers or electrode_rates
    assert "bin_centers" in result or "electrode_rates" in result or "time_bins" in result


def test_quality_metrics(synthetic_data):
    result = compute_quality_metrics(synthetic_data)
    assert isinstance(result, dict)
    assert len(result) > 0
