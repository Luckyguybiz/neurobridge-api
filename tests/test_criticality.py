"""Tests for criticality analysis module."""

from analysis.criticality import detect_avalanches


def test_avalanches(synthetic_data):
    result = detect_avalanches(synthetic_data)
    assert isinstance(result, dict)
    assert "avalanches" in result or "n_avalanches" in result or "sizes" in result
