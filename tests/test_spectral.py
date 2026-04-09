"""Tests for spectral analysis module."""

from analysis.spectral import compute_power_spectrum, compute_coherence


def test_power_spectrum(synthetic_data):
    result = compute_power_spectrum(synthetic_data)
    assert isinstance(result, dict)
    assert "per_electrode" in result or "frequencies" in result


def test_coherence(synthetic_data):
    result = compute_coherence(synthetic_data)
    assert isinstance(result, dict)
