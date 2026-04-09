"""Tests for information theory module."""

from analysis.information_theory import (
    compute_spike_train_entropy,
    compute_mutual_information,
    compute_lempel_ziv_complexity,
)


def test_entropy(synthetic_data):
    result = compute_spike_train_entropy(synthetic_data)
    assert "per_electrode" in result or "entropy" in result
    assert isinstance(result, dict)


def test_mutual_information(synthetic_data):
    result = compute_mutual_information(synthetic_data)
    assert isinstance(result, dict)


def test_lempel_ziv(synthetic_data):
    result = compute_lempel_ziv_complexity(synthetic_data)
    assert isinstance(result, dict)
