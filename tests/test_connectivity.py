"""Tests for connectivity analysis module."""

from analysis.connectivity import compute_cross_correlation, compute_connectivity_graph, compute_transfer_entropy


def test_cross_correlation(synthetic_data):
    result = compute_cross_correlation(synthetic_data)
    assert isinstance(result, dict)
    # May have pairs, correlograms, or be keyed by pair tuples
    assert len(result) > 0


def test_connectivity_graph(synthetic_data):
    result = compute_connectivity_graph(synthetic_data)
    assert "nodes" in result
    assert "edges" in result
    assert len(result["nodes"]) == synthetic_data.n_electrodes


def test_transfer_entropy(synthetic_data):
    result = compute_transfer_entropy(synthetic_data)
    assert isinstance(result, dict)
    assert len(result) > 0
