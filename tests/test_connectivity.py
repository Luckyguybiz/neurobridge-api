"""Tests for connectivity analysis module."""

from analysis.connectivity import (
    compute_cross_correlation,
    compute_connectivity_graph,
    compute_transfer_entropy,
    compute_cofiring_rate,
    connectivity_to_dict,
    ConnectivityResult,
)


def test_cross_correlation(synthetic_data):
    result = compute_cross_correlation(synthetic_data)
    # CCGResult dataclass
    assert hasattr(result, "correlation_matrix")
    assert result.correlation_matrix.shape[0] > 0


def test_cofiring_rate(synthetic_data):
    result = compute_cofiring_rate(synthetic_data)
    assert hasattr(result, "cofiring_matrix")
    assert result.cofiring_matrix.shape[0] == len(synthetic_data.electrode_ids)


def test_connectivity_graph_fast(synthetic_data):
    """Default mode: only CCG + cofiring (fast)."""
    result = compute_connectivity_graph(synthetic_data)
    assert isinstance(result, ConnectivityResult)
    assert result.te_matrix is None  # not computed by default
    assert result.plv_matrix is None
    assert result.mi_matrix is None
    d = connectivity_to_dict(result)
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == len(synthetic_data.electrode_ids)
    assert "cofiring" in d["matrices"]


def test_connectivity_graph_include_all(synthetic_data):
    """Full mode: all measures."""
    result = compute_connectivity_graph(synthetic_data, include_all=True)
    assert isinstance(result, ConnectivityResult)
    assert result.te_matrix is not None
    assert result.plv_matrix is not None
    assert result.mi_matrix is not None
    d = connectivity_to_dict(result)
    assert "transfer_entropy" in d["matrices"]
    assert "plv" in d["matrices"]
    assert "mutual_information" in d["matrices"]


def test_transfer_entropy(synthetic_data):
    result = compute_transfer_entropy(synthetic_data)
    assert hasattr(result, "te_matrix")
    assert result.te_matrix.shape[0] > 0
