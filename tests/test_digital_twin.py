"""Tests for digital twin module."""

from analysis.digital_twin import fit_lif_parameters, simulate_lif_network, compare_real_vs_simulated


def test_fit_lif(synthetic_data):
    result = fit_lif_parameters(synthetic_data)
    assert isinstance(result, dict)
    assert "parameters" in result or "neurons" in result


def test_simulate(synthetic_data):
    params = fit_lif_parameters(synthetic_data)
    sim = simulate_lif_network(params)
    assert isinstance(sim, dict)
    assert "spikes" in sim or "spike_times" in sim or "times" in sim


def test_compare(synthetic_data):
    params = fit_lif_parameters(synthetic_data)
    sim = simulate_lif_network(params)
    comparison = compare_real_vs_simulated(synthetic_data, sim)
    assert isinstance(comparison, dict)
