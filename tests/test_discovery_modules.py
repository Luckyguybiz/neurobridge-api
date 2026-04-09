"""Tests for 10 new discovery analysis modules."""

import numpy as np
import pytest

from analysis.sleep_wake import analyze_sleep_wake
from analysis.habituation import detect_repeated_patterns
from analysis.metastability import analyze_metastability
from analysis.information_flow import compute_granger_causality
from analysis.network_motifs import enumerate_motifs
from analysis.energy_landscape import fit_ising_model
from analysis.stimulus_design import evolve_protocol
from analysis.consciousness_metrics import compute_consciousness_score
from analysis.comparative import compare_with_references
from analysis.protocol_library import list_protocols, get_protocol, suggest_protocol


class TestSleepWake:
    def test_returns_dict(self, synthetic_data):
        result = analyze_sleep_wake(synthetic_data)
        assert isinstance(result, dict)

    def test_has_states(self, synthetic_data):
        result = analyze_sleep_wake(synthetic_data)
        assert "n_up_states" in result or "up_down_states" in result or "sleep_like_score" in result

    def test_small_data(self, small_data):
        result = analyze_sleep_wake(small_data)
        assert isinstance(result, dict)


class TestHabituation:
    def test_returns_dict(self, synthetic_data):
        result = detect_repeated_patterns(synthetic_data)
        assert isinstance(result, dict)

    def test_has_habituation_flag(self, synthetic_data):
        result = detect_repeated_patterns(synthetic_data)
        assert "habituation_detected" in result or "habituated" in result or "n_events" in result

    def test_small_data(self, small_data):
        result = detect_repeated_patterns(small_data)
        assert isinstance(result, dict)


class TestMetastability:
    def test_returns_dict(self, synthetic_data):
        result = analyze_metastability(synthetic_data)
        assert isinstance(result, dict)

    def test_has_kuramoto(self, synthetic_data):
        result = analyze_metastability(synthetic_data)
        # either nested or flat structure
        has_key = (
            "kuramoto" in result
            or "kuramoto_mean" in result
            or "mean_order_parameter" in result
        )
        assert has_key

    def test_metastability_index_non_negative(self, synthetic_data):
        result = analyze_metastability(synthetic_data)
        kuramoto = result.get("kuramoto", result)
        idx = float(kuramoto.get("metastability_index", 0))
        assert idx >= 0


class TestInformationFlow:
    def test_returns_dict(self, synthetic_data):
        result = compute_granger_causality(synthetic_data)
        assert isinstance(result, dict)

    def test_has_pairs(self, synthetic_data):
        result = compute_granger_causality(synthetic_data)
        assert "pairs" in result or "gc_matrix" in result or "top_pairs" in result

    def test_small_data(self, small_data):
        result = compute_granger_causality(small_data)
        assert isinstance(result, dict)


class TestNetworkMotifs:
    def test_returns_dict(self, synthetic_data):
        result = enumerate_motifs(synthetic_data)
        assert isinstance(result, dict)

    def test_has_motif_counts(self, synthetic_data):
        result = enumerate_motifs(synthetic_data)
        assert "motif_counts" in result or "feed_forward_chains" in result or "n_motifs" in result

    def test_bursty_data(self, bursty_data):
        result = enumerate_motifs(bursty_data)
        assert isinstance(result, dict)


class TestEnergyLandscape:
    def test_returns_dict(self, synthetic_data):
        result = fit_ising_model(synthetic_data)
        assert isinstance(result, dict)

    def test_has_model(self, synthetic_data):
        result = fit_ising_model(synthetic_data)
        assert "model_type" in result or "coupling_matrix" in result or "n_attractors" in result

    def test_small_data(self, small_data):
        result = fit_ising_model(small_data)
        assert isinstance(result, dict)


class TestStimulusDesign:
    def test_returns_dict(self, small_data):
        result = evolve_protocol(small_data, generations=3, population_size=5)
        assert isinstance(result, dict)

    def test_has_best_protocol(self, small_data):
        result = evolve_protocol(small_data, generations=3, population_size=5)
        assert "best_protocol" in result or "protocol" in result

    def test_generations_parameter(self, small_data):
        result = evolve_protocol(small_data, generations=2, population_size=4)
        assert isinstance(result, dict)
        assert result.get("generations", 0) >= 1 or "best_protocol" in result


class TestConsciousnessMetrics:
    def test_returns_dict(self, synthetic_data):
        result = compute_consciousness_score(synthetic_data)
        assert isinstance(result, dict)

    def test_has_score(self, synthetic_data):
        result = compute_consciousness_score(synthetic_data)
        assert "consciousness_score" in result or "score" in result

    def test_score_range(self, synthetic_data):
        result = compute_consciousness_score(synthetic_data)
        score = float(result.get("consciousness_score", result.get("score", 0.5)))
        assert 0.0 <= score <= 1.0

    def test_small_data(self, small_data):
        result = compute_consciousness_score(small_data)
        assert isinstance(result, dict)


class TestComparative:
    def test_returns_dict(self, synthetic_data):
        result = compare_with_references(synthetic_data)
        assert isinstance(result, dict)

    def test_has_similarities(self, synthetic_data):
        result = compare_with_references(synthetic_data)
        assert "similarities" in result or "most_similar_system" in result

    def test_most_similar_is_valid(self, synthetic_data):
        result = compare_with_references(synthetic_data)
        known = {
            "cortical_slice", "c_elegans", "fruit_fly",
            "mouse_hippocampus", "organoid_typical",
        }
        most_similar = result.get("most_similar_system", "")
        assert most_similar in known


class TestProtocolLibrary:
    def test_list_returns_dict(self):
        result = list_protocols()
        assert isinstance(result, dict)
        assert "protocols" in result
        assert result.get("n_protocols", 0) >= 4

    def test_get_dishbrain(self):
        result = get_protocol("dishbrain_pong")
        assert isinstance(result, dict)
        assert "error" not in result

    def test_get_missing_returns_error(self):
        result = get_protocol("nonexistent_protocol")
        assert "error" in result

    def test_get_stdp(self):
        result = get_protocol("stdp_training")
        assert isinstance(result, dict)
        assert "parameters" in result

    def test_suggest_protocol(self, synthetic_data):
        result = suggest_protocol(synthetic_data)
        assert isinstance(result, dict)
        assert "suggested_protocol" in result or "protocol" in result
