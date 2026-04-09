"""Tests for novel/unique analysis modules."""

from analysis.plasticity import compute_stdp_matrix, detect_learning_episodes
from analysis.organoid_iq import compute_organoid_iq
from analysis.predictions import predict_firing_rates, predict_burst_probability, estimate_organoid_health
from analysis.replay import detect_replay, detect_sequence_replay
from analysis.reservoir import estimate_memory_capacity, benchmark_nonlinear_computation
from analysis.fingerprint import compute_fingerprint
from analysis.sonification import generate_sonification, compute_rhythmic_analysis
from analysis.emergence import compute_integrated_information
from analysis.attractors import map_attractor_landscape, compute_state_space_geometry
from analysis.phase_transitions import detect_phase_transitions
from analysis.predictive_coding import measure_predictive_coding
from analysis.weight_inference import infer_synaptic_weights, track_weight_changes
from analysis.multiscale import compute_multiscale_complexity


class TestPlasticity:
    def test_stdp_matrix(self, synthetic_data):
        result = compute_stdp_matrix(synthetic_data)
        assert isinstance(result, dict)

    def test_learning_episodes(self, synthetic_data):
        result = detect_learning_episodes(synthetic_data)
        assert isinstance(result, dict)


class TestOrganoidIQ:
    def test_compute_iq(self, synthetic_data):
        result = compute_organoid_iq(synthetic_data)
        assert isinstance(result, dict)
        assert "iq_score" in result or "score" in result or "dimensions" in result


class TestPredictions:
    def test_predict_rates(self, synthetic_data):
        result = predict_firing_rates(synthetic_data)
        assert isinstance(result, dict)

    def test_predict_bursts(self, synthetic_data):
        result = predict_burst_probability(synthetic_data)
        assert isinstance(result, dict)

    def test_health(self, synthetic_data):
        result = estimate_organoid_health(synthetic_data)
        assert isinstance(result, dict)


class TestReplay:
    def test_detect_replay(self, synthetic_data):
        result = detect_replay(synthetic_data)
        assert isinstance(result, dict)

    def test_sequences(self, synthetic_data):
        result = detect_sequence_replay(synthetic_data)
        assert isinstance(result, dict)


class TestReservoir:
    def test_memory_capacity(self, synthetic_data):
        result = estimate_memory_capacity(synthetic_data)
        assert isinstance(result, dict)

    def test_nonlinearity(self, synthetic_data):
        result = benchmark_nonlinear_computation(synthetic_data)
        assert isinstance(result, dict)


class TestFingerprint:
    def test_fingerprint(self, synthetic_data):
        result = compute_fingerprint(synthetic_data)
        assert isinstance(result, dict)
        assert "fingerprint_hash" in result
        assert "fingerprint_vector" in result


class TestSonification:
    def test_sonify(self, synthetic_data):
        result = generate_sonification(synthetic_data)
        assert isinstance(result, dict)
        assert "wav_base64" in result
        assert result["spikes_sonified"] > 0

    def test_rhythms(self, synthetic_data):
        result = compute_rhythmic_analysis(synthetic_data)
        assert isinstance(result, dict)


class TestEmergence:
    def test_phi(self, synthetic_data):
        result = compute_integrated_information(synthetic_data)
        assert isinstance(result, dict)


class TestAttractors:
    def test_landscape(self, synthetic_data):
        result = map_attractor_landscape(synthetic_data)
        assert isinstance(result, dict)

    def test_state_space(self, synthetic_data):
        result = compute_state_space_geometry(synthetic_data)
        assert isinstance(result, dict)


class TestPhaseTransitions:
    def test_detect(self, synthetic_data):
        result = detect_phase_transitions(synthetic_data)
        assert isinstance(result, dict)


class TestPredictiveCoding:
    def test_measure(self, synthetic_data):
        result = measure_predictive_coding(synthetic_data)
        assert isinstance(result, dict)


class TestWeightInference:
    def test_infer_weights(self, synthetic_data):
        result = infer_synaptic_weights(synthetic_data)
        assert isinstance(result, dict)

    def test_track_weights(self, synthetic_data):
        result = track_weight_changes(synthetic_data)
        assert isinstance(result, dict)


class TestMultiscale:
    def test_complexity(self, synthetic_data):
        result = compute_multiscale_complexity(synthetic_data)
        assert isinstance(result, dict)
