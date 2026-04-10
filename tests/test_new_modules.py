"""Tests for new analysis modules (Sprint 4+)."""

from analysis.sleep_wake import analyze_sleep_wake
from analysis.habituation import detect_repeated_patterns
from analysis.metastability import analyze_metastability
from analysis.information_flow import compute_granger_causality
from analysis.network_motifs import enumerate_motifs
from analysis.energy_landscape import fit_ising_model
from analysis.consciousness_metrics import compute_consciousness_score
from analysis.comparative import compare_with_references
from analysis.protocol_library import list_protocols, get_protocol
from analysis.catastrophic_forgetting import compute_retention_curve
from analysis.transfer_learning import measure_transfer
from analysis.consolidation import detect_consolidation_events
from analysis.multi_bit_memory import estimate_channel_capacity, measure_population_code_diversity
from analysis.functional_connectome import (
    build_full_connectome,
    detect_communities,
    compute_graph_theory_metrics,
)
from analysis.effective_connectivity import (
    estimate_effective_connectivity,
    compute_causal_hierarchy,
)
from analysis.topology import compute_betti_numbers, compute_topological_complexity
from analysis.turing_test import run_turing_test
from analysis.neural_architecture_search import search_optimal_protocol
from analysis.hybrid_ai import benchmark_hybrid
from analysis.genetic_programming import evolve_programs
from analysis.homeostatic_plasticity import monitor_homeostasis
from analysis.suffering_detector import detect_suffering
from analysis.welfare_report import generate_welfare_report
from analysis.swarm_organoid import simulate_swarm
from analysis.morphological_computing import analyze_morphological_computation


class TestSleepWake:
    def test_analyze(self, synthetic_data):
        result = analyze_sleep_wake(synthetic_data)
        assert isinstance(result, dict)
        assert "sleep_like_score" in result


class TestHabituation:
    def test_detect_repeated_patterns(self, synthetic_data):
        result = detect_repeated_patterns(synthetic_data)
        assert isinstance(result, dict)
        assert "habituation_detected" in result or "n_events" in result


class TestMetastability:
    def test_analyze(self, synthetic_data):
        result = analyze_metastability(synthetic_data)
        assert isinstance(result, dict)
        assert "is_metastable" in result or "kuramoto" in result


class TestInformationFlow:
    def test_granger_causality(self, synthetic_data):
        result = compute_granger_causality(synthetic_data)
        assert isinstance(result, dict)
        assert "pairs" in result or "gc_matrix" in result or "hubs" in result


class TestNetworkMotifs:
    def test_enumerate(self, synthetic_data):
        result = enumerate_motifs(synthetic_data)
        assert isinstance(result, dict)
        assert "motifs" in result or "motif_counts" in result


class TestEnergyLandscape:
    def test_fit_ising(self, synthetic_data):
        result = fit_ising_model(synthetic_data)
        assert isinstance(result, dict)
        assert "bias_terms" in result or "coupling_matrix" in result or "energy_landscape" in result


class TestConsciousnessMetrics:
    def test_score(self, synthetic_data):
        result = compute_consciousness_score(synthetic_data)
        assert isinstance(result, dict)
        assert "consciousness_score" in result or "composite_score" in result or "dimensions" in result


class TestComparative:
    def test_compare(self, synthetic_data):
        result = compare_with_references(synthetic_data)
        assert isinstance(result, dict)
        assert "comparisons" in result or "most_similar" in result or "organoid_stats" in result


class TestProtocolLibrary:
    def test_list(self):
        result = list_protocols()
        assert isinstance(result, dict)
        assert "protocols" in result or "count" in result or len(result) > 0

    def test_get(self):
        result = get_protocol("dishbrain_pong")
        assert isinstance(result, dict)
        assert "name" in result or "description" in result or "encoding" in result


class TestCatastrophicForgetting:
    def test_retention_curve(self, synthetic_data):
        result = compute_retention_curve(synthetic_data)
        assert isinstance(result, dict)
        assert "retention_scores" in result or "forgetting_index" in result or "windows" in result


class TestTransferLearning:
    def test_measure(self, synthetic_data):
        result = measure_transfer(synthetic_data)
        assert isinstance(result, dict)
        assert "has_positive_transfer" in result or "max_transfer_gain" in result


class TestConsolidation:
    def test_detect(self, synthetic_data):
        result = detect_consolidation_events(synthetic_data)
        assert isinstance(result, dict)
        assert "events" in result or "consolidation_detected" in result or "replay_events" in result


class TestMultiBitMemory:
    def test_channel_capacity(self, synthetic_data):
        result = estimate_channel_capacity(synthetic_data)
        assert isinstance(result, dict)
        assert "channel_capacity_bits" in result or "mutual_information_bits" in result

    def test_population_code_diversity(self, synthetic_data):
        result = measure_population_code_diversity(synthetic_data)
        assert isinstance(result, dict)
        assert "n_unique_states" in result or "capacity_bits" in result or "state_entropy" in result


class TestFunctionalConnectome:
    def test_build(self, synthetic_data):
        result = build_full_connectome(synthetic_data)
        assert isinstance(result, dict)
        assert "correlation_matrix" in result or "thresholded_matrix" in result or "electrode_ids" in result

    def test_communities(self, synthetic_data):
        result = detect_communities(synthetic_data)
        assert isinstance(result, dict)
        assert "communities" in result or "modularity" in result or "n_communities" in result

    def test_graph_metrics(self, synthetic_data):
        result = compute_graph_theory_metrics(synthetic_data)
        assert isinstance(result, dict)
        assert "small_world_sigma" in result or "clustering_coefficient" in result or "global_efficiency" in result


class TestEffectiveConnectivity:
    def test_estimate(self, synthetic_data):
        result = estimate_effective_connectivity(synthetic_data)
        assert isinstance(result, dict)
        assert "ec_matrix" in result or "significant_connections" in result or "electrode_ids" in result

    def test_causal_hierarchy(self, synthetic_data):
        result = compute_causal_hierarchy(synthetic_data)
        assert isinstance(result, dict)
        assert "hierarchy_levels" in result or "driver_nodes" in result or "flow_direction" in result


class TestTopology:
    def test_betti_numbers(self, synthetic_data):
        result = compute_betti_numbers(synthetic_data)
        assert isinstance(result, dict)
        assert "betti_curves" in result or "max_betti" in result or "mean_betti" in result

    def test_topological_complexity(self, synthetic_data):
        result = compute_topological_complexity(synthetic_data)
        assert isinstance(result, dict)
        assert "complexity_score" in result or "betti_summary" in result or "effective_dimension" in result


class TestTuringTest:
    def test_run(self, synthetic_data):
        result = run_turing_test(synthetic_data)
        assert isinstance(result, dict)
        assert "verdict" in result or "score" in result or "classification" in result


class TestNeuralArchitectureSearch:
    def test_search(self, synthetic_data):
        result = search_optimal_protocol(synthetic_data, generations=3, population_size=5)
        assert isinstance(result, dict)
        assert "best" in result or "best_architecture" in result or "fitness" in result or "generations" in result


class TestHybridAI:
    def test_benchmark(self, synthetic_data):
        result = benchmark_hybrid(synthetic_data)
        assert isinstance(result, dict)
        assert "accuracies" in result or "hybrid_advantage" in result or "biological_contribution" in result


class TestGeneticProgramming:
    def test_evolve(self, synthetic_data):
        result = evolve_programs(synthetic_data, generations=5, population_size=5)
        assert isinstance(result, dict)
        assert "best" in result or "best_program" in result or "fitness" in result


class TestHomeostaticPlasticity:
    def test_monitor(self, synthetic_data):
        result = monitor_homeostasis(synthetic_data)
        assert isinstance(result, dict)
        assert "homeostasis_active" in result or "mean_cv_over_time" in result


class TestSufferingDetector:
    def test_detect(self, synthetic_data):
        result = detect_suffering(synthetic_data)
        assert isinstance(result, dict)
        assert "suffering_detected" in result or "risk_level" in result or "alerts" in result


class TestWelfareReport:
    def test_generate(self, synthetic_data):
        result = generate_welfare_report(synthetic_data)
        assert isinstance(result, dict)
        assert "welfare" in result or "overall_status" in result or "report" in result or "health" in result


class TestSwarmOrganoid:
    def test_simulate(self, synthetic_data):
        result = simulate_swarm(synthetic_data)
        assert isinstance(result, dict)
        assert "swarm" in result or "collective_performance" in result or "n_organoids" in result


class TestMorphologicalComputing:
    def test_analyze(self, synthetic_data):
        result = analyze_morphological_computation(synthetic_data)
        assert isinstance(result, dict)
        assert "spatial_entropy" in result or "electrode_coverage" in result or "morphological_computation_index" in result
