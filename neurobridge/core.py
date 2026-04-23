"""Core API for the neurobridge package.

Provides simple high-level functions:
    load()              — Load spike data from file
    generate_synthetic() — Generate synthetic test data
    analyze()           — Run a specific analysis
    full_report()       — Run all analyses at once
"""

import numpy as np

from analysis.loader import SpikeData, load_file


def load(filepath: str, sampling_rate: float = 30000.0) -> SpikeData:
    """Load spike data from a file.

    Supported formats: CSV, HDF5, Parquet, JSON, NWB.

    Args:
        filepath: Path to the data file.
        sampling_rate: Sampling rate in Hz (default: 30000 for FinalSpark).

    Returns:
        SpikeData object with times, electrodes, amplitudes, and optional waveforms.

    Example:
        >>> data = nb.load("my_recording.csv")
        >>> print(f"{data.n_spikes} spikes, {data.n_electrodes} electrodes")
    """
    return load_file(filepath, sampling_rate=sampling_rate)


def generate_synthetic(
    duration: float = 30.0,
    n_electrodes: int = 8,
    base_rate: float = 10.0,
    burst_probability: float = 0.15,
    seed: int = None,
) -> SpikeData:
    """Generate synthetic spike data for testing.

    Args:
        duration: Recording duration in seconds.
        n_electrodes: Number of electrodes.
        base_rate: Base firing rate per electrode (Hz).
        burst_probability: Probability of burst events.
        seed: Random seed for reproducibility.

    Returns:
        SpikeData with synthetic spikes.

    Example:
        >>> data = nb.generate_synthetic(duration=60, n_electrodes=8)
    """
    if seed is not None:
        np.random.seed(seed)

    all_times = []
    all_electrodes = []
    all_amplitudes = []

    for e in range(n_electrodes):
        rate = base_rate * np.random.uniform(0.5, 1.5)
        n_spikes = int(rate * duration)
        times = np.sort(np.random.uniform(0, duration, n_spikes))

        # Add bursts
        n_bursts = int(duration * burst_probability)
        for _ in range(n_bursts):
            burst_center = np.random.uniform(0, duration)
            burst_spikes = np.random.normal(burst_center, 0.01, np.random.randint(5, 20))
            burst_spikes = burst_spikes[(burst_spikes >= 0) & (burst_spikes <= duration)]
            times = np.sort(np.concatenate([times, burst_spikes]))

        all_times.extend(times)
        all_electrodes.extend([e] * len(times))
        all_amplitudes.extend(np.random.normal(-50, 15, len(times)))

    return SpikeData(
        times=np.array(all_times),
        electrodes=np.array(all_electrodes),
        amplitudes=np.array(all_amplitudes),
        sampling_rate=30000.0,
    )


# Analysis module registry
_ANALYSES = {
    "summary": ("analysis.stats", "compute_full_summary"),
    "quality": ("analysis.stats", "compute_quality_metrics"),
    "firing_rates": ("analysis.spikes", "compute_firing_rates"),
    "isi": ("analysis.spikes", "compute_isi"),
    "amplitudes": ("analysis.spikes", "compute_amplitude_stats"),
    "bursts": ("analysis.bursts", "detect_bursts"),
    "connectivity": ("analysis.connectivity", "compute_connectivity_graph"),
    "cross_correlation": ("analysis.connectivity", "compute_cross_correlation"),
    "transfer_entropy": ("analysis.connectivity", "compute_transfer_entropy"),
    "entropy": ("analysis.information_theory", "compute_spike_train_entropy"),
    "mutual_information": ("analysis.information_theory", "compute_mutual_information"),
    "complexity": ("analysis.information_theory", "compute_lempel_ziv_complexity"),
    "power_spectrum": ("analysis.spectral", "compute_power_spectrum"),
    "coherence": ("analysis.spectral", "compute_coherence"),
    "avalanches": ("analysis.criticality", "detect_avalanches"),
    "digital_twin": ("analysis.digital_twin", "fit_lif_parameters"),
    "anomalies": ("analysis.ml_pipeline", "detect_anomalies"),
    "states": ("analysis.ml_pipeline", "classify_states"),
    "pca": ("analysis.ml_pipeline", "compute_pca_embedding"),
    "features": ("analysis.ml_pipeline", "extract_features"),
    "stdp": ("analysis.plasticity", "compute_stdp_matrix"),
    "learning": ("analysis.plasticity", "detect_learning_episodes"),
    "organoid_iq": ("analysis.organoid_iq", "compute_organoid_iq"),
    "predict_rates": ("analysis.predictions", "predict_firing_rates"),
    "predict_bursts": ("analysis.predictions", "predict_burst_probability"),
    "health": ("analysis.predictions", "estimate_organoid_health"),
    "replay": ("analysis.replay", "detect_replay"),
    "sequences": ("analysis.replay", "detect_sequence_replay"),
    "memory_capacity": ("analysis.reservoir", "estimate_memory_capacity"),
    "nonlinearity": ("analysis.reservoir", "benchmark_nonlinear_computation"),
    "fingerprint": ("analysis.fingerprint", "compute_fingerprint"),
    "emergence": ("analysis.emergence", "compute_integrated_information"),
    "attractors": ("analysis.attractors", "map_attractor_landscape"),
    "state_space": ("analysis.attractors", "compute_state_space_geometry"),
    "phase_transitions": ("analysis.phase_transitions", "detect_phase_transitions"),
    "predictive_coding": ("analysis.predictive_coding", "measure_predictive_coding"),
    "weights": ("analysis.weight_inference", "infer_synaptic_weights"),
    "weight_tracking": ("analysis.weight_inference", "track_weight_changes"),
    "multiscale": ("analysis.multiscale", "compute_multiscale_complexity"),
    "sleep_wake": ("analysis.sleep_wake", "analyze_sleep_wake"),
    "habituation": ("analysis.habituation", "detect_repeated_patterns"),
    "metastability": ("analysis.metastability", "analyze_metastability"),
    "information_flow": ("analysis.information_flow", "compute_granger_causality"),
    "motifs": ("analysis.network_motifs", "enumerate_motifs"),
    "energy_landscape": ("analysis.energy_landscape", "fit_ising_model"),
    "consciousness": ("analysis.consciousness_metrics", "compute_consciousness_score"),
    "comparative": ("analysis.comparative", "compare_with_references"),
    "turing_test": ("analysis.turing_test", "run_turing_test"),
    "hybrid": ("analysis.hybrid_ai", "benchmark_hybrid"),
    "forgetting": ("analysis.catastrophic_forgetting", "compute_retention_curve"),
    "transfer": ("analysis.transfer_learning", "measure_transfer"),
    "consolidation": ("analysis.consolidation", "detect_consolidation_events"),
    "topology": ("analysis.topology", "compute_betti_numbers"),
    "connectome": ("analysis.functional_connectome", "build_full_connectome"),
    "homeostasis": ("analysis.homeostatic_plasticity", "monitor_homeostasis"),
    "suffering": ("analysis.suffering_detector", "detect_suffering"),
    "welfare": ("analysis.welfare_report", "generate_welfare_report"),
    "swarm": ("analysis.swarm_organoid", "simulate_swarm"),
    "morphology": ("analysis.morphological_computing", "analyze_morphological_computation"),
}


def analyze(data: SpikeData, analysis: str, **kwargs) -> dict:
    """Run a specific analysis on spike data.

    Args:
        data: SpikeData object.
        analysis: Name of analysis (e.g., "organoid_iq", "bursts", "connectivity").
        **kwargs: Additional parameters for the analysis function.

    Returns:
        Dict with analysis results.

    Available analyses:
        summary, quality, firing_rates, isi, amplitudes, bursts,
        connectivity, cross_correlation, transfer_entropy, entropy,
        mutual_information, complexity, power_spectrum, coherence,
        avalanches, digital_twin, anomalies, states, pca, features,
        stdp, learning, organoid_iq, predict_rates, predict_bursts,
        health, replay, sequences, memory_capacity, nonlinearity,
        fingerprint, emergence, attractors, state_space, phase_transitions,
        predictive_coding, weights, weight_tracking, multiscale,
        sleep_wake, habituation, metastability, information_flow,
        motifs, energy_landscape, consciousness, comparative,
        turing_test, hybrid, forgetting, transfer, consolidation,
        topology, connectome, homeostasis, suffering, welfare,
        swarm, morphology

    Example:
        >>> result = nb.analyze(data, "organoid_iq")
        >>> print(result["iq_score"])
    """
    if analysis not in _ANALYSES:
        available = ", ".join(sorted(_ANALYSES.keys()))
        raise ValueError(f"Unknown analysis '{analysis}'. Available: {available}")

    module_path, func_name = _ANALYSES[analysis]
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func(data, **kwargs)


def full_report(data: SpikeData) -> dict:
    """Run all analyses and return a comprehensive report.

    Args:
        data: SpikeData object.

    Returns:
        Dict with all analysis results keyed by analysis name.

    Example:
        >>> report = nb.full_report(data)
        >>> print(report["organoid_iq"]["iq_score"])
        >>> print(report["emergence"]["phi"])
    """
    from analysis.report import generate_full_report
    return generate_full_report(data)


def list_analyses() -> list[str]:
    """List all available analysis names."""
    return sorted(_ANALYSES.keys())
