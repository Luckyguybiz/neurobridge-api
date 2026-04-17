"""Full analysis report generator.

Runs ALL available analyses on a dataset and produces a comprehensive JSON report.
This is the "one-click analyze everything" endpoint.
"""

import time
from .loader import SpikeData
from . import spikes, bursts, connectivity, stats, information_theory, spectral, criticality, ml_pipeline, digital_twin


def generate_full_report(data: SpikeData) -> dict:
    """Run every analysis module and compile into one report.

    This is the ultimate analysis — every metric, every perspective,
    every insight that can be extracted from the data.
    """
    t0 = time.time()
    report = {"_meta": {"started_at": time.time()}}
    errors = []

    # 1. Summary statistics
    try:
        report["summary"] = stats.compute_full_summary(data)
    except Exception as e:
        errors.append(f"summary: {e}")

    # 2. Quality metrics
    try:
        report["quality"] = stats.compute_quality_metrics(data)
    except Exception as e:
        errors.append(f"quality: {e}")

    # 3. Firing rates
    try:
        report["firing_rates"] = spikes.compute_firing_rates(data, bin_size_sec=1.0)
    except Exception as e:
        errors.append(f"firing_rates: {e}")

    # 4. ISI analysis
    try:
        report["isi"] = spikes.compute_isi(data)
    except Exception as e:
        errors.append(f"isi: {e}")

    # 5. Amplitude statistics
    try:
        report["amplitudes"] = spikes.compute_amplitude_stats(data)
    except Exception as e:
        errors.append(f"amplitudes: {e}")

    # 6. Temporal dynamics
    try:
        bin_size = max(1.0, data.duration / 30)
        report["temporal_dynamics"] = stats.compute_temporal_dynamics(data, bin_size_sec=bin_size)
    except Exception as e:
        errors.append(f"temporal: {e}")

    # 7. Network burst detection
    try:
        report["bursts"] = bursts.analyze_bursts(data)
    except Exception as e:
        errors.append(f"bursts: {e}")

    # 8. Single-channel bursts
    try:
        sc_bursts = {}
        for e in data.electrode_ids:
            e_times = data.times[data.electrodes == e]
            sc_bursts[int(e)] = bursts.detect_single_channel_bursts(e_times)
        report["single_channel_bursts"] = {
            "per_electrode": {e: {"n_bursts": len(b)} for e, b in sc_bursts.items()},
        }
    except Exception as e:
        errors.append(f"sc_bursts: {e}")

    # 9. Functional connectivity
    try:
        conn = connectivity.compute_connectivity_graph(data)
        report["connectivity"] = connectivity.connectivity_to_dict(conn)
    except Exception as e:
        errors.append(f"connectivity: {e}")

    # 10. Cross-correlation
    try:
        ccg = connectivity.compute_cross_correlation(data)
        report["cross_correlation"] = ccg.to_dict() if hasattr(ccg, 'to_dict') else ccg
    except Exception as e:
        errors.append(f"cross_corr: {e}")

    # 11. Transfer entropy
    try:
        te = connectivity.compute_transfer_entropy(data)
        report["transfer_entropy"] = te.to_dict() if hasattr(te, 'to_dict') else te
    except Exception as e:
        errors.append(f"transfer_entropy: {e}")

    # 12. Information theory
    try:
        report["entropy"] = information_theory.compute_spike_train_entropy(data)
        if "per_electrode" in report["entropy"]:
            for e in report["entropy"]["per_electrode"]:
                report["entropy"]["per_electrode"][e].pop("n_unique_words", None)
    except Exception as e:
        errors.append(f"entropy: {e}")

    # 13. Mutual information
    try:
        mi = information_theory.compute_mutual_information(data)
        mi.pop("mi_matrix", None)  # compact
        report["mutual_information"] = mi
    except Exception as e:
        errors.append(f"mutual_info: {e}")

    # 14. Lempel-Ziv complexity
    try:
        report["complexity"] = information_theory.compute_lempel_ziv_complexity(data)
    except Exception as e:
        errors.append(f"complexity: {e}")

    # 15. Power spectrum
    try:
        psd = spectral.compute_power_spectrum(data)
        for e in psd.get("per_electrode", {}):
            psd["per_electrode"][e].pop("frequencies", None)
            psd["per_electrode"][e].pop("psd", None)
        report["spectral"] = psd
    except Exception as e:
        errors.append(f"spectral: {e}")

    # 16. Coherence
    try:
        report["coherence"] = spectral.compute_coherence(data)
    except Exception as e:
        errors.append(f"coherence: {e}")

    # 17. Criticality / avalanches
    try:
        crit = criticality.analyse_criticality(data)
        crit.pop("avalanches", None)  # compact
        crit.pop("size_distribution", None)
        report["criticality"] = crit
    except Exception as e:
        errors.append(f"criticality: {e}")

    # 18. Digital twin parameters
    try:
        report["digital_twin"] = digital_twin.fit_lif_parameters(data)
    except Exception as e:
        errors.append(f"digital_twin: {e}")

    # 19. State classification
    try:
        states = ml_pipeline.classify_states(data)
        states.pop("timeline", None)  # compact
        report["state_classification"] = states
    except Exception as e:
        errors.append(f"states: {e}")

    # 20. Anomaly detection
    try:
        anomalies = ml_pipeline.detect_anomalies(data)
        anomalies.pop("anomaly_scores", None)  # compact
        report["anomalies"] = anomalies
    except Exception as e:
        errors.append(f"anomalies: {e}")

    # 21. PCA embedding
    try:
        pca = ml_pipeline.compute_pca_embedding(data)
        pca.pop("embedding", None)
        pca.pop("window_times", None)
        report["pca"] = pca
    except Exception as e:
        errors.append(f"pca: {e}")

    total_time = (time.time() - t0) * 1000
    report["_meta"] = {
        "total_computation_ms": round(total_time, 1),
        "n_analyses": 21 - len(errors),
        "n_errors": len(errors),
        "errors": errors,
        "dataset_id": data.metadata.get("source", "unknown"),
        "n_spikes": data.n_spikes,
        "n_electrodes": data.n_electrodes,
        "duration_s": round(data.duration, 3),
    }

    return report
