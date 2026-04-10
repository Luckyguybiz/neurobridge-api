"""Morphological computing analysis.

Scientific basis:
    The physical structure of a neural network influences its computation.
    Organoid size, electrode coverage, and spatial activity patterns
    determine what computations are possible. This module measures how
    morphology (structure) relates to function (computation).
"""
import numpy as np
from .loader import SpikeData


def analyze_morphological_computation(data: SpikeData) -> dict:
    """Analyze how organoid structure relates to computational ability."""
    # Spatial activity distribution
    electrode_rates = {}
    for eid in data.electrode_ids:
        mask = data.electrodes == eid
        electrode_rates[int(eid)] = float(np.sum(mask)) / max(data.duration, 0.001)

    rates = list(electrode_rates.values())
    total_rate = sum(rates)

    # Spatial entropy (how evenly distributed is activity?)
    if total_rate > 0:
        probs = [r / total_rate for r in rates if r > 0]
        spatial_entropy = float(-sum(p * np.log2(p) for p in probs))
        max_entropy = np.log2(max(len(rates), 1))
        spatial_uniformity = spatial_entropy / max(max_entropy, 1e-10)
    else:
        spatial_entropy = 0.0
        spatial_uniformity = 0.0

    # Active electrode fraction
    active_fraction = float(sum(1 for r in rates if r > 0.1) / max(len(rates), 1))

    # Spatial clustering (do active electrodes cluster together?)
    active_ids = [eid for eid, r in electrode_rates.items() if r > 0.5]
    if len(active_ids) > 1:
        # Simple clustering: mean distance between active electrode indices
        indices = [data.electrode_ids.index(eid) for eid in active_ids if eid in data.electrode_ids]
        if len(indices) > 1:
            diffs = np.diff(sorted(indices))
            clustering = float(1.0 / (1.0 + np.mean(diffs)))
        else:
            clustering = 0.0
    else:
        clustering = 0.0

    # Rate heterogeneity (CV of rates)
    rate_cv = float(np.std(rates) / max(np.mean(rates), 0.01)) if rates else 0.0

    # Morphological computing score
    # High score = spatially distributed, active, heterogeneous (= more computational capacity)
    morph_score = (spatial_uniformity * 0.3 + active_fraction * 0.3 + rate_cv * 0.2 + clustering * 0.2)
    morph_score = min(1.0, morph_score)

    return {
        "morphological_score": float(morph_score),
        "spatial_entropy": spatial_entropy,
        "spatial_uniformity": spatial_uniformity,
        "active_electrode_fraction": active_fraction,
        "n_active_electrodes": int(sum(1 for r in rates if r > 0.1)),
        "n_total_electrodes": data.n_electrodes,
        "spatial_clustering": clustering,
        "rate_heterogeneity_cv": rate_cv,
        "electrode_rates": electrode_rates,
        "interpretation": (
            "High morphological complexity — distributed, heterogeneous activity"
            if morph_score > 0.6
            else "Moderate — partially distributed activity"
            if morph_score > 0.3
            else "Low — activity concentrated in few electrodes"
        ),
    }
