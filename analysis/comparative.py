"""Comparative analysis — how does this organoid compare to known neural systems?

Reference statistics from: cortical slice, C. elegans, fruit fly, mouse hippocampus.
"""
import numpy as np
from .loader import SpikeData
from .stats import compute_full_summary

# Reference neural system statistics (from literature)
REFERENCE_SYSTEMS = {
    "cortical_slice": {
        "firing_rate_hz": 5.0,
        "burst_rate_per_min": 3.0,
        "cv_isi": 1.2,
        "synchrony": 0.4,
        "description": "Mouse cortical slice (in vitro)",
    },
    "c_elegans": {
        "firing_rate_hz": 0.5,
        "burst_rate_per_min": 0.5,
        "cv_isi": 2.0,
        "synchrony": 0.2,
        "description": "C. elegans (302 neurons)",
    },
    "fruit_fly": {
        "firing_rate_hz": 8.0,
        "burst_rate_per_min": 5.0,
        "cv_isi": 1.0,
        "synchrony": 0.5,
        "description": "Drosophila mushroom body",
    },
    "mouse_hippocampus": {
        "firing_rate_hz": 2.0,
        "burst_rate_per_min": 8.0,
        "cv_isi": 1.5,
        "synchrony": 0.6,
        "description": "Mouse hippocampal CA1",
    },
    "organoid_typical": {
        "firing_rate_hz": 3.0,
        "burst_rate_per_min": 2.0,
        "cv_isi": 1.3,
        "synchrony": 0.3,
        "description": "Typical cortical organoid (3-6 months)",
    },
}

def compare_with_references(data: SpikeData) -> dict:
    """Compare organoid with reference neural systems."""
    summary = compute_full_summary(data)

    pop = summary.get("population", {})
    firing_rate = float(pop.get("mean_firing_rate_hz", 0))

    # Compute ISI CV
    from .spikes import compute_isi
    isi = compute_isi(data)
    cv_values = []
    for k, v in isi.items():
        if isinstance(v, dict) and "cv" in v:
            cv_values.append(v["cv"])
    cv_isi = float(np.mean(cv_values)) if cv_values else 1.0

    organoid_stats = {
        "firing_rate_hz": firing_rate,
        "cv_isi": cv_isi,
    }

    similarities = {}
    for name, ref in REFERENCE_SYSTEMS.items():
        # Euclidean distance in normalized feature space
        diff_rate = (organoid_stats["firing_rate_hz"] - ref["firing_rate_hz"]) / max(ref["firing_rate_hz"], 0.1)
        diff_cv = (organoid_stats["cv_isi"] - ref["cv_isi"]) / max(ref["cv_isi"], 0.1)
        distance = np.sqrt(diff_rate**2 + diff_cv**2)
        similarity = float(1.0 / (1.0 + distance))

        similarities[name] = {
            "similarity": similarity,
            "description": ref["description"],
            "distance": float(distance),
        }

    # Find most similar
    most_similar = max(similarities.items(), key=lambda x: x[1]["similarity"])

    return {
        "organoid_stats": organoid_stats,
        "similarities": similarities,
        "most_similar_system": most_similar[0],
        "most_similar_score": most_similar[1]["similarity"],
        "most_similar_description": most_similar[1]["description"],
    }
