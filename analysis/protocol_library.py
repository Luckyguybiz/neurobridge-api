"""Protocol library — standardized stimulation protocols.

JSON library of standard and experimental protocols for biocomputing research.
Based on DishBrain (Kagan 2022), Brainoware (Cai 2023), UCSC cart-pole (2026).
"""

PROTOCOLS = {
    "dishbrain_pong": {
        "name": "DishBrain Pong Protocol",
        "description": "Free energy principle-based Pong training (Kagan et al. 2022)",
        "reference": "Kagan et al., Neuron, 2022",
        "encoding": {
            "method": "frequency_modulation",
            "input_electrodes": [0, 1, 2, 3],
            "frequency_range_hz": [4, 40],
            "variable": "ball_position",
        },
        "decoding": {
            "method": "spike_rate_comparison",
            "output_electrodes": [4, 5],
            "decision": "higher_rate_wins",
        },
        "reward": {
            "hit": "predictable_stimulation",
            "miss": "random_stimulation",
            "principle": "free_energy_minimization",
        },
        "parameters": {
            "trial_duration_ms": 500,
            "inter_trial_interval_ms": 1000,
            "training_trials": 500,
            "test_trials": 100,
        },
    },
    "stdp_training": {
        "name": "STDP Paired-Pulse Protocol",
        "description": "Map spike-timing dependent plasticity curves",
        "reference": "Bi & Poo, J Neuroscience, 1998",
        "parameters": {
            "pre_electrode": 0,
            "post_electrode": 1,
            "delays_ms": list(range(-50, 51, 5)),
            "n_repetitions": 60,
            "pulse_amplitude_uv": 50,
            "pulse_width_ms": 0.5,
            "inter_pair_interval_ms": 5000,
        },
    },
    "theta_burst": {
        "name": "Theta Burst Stimulation",
        "description": "Induce LTP/LTD via theta-burst patterns (Johns Hopkins protocol)",
        "reference": "Smirnova et al., Frontiers, 2023",
        "parameters": {
            "burst_frequency_hz": 100,
            "burst_duration_ms": 50,
            "inter_burst_interval_ms": 200,
            "n_bursts_per_train": 10,
            "n_trains": 3,
            "inter_train_interval_s": 10,
            "electrode": 0,
            "amplitude_uv": 50,
        },
    },
    "sleep_wake_induction": {
        "name": "Sleep-Wake Cycle Induction",
        "description": "Alternate stimulation and rest to promote memory consolidation",
        "parameters": {
            "training_phase_min": 30,
            "rest_phase_min": 120,
            "n_cycles": 3,
            "training_frequency_hz": 5,
            "rest_frequency_hz": 0,
            "test_after_each_cycle": True,
        },
    },
    "reservoir_training": {
        "name": "Reservoir Computing Training",
        "description": "Random input stimulation for reservoir computing benchmark",
        "reference": "Cai et al., Nature Electronics, 2023",
        "parameters": {
            "input_electrodes": [0, 1],
            "readout_electrodes": [2, 3, 4, 5, 6, 7],
            "input_type": "random_binary",
            "stimulus_duration_ms": 200,
            "inter_stimulus_ms": 300,
            "n_training_samples": 200,
            "n_test_samples": 40,
            "readout_method": "linear_regression",
        },
    },
    "curriculum_basic": {
        "name": "Basic Curriculum Learning",
        "description": "Gradually increasing complexity stimulation",
        "parameters": {
            "stages": [
                {"name": "familiarization", "frequency_hz": 0.5, "electrodes": 1, "duration_days": 3},
                {"name": "simple_patterns", "frequency_hz": 2, "electrodes": 2, "duration_days": 4},
                {"name": "complex_patterns", "frequency_hz": 5, "electrodes": 4, "duration_days": 7},
                {"name": "task_training", "frequency_hz": 10, "electrodes": 8, "duration_days": 14},
            ],
            "advancement_criterion": "burst_synchrony_increase",
        },
    },
}

def list_protocols() -> dict:
    """List all available protocols with metadata."""
    return {
        "protocols": {
            name: {
                "name": p["name"],
                "description": p["description"],
                "reference": p.get("reference", "NeuroBridge original"),
            }
            for name, p in PROTOCOLS.items()
        },
        "n_protocols": len(PROTOCOLS),
    }

def get_protocol(name: str) -> dict:
    """Get full protocol details by name."""
    if name not in PROTOCOLS:
        return {"error": f"Protocol '{name}' not found", "available": list(PROTOCOLS.keys())}
    return PROTOCOLS[name]

def suggest_protocol(data) -> dict:
    """Suggest best protocol based on current organoid state."""
    # Simple heuristic based on firing rate
    if hasattr(data, 'times') and len(data.times) > 0:
        rate = len(data.times) / max(data.duration, 0.001)
        if rate < 5:
            suggestion = "curriculum_basic"
            reason = "Low firing rate suggests immature network - start with curriculum"
        elif rate < 20:
            suggestion = "stdp_training"
            reason = "Moderate activity - good candidate for plasticity mapping"
        else:
            suggestion = "dishbrain_pong"
            reason = "High activity level - ready for complex task training"
    else:
        suggestion = "curriculum_basic"
        reason = "No data available - start with basic curriculum"

    return {
        "suggested_protocol": suggestion,
        "reason": reason,
        "protocol": PROTOCOLS[suggestion],
    }
