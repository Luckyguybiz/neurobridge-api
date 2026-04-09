"""Digital Twin module — Leaky Integrate-and-Fire (LIF) neuron model.

Builds a computational model of the organoid based on recorded data:
- Fits LIF parameters to match observed firing statistics
- Simulates network activity
- Predicts response to stimulation protocols
- A/B tests protocols on the twin before real experiments
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


def fit_lif_parameters(data: SpikeData) -> dict:
    """Fit LIF neuron model parameters from observed spike data.

    For each electrode, estimates:
    - tau_m: membrane time constant (ms)
    - v_rest: resting potential (mV)
    - v_threshold: spike threshold (mV)
    - v_reset: reset potential after spike (mV)
    - tau_ref: refractory period (ms)
    - i_background: background current (nA)
    """
    params = {}

    for e in data.electrode_ids:
        e_times = data.times[data.electrodes == e]
        e_amps = data.amplitudes[data.electrodes == e]

        if len(e_times) < 10:
            continue

        # Estimate from ISI distribution
        isi = np.diff(e_times) * 1000  # ms
        isi = isi[isi > 0]

        mean_isi = float(np.mean(isi))
        min_isi = float(np.min(isi)) if len(isi) > 0 else 2.0
        cv_isi = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 1.0

        # LIF parameter estimation (Brunel & Sergi, 1998 method)
        # tau_ref from minimum ISI
        tau_ref = max(1.0, min_isi * 0.8)

        # tau_m from CV of ISI (higher CV = longer tau_m relative to mean ISI)
        tau_m = mean_isi * 0.3 * (1 + cv_isi)

        # Background current from mean firing rate
        firing_rate = len(e_times) / data.duration if data.duration > 0 else 0
        i_background = 0.5 + firing_rate * 0.05  # nA, rough estimate

        # Amplitude statistics for threshold estimation
        mean_amp = float(np.mean(np.abs(e_amps))) if len(e_amps) > 0 else 100.0
        v_threshold = -40 - mean_amp * 0.05  # mV

        params[int(e)] = {
            "tau_m": round(tau_m, 2),
            "tau_ref": round(tau_ref, 2),
            "v_rest": -65.0,
            "v_threshold": round(v_threshold, 2),
            "v_reset": -70.0,
            "i_background": round(i_background, 3),
            "firing_rate_hz": round(firing_rate, 3),
            "mean_isi_ms": round(mean_isi, 2),
            "cv_isi": round(cv_isi, 3),
        }

    return {"parameters": params, "n_neurons": len(params), "model": "LIF"}


def simulate_lif_network(
    params: dict,
    duration_ms: float = 5000.0,
    dt: float = 0.1,
    connectivity_matrix: Optional[np.ndarray] = None,
    stimulus: Optional[dict] = None,
) -> dict:
    """Simulate LIF neural network.

    Args:
        params: Per-neuron LIF parameters from fit_lif_parameters
        duration_ms: Simulation duration in ms
        dt: Time step in ms
        connectivity_matrix: NxN weight matrix (None = no connections)
        stimulus: {"electrode": int, "times_ms": [...], "current_nA": float}
    """
    neuron_params = params.get("parameters", params)
    neuron_ids = sorted(neuron_params.keys(), key=int)
    n = len(neuron_ids)

    if n == 0:
        return {"error": "No neurons to simulate"}

    n_steps = int(duration_ms / dt)

    # State arrays
    v = np.full(n, -65.0)  # membrane potential
    refractory = np.zeros(n)  # refractory countdown

    # Record spikes
    spike_times = []
    spike_neurons = []

    # Optional connectivity
    weights = connectivity_matrix if connectivity_matrix is not None else np.zeros((n, n))
    syn_current = np.zeros(n)
    tau_syn = 5.0  # synaptic time constant ms

    # Stimulus
    stim_current = np.zeros(n)

    for step in range(n_steps):
        t = step * dt

        # Reset stimulus
        stim_current[:] = 0

        # Apply external stimulus
        if stimulus:
            stim_idx = neuron_ids.index(stimulus["electrode"]) if stimulus["electrode"] in neuron_ids else -1
            if stim_idx >= 0:
                for stim_t in stimulus.get("times_ms", []):
                    if abs(t - stim_t) < dt:
                        stim_current[stim_idx] = stimulus.get("current_nA", 1.0)

        for i, nid in enumerate(neuron_ids):
            p = neuron_params[nid]

            if refractory[i] > 0:
                refractory[i] -= dt
                v[i] = p["v_reset"]
                continue

            # LIF dynamics: tau_m * dV/dt = -(V - V_rest) + R * I
            I_total = p["i_background"] + syn_current[i] + stim_current[i]
            I_total += np.random.randn() * 0.1  # noise

            dv = (-(v[i] - p["v_rest"]) + 10.0 * I_total) * dt / p["tau_m"]
            v[i] += dv

            # Spike?
            if v[i] >= p["v_threshold"]:
                spike_times.append(round(t, 2))
                spike_neurons.append(int(nid))
                v[i] = p["v_reset"]
                refractory[i] = p["tau_ref"]

                # Propagate to connected neurons
                for j in range(n):
                    if weights[i, j] != 0:
                        syn_current[j] += weights[i, j]

        # Synaptic current decay
        syn_current *= np.exp(-dt / tau_syn)

    # Statistics
    sim_spikes = {}
    for nid in neuron_ids:
        nid_int = int(nid)
        n_spikes = sum(1 for sn in spike_neurons if sn == nid_int)
        sim_spikes[nid_int] = {
            "n_spikes": n_spikes,
            "firing_rate_hz": n_spikes / (duration_ms / 1000),
        }

    return {
        "spike_times": spike_times,
        "spike_neurons": spike_neurons,
        "n_total_spikes": len(spike_times),
        "duration_ms": duration_ms,
        "per_neuron": sim_spikes,
        "model": "LIF",
        "dt_ms": dt,
    }


def compare_real_vs_simulated(data: SpikeData, sim_result: dict) -> dict:
    """Compare real organoid data with digital twin simulation."""
    comparison = {}

    for e in data.electrode_ids:
        real_n = int(np.sum(data.electrodes == e))
        real_rate = real_n / data.duration if data.duration > 0 else 0

        sim_info = sim_result.get("per_neuron", {}).get(e, {})
        sim_rate = sim_info.get("firing_rate_hz", 0)

        error = abs(real_rate - sim_rate) / real_rate * 100 if real_rate > 0 else 0

        comparison[int(e)] = {
            "real_rate_hz": round(real_rate, 3),
            "simulated_rate_hz": round(sim_rate, 3),
            "error_pct": round(error, 1),
        }

    errors = [c["error_pct"] for c in comparison.values()]
    return {
        "comparison": comparison,
        "mean_error_pct": round(float(np.mean(errors)), 1) if errors else 0,
        "max_error_pct": round(float(np.max(errors)), 1) if errors else 0,
        "quality": "good" if np.mean(errors) < 20 else "needs_tuning" if np.mean(errors) < 50 else "poor",
    }
