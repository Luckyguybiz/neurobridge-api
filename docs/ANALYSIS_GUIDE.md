# NeuroBridge Analysis Guide

What each analysis means and how to interpret the results.

---

## Standard Neuroscience Analysis

### Firing Rates (`/firing-rates`)
**What:** Number of spikes per second for each electrode.
**Interpret:** Higher rates = more active neurons near that electrode. Typical organoid rates: 1-20 Hz. Rates > 50 Hz may indicate noise or multi-unit activity.

### Inter-Spike Interval (`/isi`)
**What:** Time between consecutive spikes on each electrode.
**Interpret:**
- **Regular ISI** (low CV) → pacemaker-like neuron
- **Random ISI** (CV ~1.0) → Poisson-like firing (most organoid neurons)
- **Bursty ISI** (CV > 1.5) → neuron fires in bursts with quiet periods
- **Bimodal ISI** → two distinct firing modes

### Spike Sorting (`/spike-sorting`)
**What:** Groups spikes by waveform shape. Each cluster = putatively one neuron.
**Interpret:** More clusters = more distinguishable neurons near the electrode. Well-separated clusters indicate reliable unit isolation.

### Burst Detection (`/bursts`)
**What:** Episodes where multiple electrodes fire together within a short window.
**Interpret:** Network bursts are a hallmark of organoid maturation. More frequent, more synchronized bursts = more network-level organization.

### Connectivity Graph (`/connectivity`)
**What:** Which electrodes fire together. Edges = functional connections, weighted by co-firing frequency.
**Interpret:**
- **High clustering coefficient** → local circuit organization
- **Short path length** → efficient information transmission
- **Hub nodes** → electrodes near highly connected neurons

### Cross-Correlation (`/cross-correlation`)
**What:** Temporal relationship between spike trains of two electrodes.
**Interpret:**
- Peak at lag > 0 → electrode A drives electrode B
- Peak at lag < 0 → electrode B drives electrode A
- Peak at lag = 0 → common input or direct connection
- Flat correlogram → no functional relationship

### Transfer Entropy (`/transfer-entropy`)
**What:** Directional information flow. How much does knowing electrode A's past help predict electrode B's future?
**Interpret:** Higher TE = stronger causal influence. Asymmetric TE between A and B reveals direction of information flow.

---

## Information Theory

### Entropy (`/entropy`)
**What:** Randomness/unpredictability of each electrode's spike train (bits).
**Interpret:** Higher entropy = less predictable. Maximum entropy = random firing. Low entropy = highly regular or stereotyped patterns.

### Mutual Information (`/mutual-information`)
**What:** How much information two electrodes share about each other's activity (bits).
**Interpret:** High MI = electrodes encode redundant information. Zero MI = completely independent.

### Lempel-Ziv Complexity (`/complexity`)
**What:** Compressibility of the spike train. Low complexity = repetitive patterns.
**Interpret:** LZ complexity near 1.0 = random (no structure). Significantly < 1.0 = regular patterns exist. Useful for detecting hidden temporal structure.

---

## Spectral Analysis

### Power Spectrum (`/power-spectrum`)
**What:** Frequency content of binned spike trains.
**Interpret:**
- Peaks at specific frequencies → oscillatory activity
- 1/f spectrum → scale-free dynamics (sign of criticality)
- Flat spectrum → random firing

### Coherence (`/coherence`)
**What:** Frequency-specific correlation between electrode pairs.
**Interpret:** High coherence at a frequency band = electrodes oscillate together at that frequency. Useful for detecting synchronization patterns.

---

## Criticality

### Avalanches (`/avalanches`)
**What:** Cascades of neural activity — one spike triggers more spikes.
**Interpret:**
- **Power-law distributed** sizes → system is near criticality (optimal computation)
- **Branching ratio ~1.0** → critical state
- **Branching ratio > 1.0** → supercritical (epileptic-like activity)
- **Branching ratio < 1.0** → subcritical (dying activity)

---

## Digital Twin

### LIF Model Fit (`/digital-twin/fit`)
**What:** Fits a Leaky Integrate-and-Fire neuron model to reproduce observed firing statistics.
**Interpret:** Good fit (low error) means the network is well-described by simple LIF dynamics. Poor fit suggests more complex mechanisms.

### Simulation (`/digital-twin/simulate`)
**What:** Runs the fitted model and compares output to real data.
**Interpret:** Matching statistics = model captures essential dynamics. Divergence highlights what the model misses.

---

## ML Pipeline

### Feature Extraction (`/features`)
**What:** Computes multi-scale features (rate, burstiness, synchrony, complexity) for each time window.
**Use for:** Input to machine learning models, comparing activity across conditions.

### Anomaly Detection (`/anomalies`)
**What:** Finds unusual time windows using Isolation Forest.
**Interpret:** Anomalous windows may indicate: responses to stimulation, artifacts, state transitions, or biologically interesting events.

### State Classification (`/states`)
**What:** Labels each time window as: resting, active, or bursting.
**Use for:** Understanding the organoid's behavioral repertoire and state transitions.

### PCA Embedding (`/pca`)
**What:** Dimensionality reduction of activity patterns.
**Interpret:** Clusters in PCA space = distinct network states. Trajectories = state transitions.

---

## Novel Analysis (Unique to NeuroBridge)

### STDP Mapping (`/stdp`)
**What:** Spike-Timing-Dependent Plasticity curves — the fundamental learning rule of biological neurons. If neuron A fires before B, the connection strengthens (LTP). If A fires after B, it weakens (LTD).
**Interpret:** Clear STDP curves = evidence of plasticity mechanisms. No STDP = neurons may not be forming directed connections.

### Learning Episodes (`/learning`)
**What:** Detects periods where plasticity patterns change significantly — potential learning events.
**Interpret:** More learning episodes = more dynamic network. Linked to stimulation = evidence of training effect.

### Organoid IQ (`/iq`)
**What:** Composite intelligence score (0-100) across 6 dimensions:
1. **Complexity** — Information-theoretic richness
2. **Synchrony** — Network coordination
3. **Adaptability** — Temporal variability
4. **Integration** — Information sharing across electrodes
5. **Temporal Structure** — Non-random patterns over time
6. **Responsiveness** — Dynamic range of activity

**Interpret:** Score 0-30 = minimal organization (young or unhealthy). 30-60 = developing network. 60-80 = mature, organized. 80-100 = exceptional (unlikely for current organoids).

**Grades:** A (90+), B (75-89), C (50-74), D (25-49), F (0-24)

### Predictions (`/predict/firing-rates`, `/predict/bursts`)
**What:** Forecasts future firing rates and burst probability based on trends.
**Interpret:** Stable predictions = steady-state organoid. Increasing rates = maturing or activated. Decreasing = degrading.

### Health Assessment (`/health`)
**What:** Estimates organoid viability from signal metrics.
**Interpret:** Combines signal-to-noise ratio, activity level, inter-electrode variability, and stability.

### Neural Replay (`/replay`)
**What:** Detects repeated patterns during "rest" that match previously observed sequences — memory consolidation.
**Interpret:** Replay events are strong evidence that the network is storing and reinforcing activity patterns.

### Memory Capacity (`/memory-capacity`)
**What:** Reservoir computing benchmark — how many past time steps can the network "remember"?
**Interpret:** Higher MC = better short-term memory. Typical organoid: MC 0.01-0.1. Digital ESN: MC 0.3-0.5.

### Fingerprint (`/fingerprint`)
**What:** Unique identity hash computed from multi-scale activity features.
**Use for:** Tracking the same organoid over time, comparing organoids, quality control.

### Sonification (`/sonify`)
**What:** Converts spike activity to audio — each electrode mapped to a musical note.
**Use for:** Auditory exploration of activity patterns, presentations, public engagement.

### Emergence / Phi (`/emergence`)
**What:** Integrated Information Theory (IIT) metric — how much information is generated by the system as a whole beyond its parts.
**Interpret:** Higher Phi = more integrated processing. Phi > 0 = system is "more than the sum of its parts."

### Attractor Landscape (`/attractors`)
**What:** Maps network states as points in phase space. Attractors = states the network repeatedly returns to.
**Interpret:**
- **Memory candidates** — attractors that reappear after perturbation
- **Number of attractors** — repertoire of stable states
- **Basin size** — how robust each memory is

### Phase Transitions (`/phase-transitions`)
**What:** Detects moments of sudden network reorganization.
**Interpret:** Phase transitions are optimal moments for stimulation — the network is most plastic during transitions.

### Predictive Coding (`/predictive-coding`)
**What:** Tests whether the organoid minimizes prediction error (Friston's free energy principle).
**Interpret:** True = network builds internal models. False = purely reactive (typical for current organoids).

### Weight Inference (`/weights`)
**What:** Infers a virtual "connectome" — synaptic weight matrix from spike timing.
**Interpret:** Excitatory/inhibitory ratio. E/I balance is crucial for healthy network function (typical brain: E/I ~4:1).

### Weight Tracking (`/weight-tracking`)
**What:** Monitors how inferred weights change over time.
**Interpret:** Significant changes = learning is occurring. Stable weights = consolidated or inactive network.

### Multi-Timescale Complexity (`/multiscale`)
**What:** Measures complexity at 12 different timescales (1ms to 10s).
**Interpret:**
- **Optimal timescale** — where the organoid is most complex
- **Operating frequency** — the natural rhythm
- **Rich multi-scale structure** → mature, information-processing network

---

## Full Report (`/full-report`)

Runs all analyses in one call. Takes ~5 seconds on 30s of data.

Sections in the report:
1. Summary statistics
2. Quality metrics
3. Burst analysis
4. Connectivity
5. Information theory
6. Spectral analysis
7. Criticality
8. Digital twin
9. ML features
10. STDP & learning
11. Organoid IQ
12. Predictions
13. Health
14. Replay & sequences
15. Reservoir computing
16. Fingerprint
17. Emergence (Phi)
18. Attractors
19. Phase transitions
20. Predictive coding
21. Weight inference
22. Multi-timescale complexity

---

*Guide written for NeuroBridge v0.2.0 — April 2026*
