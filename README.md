# NeuroBridge API

**The analysis engine for biological neural networks.**

57 analysis modules · 125 API endpoints · One-click full report on any organoid recording.

[![Live API](https://img.shields.io/badge/API-live-brightgreen)](https://api.neurocomputers.io/docs)
[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Built for researchers working with brain organoids on [FinalSpark](https://finalspark.com), [Cortical Labs](https://corticallabs.com), and university MEA platforms.

**Live:** [neurocomputers.io](https://neurocomputers.io) | **API Docs:** [api.neurocomputers.io/docs](https://api.neurocomputers.io/docs)

---

## Quick Start

```bash
# Clone
git clone https://github.com/Luckyguybiz/neurobridge-api.git
cd neurobridge-api

# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python main.py
# → http://localhost:8847/docs (Swagger UI)
```

### Generate synthetic data and analyze:
```bash
# Generate 30s of synthetic spike data
curl -X POST "http://localhost:8847/api/generate?duration=30"
# → {"dataset_id": "abc12345", "n_spikes": 2141, ...}

# Run ALL 25+ analyses in one call
curl "http://localhost:8847/api/analysis/abc12345/full-report"

# Get Organoid IQ score
curl "http://localhost:8847/api/analysis/abc12345/iq"
# → {"iq_score": 49.2, "grade": "C", ...}
```

### Upload real data:
```bash
curl -X POST "http://localhost:8847/api/upload" \
  -F "file=@my_spikes.csv"
```

Supported formats: CSV, HDF5, Parquet, JSON, NWB.

---

## Analysis Modules

### Standard Neuroscience (8 modules)

| Module | Endpoint | Description |
|--------|----------|-------------|
| Spike Analysis | `/analysis/{id}/firing-rates` | Spike detection, ISI, firing rates, amplitudes |
| Spike Sorting | `/analysis/{id}/spike-sorting` | PCA + K-means / HDBSCAN clustering |
| Burst Detection | `/analysis/{id}/bursts` | Network bursts, single-channel, profiles |
| Connectivity | `/analysis/{id}/connectivity` | Co-firing graph, degree, clustering |
| Cross-Correlation | `/analysis/{id}/cross-correlation` | Pairwise correlograms |
| Transfer Entropy | `/analysis/{id}/transfer-entropy` | Directed information flow |
| Statistics | `/analysis/{id}/summary` | Full summary, quality, temporal dynamics |
| Spectral | `/analysis/{id}/power-spectrum` | PSD, frequency bands, coherence |

### Novel Analysis (unique to NeuroBridge)

| Module | Endpoint | What it does |
|--------|----------|-------------|
| **Organoid IQ** | `/analysis/{id}/iq` | Composite intelligence score (0-100) across 6 dimensions |
| **STDP Mapping** | `/analysis/{id}/stdp` | Spike-timing dependent plasticity curves |
| **Learning Detection** | `/analysis/{id}/learning` | Temporal changes in plasticity = learning episodes |
| **Attractor Landscape** | `/analysis/{id}/attractors` | Memory as dynamical attractors (Hopfield theory) |
| **Phase Transitions** | `/analysis/{id}/phase-transitions` | Neural reorganization moments + optimal stim timing |
| **Predictive Coding** | `/analysis/{id}/predictive-coding` | Does the organoid minimize prediction error? |
| **Weight Inference** | `/analysis/{id}/weights` | Infer synaptic connectome from spike timing |
| **Weight Tracking** | `/analysis/{id}/weight-tracking` | Watch learning happen in real-time |
| **Neural Replay** | `/analysis/{id}/replay` | Memory consolidation during "rest" periods |
| **Reservoir Computing** | `/analysis/{id}/memory-capacity` | Memory capacity benchmark |
| **Causal Emergence** | `/analysis/{id}/emergence` | Integrated information (Phi) |
| **Multi-Timescale** | `/analysis/{id}/multiscale` | Complexity at 12 timescales |
| **Fingerprinting** | `/analysis/{id}/fingerprint` | Unique organoid identity signature |
| **Sonification** | `/analysis/{id}/sonify` | Neural activity → audio WAV |
| **Health Monitor** | `/analysis/{id}/health` | Organoid viability assessment |
| **Predictions** | `/analysis/{id}/predict/bursts` | Burst probability forecast |
| **Anomaly Detection** | `/analysis/{id}/anomalies` | Isolation Forest on neural features |
| **State Classification** | `/analysis/{id}/states` | Resting / active / bursting states |

### Discovery Analysis (10 modules)

| Module | Endpoint | What it does |
|--------|----------|-------------|
| **Sleep-Wake** | `/analysis/{id}/sleep-wake` | UP/DOWN state detection, slow-wave oscillations |
| **Habituation** | `/analysis/{id}/habituation` | Response decay — simplest form of learning |
| **Metastability** | `/analysis/{id}/metastability` | Kuramoto synchronization, brain-like state switching |
| **Information Flow** | `/analysis/{id}/information-flow` | Granger causality, hub detection |
| **Network Motifs** | `/analysis/{id}/motifs` | 3-node subgraph patterns vs random |
| **Energy Landscape** | `/analysis/{id}/energy-landscape` | Ising model, attractor basins |
| **Consciousness** | `/analysis/{id}/consciousness` | PCI + Phi + recurrence composite score |
| **Comparative** | `/analysis/{id}/comparative` | vs cortical slice, C. elegans, fruit fly, mouse hippocampus |
| **Turing Test** | `/analysis/{id}/turing-test` | Can you distinguish from Poisson/LIF? |
| **Welfare** | `/analysis/{id}/welfare` | Health + suffering detection + recommendations |

### Learning & Memory (4 modules)

| Module | Endpoint | What it does |
|--------|----------|-------------|
| **Catastrophic Forgetting** | `/analysis/{id}/forgetting` | Do early patterns survive over time? |
| **Transfer Learning** | `/analysis/{id}/transfer` | Does learning task A help task B? |
| **Consolidation** | `/analysis/{id}/consolidation` | Offline memory consolidation events |
| **Channel Capacity** | `/analysis/{id}/channel-capacity` | Multi-bit information capacity |

### Connectomics (5 endpoints)

| Module | Endpoint | What it does |
|--------|----------|-------------|
| **Full Connectome** | `/analysis/{id}/connectome` | Weighted adjacency, communities, modularity |
| **Graph Theory** | `/analysis/{id}/graph-theory` | Rich-club, small-world, efficiency, centrality |
| **Effective Connectivity** | `/analysis/{id}/effective-connectivity` | Directed causal connections |
| **Topology** | `/analysis/{id}/topology` | Betti numbers, persistent homology |

### Experiments (6 endpoints)

| Module | Endpoint | What it does |
|--------|----------|-------------|
| **Pong** | `POST /experiments/{id}/pong/simulate` | DishBrain Pong simulation |
| **Logic Gates** | `POST /experiments/{id}/logic/benchmark` | AND, OR, XOR, NAND benchmark |
| **Vowels** | `POST /experiments/{id}/vowels/simulate` | Brainoware 240-vowel classification |
| **Memory Tests** | `/experiments/{id}/memory-tests` | Working, short-term, long-term, associative |
| **Closed-Loop** | `POST /experiments/{id}/closed-loop/simulate` | DishBrain mode simulation |
| **Architecture Search** | `POST /analysis/{id}/architecture-search` | NAS for optimal stimulation |

### Full Report

```bash
# Run ALL analyses in one call (~5 seconds)
curl "http://localhost:8847/api/analysis/{id}/full-report"
```

Returns 21+ analysis results in a single JSON response.

---

## Data Formats

### Input
- **CSV**: columns `time, electrode, amplitude`
- **HDF5**: FinalSpark format or generic arrays
- **Parquet**: columnar format
- **JSON**: array of objects or nested structure
- **NWB**: Neurodata Without Borders

Column names are auto-detected. FinalSpark electrode indexing (modulo 32) is handled automatically.

### FinalSpark Compatibility
- Sampling rate: 30 kHz (configurable)
- Spike detection: 6× median(σ) threshold (Section 4.2 of Jordan et al. 2024)
- MEA: 4 × 8 electrodes = 32 channels
- Electrode index > 32 → modulo 32 applied automatically

---

## Tech Stack

- **Python 3.12** + FastAPI
- **NumPy / SciPy / Pandas** — core computation
- **scikit-learn** — ML (Isolation Forest, KMeans, Ridge, PCA)
- **h5py** — HDF5 file support

---

## API Reference

### Data Management
```
POST   /api/upload              Upload spike data file
POST   /api/generate            Generate synthetic data
GET    /api/datasets             List loaded datasets
GET    /api/datasets/{id}        Dataset info
GET    /api/datasets/{id}/spikes Spike data with filters
```

### Analysis
```
GET    /api/analysis/{id}/summary           Full statistics
GET    /api/analysis/{id}/quality            Data quality metrics
GET    /api/analysis/{id}/firing-rates       Time-binned rates
GET    /api/analysis/{id}/isi                Inter-spike intervals
GET    /api/analysis/{id}/amplitudes         Amplitude distributions
GET    /api/analysis/{id}/temporal           Temporal dynamics
POST   /api/analysis/{id}/spike-sorting      Spike sorting
GET    /api/analysis/{id}/bursts             Network bursts
GET    /api/analysis/{id}/connectivity       Connectivity graph
GET    /api/analysis/{id}/cross-correlation  Correlograms
GET    /api/analysis/{id}/transfer-entropy   Directed information
GET    /api/analysis/{id}/entropy            Shannon entropy
GET    /api/analysis/{id}/mutual-information Mutual information
GET    /api/analysis/{id}/complexity         LZ complexity
GET    /api/analysis/{id}/power-spectrum     Power spectral density
GET    /api/analysis/{id}/coherence          Spectral coherence
GET    /api/analysis/{id}/avalanches         Criticality
GET    /api/analysis/{id}/digital-twin/fit   LIF model fitting
POST   /api/analysis/{id}/digital-twin/simulate  Simulate twin
GET    /api/analysis/{id}/anomalies          Anomaly detection
GET    /api/analysis/{id}/states             State classification
GET    /api/analysis/{id}/pca                PCA embedding
GET    /api/analysis/{id}/stdp               STDP plasticity
GET    /api/analysis/{id}/learning           Learning episodes
GET    /api/analysis/{id}/iq                 Organoid IQ
GET    /api/analysis/{id}/predict/firing-rates  Rate forecast
GET    /api/analysis/{id}/predict/bursts     Burst probability
GET    /api/analysis/{id}/health             Health assessment
GET    /api/analysis/{id}/replay             Neural replay
GET    /api/analysis/{id}/sequences          Neural circuits
GET    /api/analysis/{id}/memory-capacity    Reservoir MC
GET    /api/analysis/{id}/nonlinearity       Computation benchmark
GET    /api/analysis/{id}/fingerprint        Organoid fingerprint
GET    /api/analysis/{id}/sonify             Audio sonification
GET    /api/analysis/{id}/rhythms            Rhythmic analysis
GET    /api/analysis/{id}/emergence          Phi / causal emergence
GET    /api/analysis/{id}/attractors         Attractor landscape
GET    /api/analysis/{id}/state-space        State space geometry
GET    /api/analysis/{id}/phase-transitions  Phase transitions
GET    /api/analysis/{id}/predictive-coding  Predictive coding
GET    /api/analysis/{id}/weights            Synaptic weights
GET    /api/analysis/{id}/weight-tracking    Weight changes
GET    /api/analysis/{id}/multiscale         Multi-timescale
GET    /api/analysis/{id}/full-report        ALL analyses
```

### Export
```
GET    /api/export/{id}/csv    Download as CSV
GET    /api/export/{id}/json   Download as JSON
```

---

## License

MIT

---

*NeuroBridge — Biocomputing-as-a-Service Platform*
