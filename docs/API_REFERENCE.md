# NeuroBridge API Reference

Base URL: `http://localhost:8847`

Interactive docs: `http://localhost:8847/docs` (Swagger UI)

---

## Health

### `GET /health`
Returns API status.

```bash
curl http://localhost:8847/health
```

---

## Data Management

### `POST /api/generate`
Generate synthetic spike data for testing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| duration | float | 30.0 | Duration in seconds |
| n_electrodes | int | 8 | Number of electrodes |
| burst_probability | float | 0.3 | Probability of burst events |

```bash
curl -X POST "http://localhost:8847/api/generate?duration=60&n_electrodes=8"
```

### `POST /api/upload`
Upload a spike data file.

| Parameter | Type | Description |
|-----------|------|-------------|
| file | File | CSV, HDF5, Parquet, JSON, or NWB file |
| sampling_rate | float | Sampling rate in Hz (default: 30000) |

```bash
curl -X POST "http://localhost:8847/api/upload" \
  -F "file=@recording.csv" \
  -F "sampling_rate=30000"
```

### `GET /api/datasets`
List all loaded datasets.

```bash
curl http://localhost:8847/api/datasets
```

### `GET /api/datasets/{dataset_id}`
Get detailed info about a dataset.

```bash
curl http://localhost:8847/api/datasets/abc12345
```

### `GET /api/datasets/{dataset_id}/spikes`
Get spike data with optional filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| start | float | 0 | Start time (seconds) |
| end | float | max | End time (seconds) |
| electrodes | string | all | Comma-separated electrode IDs |
| limit | int | 10000 | Max spikes returned |

```bash
curl "http://localhost:8847/api/datasets/abc12345/spikes?start=0&end=10&electrodes=0,1,2&limit=5000"
```

---

## Standard Analysis

### `GET /api/analysis/{id}/summary`
Full dataset summary — population statistics, per-electrode stats, burst overview.

```bash
curl http://localhost:8847/api/analysis/abc12345/summary
```

### `GET /api/analysis/{id}/quality`
Data quality metrics — SNR, refractory violations, gaps, stationarity.

```bash
curl http://localhost:8847/api/analysis/abc12345/quality
```

### `GET /api/analysis/{id}/firing-rates`
Time-binned firing rates per electrode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| bin_size | float | 1.0 | Bin size in seconds |

```bash
curl "http://localhost:8847/api/analysis/abc12345/firing-rates?bin_size=1.0"
```

### `GET /api/analysis/{id}/isi`
Inter-spike interval analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| electrode | int | all | Specific electrode (optional) |

```bash
curl "http://localhost:8847/api/analysis/abc12345/isi?electrode=0"
```

### `GET /api/analysis/{id}/amplitudes`
Amplitude distribution statistics per electrode.

```bash
curl http://localhost:8847/api/analysis/abc12345/amplitudes
```

### `GET /api/analysis/{id}/temporal`
Temporal dynamics — trends, rate changes over time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| bin_size | float | 60.0 | Bin size in seconds |

```bash
curl "http://localhost:8847/api/analysis/abc12345/temporal?bin_size=30"
```

### `POST /api/analysis/{id}/spike-sorting`
Sort spikes by waveform shape using PCA + K-means.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method | string | pca_kmeans | Sorting method |
| n_components | int | 3 | PCA components |
| n_clusters | int | auto | Number of clusters |

```bash
curl -X POST "http://localhost:8847/api/analysis/abc12345/spike-sorting" \
  -H "Content-Type: application/json" \
  -d '{"method": "pca_kmeans", "n_clusters": 4}'
```

---

## Burst Analysis

### `GET /api/analysis/{id}/bursts`
Detect network bursts (multiple electrodes firing together).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| min_electrodes | int | 3 | Minimum electrodes participating |
| window_ms | float | 50.0 | Detection window (ms) |

```bash
curl "http://localhost:8847/api/analysis/abc12345/bursts?min_electrodes=3&window_ms=50"
```

### `GET /api/analysis/{id}/bursts/profiles`
Detailed burst profiles — duration, recruitment order, peak rate.

```bash
curl http://localhost:8847/api/analysis/abc12345/bursts/profiles
```

### `GET /api/analysis/{id}/bursts/single-channel`
Single-electrode burst detection.

```bash
curl http://localhost:8847/api/analysis/abc12345/bursts/single-channel
```

---

## Connectivity

### `GET /api/analysis/{id}/connectivity`
Functional connectivity graph (co-firing analysis).

```bash
curl http://localhost:8847/api/analysis/abc12345/connectivity
```

Returns nodes (electrodes) and edges (functional connections with strength).

### `GET /api/analysis/{id}/cross-correlation`
Pairwise cross-correlograms.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_lag_ms | float | 50.0 | Maximum lag (ms) |
| bin_size_ms | float | 1.0 | Bin size (ms) |

```bash
curl "http://localhost:8847/api/analysis/abc12345/cross-correlation?max_lag_ms=50"
```

### `GET /api/analysis/{id}/transfer-entropy`
Directed information flow between electrode pairs.

```bash
curl http://localhost:8847/api/analysis/abc12345/transfer-entropy
```

---

## Information Theory

### `GET /api/analysis/{id}/entropy`
Shannon entropy of spike trains per electrode.

```bash
curl http://localhost:8847/api/analysis/abc12345/entropy
```

### `GET /api/analysis/{id}/mutual-information`
Pairwise mutual information between electrodes.

```bash
curl http://localhost:8847/api/analysis/abc12345/mutual-information
```

### `GET /api/analysis/{id}/complexity`
Lempel-Ziv complexity — measure of spike train compressibility.

```bash
curl http://localhost:8847/api/analysis/abc12345/complexity
```

---

## Spectral Analysis

### `GET /api/analysis/{id}/power-spectrum`
Power spectral density of binned spike trains.

```bash
curl http://localhost:8847/api/analysis/abc12345/power-spectrum
```

### `GET /api/analysis/{id}/coherence`
Spectral coherence between electrode pairs.

```bash
curl http://localhost:8847/api/analysis/abc12345/coherence
```

---

## Criticality

### `GET /api/analysis/{id}/avalanches`
Detect neuronal avalanches — signature of critical dynamics.

```bash
curl http://localhost:8847/api/analysis/abc12345/avalanches
```

---

## Digital Twin

### `GET /api/analysis/{id}/digital-twin/fit`
Fit Leaky Integrate-and-Fire (LIF) neuron model to recorded data.

```bash
curl http://localhost:8847/api/analysis/abc12345/digital-twin/fit
```

### `POST /api/analysis/{id}/digital-twin/simulate`
Simulate the fitted LIF network and compare with real data.

```bash
curl -X POST http://localhost:8847/api/analysis/abc12345/digital-twin/simulate
```

---

## ML Pipeline

### `GET /api/analysis/{id}/features`
Extract multi-scale features from spike data.

```bash
curl http://localhost:8847/api/analysis/abc12345/features
```

### `GET /api/analysis/{id}/anomalies`
Detect anomalous time windows (Isolation Forest).

```bash
curl http://localhost:8847/api/analysis/abc12345/anomalies
```

### `GET /api/analysis/{id}/states`
Classify neural states: resting, active, bursting.

```bash
curl http://localhost:8847/api/analysis/abc12345/states
```

### `GET /api/analysis/{id}/pca`
PCA embedding of neural activity windows.

```bash
curl http://localhost:8847/api/analysis/abc12345/pca
```

---

## Novel Analysis (Unique to NeuroBridge)

### `GET /api/analysis/{id}/stdp`
Map spike-timing-dependent plasticity (STDP) curves.

```bash
curl http://localhost:8847/api/analysis/abc12345/stdp
```

### `GET /api/analysis/{id}/learning`
Detect learning episodes — temporal changes in plasticity.

```bash
curl http://localhost:8847/api/analysis/abc12345/learning
```

### `GET /api/analysis/{id}/iq`
Compute Organoid Intelligence Quotient (0-100 composite score).

```bash
curl http://localhost:8847/api/analysis/abc12345/iq
```

Returns score across 6 dimensions: complexity, synchrony, adaptability, information integration, temporal structure, responsiveness.

### `GET /api/analysis/{id}/predict/firing-rates`
Forecast future firing rates using trend analysis.

```bash
curl http://localhost:8847/api/analysis/abc12345/predict/firing-rates
```

### `GET /api/analysis/{id}/predict/bursts`
Predict burst probability in upcoming time windows.

```bash
curl http://localhost:8847/api/analysis/abc12345/predict/bursts
```

### `GET /api/analysis/{id}/health`
Assess organoid viability — signal quality, activity level, stability.

```bash
curl http://localhost:8847/api/analysis/abc12345/health
```

### `GET /api/analysis/{id}/replay`
Detect neural replay events — memory consolidation signatures.

```bash
curl http://localhost:8847/api/analysis/abc12345/replay
```

### `GET /api/analysis/{id}/sequences`
Find repeated neural circuit patterns.

```bash
curl http://localhost:8847/api/analysis/abc12345/sequences
```

### `GET /api/analysis/{id}/memory-capacity`
Estimate reservoir computing memory capacity.

```bash
curl http://localhost:8847/api/analysis/abc12345/memory-capacity
```

### `GET /api/analysis/{id}/nonlinearity`
Benchmark nonlinear computation ability.

```bash
curl http://localhost:8847/api/analysis/abc12345/nonlinearity
```

### `GET /api/analysis/{id}/fingerprint`
Compute unique organoid identity signature (hash + feature vector).

```bash
curl http://localhost:8847/api/analysis/abc12345/fingerprint
```

### `GET /api/analysis/{id}/sonify`
Convert neural activity to audio (returns base64 WAV).

```bash
curl http://localhost:8847/api/analysis/abc12345/sonify
```

### `GET /api/analysis/{id}/rhythms`
Analyze rhythmic structure in activity patterns.

```bash
curl http://localhost:8847/api/analysis/abc12345/rhythms
```

### `GET /api/analysis/{id}/emergence`
Compute integrated information (Phi) — measure of consciousness-like properties.

```bash
curl http://localhost:8847/api/analysis/abc12345/emergence
```

### `GET /api/analysis/{id}/attractors`
Map the attractor landscape — memory as dynamical attractors.

```bash
curl http://localhost:8847/api/analysis/abc12345/attractors
```

### `GET /api/analysis/{id}/state-space`
Analyze state space geometry.

```bash
curl http://localhost:8847/api/analysis/abc12345/state-space
```

### `GET /api/analysis/{id}/phase-transitions`
Detect neural reorganization moments and optimal stimulation timing.

```bash
curl http://localhost:8847/api/analysis/abc12345/phase-transitions
```

### `GET /api/analysis/{id}/predictive-coding`
Measure whether the organoid minimizes prediction error (free energy principle).

```bash
curl http://localhost:8847/api/analysis/abc12345/predictive-coding
```

### `GET /api/analysis/{id}/weights`
Infer synaptic weight matrix from spike timing.

```bash
curl http://localhost:8847/api/analysis/abc12345/weights
```

### `GET /api/analysis/{id}/weight-tracking`
Track synaptic weight changes over time (learning detection).

```bash
curl http://localhost:8847/api/analysis/abc12345/weight-tracking
```

### `GET /api/analysis/{id}/multiscale`
Compute complexity at 12 different timescales.

```bash
curl http://localhost:8847/api/analysis/abc12345/multiscale
```

---

## Full Report

### `GET /api/analysis/{id}/full-report`
Run ALL analyses in a single call. Returns 21+ analysis results.

```bash
curl http://localhost:8847/api/analysis/abc12345/full-report
```

---

## Export

### `GET /api/export/{id}/csv`
Download spike data as CSV.

```bash
curl -O http://localhost:8847/api/export/abc12345/csv
```

### `GET /api/export/{id}/json`
Download spike data as JSON.

```bash
curl -O http://localhost:8847/api/export/abc12345/json
```

---

## Error Handling

All endpoints return errors as:

```json
{
  "detail": "Dataset abc12345 not found"
}
```

Common HTTP status codes:
- `200` — Success
- `404` — Dataset not found
- `422` — Invalid parameters
- `500` — Analysis error
