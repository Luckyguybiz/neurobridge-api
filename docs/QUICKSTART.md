# NeuroBridge API — Quick Start

Get from zero to your first analysis in 5 minutes.

## 1. Install

```bash
git clone https://github.com/Luckyguybiz/neurobridge-api.git
cd neurobridge-api

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Start the server

```bash
python main.py
```

The API is now running at `http://localhost:8847`. Open `http://localhost:8847/docs` for the interactive Swagger UI.

## 3. Generate synthetic data

```bash
curl -X POST "http://localhost:8847/api/generate?duration=30&n_electrodes=8"
```

Response:
```json
{
  "dataset_id": "abc12345",
  "n_spikes": 2141,
  "n_electrodes": 8,
  "duration_s": 30.0
}
```

Save the `dataset_id` — you'll use it for all analysis calls.

## 4. Run your first analysis

### Get a full summary:
```bash
curl "http://localhost:8847/api/analysis/abc12345/summary"
```

### Detect bursts:
```bash
curl "http://localhost:8847/api/analysis/abc12345/bursts"
```

### Get the Organoid IQ score:
```bash
curl "http://localhost:8847/api/analysis/abc12345/iq"
```

### Run ALL 25+ analyses at once:
```bash
curl "http://localhost:8847/api/analysis/abc12345/full-report"
```

## 5. Upload real data

```bash
curl -X POST "http://localhost:8847/api/upload" \
  -F "file=@my_recording.csv"
```

CSV format: `time,electrode,amplitude` (one spike per row).

Supported formats: CSV, HDF5, Parquet, JSON, NWB.

## 6. Export results

```bash
# Download as CSV
curl -O "http://localhost:8847/api/export/abc12345/csv"

# Download as JSON
curl -O "http://localhost:8847/api/export/abc12345/json"
```

## 7. Connect the dashboard

Start the [NeuroBridge frontend](https://github.com/Luckyguybiz/neurobridge):

```bash
cd ../neurobridge
npm install && npm run dev
# Open http://localhost:3000/dashboard
```

The dashboard auto-connects to the API at port 8847.

---

## What's next?

- Read the [API Reference](API_REFERENCE.md) for all 60+ endpoints
- Read the [Analysis Guide](ANALYSIS_GUIDE.md) to understand what each analysis means
- Try the interactive Swagger UI at `/docs`
