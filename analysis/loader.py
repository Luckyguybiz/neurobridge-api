"""Data loading module — handles CSV, HDF5, Parquet, JSON formats.

FinalSpark dataset structure:
- 30kHz sampling rate, 16-bit resolution (0.15 µV)
- 4 MEA x 8 electrodes = 32 channels
- Spike detection threshold: 6 x median(std_dev)
- If electrode index > 32, apply modulo 32
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import json
import time


class SpikeData:
    """Container for spike data with efficient access patterns."""

    def __init__(
        self,
        times: np.ndarray,
        electrodes: np.ndarray,
        amplitudes: np.ndarray,
        waveforms: Optional[np.ndarray] = None,
        sampling_rate: float = 30000.0,
        metadata: Optional[dict] = None,
    ):
        self.times = np.asarray(times, dtype=np.float64)
        self.electrodes = np.asarray(electrodes, dtype=np.int32)
        self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
        self.waveforms = waveforms  # shape: (n_spikes, waveform_length) or None
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}

        # Sort by time
        sort_idx = np.argsort(self.times)
        self.times = self.times[sort_idx]
        self.electrodes = self.electrodes[sort_idx]
        self.amplitudes = self.amplitudes[sort_idx]
        if self.waveforms is not None:
            self.waveforms = self.waveforms[sort_idx]

        # Pre-compute per-electrode indices for fast lookup
        self._electrode_indices: dict[int, np.ndarray] = {}
        for e in np.unique(self.electrodes):
            self._electrode_indices[int(e)] = np.where(self.electrodes == e)[0]

    @property
    def n_spikes(self) -> int:
        return len(self.times)

    @property
    def n_electrodes(self) -> int:
        return len(self._electrode_indices)

    @property
    def electrode_ids(self) -> list[int]:
        return sorted(self._electrode_indices.keys())

    @property
    def duration(self) -> float:
        if self.n_spikes == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])

    @property
    def time_range(self) -> tuple[float, float]:
        if self.n_spikes == 0:
            return (0.0, 0.0)
        return (float(self.times[0]), float(self.times[-1]))

    def get_electrode(self, electrode_id: int) -> "SpikeData":
        """Get spikes for a single electrode."""
        idx = self._electrode_indices.get(electrode_id, np.array([], dtype=int))
        return SpikeData(
            times=self.times[idx],
            electrodes=self.electrodes[idx],
            amplitudes=self.amplitudes[idx],
            waveforms=self.waveforms[idx] if self.waveforms is not None else None,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )

    def get_time_range(self, start: float, end: float) -> "SpikeData":
        """Get spikes within a time range."""
        mask = (self.times >= start) & (self.times <= end)
        return SpikeData(
            times=self.times[mask],
            electrodes=self.electrodes[mask],
            amplitudes=self.amplitudes[mask],
            waveforms=self.waveforms[mask] if self.waveforms is not None else None,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )

    def get_filtered(
        self,
        electrodes: Optional[list[int]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> "SpikeData":
        """Get filtered subset of spikes."""
        mask = np.ones(self.n_spikes, dtype=bool)
        if start is not None:
            mask &= self.times >= start
        if end is not None:
            mask &= self.times <= end
        if electrodes is not None:
            mask &= np.isin(self.electrodes, electrodes)
        return SpikeData(
            times=self.times[mask],
            electrodes=self.electrodes[mask],
            amplitudes=self.amplitudes[mask],
            waveforms=self.waveforms[mask] if self.waveforms is not None else None,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            "time": self.times,
            "electrode": self.electrodes,
            "amplitude": self.amplitudes,
        })
        return df

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "n_spikes": self.n_spikes,
            "n_electrodes": self.n_electrodes,
            "duration": self.duration,
            "time_range": self.time_range,
            "sampling_rate": self.sampling_rate,
            "electrode_ids": self.electrode_ids,
            "metadata": self.metadata,
            "times": self.times.tolist(),
            "electrodes": self.electrodes.tolist(),
            "amplitudes": self.amplitudes.tolist(),
        }


def load_csv(filepath: str, **kwargs) -> SpikeData:
    """Load spike data from CSV file.

    Expected columns: time, electrode, amplitude
    Also supports FinalSpark format: _time (ISO 8601), _value (amplitude µV), index (electrode)
    Optional columns: waveform (comma-separated values in a single column)
    """
    df = pd.read_csv(filepath, **{k: v for k, v in kwargs.items() if k == "sep"})

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("time", "timestamp", "t", "time_s", "time_sec", "_time"):
            col_map["time"] = col
        elif lower in ("electrode", "channel", "ch", "electrode_id", "chan", "index"):
            col_map["electrode"] = col
        elif lower in ("amplitude", "amp", "voltage", "value", "amplitude_uv", "_value"):
            col_map["amplitude"] = col

    if "time" not in col_map:
        # Try first numeric column as time
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col_map["time"] = numeric_cols[0]
            col_map["electrode"] = numeric_cols[1]
            if len(numeric_cols) >= 3:
                col_map["amplitude"] = numeric_cols[2]

    # Handle ISO 8601 timestamps (FinalSpark _time column)
    time_col = df[col_map["time"]]
    if time_col.dtype == object or pd.api.types.is_string_dtype(time_col):
        # Parse datetime strings and convert to seconds from recording start
        parsed = pd.to_datetime(time_col, utc=True)
        t0 = parsed.min()
        times = (parsed - t0).dt.total_seconds().values
    else:
        times = time_col.values

    electrodes = df[col_map["electrode"]].values.astype(np.int32)

    # Apply modulo 32 for FinalSpark electrode indexing
    electrodes = electrodes % 32

    amplitudes = df[col_map.get("amplitude", col_map["time"])].values if "amplitude" in col_map else np.zeros_like(times)

    return SpikeData(
        times=times,
        electrodes=electrodes,
        amplitudes=amplitudes.astype(np.float64),
        sampling_rate=kwargs.get("sampling_rate", 30000.0),
        metadata={"source": filepath, "format": "csv"},
    )


def load_hdf5(filepath: str, **kwargs) -> SpikeData:
    """Load spike data from HDF5 file (FinalSpark or NWB format)."""
    import h5py

    with h5py.File(filepath, "r") as f:
        # Try FinalSpark format
        if "spikes" in f:
            grp = f["spikes"]
            times = grp["times"][:]
            electrodes = grp["electrodes"][:] if "electrodes" in grp else grp["channels"][:]
            amplitudes = grp["amplitudes"][:] if "amplitudes" in grp else np.zeros_like(times)
            waveforms = grp["waveforms"][:] if "waveforms" in grp else None
        # Try raw recording format
        elif "recordings" in f or "data" in f:
            key = "recordings" if "recordings" in f else "data"
            raw_data = f[key][:]
            # Will need spike detection on raw data
            return _detect_spikes_from_raw(raw_data, kwargs.get("sampling_rate", 30000.0), filepath)
        # Try flat arrays
        else:
            keys = list(f.keys())
            times = f[keys[0]][:] if len(keys) > 0 else np.array([])
            electrodes = f[keys[1]][:] if len(keys) > 1 else np.zeros_like(times, dtype=np.int32)
            amplitudes = f[keys[2]][:] if len(keys) > 2 else np.zeros_like(times)
            waveforms = None

        electrodes = electrodes.astype(np.int32) % 32

        metadata = {"source": filepath, "format": "hdf5"}
        if "metadata" in f:
            for k in f["metadata"].attrs:
                metadata[k] = f["metadata"].attrs[k]

    return SpikeData(
        times=times,
        electrodes=electrodes,
        amplitudes=amplitudes,
        waveforms=waveforms,
        sampling_rate=kwargs.get("sampling_rate", 30000.0),
        metadata=metadata,
    )


def load_parquet(filepath: str, **kwargs) -> SpikeData:
    """Load spike data from Parquet file."""
    df = pd.read_parquet(filepath)
    return _from_dataframe(df, filepath, "parquet", **kwargs)


def load_json(filepath: str, **kwargs) -> SpikeData:
    """Load spike data from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
        return _from_dataframe(df, filepath, "json", **kwargs)
    elif isinstance(data, dict):
        times = np.array(data.get("times", data.get("time", [])))
        electrodes = np.array(data.get("electrodes", data.get("electrode", data.get("channels", []))), dtype=np.int32)
        amplitudes = np.array(data.get("amplitudes", data.get("amplitude", np.zeros_like(times))))
        return SpikeData(
            times=times, electrodes=electrodes % 32, amplitudes=amplitudes,
            sampling_rate=kwargs.get("sampling_rate", 30000.0),
            metadata={"source": filepath, "format": "json"},
        )

    raise ValueError(f"Unsupported JSON structure in {filepath}")


def _from_dataframe(df: pd.DataFrame, filepath: str, fmt: str, **kwargs) -> SpikeData:
    """Convert DataFrame to SpikeData with column auto-detection."""
    col_candidates = {
        "time": ["time", "timestamp", "t", "time_s", "_time"],
        "electrode": ["electrode", "channel", "ch", "electrode_id", "index"],
        "amplitude": ["amplitude", "amp", "voltage", "value", "_value"],
    }
    col_map = {}
    for field, candidates in col_candidates.items():
        for c in candidates:
            matches = [col for col in df.columns if col.lower().strip() == c]
            if matches:
                col_map[field] = matches[0]
                break

    if "time" not in col_map:
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric) >= 2:
            col_map["time"] = numeric[0]
            col_map["electrode"] = numeric[1]
            if len(numeric) >= 3:
                col_map["amplitude"] = numeric[2]

    # Handle ISO 8601 timestamps
    time_col = df[col_map["time"]]
    if time_col.dtype == object or pd.api.types.is_string_dtype(time_col):
        parsed = pd.to_datetime(time_col, utc=True)
        t0 = parsed.min()
        times = (parsed - t0).dt.total_seconds().values
    else:
        times = time_col.values

    return SpikeData(
        times=times,
        electrodes=df[col_map["electrode"]].values.astype(np.int32) % 32,
        amplitudes=df[col_map.get("amplitude", col_map["time"])].values.astype(np.float64) if "amplitude" in col_map else np.zeros(len(df)),
        sampling_rate=kwargs.get("sampling_rate", 30000.0),
        metadata={"source": filepath, "format": fmt},
    )


def _detect_spikes_from_raw(
    raw_data: np.ndarray,
    sampling_rate: float,
    filepath: str,
    threshold_factor: float = 6.0,
    window_samples: int = 900,  # 30ms at 30kHz
) -> SpikeData:
    """Detect spikes from raw multi-channel recording.

    Uses FinalSpark method: T = 6 × Median({σi})
    where σi is std dev over 30ms windows.
    """
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    n_channels, n_samples = raw_data.shape
    times_list, electrodes_list, amplitudes_list = [], [], []

    for ch in range(n_channels):
        signal = raw_data[ch]
        # Compute windowed std dev
        n_windows = n_samples // window_samples
        stds = np.array([
            np.std(signal[i * window_samples:(i + 1) * window_samples])
            for i in range(n_windows)
        ])
        threshold = threshold_factor * np.median(stds)

        # Find threshold crossings (negative peaks)
        below = signal < -threshold
        crossings = np.where(np.diff(below.astype(int)) == 1)[0]

        for cx in crossings:
            # Find negative peak within 1ms
            search_end = min(cx + int(sampling_rate * 0.001), n_samples)
            peak_idx = cx + np.argmin(signal[cx:search_end])
            times_list.append(peak_idx / sampling_rate)
            electrodes_list.append(ch % 32)
            amplitudes_list.append(float(signal[peak_idx]))

    return SpikeData(
        times=np.array(times_list),
        electrodes=np.array(electrodes_list, dtype=np.int32),
        amplitudes=np.array(amplitudes_list),
        sampling_rate=sampling_rate,
        metadata={"source": filepath, "format": "raw", "spike_detection": "threshold_6x_std"},
    )


def load_file(filepath: str, **kwargs) -> SpikeData:
    """Auto-detect format and load spike data."""
    path = Path(filepath)
    ext = path.suffix.lower()

    loaders = {
        ".csv": load_csv,
        ".tsv": lambda f, **kw: load_csv(f, sep="\t", **kw),
        ".h5": load_hdf5,
        ".hdf5": load_hdf5,
        ".hdf": load_hdf5,
        ".nwb": load_hdf5,
        ".parquet": load_parquet,
        ".pq": load_parquet,
        ".json": load_json,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {list(loaders.keys())}")

    return loader(filepath, **kwargs)
