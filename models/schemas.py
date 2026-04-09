"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DataFormat(str, Enum):
    CSV = "csv"
    HDF5 = "hdf5"
    PARQUET = "parquet"
    JSON = "json"
    NWB = "nwb"


class MEAConfig(BaseModel):
    """MEA hardware configuration."""
    num_electrodes: int = 8
    num_mea: int = 4
    sampling_rate: float = 30000.0  # Hz
    resolution_bits: int = 16
    voltage_resolution: float = 0.15  # µV per bit


class DatasetInfo(BaseModel):
    """Summary of loaded dataset."""
    filename: str
    format: DataFormat
    num_electrodes: int
    num_spikes: int
    duration_seconds: float
    time_range: tuple[float, float]
    sampling_rate: float
    file_size_mb: float
    organoid_age_days: Optional[int] = None
    mea_id: Optional[int] = None
    temperature: Optional[float] = None


class SpikeDetectionParams(BaseModel):
    """Parameters for spike detection algorithm."""
    threshold_factor: float = Field(6.0, ge=1.0, le=20.0, description="Multiplier for std dev threshold (FinalSpark uses 6)")
    window_ms: float = Field(30.0, ge=5.0, le=100.0, description="Window size for std dev calculation in ms")
    refractory_ms: float = Field(1.0, ge=0.5, le=5.0, description="Minimum inter-spike interval in ms")
    waveform_pre_ms: float = Field(1.0, description="Time before spike peak to extract")
    waveform_post_ms: float = Field(2.0, description="Time after spike peak to extract")


class BurstDetectionParams(BaseModel):
    """Parameters for network burst detection."""
    min_electrodes: int = Field(3, ge=2, le=32, description="Minimum electrodes firing simultaneously")
    window_ms: float = Field(50.0, ge=10.0, le=200.0, description="Time window for coincidence detection")
    min_spikes_per_electrode: int = Field(2, ge=1, description="Min spikes per electrode within burst")


class SpikeSortingParams(BaseModel):
    """Parameters for spike sorting (clustering)."""
    method: str = Field("pca_kmeans", description="Method: pca_kmeans, pca_hdbscan, umap_hdbscan")
    n_components: int = Field(3, ge=2, le=10, description="PCA components")
    n_clusters: Optional[int] = Field(None, description="Number of clusters (None for auto)")
    min_cluster_size: int = Field(20, ge=5, description="Min cluster size for HDBSCAN")


class ConnectivityParams(BaseModel):
    """Parameters for functional connectivity analysis."""
    method: str = Field("cross_correlation", description="Method: cross_correlation, transfer_entropy, granger")
    max_lag_ms: float = Field(50.0, ge=5.0, le=200.0, description="Maximum lag for cross-correlation")
    bin_size_ms: float = Field(1.0, ge=0.1, le=10.0, description="Bin size for cross-correlogram")
    significance_threshold: float = Field(0.01, description="P-value threshold for significance")


class TimeRangeFilter(BaseModel):
    """Time range filter for analysis."""
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    electrodes: Optional[list[int]] = None  # None = all


class AnalysisRequest(BaseModel):
    """Generic analysis request."""
    dataset_id: str
    time_range: Optional[TimeRangeFilter] = None
    params: dict = {}


class AnalysisResult(BaseModel):
    """Generic analysis result."""
    analysis_type: str
    dataset_id: str
    computed_at: str
    duration_ms: float
    summary: dict
    data: dict


class ElectrodeStats(BaseModel):
    """Per-electrode statistics."""
    electrode_id: int
    num_spikes: int
    firing_rate_hz: float
    mean_amplitude_uv: float
    std_amplitude_uv: float
    mean_isi_ms: float
    cv_isi: float  # coefficient of variation
    burst_rate: float  # bursts per minute
