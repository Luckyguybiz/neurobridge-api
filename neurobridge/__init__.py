"""NeuroBridge — Analysis engine for brain organoid data.

57 analysis modules for spike sorting, burst detection, connectivity mapping,
Organoid IQ scoring, attractor landscapes, consciousness metrics, and more.

Quick start:
    >>> import neurobridge as nb
    >>> data = nb.load("recording.csv")
    >>> report = nb.analyze(data)
    >>> print(report["organoid_iq"]["iq_score"])
"""

__version__ = "0.1.0"

from neurobridge.core import load, analyze, generate_synthetic, full_report
from neurobridge.core import SpikeData

__all__ = [
    "load",
    "analyze",
    "generate_synthetic",
    "full_report",
    "SpikeData",
    "__version__",
]
