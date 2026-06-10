"""Tests for common-mode artifact detection.

Strategy: build synthetic data with a KNOWN ground truth — two independent
"wells" of Poisson biology plus a set of injected synchronous transients that
hit every electrode at the same instant — and verify the detector recovers the
injected artifacts without flagging the independent biology.
"""

import numpy as np
import pytest

from analysis.loader import SpikeData
from analysis import artifact_rejection as ar


def _synthetic_with_artifacts(seed=0, n_bio_per_elec=2000, n_artifacts=500):
    rng = np.random.default_rng(seed)
    dur = 600.0
    times, elecs, amps = [], [], []

    # Independent biology: 16 electrodes (2 wells of 8), Poisson, low amplitude.
    for e in range(16):
        tt = np.sort(rng.uniform(0, dur, n_bio_per_elec))
        times.append(tt)
        elecs.append(np.full(n_bio_per_elec, e))
        amps.append(rng.normal(-30, 8, n_bio_per_elec))  # ~30 µV biology

    # Injected common-mode artifacts: at each artifact time, EVERY electrode
    # fires simultaneously with high amplitude. This is the non-biological
    # ground truth.
    art_times = np.sort(rng.uniform(0, dur, n_artifacts))
    for e in range(16):
        jitter = rng.normal(0, 0.0002, n_artifacts)  # 0.2 ms scatter
        times.append(art_times + jitter)
        elecs.append(np.full(n_artifacts, e))
        amps.append(rng.normal(-800, 100, n_artifacts))  # high amplitude

    t = np.concatenate(times)
    e = np.concatenate(elecs).astype(np.int32)
    a = np.concatenate(amps)
    # ground-truth artifact label: the last 16*n_artifacts entries
    is_art = np.zeros(len(t), bool)
    is_art[16 * n_bio_per_elec:] = True

    order = np.argsort(t)
    return SpikeData(t[order], e[order], a[order], sampling_rate=30000.0), is_art[order]


def test_detects_injected_artifacts():
    data, truth = _synthetic_with_artifacts()
    rep = ar.detect_common_mode_artifacts(data, window_ms=2.0, group_size=8)

    # The detector should flag a substantial fraction and be enriched over chance.
    assert rep["artifact_fraction"] > 0.05
    assert rep["enrichment_over_chance"] is not None and rep["enrichment_over_chance"] > 3
    assert "CONTAMINATED" in rep["verdict"]

    # Recover the flagged mask and compare to ground truth.
    flagged = np.zeros(data.n_spikes, bool)
    flagged[rep["artifact_indices"]] = True
    # Recall on injected artifacts should be high (they hit all 16 electrodes).
    recall = flagged[truth].mean()
    assert recall > 0.9, f"recall too low: {recall:.2f}"
    # Precision: most low-amplitude biology should NOT be flagged.
    bio_flagged_rate = flagged[~truth].mean()
    assert bio_flagged_rate < 0.1, f"too much biology flagged: {bio_flagged_rate:.2f}"


def test_amplitude_dissociation():
    data, _ = _synthetic_with_artifacts()
    rep = ar.detect_common_mode_artifacts(data, window_ms=2.0, group_size=8)
    diss = rep["amplitude_dissociation"]
    # High-amplitude spikes should be far more artifact-laden than low-amplitude.
    hi = diss["high_amp_gt300uV"]["artifact_fraction"]
    lo = diss["low_amp_lt50uV"]["artifact_fraction"]
    assert hi > lo
    assert hi > 0.8


def test_clean_removes_artifacts():
    data, truth = _synthetic_with_artifacts()
    cleaned = ar.clean(data, window_ms=2.0, group_size=8)
    assert cleaned.n_spikes < data.n_spikes
    assert cleaned.metadata.get("artifact_cleaned") is True
    # Removed count should be in the ballpark of the injected artifacts.
    assert cleaned.metadata["artifact_removed"] >= 0.9 * truth.sum()


def test_clean_recording_not_flagged():
    """Independent biology with no artifacts → low enrichment, not CONTAMINATED.

    Uses realistic organoid firing rates (~0.25 Hz/electrode). The principled
    contamination signal is enrichment over the jittered-null chance level, not
    the absolute coincidence fraction (which scales with firing rate).
    """
    rng = np.random.default_rng(1)
    dur = 600.0
    times, elecs, amps = [], [], []
    for e in range(16):
        n = 150  # ~0.25 Hz, realistic for organoid electrodes
        tt = np.sort(rng.uniform(0, dur, n))
        times.append(tt); elecs.append(np.full(n, e)); amps.append(rng.normal(-30, 8, n))
    data = SpikeData(np.concatenate(times), np.concatenate(elecs).astype(np.int32),
                     np.concatenate(amps), sampling_rate=30000.0)
    rep = ar.detect_common_mode_artifacts(data, window_ms=2.0, group_size=8)
    # Independent biology: coincidence stays near its own chance level → low
    # enrichment, and the verdict must NOT call it contaminated.
    assert rep["enrichment_over_chance"] is None or rep["enrichment_over_chance"] < 3
    assert "CONTAMINATED" not in rep["verdict"]


def test_empty_data():
    data = SpikeData(np.array([]), np.array([], dtype=np.int32), np.array([]))
    rep = ar.detect_common_mode_artifacts(data)
    assert rep["artifact_fraction"] == 0.0
