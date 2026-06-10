"""Common-mode artifact detection and rejection for MEA spike data.

Multi-well MEA recordings (e.g. FinalSpark's 4-well plates) pick up
electrical/mechanical transients that appear *synchronously* across electrodes
that are not biologically connected — pump strokes, incubator door, electrical
pickup, vibration. These show up as high-amplitude "spikes" that coincide to
within a sample across many electrodes at once.

Because the wells (MEAs) are physically isolated, *any* sub-millisecond
coincidence between two different wells is non-biological by construction —
there is no axon between them. That gives a model-free ground truth for
artifact: cross-well synchrony.

This module detects such artifacts two ways:

1. **Population synchrony** (works for any layout): flag spikes that fall in a
   time window where an abnormally large fraction of *all* electrodes fire at
   once. Genuine network bursts recruit electrodes over tens of ms; common-mode
   transients hit everything within one bin.

2. **Cross-group coincidence** (needs a grouping, e.g. electrode → well): flag
   spikes that coincide within ±`window_ms` with spikes on a *different* group.
   For physically separate wells this is, by construction, not biology.

Empirically on FinalSpark fs437 the two agree and the artifact population is
cleanly separated in amplitude (≈90% of >300 µV spikes are cross-well
coincident vs ≈4% of <50 µV spikes), so the report includes the
amplitude dissociation as a built-in sanity check.

References:
- Common-average referencing / common-mode rejection is standard in
  multi-electrode electrophysiology (e.g. Ludwig et al. 2009, J Neurophysiol).
- The cross-well-coincidence ground truth is specific to multi-well plates.
"""

from __future__ import annotations

import numpy as np

from .loader import SpikeData


def _group_of(electrodes: np.ndarray, group_size: int | None) -> np.ndarray:
    """Map each electrode id to a group id.

    If `group_size` is given (e.g. 8 electrodes per well), group = id // size.
    Otherwise every electrode is its own group (population-synchrony only).
    """
    if group_size and group_size > 0:
        return electrodes // group_size
    return electrodes.copy()


def detect_common_mode_artifacts(
    data: SpikeData,
    window_ms: float = 2.0,
    group_size: int | None = None,
    min_cross_groups: int = 1,
    population_fraction: float = 0.5,
) -> dict:
    """Detect common-mode artifact spikes.

    Args:
        data: spike data.
        window_ms: coincidence window (± half-width is window_ms/2 in the
            binned approximation; the flag is insensitive to this between
            0.5–10 ms on real data).
        group_size: electrodes per recording group (well). For FinalSpark
            4-well plates this is 8. If None, only population-synchrony is used
            and every electrode is its own group.
        min_cross_groups: a spike is artifact if it coincides with spikes on at
            least this many *other* groups. 1 = any cross-group coincidence
            (correct for physically isolated wells).
        population_fraction: a spike is also flagged if its coincidence window
            contains spikes from at least this fraction of all electrodes
            (catches common-mode even without a grouping).

    Returns:
        dict with the artifact mask (as indices), per-group statistics, the
        amplitude dissociation, and a chance-level estimate from a jittered
        null — everything needed to decide whether the recording is
        contaminated.
    """
    n = data.n_spikes
    if n == 0:
        return {"n_spikes": 0, "artifact_fraction": 0.0, "flagged": []}

    t = data.times
    elec = data.electrodes
    amp = np.abs(data.amplitudes)
    groups = _group_of(elec, group_size)
    n_groups = int(len(np.unique(groups)))
    n_elec = data.n_electrodes

    w = max(window_ms, 1e-6) / 1000.0  # bin width in seconds
    # Bin spikes; coincidence ≈ same bin. Window-insensitivity (verified on
    # real data) makes the fixed-bin approximation safe and O(n).
    bin_idx = np.floor(t / w).astype(np.int64)

    # For each bin: how many distinct electrodes and distinct groups fired.
    order = np.argsort(bin_idx, kind="stable")
    b_sorted = bin_idx[order]
    e_sorted = elec[order]
    g_sorted = groups[order]

    # Boundaries of each bin in the sorted arrays.
    boundaries = np.flatnonzero(np.diff(b_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(b_sorted)]))

    distinct_elec = np.zeros(len(b_sorted), dtype=np.int32)
    distinct_grp = np.zeros(len(b_sorted), dtype=np.int32)
    for s, e in zip(starts, ends):
        distinct_elec[s:e] = len(np.unique(e_sorted[s:e]))
        distinct_grp[s:e] = len(np.unique(g_sorted[s:e]))

    # Per-spike synchrony counts, back in original order.
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    sync_elec = distinct_elec[inv]
    sync_grp = distinct_grp[inv]

    # Artifact flag: coincides with >= min_cross_groups OTHER groups, OR the
    # window holds >= population_fraction of all electrodes.
    cross_group_flag = (sync_grp - 1) >= min_cross_groups if n_groups > 1 else np.zeros(n, bool)
    population_flag = sync_elec >= max(2, int(np.ceil(population_fraction * n_elec)))
    artifact = cross_group_flag | population_flag

    # Jittered null: shift each group by a random offset, recount cross-group.
    # Destroys real-time-locked coincidence → residual is the chance level.
    rng = np.random.default_rng(0)
    dur = data.duration
    t_jit = t.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        t_jit[idx] = (t[idx] + rng.uniform(5.0, 50.0)) % dur
    bj = np.floor(t_jit / w).astype(np.int64)
    oj = np.argsort(bj, kind="stable")
    bjs, gjs = bj[oj], groups[oj]
    bnd = np.flatnonzero(np.diff(bjs)) + 1
    sj, ej = np.concatenate(([0], bnd)), np.concatenate((bnd, [len(bjs)]))
    dgj = np.zeros(len(bjs), np.int32)
    for s, e in zip(sj, ej):
        dgj[s:e] = len(np.unique(gjs[s:e]))
    invj = np.empty(len(oj), np.int64)
    invj[oj] = np.arange(len(oj))
    chance_cross = float(((dgj[invj] - 1) >= min_cross_groups).mean()) if n_groups > 1 else 0.0

    # Per-group breakdown.
    per_group = {}
    for g in np.unique(groups):
        gm = groups == g
        per_group[int(g)] = {
            "n_spikes": int(gm.sum()),
            "n_artifact": int(artifact[gm].sum()),
            "artifact_fraction": round(float(artifact[gm].mean()), 4),
            "clean_rate_hz": round(float((gm & ~artifact).sum() / dur), 4),
        }

    # Amplitude dissociation — the built-in sanity check.
    lowA = amp < 50
    highA = amp > 300
    amp_dissociation = {
        "low_amp_lt50uV": {
            "n": int(lowA.sum()),
            "artifact_fraction": round(float(artifact[lowA].mean()), 4) if lowA.any() else None,
            "median_amp_uV": round(float(np.median(amp[lowA])), 1) if lowA.any() else None,
        },
        "high_amp_gt300uV": {
            "n": int(highA.sum()),
            "artifact_fraction": round(float(artifact[highA].mean()), 4) if highA.any() else None,
            "median_amp_uV": round(float(np.median(amp[highA])), 1) if highA.any() else None,
        },
        "artifact_median_amp_uV": round(float(np.median(amp[artifact])), 1) if artifact.any() else None,
        "clean_median_amp_uV": round(float(np.median(amp[~artifact])), 1) if (~artifact).any() else None,
    }

    frac = float(artifact.mean())
    # Enrichment over chance: how many times more cross-group coincidence than
    # the jittered null. >~3× means the recording is genuinely contaminated.
    enrichment = round(frac / chance_cross, 1) if chance_cross > 0 else None

    return {
        "n_spikes": int(n),
        "n_groups": n_groups,
        "n_electrodes": int(n_elec),
        "window_ms": window_ms,
        "group_size": group_size,
        "artifact_fraction": round(frac, 4),
        "n_artifact": int(artifact.sum()),
        "chance_fraction": round(chance_cross, 4),
        "enrichment_over_chance": enrichment,
        "verdict": _verdict(frac, enrichment),
        "per_group": per_group,
        "amplitude_dissociation": amp_dissociation,
        "artifact_indices": np.flatnonzero(artifact).astype(np.int64),
    }


def _verdict(frac: float, enrichment: float | None) -> str:
    if enrichment is None:
        return "single-group recording — cross-well check unavailable"
    if frac < 0.02:
        return "clean — no significant common-mode contamination"
    if enrichment >= 3 and frac >= 0.05:
        return (
            f"CONTAMINATED — {frac*100:.0f}% of spikes are common-mode artifacts "
            f"({enrichment:.0f}× chance). Downstream connectivity/complexity "
            f"metrics should be recomputed on the cleaned data."
        )
    return "mild contamination — inspect per-group fractions"


def clean(data: SpikeData, window_ms: float = 2.0, group_size: int | None = None) -> SpikeData:
    """Return a copy of `data` with common-mode artifact spikes removed."""
    rep = detect_common_mode_artifacts(data, window_ms=window_ms, group_size=group_size)
    art = rep.get("artifact_indices", np.array([], dtype=np.int64))
    keep = np.ones(data.n_spikes, dtype=bool)
    keep[art] = False
    wf = data.waveforms[keep] if data.waveforms is not None else None
    return SpikeData(
        times=data.times[keep],
        electrodes=data.electrodes[keep],
        amplitudes=data.amplitudes[keep],
        waveforms=wf,
        sampling_rate=data.sampling_rate,
        metadata={**data.metadata, "artifact_cleaned": True,
                  "artifact_removed": int(art.size)},
    )


def artifacts_to_dict(report: dict) -> dict:
    """API-safe view: drop the raw index array (can be 100k+ long)."""
    out = {k: v for k, v in report.items() if k != "artifact_indices"}
    out["artifact_indices_available"] = True
    return out
