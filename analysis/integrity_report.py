"""Client-ready MEA Data Integrity Report.

Turns the common-mode artifact analysis into a self-contained, branded HTML
document (prints cleanly to PDF) that can be sent to a lab, a hardware vendor,
or a pharma/CRO client. *The report is the product* for the data-integrity /
QC business line — it is generated fully automatically, no manual writeup.

Design constraints:
- zero external dependencies (no matplotlib/plotly) — charts are inline SVG
- self-contained single HTML string — no asset files, prints to PDF as-is
- credible for a scientist reader: explicit methods + reproducibility footer
"""

from __future__ import annotations

import html
import numpy as np

from .loader import SpikeData
from .artifact_rejection import detect_common_mode_artifacts


# ── tiny inline-SVG chart helpers ───────────────────────────────────────────

def _bar_chart(labels, values, *, unit="%", color="#0ea5e9", danger_above=None, width=520, bar_h=26, gap=10):
    """Horizontal bar chart as inline SVG. values in 0..100 if unit='%'."""
    if not values:
        return ""
    vmax = max(max(values), 1e-9)
    scale = (width - 150) / (100 if unit == "%" else vmax)
    rows = []
    y = 0
    for lab, v in zip(labels, values):
        w = max(v * scale, 1)
        c = "#ef4444" if (danger_above is not None and v >= danger_above) else color
        rows.append(
            f'<text x="0" y="{y+bar_h*0.68}" font-size="13" fill="#334155">{html.escape(str(lab))}</text>'
            f'<rect x="110" y="{y}" width="{w:.0f}" height="{bar_h}" rx="4" fill="{c}"/>'
            f'<text x="{110+w+8:.0f}" y="{y+bar_h*0.68}" font-size="12" fill="#475569">{v:.0f}{unit}</text>'
        )
        y += bar_h + gap
    h = y
    return f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px">{"".join(rows)}</svg>'


def _timeline(hourly_frac, width=720, height=120):
    """Hourly artifact-fraction timeline as an inline SVG area chart."""
    n = len(hourly_frac)
    if n == 0:
        return ""
    dx = width / max(n - 1, 1)
    pts = " ".join(f"{i*dx:.1f},{height-(f*height):.1f}" for i, f in enumerate(hourly_frac))
    area = f"0,{height} {pts} {width},{height}"
    grid = "".join(
        f'<line x1="0" y1="{height*frac:.0f}" x2="{width}" y2="{height*frac:.0f}" stroke="#e2e8f0"/>'
        for frac in (0.25, 0.5, 0.75)
    )
    return (
        f'<svg viewBox="0 0 {width} {height+18}" width="100%" style="max-width:{width}px">'
        f'{grid}'
        f'<polygon points="{area}" fill="#fca5a5" opacity="0.5"/>'
        f'<polyline points="{pts}" fill="none" stroke="#ef4444" stroke-width="2"/>'
        f'<text x="0" y="{height+14}" font-size="11" fill="#94a3b8">recording start</text>'
        f'<text x="{width}" y="{height+14}" font-size="11" fill="#94a3b8" text-anchor="end">end</text>'
        f'</svg>'
    )


def _badge(verdict: str) -> str:
    contaminated = "CONTAMINATED" in verdict
    color = "#b91c1c" if contaminated else "#15803d"
    bg = "#fef2f2" if contaminated else "#f0fdf4"
    label = "CONTAMINATED" if contaminated else "CLEAN"
    return (
        f'<span style="display:inline-block;padding:6px 14px;border-radius:999px;'
        f'background:{bg};color:{color};font-weight:700;font-size:13px;'
        f'letter-spacing:0.04em;border:1px solid {color}33">{label}</span>'
    )


# ── report ──────────────────────────────────────────────────────────────────

def generate_integrity_report(
    data: SpikeData,
    dataset_name: str = "MEA dataset",
    group_size: int | None = None,
    window_ms: float = 2.0,
) -> str:
    """Generate a self-contained branded HTML data-integrity report."""
    gs = group_size if group_size is not None else (8 if data.n_electrodes == 32 else None)
    rep = detect_common_mode_artifacts(data, window_ms=window_ms, group_size=gs)

    art_idx = rep.get("artifact_indices", np.array([], dtype=np.int64))
    n = data.n_spikes
    frac = rep["artifact_fraction"]
    contaminated = "CONTAMINATED" in rep["verdict"]

    # per-well chart
    wells = sorted(rep["per_group"].keys())
    well_labels = [f"Well {w}" for w in wells]
    well_vals = [rep["per_group"][w]["artifact_fraction"] * 100 for w in wells]

    # amplitude dissociation chart
    ad = rep["amplitude_dissociation"]
    amp_labels = ["spikes <50 µV", "spikes >300 µV"]
    amp_vals = [
        (ad["low_amp_lt50uV"]["artifact_fraction"] or 0) * 100,
        (ad["high_amp_gt300uV"]["artifact_fraction"] or 0) * 100,
    ]

    # hourly artifact timeline
    timeline_svg = ""
    if art_idx.size and data.duration > 0:
        edges = np.arange(0, data.duration + 3600, 3600)
        all_h, _ = np.histogram(data.times, bins=edges)
        art_h, _ = np.histogram(data.times[art_idx], bins=edges)
        hourly = np.divide(art_h, all_h, out=np.zeros_like(art_h, float), where=all_h > 0)
        timeline_svg = _timeline(list(hourly))

    # downstream impact (only if grouped & contaminated): cross-well corr drop is
    # described qualitatively to avoid recomputing heavy metrics in the report.
    enrich = rep.get("enrichment_over_chance")
    clean_n = n - rep["n_artifact"]

    rows_summary = f"""
      <tr><td>Total spikes</td><td><b>{n:,}</b></td></tr>
      <tr><td>Electrodes / wells</td><td>{data.n_electrodes} / {rep['n_groups']}</td></tr>
      <tr><td>Recording duration</td><td>{data.duration/3600:.1f} h</td></tr>
      <tr><td>Coincidence window</td><td>±{window_ms:.1f} ms</td></tr>
    """

    rows_findings = f"""
      <tr><td>Common-mode artifact spikes</td><td><b style="color:{'#b91c1c' if contaminated else '#15803d'}">{rep['n_artifact']:,} ({frac*100:.1f}%)</b></td></tr>
      <tr><td>Enrichment over chance (jitter null)</td><td><b>{enrich}×</b> (chance {rep['chance_fraction']*100:.1f}%)</td></tr>
      <tr><td>Median amplitude — artifact vs clean</td><td>{ad['artifact_median_amp_uV']} µV vs {ad['clean_median_amp_uV']} µV</td></tr>
      <tr><td>Clean (biological) spikes retained</td><td>{clean_n:,}</td></tr>
    """

    impact_block = ""
    if contaminated:
        impact_block = """
        <h2>Why this matters</h2>
        <ul>
          <li><b>Connectivity / functional networks are inflated.</b> Synchronous transients create spurious cross-electrode and cross-well "connections" that are not biology.</li>
          <li><b>Complexity / criticality is biased.</b> A single common-mode transient registers as a giant population avalanche, shifting branching-ratio and power-law estimates.</li>
          <li><b>Slow rhythms can be confounded.</b> If the artifact source follows a wall-clock schedule (building HVAC, scheduled pumps), apparent day/night or circadian structure can be artifactual rather than endogenous.</li>
          <li><b>Low-activity wells are most affected</b> — where real spikes are few, the artifact can dominate the recording.</li>
        </ul>"""

    methods = f"""
      <h2>Methods &amp; reproducibility</h2>
      <p class="methods">Artifact ground truth is model-free: in a multi-well plate the wells are
      physically isolated, so any sub-millisecond coincidence <i>between different wells</i> cannot be
      biological — there is no axon between them. A spike is flagged when it coincides within ±{window_ms:.1f} ms
      with a spike on a different well, or when an abnormal fraction of all electrodes fire in the same window.
      Significance is established against a jittered null (each well randomly time-shifted), reported as
      enrichment over chance. The amplitude dissociation (artifact spikes are high-amplitude, biological
      spikes low-amplitude) is an independent corroboration, not an input to the flag.</p>
      <p class="methods">Generated by NeuroBridge · neurocomputers.io · analysis/artifact_rejection.py.
      Re-run: <code>detect_common_mode_artifacts(data, window_ms={window_ms}, group_size={gs})</code>.</p>
    """

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Data Integrity Report — {html.escape(dataset_name)}</title>
<style>
  * {{ box-sizing:border-box }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#0f172a;
         max-width:820px; margin:0 auto; padding:48px 40px; line-height:1.55; background:#fff }}
  .brand {{ font-size:13px; letter-spacing:0.12em; text-transform:uppercase; color:#0ea5e9; font-weight:700 }}
  h1 {{ font-size:26px; margin:6px 0 2px }}
  h2 {{ font-size:17px; margin:32px 0 10px; padding-bottom:6px; border-bottom:1px solid #e2e8f0 }}
  .sub {{ color:#64748b; font-size:14px; margin-bottom:18px }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin:8px 0 }}
  td {{ padding:7px 4px; border-bottom:1px solid #f1f5f9 }}
  td:first-child {{ color:#475569 }}
  td:last-child {{ text-align:right }}
  .verdict {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:18px 20px; margin:18px 0 }}
  .verdict p {{ margin:10px 0 0; font-size:14px; color:#334155 }}
  ul {{ font-size:14px; color:#334155; padding-left:20px }} li {{ margin:5px 0 }}
  .methods {{ font-size:12.5px; color:#64748b }}
  .chartcap {{ font-size:12px; color:#94a3b8; margin:2px 0 14px }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12px }}
  footer {{ margin-top:40px; padding-top:14px; border-top:1px solid #e2e8f0; font-size:12px; color:#94a3b8 }}
</style></head>
<body>
  <div class="brand">NeuroBridge · Data Integrity Report</div>
  <h1>{html.escape(dataset_name)}</h1>
  <div class="sub">Common-mode artifact audit · multi-well MEA recording</div>

  <div class="verdict">
    {_badge(rep['verdict'])}
    <p>{html.escape(rep['verdict'])}</p>
  </div>

  <h2>Dataset</h2>
  <table>{rows_summary}</table>

  <h2>Findings</h2>
  <table>{rows_findings}</table>

  <h2>Artifact by well</h2>
  {_bar_chart(well_labels, well_vals, danger_above=40)}
  <div class="chartcap">Fraction of each well's spikes that are cross-well-synchronous (artifact). Red ≥ 40%.</div>

  <h2>Amplitude dissociation</h2>
  {_bar_chart(amp_labels, amp_vals, color="#6366f1", danger_above=50)}
  <div class="chartcap">High-amplitude spikes are overwhelmingly artifact; low-amplitude spikes are near chance — an independent confirmation.</div>

  {"<h2>Artifact over time</h2>" + timeline_svg + '<div class="chartcap">Hourly artifact fraction. Time-clustered (vs uniform) implicates an intermittent external source.</div>' if timeline_svg else ""}

  {impact_block}

  <h2>Recommendation</h2>
  <p style="font-size:14px;color:#334155">{'Remove flagged artifact spikes (or amplitude-gate) before computing connectivity, complexity, criticality, or rhythm metrics. NeuroBridge can deliver a cleaned dataset and re-run all downstream analyses on request.' if contaminated else 'No significant common-mode contamination detected. Standard analyses can proceed on the raw spike data.'}</p>

  {methods}

  <footer>NeuroBridge — open-source MEA analysis · neurocomputers.io · MIT-licensed · Report auto-generated, no manual editing.</footer>
</body></html>"""
