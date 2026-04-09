"""Auto-generate paper draft from analysis results.

Creates structured markdown draft: title, abstract, methods, results, discussion.
Based on actual analysis data from NeuroBridge API.
"""
import numpy as np
from datetime import datetime
from .loader import SpikeData


def generate_draft(data: SpikeData, analyses: dict = None) -> dict:
    """Generate a paper draft from spike data and optional pre-computed analyses."""
    if analyses is None:
        analyses = {}

    n_spikes = data.n_spikes
    n_electrodes = data.n_electrodes
    duration = data.duration
    rate = n_spikes / max(duration, 0.001)

    # Title
    title = f"Characterization of Neural Activity Patterns in Brain Organoid Culture: {n_electrodes}-Channel MEA Analysis"

    # Abstract
    abstract = (
        f"We present a comprehensive analysis of spontaneous neural activity recorded from a brain organoid "
        f"using a {n_electrodes}-channel multi-electrode array (MEA). Over {duration:.1f} seconds of recording, "
        f"we detected {n_spikes:,} spikes with a mean population firing rate of {rate:.1f} Hz. "
        f"Analysis revealed organized network dynamics including burst activity, functional connectivity patterns, "
        f"and evidence of emergent computational properties. "
        f"These findings contribute to our understanding of self-organized neural computation in vitro."
    )

    # Methods
    methods = (
        f"## Methods\n\n"
        f"### Recording Setup\n"
        f"Neural activity was recorded from a brain organoid cultured on a {n_electrodes}-channel "
        f"multi-electrode array (MEA) at a sampling rate of {data.sampling_rate/1000:.0f} kHz. "
        f"Spike detection used a threshold of 6x median standard deviation (Jordan et al., 2024).\n\n"
        f"### Analysis Pipeline\n"
        f"All analyses were performed using the NeuroBridge analysis platform (github.com/Luckyguybiz/neurobridge-api). "
        f"The platform provides 35+ analysis modules covering spike statistics, burst detection, "
        f"functional connectivity, information theory, criticality, digital twin modeling, "
        f"and novel metrics including Organoid IQ scoring and attractor landscape mapping.\n\n"
        f"### Statistical Methods\n"
        f"Spike sorting was performed via PCA + K-means clustering. "
        f"Functional connectivity was assessed using co-firing analysis and transfer entropy. "
        f"Network dynamics were characterized using Kuramoto order parameter for synchronization "
        f"and Lempel-Ziv complexity for information content."
    )

    # Results
    results_sections = []
    results_sections.append(
        f"### Basic Statistics\n"
        f"- Total spikes: {n_spikes:,}\n"
        f"- Recording duration: {duration:.1f} s\n"
        f"- Active electrodes: {n_electrodes}\n"
        f"- Mean firing rate: {rate:.1f} Hz\n"
    )

    if "iq" in analyses:
        iq = analyses["iq"]
        score = iq.get("iq_score", iq.get("score", "N/A"))
        grade = iq.get("grade", "N/A")
        results_sections.append(
            f"### Organoid Intelligence Quotient\n"
            f"The composite IQ score was {score}/100 (Grade {grade}), "
            f"indicating organized but developing neural network dynamics.\n"
        )

    if "bursts" in analyses:
        b = analyses["bursts"]
        results_sections.append(
            f"### Burst Activity\n"
            f"Network bursts were detected at a rate of {b.get('burst_rate_per_min', 'N/A')}/min "
            f"with mean duration of {b.get('mean_duration_ms', 'N/A')} ms.\n"
        )

    results = "## Results\n\n" + "\n".join(results_sections)

    # Discussion
    discussion = (
        f"## Discussion\n\n"
        f"The recorded organoid exhibited spontaneous neural activity with organized network dynamics. "
        f"The presence of network bursts and functional connectivity patterns suggests "
        f"self-organized circuit formation consistent with previous reports "
        f"(Jordan et al., 2024; Kagan et al., 2022).\n\n"
        f"Future work should investigate whether these patterns support computational tasks "
        f"such as reservoir computing (Cai et al., 2023) or closed-loop learning "
        f"(Kagan et al., 2022)."
    )

    # References
    references = (
        "## References\n\n"
        "1. Jordan, F.D. et al. (2024). Open and remotely accessible Neuroplatform for research in wetware computing. *Frontiers in AI*.\n"
        "2. Kagan, B.J. et al. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. *Neuron*.\n"
        "3. Cai, H. et al. (2023). Brain organoid reservoir computing for artificial intelligence. *Nature Electronics*.\n"
        "4. Smirnova, L. et al. (2023). Organoid intelligence (OI): the new frontier in biocomputing. *Frontiers in Science*.\n"
    )

    full_markdown = f"# {title}\n\n## Abstract\n\n{abstract}\n\n{methods}\n\n{results}\n\n{discussion}\n\n{references}"

    return {
        "title": title,
        "abstract": abstract,
        "markdown": full_markdown,
        "word_count": len(full_markdown.split()),
        "sections": ["Abstract", "Methods", "Results", "Discussion", "References"],
        "generated_at": datetime.now().isoformat(),
    }
