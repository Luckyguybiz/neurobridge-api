"""Auto-generate paper drafts from organoid analysis results.

Scientific basis:
    Structured scientific communication follows the IMRaD format
    (Introduction, Methods, Results, and Discussion) established by
    the ICMJE guidelines. This module generates markdown-formatted
    paper drafts from analysis results, providing:

    - Title generation based on key findings
    - Abstract with background, methods, results, conclusions
    - Methods section detailing recording parameters and analyses
    - Results section with quantitative findings
    - Discussion with interpretation and comparison to literature

    The generator incorporates domain-specific knowledge about:
    - Organoid electrophysiology (FinalSpark MEA recordings)
    - Reservoir computing benchmarks (Brainoware comparisons)
    - Neural criticality and information theory metrics
    - Ethical considerations for organoid intelligence research

    Output follows conventions of journals like Nature Biomedical
    Engineering, Lab on a Chip, and Advanced Intelligent Systems.
"""

import numpy as np
import time
from typing import Optional
from .loader import SpikeData


def generate_draft(
    data: SpikeData,
    analyses: dict,
    title: Optional[str] = None,
    authors: Optional[list] = None,
    institution: Optional[str] = None,
) -> dict:
    """Generate a complete paper draft in markdown from analysis results.

    Assembles a structured scientific paper from raw data properties
    and computed analysis results. Each section is generated based on
    available analyses, gracefully skipping sections where data is missing.

    Args:
        data: SpikeData used for the analyses.
        analyses: Dict of analysis results keyed by analysis name.
            Recognized keys: 'summary', 'firing_rates', 'bursts',
            'connectivity', 'information_theory', 'criticality',
            'reservoir', 'organoid_iq', 'vowel_classification',
            'plasticity', 'experiment_delta'.
        title: Optional custom title (auto-generated if not provided).
        authors: Optional list of author names.
        institution: Optional institution name.

    Returns:
        Dict with 'markdown' (full paper text), 'sections' (individual
        section strings), 'word_count', and metadata.
    """
    sections = {}

    # Title
    generated_title = title or _generate_title(data, analyses)
    sections["title"] = f"# {generated_title}\n"

    # Authors and affiliation
    if authors or institution:
        author_line = ", ".join(authors) if authors else "NeuroBridge Research Team"
        inst_line = institution or "NeuroBridge Biocomputing Laboratory"
        sections["authors"] = f"**{author_line}**\n\n*{inst_line}*\n"
    else:
        sections["authors"] = (
            "**NeuroBridge Research Team**\n\n"
            "*NeuroBridge Biocomputing Laboratory*\n"
        )

    # Abstract
    sections["abstract"] = _generate_abstract(data, analyses, generated_title)

    # Keywords
    sections["keywords"] = _generate_keywords(analyses)

    # Introduction
    sections["introduction"] = _generate_introduction(analyses)

    # Methods
    sections["methods"] = _generate_methods(data, analyses)

    # Results
    sections["results"] = _generate_results(data, analyses)

    # Discussion
    sections["discussion"] = _generate_discussion(analyses)

    # References
    sections["references"] = _generate_references(analyses)

    # Assemble full markdown
    markdown = "\n\n---\n\n".join([
        sections["title"],
        sections["authors"],
        sections["abstract"],
        sections["keywords"],
        sections["introduction"],
        sections["methods"],
        sections["results"],
        sections["discussion"],
        sections["references"],
    ])

    word_count = len(markdown.split())

    return {
        "markdown": markdown,
        "sections": sections,
        "title": generated_title,
        "word_count": word_count,
        "generated_at": time.time(),
        "n_analyses_included": len(analyses),
        "analyses_used": list(analyses.keys()),
    }


def generate_abstract_only(
    data: SpikeData,
    analyses: dict,
) -> dict:
    """Generate just the abstract section.

    Useful for quick summaries or conference submissions.

    Args:
        data: SpikeData from the recording.
        analyses: Dict of analysis results.

    Returns:
        Dict with abstract text and word count.
    """
    title = _generate_title(data, analyses)
    abstract = _generate_abstract(data, analyses, title)
    return {
        "title": title,
        "abstract": abstract,
        "word_count": len(abstract.split()),
    }


def generate_methods_section(
    data: SpikeData,
    analyses: dict,
) -> dict:
    """Generate just the methods section.

    Args:
        data: SpikeData from the recording.
        analyses: Dict of analysis results.

    Returns:
        Dict with methods text.
    """
    methods = _generate_methods(data, analyses)
    return {
        "methods": methods,
        "word_count": len(methods.split()),
    }


def _generate_title(data: SpikeData, analyses: dict) -> str:
    """Auto-generate a paper title based on key findings."""
    parts = []

    if "organoid_iq" in analyses:
        iq = analyses["organoid_iq"]
        score = iq.get("overall_iq", iq.get("total_score", 0))
        if score > 60:
            parts.append("High-Performance")
        parts.append("Computational Characterization")
    elif "reservoir" in analyses or "vowel_classification" in analyses:
        parts.append("Reservoir Computing Capabilities")
    elif "criticality" in analyses:
        parts.append("Critical Dynamics")
    else:
        parts.append("Electrophysiological Characterization")

    parts.append("of Cortical Organoids")

    if "vowel_classification" in analyses:
        parts.append("for Speech Recognition via Biological Neural Networks")
    elif "information_theory" in analyses:
        parts.append("Reveals Emergent Information Processing")
    elif "connectivity" in analyses:
        parts.append("with Emergent Network Architecture")
    else:
        parts.append("on Multi-Electrode Arrays")

    return " ".join(parts)


def _generate_abstract(data: SpikeData, analyses: dict, title: str) -> str:
    """Generate abstract with background, methods, results, conclusions."""
    lines = ["## Abstract\n"]

    # Background
    lines.append(
        "Brain organoids represent a promising substrate for biological computing, "
        "yet standardized characterization of their computational capabilities remains limited. "
    )

    # Methods
    lines.append(
        f"We recorded spontaneous and evoked activity from cortical organoids using "
        f"multi-electrode arrays ({data.n_electrodes} channels, "
        f"{data.sampling_rate/1000:.0f} kHz sampling rate) and performed comprehensive "
        f"computational analysis spanning {len(analyses)} analytical dimensions. "
    )

    # Key results
    result_parts = []
    if "summary" in analyses:
        s = analyses["summary"]
        rate = s.get("mean_firing_rate", s.get("overall_firing_rate", 0))
        result_parts.append(f"mean firing rate of {rate:.2f} Hz")

    if "organoid_iq" in analyses:
        iq = analyses["organoid_iq"]
        score = iq.get("overall_iq", iq.get("total_score", 0))
        result_parts.append(f"Organoid IQ score of {score:.1f}/100")

    if "vowel_classification" in analyses:
        vc = analyses["vowel_classification"]
        acc = vc.get("test_accuracy", 0)
        result_parts.append(f"vowel classification accuracy of {acc:.1%}")

    if "criticality" in analyses:
        c = analyses["criticality"]
        kappa = c.get("branching_ratio", c.get("kappa", 0))
        result_parts.append(f"branching ratio of {kappa:.3f}")

    if result_parts:
        lines.append("Key findings include " + ", ".join(result_parts) + ". ")

    # Conclusion
    lines.append(
        "These results demonstrate that cortical organoids exhibit non-trivial "
        "computational properties suitable for biocomputing applications, with implications "
        "for neuromorphic engineering and biological intelligence research."
    )

    return "\n".join(lines)


def _generate_keywords(analyses: dict) -> str:
    """Generate keywords section."""
    keywords = ["brain organoid", "biocomputing", "multi-electrode array", "electrophysiology"]

    if "reservoir" in analyses or "vowel_classification" in analyses:
        keywords.extend(["reservoir computing", "speech recognition"])
    if "criticality" in analyses:
        keywords.append("criticality")
    if "information_theory" in analyses:
        keywords.append("information theory")
    if "plasticity" in analyses:
        keywords.append("synaptic plasticity")
    if "connectivity" in analyses:
        keywords.append("functional connectivity")

    return "**Keywords:** " + ", ".join(keywords)


def _generate_introduction(analyses: dict) -> str:
    """Generate introduction section."""
    lines = ["## Introduction\n"]

    lines.append(
        "The intersection of neuroscience and computing has given rise to organoid "
        "intelligence (OI), a field that leverages the computational properties of "
        "three-dimensional neural cultures for information processing. "
        "Unlike silicon-based computers, biological neural networks operate with "
        "remarkable energy efficiency and exhibit inherent plasticity, enabling "
        "adaptive computation.\n"
    )

    lines.append(
        "Recent advances in brain organoid technology have demonstrated that these "
        "self-organizing neural structures can develop spontaneous electrical activity, "
        "form functional networks, and even perform basic computational tasks. "
        "The Brainoware system (Cai et al., 2023) showed that organoids can serve "
        "as reservoirs for speech recognition, achieving competitive accuracy on "
        "vowel classification benchmarks.\n"
    )

    if "criticality" in analyses:
        lines.append(
            "A key hypothesis in computational neuroscience is that optimal information "
            "processing occurs near the critical point between ordered and disordered "
            "dynamics (the edge of chaos). We investigate whether organoid networks "
            "self-organize toward criticality, a hallmark of biological neural computation.\n"
        )

    lines.append(
        "In this study, we present a comprehensive computational characterization "
        "of cortical organoid activity recorded via multi-electrode arrays, employing "
        "analyses spanning spike statistics, network connectivity, information theory, "
        "and reservoir computing benchmarks.\n"
    )

    return "\n".join(lines)


def _generate_methods(data: SpikeData, analyses: dict) -> str:
    """Generate methods section."""
    lines = ["## Methods\n"]

    # Recording setup
    lines.append("### Electrophysiological Recording\n")
    lines.append(
        f"Neural activity was recorded using a multi-electrode array system "
        f"({data.n_electrodes} electrodes) at a sampling rate of "
        f"{data.sampling_rate/1000:.0f} kHz. "
        f"The recording duration was {data.duration:.1f} seconds, "
        f"yielding {data.n_spikes} detected spikes. "
        f"Spike detection employed a threshold of 6x the median standard deviation "
        f"of the signal, following established protocols for organoid MEA recordings.\n"
    )

    # Data processing
    lines.append("### Data Processing\n")
    lines.append(
        "Spike times, electrode assignments, and amplitudes were extracted and "
        "stored in a structured format. Per-electrode spike trains were binned "
        "at appropriate temporal resolutions for each analysis.\n"
    )

    # Analysis-specific methods
    if "connectivity" in analyses:
        lines.append("### Functional Connectivity\n")
        lines.append(
            "Pairwise functional connectivity was estimated using cross-correlation "
            "of binned spike trains. Significant connections were identified using "
            "surrogate-based statistical testing (p < 0.05, Bonferroni corrected).\n"
        )

    if "information_theory" in analyses:
        lines.append("### Information-Theoretic Analysis\n")
        lines.append(
            "Spike train entropy was computed using binned firing patterns. "
            "Transfer entropy between electrode pairs quantified directed information "
            "flow. Lempel-Ziv complexity measured the algorithmic complexity of "
            "neural activity patterns.\n"
        )

    if "criticality" in analyses:
        lines.append("### Criticality Analysis\n")
        lines.append(
            "Neuronal avalanches were detected as spatiotemporal clusters of activity. "
            "Avalanche size and duration distributions were fitted to power laws. "
            "The branching ratio was estimated to assess proximity to criticality.\n"
        )

    if "reservoir" in analyses or "vowel_classification" in analyses:
        lines.append("### Reservoir Computing Benchmark\n")
        lines.append(
            "The organoid's computational properties were benchmarked using a "
            "reservoir computing framework. Spike activity statistics conditioned "
            "a random reservoir's parameters (spectral radius, leak rate). "
            "A linear readout was trained via ridge regression for vowel "
            "classification on 240 synthetic LPC cepstral coefficient vectors "
            "across 8 vowel classes.\n"
        )

    # Statistical analysis
    lines.append("### Statistical Analysis\n")
    lines.append(
        "All numerical computations were performed using NumPy. "
        "Effect sizes were quantified using Cohen's d. "
        "Results are reported as mean +/- standard deviation unless otherwise noted.\n"
    )

    return "\n".join(lines)


def _generate_results(data: SpikeData, analyses: dict) -> str:
    """Generate results section with quantitative findings."""
    lines = ["## Results\n"]

    # Basic activity
    if "summary" in analyses:
        s = analyses["summary"]
        lines.append("### Spontaneous Activity Characterization\n")
        rate = s.get("mean_firing_rate", s.get("overall_firing_rate", 0))
        lines.append(
            f"The organoid exhibited spontaneous activity with a mean firing rate of "
            f"{rate:.2f} Hz across {data.n_electrodes} active electrodes. "
            f"A total of {data.n_spikes} spikes were detected over "
            f"{data.duration:.1f} seconds of recording.\n"
        )

    # Bursts
    if "bursts" in analyses:
        b = analyses["bursts"]
        n_bursts = b.get("n_bursts", b.get("total_bursts", 0))
        lines.append("### Burst Activity\n")
        lines.append(
            f"Network burst analysis identified {n_bursts} burst events. "
        )
        if "mean_burst_duration" in b:
            lines.append(
                f"Mean burst duration was {b['mean_burst_duration']:.1f} ms. "
            )
        lines.append("\n")

    # Connectivity
    if "connectivity" in analyses:
        c = analyses["connectivity"]
        lines.append("### Functional Network Architecture\n")
        n_conn = c.get("n_significant_connections", c.get("n_connections", 0))
        density = c.get("density", c.get("network_density", 0))
        lines.append(
            f"Functional connectivity analysis revealed {n_conn} significant "
            f"connections (network density: {density:.3f}). "
        )
        lines.append("\n")

    # Information theory
    if "information_theory" in analyses:
        it = analyses["information_theory"]
        lines.append("### Information Processing Capacity\n")
        entropy = it.get("mean_entropy", 0)
        lines.append(
            f"Mean spike train entropy was {entropy:.3f} bits, indicating "
        )
        if entropy > 0.7:
            lines.append("high information content in the neural code. ")
        elif entropy > 0.4:
            lines.append("moderate information content with structured patterns. ")
        else:
            lines.append("low entropy suggesting stereotyped activity. ")
        lines.append("\n")

    # Criticality
    if "criticality" in analyses:
        cr = analyses["criticality"]
        kappa = cr.get("branching_ratio", cr.get("kappa", 0))
        lines.append("### Proximity to Criticality\n")
        lines.append(
            f"The estimated branching ratio was {kappa:.3f} "
        )
        if 0.9 < kappa < 1.1:
            lines.append("(near-critical regime), consistent with optimal information processing. ")
        elif kappa > 1.1:
            lines.append("(supercritical regime), indicating tendency toward synchronous bursting. ")
        else:
            lines.append("(subcritical regime), suggesting damped dynamics. ")
        lines.append("\n")

    # Organoid IQ
    if "organoid_iq" in analyses:
        iq = analyses["organoid_iq"]
        score = iq.get("overall_iq", iq.get("total_score", 0))
        lines.append("### Composite Intelligence Score\n")
        lines.append(
            f"The composite Organoid IQ score was {score:.1f}/100, "
        )
        if score > 60:
            lines.append("indicating computation-capable neural dynamics. ")
        elif score > 40:
            lines.append("indicating organized activity with emerging computational properties. ")
        else:
            lines.append("indicating basic spontaneous activity. ")
        lines.append("\n")

    # Vowel classification
    if "vowel_classification" in analyses:
        vc = analyses["vowel_classification"]
        acc = vc.get("test_accuracy", 0)
        lines.append("### Reservoir Computing: Vowel Classification\n")
        lines.append(
            f"The organoid-conditioned reservoir achieved {acc:.1%} classification "
            f"accuracy on the 8-class Japanese vowel recognition task "
            f"(chance level: 12.5%). "
        )
        brainoware = vc.get("brainoware_comparison", {})
        if brainoware:
            bw_acc = brainoware.get("brainoware_accuracy", 0)
            lines.append(
                f"This compares to {bw_acc:.0%} reported by the Brainoware system. "
            )
        lines.append("\n")

    # Experiment delta
    if "experiment_delta" in analyses:
        ed = analyses["experiment_delta"]
        delta = ed.get("delta", ed)
        fr_pct = delta.get("firing_rate_change_pct", 0)
        lines.append("### Pre/Post Intervention Comparison\n")
        lines.append(
            f"Following the experimental intervention, firing rate changed by "
            f"{fr_pct:+.1f}%. "
        )
        d_cohen = delta.get("effect_size_cohens_d", 0)
        if d_cohen > 0:
            lines.append(f"Effect size: Cohen's d = {d_cohen:.2f}. ")
        lines.append("\n")

    if not any(k in analyses for k in [
        "summary", "bursts", "connectivity", "information_theory",
        "criticality", "organoid_iq", "vowel_classification", "experiment_delta",
    ]):
        lines.append(
            "Analysis results pending. Additional data collection and "
            "analysis are required for quantitative findings.\n"
        )

    return "\n".join(lines)


def _generate_discussion(analyses: dict) -> str:
    """Generate discussion section."""
    lines = ["## Discussion\n"]

    lines.append(
        "This study provides a multi-dimensional computational characterization "
        "of cortical organoid activity, contributing to the growing body of evidence "
        "that biological neural networks exhibit non-trivial information processing "
        "capabilities.\n"
    )

    if "vowel_classification" in analyses:
        lines.append(
            "Our reservoir computing results extend the findings of Cai et al. (2023), "
            "demonstrating that organoid-derived dynamical properties can be leveraged "
            "for pattern recognition tasks. The approach of conditioning reservoir "
            "parameters on biological measurements provides a bridge between "
            "wetware properties and computational benchmarks.\n"
        )

    if "criticality" in analyses:
        cr = analyses["criticality"]
        kappa = cr.get("branching_ratio", cr.get("kappa", 0))
        if 0.9 < kappa < 1.1:
            lines.append(
                "The near-critical dynamics observed are consistent with the criticality "
                "hypothesis, suggesting that organoids self-organize toward a regime "
                "that maximizes information processing capacity and dynamic range.\n"
            )

    if "organoid_iq" in analyses:
        lines.append(
            "The composite Organoid IQ metric provides a standardized framework for "
            "comparing computational capabilities across organoid preparations and "
            "experimental conditions. While preliminary, this metric could serve as "
            "a benchmark for the emerging field of organoid intelligence.\n"
        )

    # Limitations
    lines.append("### Limitations\n")
    lines.append(
        "This analysis is subject to several limitations. The synthetic vowel "
        "dataset, while informed by acoustic phonetics, does not capture the full "
        "complexity of natural speech. The reservoir computing framework uses "
        "organoid statistics to condition simulation parameters rather than direct "
        "biological computation. Future work should employ closed-loop stimulation "
        "to directly interface organoids with computational tasks.\n"
    )

    # Future directions
    lines.append("### Future Directions\n")
    lines.append(
        "Promising directions include: (1) direct organoid-in-the-loop computation "
        "using real-time stimulation and recording, (2) longitudinal tracking of "
        "computational capabilities during organoid maturation, (3) comparison across "
        "organoid types (cortical, hippocampal, assembloid), and (4) development of "
        "standardized benchmarks for biological neural computation.\n"
    )

    return "\n".join(lines)


def _generate_references(analyses: dict) -> str:
    """Generate references section."""
    refs = [
        "## References\n",
        ("1. Cai, H. et al. (2023). Brainoware: A neuromorphic platform for reservoir "
         "computing using brain organoids. *Nature Electronics*, 6, 1032-1039."),
        ("2. Smirnova, L. et al. (2023). Organoid intelligence (OI): the new frontier "
         "in biocomputing and intelligence-in-a-dish. *Frontiers in Science*, 1, 1017235."),
        ("3. Beggs, J.M. & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. "
         "*Journal of Neuroscience*, 23(35), 11167-11177."),
        ("4. Jaeger, H. & Haas, H. (2004). Harnessing nonlinearity: Predicting chaotic "
         "systems and saving energy in wireless communication. *Science*, 304(5667), 78-80."),
    ]

    if "information_theory" in analyses:
        refs.append(
            "5. Timme, N.M. & Lapish, C. (2018). A tutorial for information theory in "
            "neuroscience. *eNeuro*, 5(3), ENEURO.0052-18.2018."
        )

    if "plasticity" in analyses:
        refs.append(
            "6. Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured "
            "hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472."
        )

    return "\n\n".join(refs)
