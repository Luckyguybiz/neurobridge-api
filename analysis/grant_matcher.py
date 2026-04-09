"""Grant matching based on organoid analysis results.

Scientific basis:
    Biocomputing and organoid intelligence research is supported by
    several major funding programs worldwide. This module matches
    organoid experimental results against grant program criteria to
    identify the best-fit funding opportunities.

    Supported grant programs:

    1. NSF BEGIN OI (Biocomputing Enabled by Genomics and Intelligence
       of Organoid Interfaces) -- NSF's flagship program for organoid
       computing. Focuses on: biocomputing, neural interfaces, reservoir
       computing, learning in biological systems.

    2. Astana Hub (Kazakhstan Innovation Fund) -- Supports biotech
       and AI convergence projects. Relevant for: computational
       neuroscience, AI-bio interfaces, novel computing paradigms.

    3. EU Horizon Europe -- Cluster 1 (Health) and Cluster 4 (Digital).
       Relevant topics: brain-inspired computing, organoid technology,
       neural interfaces, ethical AI/bio research.

    4. DARPA (Defense Advanced Research Projects Agency) -- Programs
       like N3 (Next-Generation Nonsurgical Neurotechnology) and
       BTO (Biological Technologies Office). Focus: neural computation,
       biological computing substrates, hybrid bio-digital systems.

    5. NIH BRAIN Initiative -- Supports fundamental neuroscience and
       neurotechnology. Relevant for: neural recording, brain-machine
       interfaces, computational neuroscience tools.

    Match scoring considers:
    - Alignment of research focus with grant priorities
    - Technical maturity of the organoid system (derived from metrics)
    - Novelty of findings
    - Ethical compliance readiness
    - Team/infrastructure requirements
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# Grant program database
_GRANTS = {
    "NSF_BEGIN_OI": {
        "name": "NSF BEGIN OI: Biocomputing Enabled by Genomics and Intelligence of Organoid Interfaces",
        "agency": "National Science Foundation (NSF)",
        "country": "USA",
        "max_funding_usd": 1_500_000,
        "duration_years": 3,
        "deadline": "2026-09-15",
        "url": "https://www.nsf.gov/pubs/2024/nsf24523/nsf24523.htm",
        "focus_areas": [
            "biocomputing",
            "organoid_intelligence",
            "reservoir_computing",
            "neural_interfaces",
            "learning_plasticity",
        ],
        "requirements": {
            "min_electrodes": 8,
            "needs_computation_evidence": True,
            "needs_ethics_plan": True,
            "us_institution_required": True,
        },
        "weight_factors": {
            "reservoir_computing": 0.25,
            "plasticity": 0.20,
            "organoid_iq": 0.15,
            "information_theory": 0.15,
            "criticality": 0.10,
            "ethics_readiness": 0.15,
        },
    },
    "ASTANA_HUB": {
        "name": "Astana Hub Innovation Grant: Biotech & AI Convergence",
        "agency": "Astana Hub International Technopark",
        "country": "Kazakhstan",
        "max_funding_usd": 200_000,
        "duration_years": 2,
        "deadline": "2026-12-01",
        "url": "https://astanahub.com/en/grants",
        "focus_areas": [
            "biotech_ai_convergence",
            "computational_neuroscience",
            "novel_computing",
            "international_collaboration",
        ],
        "requirements": {
            "min_electrodes": 4,
            "needs_computation_evidence": False,
            "needs_ethics_plan": False,
            "registration_in_kazakhstan": False,
        },
        "weight_factors": {
            "novelty": 0.30,
            "technical_maturity": 0.25,
            "international_collaboration": 0.20,
            "commercial_potential": 0.15,
            "data_quality": 0.10,
        },
    },
    "EU_HORIZON": {
        "name": "EU Horizon Europe: Brain-Inspired Technologies (Cluster 4, Destination 3)",
        "agency": "European Commission",
        "country": "EU",
        "max_funding_usd": 4_000_000,
        "duration_years": 4,
        "deadline": "2026-10-22",
        "url": "https://ec.europa.eu/info/funding-tenders/opportunities/portal",
        "focus_areas": [
            "brain_inspired_computing",
            "organoid_technology",
            "neural_interfaces",
            "ethical_research",
            "neuromorphic_engineering",
        ],
        "requirements": {
            "min_electrodes": 8,
            "needs_computation_evidence": True,
            "needs_ethics_plan": True,
            "eu_consortium_required": True,
            "min_partners": 3,
        },
        "weight_factors": {
            "scientific_excellence": 0.30,
            "ethics_compliance": 0.20,
            "impact_potential": 0.20,
            "organoid_iq": 0.15,
            "consortium_readiness": 0.15,
        },
    },
    "DARPA_BTO": {
        "name": "DARPA BTO: Biological Computing Substrates",
        "agency": "Defense Advanced Research Projects Agency (DARPA)",
        "country": "USA",
        "max_funding_usd": 5_000_000,
        "duration_years": 4,
        "deadline": "2026-08-01",
        "url": "https://www.darpa.mil/work-with-us/opportunities",
        "focus_areas": [
            "biological_computing",
            "hybrid_bio_digital",
            "neural_computation",
            "unconventional_computing",
            "low_power_computing",
        ],
        "requirements": {
            "min_electrodes": 16,
            "needs_computation_evidence": True,
            "needs_reservoir_benchmark": True,
            "us_institution_required": True,
        },
        "weight_factors": {
            "reservoir_computing": 0.30,
            "computation_capability": 0.25,
            "technical_maturity": 0.20,
            "novelty": 0.15,
            "scalability": 0.10,
        },
    },
    "NIH_BRAIN": {
        "name": "NIH BRAIN Initiative: Neural Circuit Dynamics (R01)",
        "agency": "National Institutes of Health (NIH)",
        "country": "USA",
        "max_funding_usd": 750_000,
        "duration_years": 5,
        "deadline": "2026-06-05",
        "url": "https://braininitiative.nih.gov/funding/funding-opportunities",
        "focus_areas": [
            "neural_dynamics",
            "circuit_analysis",
            "recording_technology",
            "computational_neuroscience",
            "neural_coding",
        ],
        "requirements": {
            "min_electrodes": 8,
            "needs_computation_evidence": False,
            "needs_ethics_plan": True,
            "us_institution_required": True,
        },
        "weight_factors": {
            "data_quality": 0.25,
            "scientific_rigor": 0.25,
            "connectivity_analysis": 0.20,
            "information_theory": 0.15,
            "reproducibility": 0.15,
        },
    },
}


def match_grants(
    data: SpikeData,
    analyses: Optional[dict] = None,
    country_filter: Optional[str] = None,
    min_funding_usd: Optional[int] = None,
) -> dict:
    """Match organoid experimental results against available grants.

    Evaluates how well the current research output aligns with each
    grant program's priorities and requirements. Returns ranked list
    of grants with match scores and specific recommendations.

    Args:
        data: SpikeData from the organoid recording.
        analyses: Dict of pre-computed analysis results (improves matching).
            Recognized keys: 'summary', 'reservoir', 'vowel_classification',
            'organoid_iq', 'criticality', 'information_theory', 'connectivity',
            'plasticity', 'ethical_assessment'.
        country_filter: Filter grants by country (e.g., 'USA', 'EU', 'Kazakhstan').
        min_funding_usd: Minimum funding amount filter.

    Returns:
        Dict with ranked grants, match scores, and recommendations per grant.
    """
    analyses = analyses or {}

    # Compute base metrics from data
    metrics = _compute_matching_metrics(data, analyses)

    # Score each grant
    results = []
    for grant_id, grant in _GRANTS.items():
        # Apply filters
        if country_filter and grant["country"].lower() != country_filter.lower():
            continue
        if min_funding_usd and grant["max_funding_usd"] < min_funding_usd:
            continue

        score, breakdown, eligible, issues = _score_grant(
            grant_id, grant, data, metrics, analyses
        )
        recommendations = _grant_recommendations(grant_id, grant, metrics, breakdown)

        results.append({
            "grant_id": grant_id,
            "name": grant["name"],
            "agency": grant["agency"],
            "country": grant["country"],
            "max_funding_usd": grant["max_funding_usd"],
            "duration_years": grant["duration_years"],
            "deadline": grant["deadline"],
            "url": grant["url"],
            "match_score": round(score, 4),
            "match_percentage": round(score * 100, 1),
            "eligible": eligible,
            "eligibility_issues": issues,
            "score_breakdown": breakdown,
            "recommendations": recommendations,
            "focus_areas": grant["focus_areas"],
        })

    # Sort by score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)

    # Summary
    top_match = results[0] if results else None
    total_potential_funding = sum(r["max_funding_usd"] for r in results if r["eligible"])

    return {
        "grants": results,
        "n_grants_evaluated": len(results),
        "n_eligible": sum(1 for r in results if r["eligible"]),
        "top_match": {
            "name": top_match["name"],
            "score": top_match["match_score"],
            "funding": top_match["max_funding_usd"],
        } if top_match else None,
        "total_potential_funding_usd": total_potential_funding,
        "metrics_used": metrics,
        "interpretation": (
            f"Evaluated {len(results)} grant programs. "
            f"Top match: {top_match['name']} ({top_match['match_percentage']:.0f}% match). "
            f"Total potential funding across eligible grants: ${total_potential_funding:,.0f}."
            if top_match
            else "No matching grants found with current filters."
        ),
    }


def get_grant_details(grant_id: str) -> dict:
    """Get full details for a specific grant program.

    Args:
        grant_id: Grant identifier (e.g., 'NSF_BEGIN_OI', 'DARPA_BTO').

    Returns:
        Dict with complete grant program information.
    """
    if grant_id not in _GRANTS:
        return {"error": f"Grant '{grant_id}' not found. Available: {list(_GRANTS.keys())}"}
    grant = _GRANTS[grant_id]
    return {
        "grant_id": grant_id,
        **grant,
    }


def list_grants(country_filter: Optional[str] = None) -> dict:
    """List all available grant programs.

    Args:
        country_filter: Optional country filter.

    Returns:
        Dict with list of grants and summary.
    """
    grants = []
    for grant_id, grant in _GRANTS.items():
        if country_filter and grant["country"].lower() != country_filter.lower():
            continue
        grants.append({
            "grant_id": grant_id,
            "name": grant["name"],
            "agency": grant["agency"],
            "country": grant["country"],
            "max_funding_usd": grant["max_funding_usd"],
            "deadline": grant["deadline"],
        })

    return {
        "grants": grants,
        "n_grants": len(grants),
        "total_max_funding_usd": sum(g["max_funding_usd"] for g in grants),
    }


def _compute_matching_metrics(data: SpikeData, analyses: dict) -> dict:
    """Compute metrics used for grant matching."""
    duration = max(data.duration, 1e-6)

    # Basic data quality
    firing_rate = data.n_spikes / duration
    n_electrodes = data.n_electrodes

    # Firing rate variability across electrodes
    rates = []
    for eid in data.electrode_ids:
        n = int(np.sum(data.electrodes == eid))
        rates.append(n / duration)
    rate_cv = float(np.std(rates) / np.mean(rates)) if rates and np.mean(rates) > 0 else 0

    # Data quality score (0-1)
    data_quality = 0.0
    if data.n_spikes > 100:
        data_quality += 0.3
    if data.n_spikes > 1000:
        data_quality += 0.2
    if n_electrodes >= 8:
        data_quality += 0.2
    if duration > 60:
        data_quality += 0.15
    if rate_cv > 0.3:
        data_quality += 0.15
    data_quality = min(1.0, data_quality)

    # Technical maturity score (0-1)
    tech_maturity = 0.0
    n_analyses = len(analyses)
    tech_maturity += min(0.4, n_analyses * 0.05)
    if "reservoir" in analyses or "vowel_classification" in analyses:
        tech_maturity += 0.2
    if "organoid_iq" in analyses:
        tech_maturity += 0.15
    if "criticality" in analyses:
        tech_maturity += 0.1
    if "connectivity" in analyses:
        tech_maturity += 0.1
    if "ethical_assessment" in analyses:
        tech_maturity += 0.05
    tech_maturity = min(1.0, tech_maturity)

    # Computation evidence score (0-1)
    computation = 0.0
    if "reservoir" in analyses:
        mc = analyses["reservoir"].get("memory_capacity", 0)
        if isinstance(mc, dict):
            mc = mc.get("memory_capacity", 0)
        computation += min(0.3, mc / 10.0)
        nl = analyses["reservoir"].get("nonlinearity_gain", 0)
        if nl > 0.05:
            computation += 0.2
    if "vowel_classification" in analyses:
        acc = analyses["vowel_classification"].get("test_accuracy", 0)
        computation += min(0.3, acc * 0.4)
    if "organoid_iq" in analyses:
        iq_val = analyses["organoid_iq"].get("overall_iq", analyses["organoid_iq"].get("total_score", 0))
        computation += min(0.2, iq_val / 500.0)
    computation = min(1.0, computation)

    # Novelty score (0-1)
    novelty = 0.3
    if "criticality" in analyses:
        kappa = analyses["criticality"].get("branching_ratio", analyses["criticality"].get("kappa", 0))
        if 0.95 < kappa < 1.05:
            novelty += 0.2
    if "vowel_classification" in analyses:
        acc = analyses["vowel_classification"].get("test_accuracy", 0)
        if acc > 0.8:
            novelty += 0.2
    if n_analyses > 5:
        novelty += 0.15
    novelty = min(1.0, novelty)

    # Ethics readiness (0-1)
    ethics = 0.2
    if "ethical_assessment" in analyses:
        ea = analyses["ethical_assessment"]
        compliance = ea.get("guidelines_compliance", {})
        n_compliant = compliance.get("n_compliant", 0)
        n_total = compliance.get("n_guidelines_checked", 1)
        ethics = 0.5 + 0.5 * (n_compliant / n_total)
    ethics = min(1.0, ethics)

    # Plasticity score
    plasticity = 0.0
    if "plasticity" in analyses:
        p = analyses["plasticity"]
        plasticity = min(1.0, p.get("plasticity_index", p.get("stdp_score", 0.3)))

    # Connectivity analysis score
    connectivity = 0.0
    if "connectivity" in analyses:
        c = analyses["connectivity"]
        connectivity = min(1.0, c.get("density", c.get("network_density", 0)) * 5)

    return {
        "data_quality": round(data_quality, 4),
        "technical_maturity": round(tech_maturity, 4),
        "computation_evidence": round(computation, 4),
        "novelty": round(novelty, 4),
        "ethics_readiness": round(ethics, 4),
        "plasticity": round(plasticity, 4),
        "connectivity_analysis": round(connectivity, 4),
        "mean_firing_rate_hz": round(firing_rate, 2),
        "n_electrodes": n_electrodes,
        "recording_duration_sec": round(duration, 2),
        "n_analyses_available": len(analyses),
    }


def _score_grant(
    grant_id: str,
    grant: dict,
    data: SpikeData,
    metrics: dict,
    analyses: dict,
) -> tuple:
    """Score a single grant against current metrics.

    Returns (score, breakdown, eligible, issues).
    """
    issues = []
    reqs = grant.get("requirements", {})

    # Check eligibility
    if reqs.get("min_electrodes", 0) > data.n_electrodes:
        issues.append(
            f"Requires >= {reqs['min_electrodes']} electrodes (have {data.n_electrodes})"
        )
    if reqs.get("needs_computation_evidence") and metrics["computation_evidence"] < 0.1:
        issues.append("Requires evidence of computation capability")
    if reqs.get("needs_reservoir_benchmark") and "vowel_classification" not in analyses and "reservoir" not in analyses:
        issues.append("Requires reservoir computing benchmark results")
    if reqs.get("needs_ethics_plan") and "ethical_assessment" not in analyses:
        issues.append("Requires ethics assessment (run ethical_assessment module)")

    eligible = len(issues) == 0

    # Compute weighted score based on grant-specific factors
    weights = grant.get("weight_factors", {})
    breakdown = {}
    total_score = 0.0
    total_weight = 0.0

    # Map grant weight factors to metrics
    factor_mapping = {
        "reservoir_computing": "computation_evidence",
        "plasticity": "plasticity",
        "organoid_iq": "computation_evidence",
        "information_theory": "data_quality",
        "criticality": "data_quality",
        "ethics_readiness": "ethics_readiness",
        "ethics_compliance": "ethics_readiness",
        "novelty": "novelty",
        "technical_maturity": "technical_maturity",
        "commercial_potential": "technical_maturity",
        "data_quality": "data_quality",
        "scientific_excellence": "se_compute",
        "impact_potential": "ip_compute",
        "consortium_readiness": "cr_compute",
        "computation_capability": "computation_evidence",
        "scalability": "sc_compute",
        "scientific_rigor": "data_quality",
        "connectivity_analysis": "connectivity_analysis",
        "reproducibility": "data_quality",
        "international_collaboration": "ic_compute",
    }

    # Computed metrics that are derived
    computed = {
        "se_compute": (metrics["data_quality"] + metrics["novelty"]) / 2,
        "ip_compute": (metrics["novelty"] + metrics["technical_maturity"]) / 2,
        "cr_compute": 0.3,
        "sc_compute": min(1.0, data.n_electrodes / 32.0),
        "ic_compute": 0.5,
    }

    for factor_name, weight in weights.items():
        mapping = factor_mapping.get(factor_name, "data_quality")
        if mapping in computed:
            value = computed[mapping]
        else:
            value = metrics.get(mapping, 0.3)

        breakdown[factor_name] = {
            "weight": weight,
            "score": round(float(value), 4),
            "weighted_score": round(float(weight * value), 4),
        }
        total_score += weight * value
        total_weight += weight

    if total_weight > 0:
        total_score = total_score / total_weight
    else:
        total_score = 0.0

    # Penalty for eligibility issues
    if not eligible:
        total_score *= 0.5

    return float(total_score), breakdown, eligible, issues


def _grant_recommendations(
    grant_id: str,
    grant: dict,
    metrics: dict,
    breakdown: dict,
) -> list:
    """Generate specific recommendations for improving match with a grant."""
    recs = []

    # Find weakest scoring factors
    sorted_factors = sorted(
        breakdown.items(),
        key=lambda x: x[1]["score"],
    )

    for factor_name, info in sorted_factors[:3]:
        if info["score"] < 0.3:
            recs.append(_improvement_suggestion(factor_name, info["score"]))

    # Grant-specific recommendations
    if grant_id == "NSF_BEGIN_OI":
        if metrics["computation_evidence"] < 0.3:
            recs.append(
                "Run reservoir computing benchmarks (memory capacity, vowel classification) "
                "to strengthen computation evidence for NSF BEGIN OI"
            )
        if metrics["ethics_readiness"] < 0.5:
            recs.append("Complete ethical assessment to satisfy NSF ethics plan requirement")

    elif grant_id == "DARPA_BTO":
        if metrics["computation_evidence"] < 0.5:
            recs.append(
                "DARPA requires strong computation evidence. Run full reservoir "
                "benchmark suite and demonstrate nonlinear computation"
            )
        if metrics.get("n_electrodes", 0) < 16:
            recs.append("DARPA program prefers higher electrode count (>=16 channels)")

    elif grant_id == "EU_HORIZON":
        if metrics["ethics_readiness"] < 0.5:
            recs.append(
                "EU Horizon places high weight on ethics compliance. "
                "Complete ethical assessment and ensure GDPR considerations"
            )
        recs.append(
            "EU Horizon requires consortium of >= 3 partners from different EU member states"
        )

    elif grant_id == "ASTANA_HUB":
        recs.append(
            "Astana Hub values international collaboration and commercial potential. "
            "Highlight technology transfer and partnership opportunities"
        )

    elif grant_id == "NIH_BRAIN":
        if metrics["data_quality"] < 0.5:
            recs.append(
                "NIH BRAIN emphasizes data quality and scientific rigor. "
                "Ensure robust spike detection and statistical validation"
            )

    if not recs:
        recs.append("Strong match. Focus proposal on unique aspects of your organoid system.")

    return recs


def _improvement_suggestion(factor: str, score: float) -> str:
    """Generate an improvement suggestion for a weak factor."""
    suggestions = {
        "reservoir_computing": (
            "Strengthen reservoir computing evidence by running memory capacity "
            "and vowel classification benchmarks"
        ),
        "plasticity": (
            "Demonstrate plasticity by running paired stimulation protocols "
            "and analyzing STDP signatures"
        ),
        "organoid_iq": "Compute Organoid IQ score to quantify computational capability",
        "information_theory": "Run information-theoretic analyses (entropy, transfer entropy, complexity)",
        "criticality": "Analyze criticality via avalanche statistics and branching ratio estimation",
        "ethics_readiness": "Complete ethical assessment including sentience risk scoring",
        "ethics_compliance": "Run ethical assessment and address any compliance issues",
        "novelty": "Highlight unique aspects of your approach and any unexpected findings",
        "technical_maturity": "Run additional analysis modules to demonstrate comprehensive characterization",
        "data_quality": "Increase recording duration and verify spike detection quality",
        "computation_capability": "Demonstrate nonlinear computation and benchmark against linear baseline",
        "scientific_excellence": "Strengthen both data quality and novelty of findings",
        "impact_potential": "Articulate potential applications in neuromorphic computing or medicine",
        "connectivity_analysis": "Run functional connectivity analysis to characterize network architecture",
        "reproducibility": "Record multiple sessions and demonstrate consistent results",
        "scalability": "Consider scaling to higher electrode count or multiple organoids",
        "scientific_rigor": "Ensure statistical validation with surrogate testing and effect sizes",
    }
    return suggestions.get(factor, f"Improve {factor} (current score: {score:.2f})")
