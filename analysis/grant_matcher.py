"""Grant matching based on organoid analysis results.

Matches research results with relevant funding opportunities:
NSF BEGIN OI, Astana Hub, EU Horizon Europe, DARPA, NIH BRAIN.
"""
import numpy as np
from .loader import SpikeData


GRANTS = {
    "nsf_begin_oi": {
        "name": "NSF BEGIN OI (Biocomputing Empowered by Generative Intelligence for Next-generation OI)",
        "funder": "National Science Foundation (US)",
        "amount": "$2,000,000",
        "duration": "3 years",
        "focus": ["organoid intelligence", "biocomputing", "neural interfaces", "AI integration"],
        "requirements": ["US institution PI", "preliminary data", "ethical framework"],
        "url": "nsf.gov/funding",
        "deadline": "Rolling",
    },
    "astana_hub": {
        "name": "Astana Hub Tech Grant",
        "funder": "Astana Hub (Kazakhstan)",
        "amount": "$10,000-$50,000",
        "duration": "6-12 months",
        "focus": ["technology startup", "innovation", "Kazakhstan-based team"],
        "requirements": ["Kazakhstan registration", "tech product", "team"],
        "url": "astanahub.com",
        "deadline": "Quarterly",
    },
    "eu_horizon_europe": {
        "name": "EU Horizon Europe — Biotechnology",
        "funder": "European Commission",
        "amount": "EUR 1,000,000-5,000,000",
        "duration": "2-4 years",
        "focus": ["biotechnology", "neurotechnology", "ethical AI", "organ-on-chip"],
        "requirements": ["EU institution partner", "consortium", "impact plan"],
        "url": "ec.europa.eu/horizon-europe",
        "deadline": "Annual calls",
    },
    "darpa": {
        "name": "DARPA — Unconventional Computing",
        "funder": "DARPA (US)",
        "amount": "$500,000-$5,000,000",
        "duration": "2-4 years",
        "focus": ["unconventional computing", "neuromorphic", "low-power", "defense"],
        "requirements": ["US entity or partner", "technical feasibility", "defense relevance"],
        "url": "darpa.mil",
        "deadline": "BAA-specific",
    },
    "nih_brain": {
        "name": "NIH BRAIN Initiative",
        "funder": "National Institutes of Health (US)",
        "amount": "$200,000-$3,000,000",
        "duration": "1-5 years",
        "focus": ["neurotechnology", "brain recording", "neural circuits", "tools"],
        "requirements": ["US institution", "neuroscience focus", "tool development"],
        "url": "braininitiative.nih.gov",
        "deadline": "Multiple cycles/year",
    },
    "indiebio": {
        "name": "IndieBio Accelerator",
        "funder": "SOSV / IndieBio",
        "amount": "$250,000",
        "duration": "4 months",
        "focus": ["biotech startup", "life sciences", "deeptech"],
        "requirements": ["startup team", "bio/tech product", "scalable"],
        "url": "indiebio.co",
        "deadline": "Batch applications",
    },
}


def match_grants(data: SpikeData = None) -> dict:
    """Match research results with relevant grants."""
    # Compute match scores based on what we have
    results = []

    has_data = data is not None and data.n_spikes > 0
    has_platform = True  # We have NeuroBridge
    has_experiments = has_data  # If we have data, we've done experiments

    for grant_id, grant in GRANTS.items():
        # Base match score
        score = 0.3  # Base: we're in biocomputing

        # Boost for relevant focus areas
        focus = grant["focus"]
        if any("biocomputing" in f or "organoid" in f for f in focus):
            score += 0.3
        if any("technology" in f or "tool" in f or "startup" in f for f in focus):
            score += 0.1
        if any("neurotechnology" in f or "neural" in f for f in focus):
            score += 0.1
        if has_data:
            score += 0.1
        if has_platform:
            score += 0.1

        score = min(score, 1.0)

        # Gap analysis
        gaps = []
        for req in grant["requirements"]:
            if "US" in req:
                gaps.append("Need US institutional partner or entity")
            elif "EU" in req:
                gaps.append("Need EU institutional partner")
            elif "Kazakhstan" in req:
                gaps.append("Need Kazakhstan business registration (AIFC/Astana Hub)")
            elif "preliminary" in req.lower():
                if not has_data:
                    gaps.append("Need preliminary experimental data")
            elif "team" in req.lower():
                gaps.append("Need co-founder or team members")

        results.append({
            "grant_id": grant_id,
            "name": grant["name"],
            "funder": grant["funder"],
            "amount": grant["amount"],
            "match_score": float(score),
            "focus_areas": focus,
            "gaps": gaps,
            "deadline": grant["deadline"],
            "url": grant["url"],
        })

    # Sort by match score
    results.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "matches": results,
        "top_match": results[0] if results else None,
        "n_grants": len(results),
        "has_preliminary_data": has_data,
    }
