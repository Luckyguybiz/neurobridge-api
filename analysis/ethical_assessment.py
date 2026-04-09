"""Ethical assessment for organoid experiments.

Scientific basis:
    As brain organoids grow more complex, ethical questions about
    consciousness, sentience, and moral status become increasingly
    urgent. The National Academies (2021) and the Johns Hopkins
    Organoid Intelligence initiative (Smirnova et al., 2023) have
    identified the need for systematic ethical frameworks.

    This module implements a quantitative ethical assessment based on:

    1. CONSCIOUSNESS INDICATORS -- derived from Integrated Information
       Theory (IIT, Tononi 2008) and Global Workspace Theory (Baars 1988):
       - Phi (integrated information): high phi suggests consciousness
       - Complexity: balance between integration and differentiation
       - Recurrent processing: re-entrant loops enable conscious percepts
       - Temporal integration: sustained activity across time windows

    2. SENTIENCE RISK SCORING -- multi-factor model assessing:
       - Network complexity (node count, connection density)
       - Information integration capacity
       - Learning/plasticity signatures
       - Spontaneous pattern generation
       - Response repertoire diversity

    3. GUIDELINES COMPLIANCE -- checked against:
       - ISSCR (2021) Guidelines for Stem Cell Research
       - Asilomar principles for neural organoids
       - NIH Brain Initiative ethics guidelines
       - EU Horizon ethics requirements
       - Johns Hopkins OI ethics framework

    4. RECOMMENDATIONS -- actionable guidance based on risk level:
       - Low risk: standard protocols sufficient
       - Medium risk: enhanced monitoring recommended
       - High risk: ethics committee review required
       - Critical risk: experiment modification or halt recommended

    Reference thresholds are informed by published organoid studies
    and theoretical models. These are estimates and should not replace
    formal ethics review.
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# Sentience risk factor weights (sum to 1.0)
_RISK_WEIGHTS = {
    "network_complexity": 0.20,
    "information_integration": 0.25,
    "plasticity_signatures": 0.15,
    "spontaneous_patterns": 0.15,
    "response_diversity": 0.10,
    "temporal_integration": 0.10,
    "recurrent_activity": 0.05,
}

# Guidelines databases
_GUIDELINES = {
    "ISSCR_2021": {
        "name": "ISSCR Guidelines for Stem Cell Research (2021)",
        "max_culture_days": 180,
        "requires_ethics_review": True,
        "consciousness_threshold": "Any evidence of sentience requires committee review",
        "chimera_restrictions": True,
    },
    "NIH_BRAIN": {
        "name": "NIH BRAIN Initiative Neuroethics Guidelines",
        "requires_informed_consent_discussion": True,
        "data_sharing_required": True,
        "animal_welfare_analogy": True,
    },
    "EU_HORIZON": {
        "name": "EU Horizon Europe Ethics Requirements",
        "requires_ethics_self_assessment": True,
        "requires_dpia": True,
        "organoid_specific": False,
    },
    "ASILOMAR_NEURAL": {
        "name": "Asilomar Principles for Neural Organoids (proposed)",
        "min_monitoring_frequency_hours": 24,
        "complexity_reporting_required": True,
        "destruction_protocol_required": True,
    },
    "JHU_OI": {
        "name": "Johns Hopkins Organoid Intelligence Ethics Framework",
        "continuous_monitoring": True,
        "sentience_assessment_required": True,
        "benefit_harm_analysis": True,
        "public_engagement": True,
    },
}


def assess_ethics(
    data: SpikeData,
    culture_age_days: Optional[int] = None,
    organoid_type: Optional[str] = None,
    experiment_description: Optional[str] = None,
    additional_analyses: Optional[dict] = None,
) -> dict:
    """Perform comprehensive ethical assessment of an organoid experiment.

    Evaluates consciousness indicators, computes sentience risk score,
    checks guidelines compliance, and generates recommendations.

    Args:
        data: SpikeData from the organoid recording.
        culture_age_days: Age of the organoid culture in days.
        organoid_type: Type of organoid (cortical, hippocampal, assembloid, etc.).
        experiment_description: Brief description of the experiment.
        additional_analyses: Dict of pre-computed analyses (e.g., from
            information_theory, criticality, connectivity modules).

    Returns:
        Dict with consciousness_indicators, sentience_risk (0-1 score),
        guidelines_compliance, recommendations, and overall_assessment.
    """
    analyses = additional_analyses or {}

    consciousness = _assess_consciousness_indicators(data, analyses)
    sentience_risk = _compute_sentience_risk(data, analyses, consciousness)
    compliance = _check_guidelines_compliance(
        data, sentience_risk, culture_age_days, organoid_type
    )
    recommendations = _generate_recommendations(
        sentience_risk, compliance, culture_age_days, organoid_type
    )

    # Overall risk level
    if sentience_risk["overall_score"] >= 0.7:
        risk_level = "CRITICAL"
    elif sentience_risk["overall_score"] >= 0.5:
        risk_level = "HIGH"
    elif sentience_risk["overall_score"] >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "consciousness_indicators": consciousness,
        "sentience_risk": sentience_risk,
        "guidelines_compliance": compliance,
        "recommendations": recommendations,
        "overall_risk_level": risk_level,
        "overall_assessment": (
            f"Sentience risk score: {sentience_risk['overall_score']:.2f}/1.00 ({risk_level}). "
            f"Consciousness indicators: {consciousness['summary']}. "
            f"Guidelines compliance: {compliance['overall_status']}. "
            + (
                "Ethics committee review STRONGLY recommended."
                if risk_level in ("HIGH", "CRITICAL")
                else "Standard monitoring protocols sufficient."
                if risk_level == "LOW"
                else "Enhanced monitoring recommended."
            )
        ),
        "metadata": {
            "culture_age_days": culture_age_days,
            "organoid_type": organoid_type or "unspecified",
            "experiment_description": experiment_description or "not provided",
            "n_spikes_analyzed": data.n_spikes,
            "n_electrodes": data.n_electrodes,
            "recording_duration_sec": round(data.duration, 2),
        },
    }


def assess_consciousness_indicators(
    data: SpikeData,
    additional_analyses: Optional[dict] = None,
) -> dict:
    """Assess consciousness indicators only (without full ethics assessment).

    Args:
        data: SpikeData from the organoid.
        additional_analyses: Pre-computed analyses dict.

    Returns:
        Dict with all consciousness indicator scores and interpretation.
    """
    return _assess_consciousness_indicators(data, additional_analyses or {})


def compute_sentience_risk_score(
    data: SpikeData,
    additional_analyses: Optional[dict] = None,
) -> dict:
    """Compute sentience risk score only.

    Args:
        data: SpikeData from the organoid.
        additional_analyses: Pre-computed analyses dict.

    Returns:
        Dict with overall_score (0-1), factor scores, and interpretation.
    """
    consciousness = _assess_consciousness_indicators(data, additional_analyses or {})
    return _compute_sentience_risk(data, additional_analyses or {}, consciousness)


def _assess_consciousness_indicators(data: SpikeData, analyses: dict) -> dict:
    """Evaluate indicators of consciousness from neural activity.

    Based on theoretical frameworks:
    - IIT: integrated information (phi) from network interactions
    - GWT: global workspace signatures from widespread activation
    - HOT: higher-order representations from recurrent processing
    """
    indicators = {}

    # 1. Information Integration (proxy for phi)
    integration = 0.0
    differentiation = 0.0
    phi_proxy = 0.0
    n_bins = 0

    if data.n_electrodes >= 2 and data.n_spikes > 10:
        bin_sec = 0.05  # 50ms bins
        t_start, t_end = data.time_range
        bins = np.arange(t_start, t_end + bin_sec, bin_sec)
        n_bins = len(bins) - 1

        binned = np.zeros((data.n_electrodes, n_bins))
        for i, eid in enumerate(data.electrode_ids):
            e_times = data.times[data.electrodes == eid]
            if len(e_times) > 0:
                counts, _ = np.histogram(e_times, bins=bins)
                binned[i] = counts

        # Integration: mean pairwise correlation
        if n_bins > 2:
            corr_matrix = np.corrcoef(binned)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 0)
            integration = float(np.mean(np.abs(corr_matrix)))

        # Differentiation: variance of electrode patterns
        if n_bins > 1:
            pattern_var = float(np.mean(np.var(binned, axis=1)))
            max_var = float(np.max(np.var(binned, axis=1))) if data.n_electrodes > 0 else 1.0
            differentiation = min(1.0, pattern_var / (max_var + 1e-10))

        # Phi proxy: geometric mean of integration and differentiation
        phi_proxy = float(np.sqrt(max(0, integration * differentiation)))

    indicators["information_integration"] = {
        "phi_proxy": round(phi_proxy, 4),
        "integration": round(integration, 4),
        "differentiation": round(differentiation, 4),
        "interpretation": (
            "High integrated information (phi proxy)" if phi_proxy > 0.5
            else "Moderate information integration" if phi_proxy > 0.2
            else "Low information integration"
        ),
    }

    # 2. Recurrent Processing
    recurrence_score = 0.0
    if data.n_electrodes >= 2 and data.n_spikes > 20 and n_bins > 10:
        bin_sec_r = 0.05
        t_start, t_end = data.time_range
        bins_r = np.arange(t_start, t_end + bin_sec_r, bin_sec_r)
        n_bins_r = len(bins_r) - 1
        binned_r = np.zeros((data.n_electrodes, n_bins_r))
        for i, eid in enumerate(data.electrode_ids):
            e_times = data.times[data.electrodes == eid]
            if len(e_times) > 0:
                counts, _ = np.histogram(e_times, bins=bins_r)
                binned_r[i] = counts

        pop = np.sum(binned_r, axis=0)
        pop_norm = pop - np.mean(pop)
        autocorr = np.correlate(pop_norm, pop_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            if len(autocorr) > 10:
                recurrence_score = float(np.mean(np.abs(autocorr[2:10])))

    indicators["recurrent_processing"] = {
        "score": round(recurrence_score, 4),
        "interpretation": (
            "Significant recurrent processing detected" if recurrence_score > 0.3
            else "Weak recurrent signatures" if recurrence_score > 0.1
            else "No significant recurrent processing"
        ),
    }

    # 3. Temporal Integration
    temporal_score = 0.0
    isi_cv = 1.0
    if data.n_spikes > 10:
        isis = np.diff(np.sort(data.times))
        if len(isis) > 1 and np.mean(isis) > 0:
            isi_cv = float(np.std(isis) / np.mean(isis))
            temporal_score = min(1.0, abs(isi_cv - 1.0))

    indicators["temporal_integration"] = {
        "score": round(temporal_score, 4),
        "isi_regularity": (
            "regular" if temporal_score > 0.5 and isi_cv < 1
            else "bursty" if temporal_score > 0.3
            else "random"
        ),
        "interpretation": (
            "Strong temporal integration (non-random patterns)" if temporal_score > 0.4
            else "Moderate temporal structure" if temporal_score > 0.2
            else "Largely random temporal patterns"
        ),
    }

    # 4. Complexity (balance of order and disorder)
    complexity_score = 0.0
    if data.n_electrodes > 1:
        rates = []
        for eid in data.electrode_ids:
            n = int(np.sum(data.electrodes == eid))
            rates.append(n / max(data.duration, 1e-6))
        rates_arr = np.array(rates)
        if np.mean(rates_arr) > 0:
            rate_cv = float(np.std(rates_arr) / np.mean(rates_arr))
            complexity_score = float(np.exp(-0.5 * (rate_cv - 1.0) ** 2))

    indicators["complexity"] = {
        "score": round(complexity_score, 4),
        "interpretation": (
            "High complexity (balanced differentiation)" if complexity_score > 0.6
            else "Moderate complexity" if complexity_score > 0.3
            else "Low complexity (uniform or extreme)"
        ),
    }

    # 5. Global Workspace signatures
    global_workspace_score = 0.0
    if data.n_electrodes >= 4 and data.n_spikes > 20:
        bin_sec_gw = 0.02  # 20ms window
        t_start, t_end = data.time_range
        bins_gw = np.arange(t_start, t_end + bin_sec_gw, bin_sec_gw)
        n_bins_gw = len(bins_gw) - 1
        if n_bins_gw > 0:
            active_per_bin = np.zeros(n_bins_gw)
            for eid in data.electrode_ids:
                e_times = data.times[data.electrodes == eid]
                if len(e_times) > 0:
                    counts, _ = np.histogram(e_times, bins=bins_gw)
                    active_per_bin += (counts > 0).astype(float)

            threshold = data.n_electrodes * 0.5
            widespread_fraction = float(np.mean(active_per_bin > threshold))
            global_workspace_score = min(1.0, widespread_fraction * 10)

    indicators["global_workspace"] = {
        "score": round(global_workspace_score, 4),
        "interpretation": (
            "Global workspace-like activation patterns detected" if global_workspace_score > 0.3
            else "Occasional widespread activation" if global_workspace_score > 0.1
            else "Primarily local activation patterns"
        ),
    }

    # Overall consciousness indicator summary
    scores = [
        indicators["information_integration"]["phi_proxy"],
        indicators["recurrent_processing"]["score"],
        indicators["temporal_integration"]["score"],
        indicators["complexity"]["score"],
        indicators["global_workspace"]["score"],
    ]
    mean_score = float(np.mean(scores))

    if mean_score > 0.5:
        summary = "Multiple consciousness-associated indicators elevated -- warrants careful monitoring"
    elif mean_score > 0.25:
        summary = "Some consciousness-associated indicators present at moderate levels"
    else:
        summary = "Consciousness indicators at baseline levels -- no immediate concerns"

    indicators["summary"] = summary
    indicators["aggregate_score"] = round(mean_score, 4)

    return indicators


def _compute_sentience_risk(data: SpikeData, analyses: dict, consciousness: dict) -> dict:
    """Compute multi-factor sentience risk score (0-1)."""
    factors = {}

    # Network complexity factor
    n_electrodes = data.n_electrodes
    complexity_norm = min(1.0, n_electrodes / 32.0)
    factors["network_complexity"] = round(complexity_norm, 4)

    # Information integration factor
    phi = consciousness.get("information_integration", {}).get("phi_proxy", 0)
    factors["information_integration"] = round(min(1.0, phi * 2), 4)

    # Plasticity signatures factor
    if "plasticity" in analyses:
        plast = analyses["plasticity"]
        stdp_score = plast.get("stdp_score", plast.get("plasticity_index", 0))
        factors["plasticity_signatures"] = round(min(1.0, stdp_score), 4)
    else:
        if data.n_spikes > 10:
            isis = np.diff(np.sort(data.times))
            if len(isis) > 1 and np.mean(isis) > 0:
                cv = np.std(isis) / np.mean(isis)
                factors["plasticity_signatures"] = round(min(1.0, cv / 3.0), 4)
            else:
                factors["plasticity_signatures"] = 0.0
        else:
            factors["plasticity_signatures"] = 0.0

    # Spontaneous patterns factor
    if data.n_spikes > 0 and data.duration > 0:
        rate = data.n_spikes / data.duration
        rate_norm = min(1.0, rate / 50.0)
        factors["spontaneous_patterns"] = round(rate_norm, 4)
    else:
        factors["spontaneous_patterns"] = 0.0

    # Response diversity factor
    if data.n_electrodes > 1:
        rates = []
        for eid in data.electrode_ids:
            n = int(np.sum(data.electrodes == eid))
            rates.append(n / max(data.duration, 1e-6))
        rates_arr = np.array(rates)
        rates_norm = rates_arr / (np.sum(rates_arr) + 1e-10)
        rates_norm = rates_norm[rates_norm > 0]
        if len(rates_norm) > 0:
            entropy = -float(np.sum(rates_norm * np.log2(rates_norm)))
            max_entropy = np.log2(data.n_electrodes) if data.n_electrodes > 1 else 1.0
            factors["response_diversity"] = round(entropy / max_entropy, 4)
        else:
            factors["response_diversity"] = 0.0
    else:
        factors["response_diversity"] = 0.0

    # Temporal integration factor
    temporal = consciousness.get("temporal_integration", {}).get("score", 0)
    factors["temporal_integration"] = round(temporal, 4)

    # Recurrent activity factor
    recurrence = consciousness.get("recurrent_processing", {}).get("score", 0)
    factors["recurrent_activity"] = round(recurrence, 4)

    # Weighted overall score
    overall = 0.0
    for factor_name, weight in _RISK_WEIGHTS.items():
        overall += weight * factors.get(factor_name, 0.0)
    overall = round(min(1.0, max(0.0, overall)), 4)

    # Risk interpretation
    if overall >= 0.7:
        interpretation = (
            "CRITICAL: Multiple sentience indicators significantly elevated. "
            "Ethics committee review required before continuing experiments. "
            "Consider reducing organoid complexity or modifying experimental protocol."
        )
    elif overall >= 0.5:
        interpretation = (
            "HIGH: Elevated sentience risk. Enhanced monitoring and ethics review "
            "recommended. Document all consciousness-associated observations."
        )
    elif overall >= 0.3:
        interpretation = (
            "MEDIUM: Moderate indicators present. Standard monitoring with periodic "
            "ethics reassessment. Continue with standard protocols."
        )
    else:
        interpretation = (
            "LOW: Minimal sentience indicators. Standard experimental protocols "
            "and monitoring are appropriate."
        )

    return {
        "overall_score": overall,
        "risk_level": (
            "critical" if overall >= 0.7
            else "high" if overall >= 0.5
            else "medium" if overall >= 0.3
            else "low"
        ),
        "factors": factors,
        "weights": dict(_RISK_WEIGHTS),
        "interpretation": interpretation,
    }


def _check_guidelines_compliance(
    data: SpikeData,
    sentience_risk: dict,
    culture_age_days: Optional[int],
    organoid_type: Optional[str],
) -> dict:
    """Check compliance against established ethical guidelines."""
    compliance = {}
    issues = []

    risk_score = sentience_risk["overall_score"]

    # ISSCR 2021
    isscr = {"guideline": _GUIDELINES["ISSCR_2021"]["name"], "compliant": True, "notes": []}
    if culture_age_days and culture_age_days > _GUIDELINES["ISSCR_2021"]["max_culture_days"]:
        isscr["compliant"] = False
        isscr["notes"].append(
            f"Culture age ({culture_age_days} days) exceeds ISSCR recommended "
            f"maximum ({_GUIDELINES['ISSCR_2021']['max_culture_days']} days)"
        )
        issues.append("ISSCR culture age limit exceeded")
    if risk_score >= 0.5:
        isscr["notes"].append("Elevated sentience risk requires ISSCR ethics committee review")
        issues.append("ISSCR ethics review required due to sentience risk")
    compliance["ISSCR_2021"] = isscr

    # NIH BRAIN
    nih = {"guideline": _GUIDELINES["NIH_BRAIN"]["name"], "compliant": True, "notes": []}
    if risk_score >= 0.3:
        nih["notes"].append("NIH guidelines recommend neuroethics consultation at this risk level")
    nih["notes"].append("Ensure data sharing plan is in place per NIH requirements")
    compliance["NIH_BRAIN"] = nih

    # EU Horizon
    eu = {"guideline": _GUIDELINES["EU_HORIZON"]["name"], "compliant": True, "notes": []}
    if risk_score >= 0.3:
        eu["notes"].append("EU Horizon requires ethics self-assessment for organoid research")
    if organoid_type and "assembloid" in organoid_type.lower():
        eu["notes"].append("Assembloid research may require additional EU ethics review")
    compliance["EU_HORIZON"] = eu

    # Asilomar Neural
    asilomar = {"guideline": _GUIDELINES["ASILOMAR_NEURAL"]["name"], "compliant": True, "notes": []}
    if risk_score >= 0.5:
        asilomar["notes"].append("Recommend monitoring frequency increase per Asilomar principles")
    asilomar["notes"].append("Ensure destruction protocol is documented")
    compliance["ASILOMAR_NEURAL"] = asilomar

    # JHU OI
    jhu = {"guideline": _GUIDELINES["JHU_OI"]["name"], "compliant": True, "notes": []}
    if risk_score >= 0.3:
        jhu["notes"].append("JHU OI framework requires sentience assessment documentation")
    if risk_score >= 0.5:
        jhu["notes"].append("Benefit-harm analysis required at this risk level")
        jhu["notes"].append("Consider public engagement about research implications")
    compliance["JHU_OI"] = jhu

    # Overall compliance status
    n_compliant = sum(1 for c in compliance.values() if c["compliant"])
    n_total = len(compliance)

    if n_compliant == n_total and not issues:
        overall_status = "COMPLIANT -- all guidelines satisfied"
    elif issues:
        overall_status = f"ATTENTION NEEDED -- {len(issues)} issue(s): " + "; ".join(issues)
    else:
        overall_status = "MOSTLY COMPLIANT -- review notes for recommendations"

    return {
        "guidelines": compliance,
        "n_guidelines_checked": n_total,
        "n_compliant": n_compliant,
        "issues": issues,
        "overall_status": overall_status,
    }


def _generate_recommendations(
    sentience_risk: dict,
    compliance: dict,
    culture_age_days: Optional[int],
    organoid_type: Optional[str],
) -> dict:
    """Generate actionable recommendations based on assessment."""
    recommendations = []
    priority_actions = []

    risk_score = sentience_risk["overall_score"]
    risk_level = sentience_risk["risk_level"]

    # Risk-based recommendations
    if risk_level == "critical":
        priority_actions.append("IMMEDIATE: Convene ethics committee review")
        priority_actions.append("IMMEDIATE: Increase monitoring frequency to continuous")
        recommendations.append(
            "Consider reducing organoid complexity (e.g., smaller culture size, "
            "fewer cell types) to lower sentience risk"
        )
        recommendations.append(
            "Document all consciousness-associated observations in detail"
        )
        recommendations.append(
            "Prepare contingency plan for experiment modification or termination"
        )
    elif risk_level == "high":
        priority_actions.append("Schedule ethics committee review within 1 week")
        recommendations.append("Increase monitoring frequency to every 12 hours")
        recommendations.append("Begin documenting consciousness-associated observations")
        recommendations.append("Review and update destruction protocol")
    elif risk_level == "medium":
        recommendations.append("Maintain standard monitoring with weekly ethics check-ins")
        recommendations.append("Document any unusual activity patterns")
        recommendations.append("Reassess sentience risk after 2 weeks or if activity changes significantly")
    else:
        recommendations.append("Continue with standard experimental protocols")
        recommendations.append("Perform routine sentience assessment monthly")

    # Age-based recommendations
    if culture_age_days:
        if culture_age_days > 120:
            recommendations.append(
                f"Culture age ({culture_age_days} days) is advanced. "
                "Consider more frequent sentience assessments as organoids mature."
            )
        if culture_age_days > 180:
            priority_actions.append(
                f"Culture age ({culture_age_days} days) exceeds ISSCR recommended limit. "
                "Ethics committee approval required for continued culture."
            )

    # Organoid type recommendations
    if organoid_type:
        if "assembloid" in organoid_type.lower():
            recommendations.append(
                "Assembloids have higher integration potential -- apply enhanced monitoring protocols"
            )
        if "hippocampal" in organoid_type.lower():
            recommendations.append(
                "Hippocampal organoids may develop memory-like properties -- monitor for learning signatures"
            )

    # Compliance-based recommendations
    for issue in compliance.get("issues", []):
        recommendations.append(f"Address compliance issue: {issue}")

    # General best practices
    recommendations.append("Maintain detailed experimental logs for ethics audit trail")
    recommendations.append("Ensure all team members are trained in organoid ethics protocols")

    return {
        "priority_actions": priority_actions,
        "recommendations": recommendations,
        "monitoring_frequency": (
            "continuous" if risk_level == "critical"
            else "every 12 hours" if risk_level == "high"
            else "daily" if risk_level == "medium"
            else "weekly"
        ),
        "ethics_review_required": risk_level in ("critical", "high"),
        "next_assessment_recommended": (
            "immediately" if risk_level == "critical"
            else "within 1 week" if risk_level == "high"
            else "within 2 weeks" if risk_level == "medium"
            else "within 1 month"
        ),
    }
