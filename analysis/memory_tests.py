"""Memory Battery — Standardized tests for neural organoid memory.

Battery of 4 memory systems, each with dedicated protocol:

1. Working Memory (WM):
   Delay-match-to-sample. Present pattern, delay 0.5–5s, present again.
   Score = how well organoid distinguishes match vs non-match after delay.
   Metric: d-prime (signal detection theory).

2. Short-Term Memory (STM):
   Digit-span analogue. Present N items sequentially, test recall order.
   Measure capacity (Miller's 7±2 analogue for organoids).
   Metric: span length where accuracy drops below 50%.

3. Long-Term Memory (LTM):
   Spaced repetition. Same pattern presented at 0s, 60s, 300s intervals.
   Measure retention via response similarity over time.
   Metric: retention ratio (late / early response similarity).

4. Associative Memory (AM):
   Paired-associate learning. Pattern A → Pattern B.
   After training, present A, measure if B-like response occurs.
   Metric: completion score (Hopfield network analogue).
"""

import numpy as np
from typing import Optional
from .loader import SpikeData


# ── Full Battery ──────────────────────────────────────────────────────────────

def run_memory_battery(
    data: SpikeData,
    seed: int = 42,
) -> dict:
    """Run complete memory battery: WM + STM + LTM + AM.

    Args:
        data: SpikeData

    Returns:
        dict with all 4 memory test results, composite score, interpretation
    """
    rng = np.random.default_rng(seed)

    results = {
        "working_memory": test_working_memory(data, rng=rng),
        "short_term_memory": test_short_term_memory(data, rng=rng),
        "long_term_memory": test_long_term_memory(data, rng=rng),
        "associative_memory": test_associative_memory(data, rng=rng),
    }

    # Composite score (0-100)
    scores = {
        "working_memory": results["working_memory"]["score"],
        "short_term_memory": results["short_term_memory"]["score"],
        "long_term_memory": results["long_term_memory"]["score"],
        "associative_memory": results["associative_memory"]["score"],
    }

    composite = float(np.mean(list(scores.values())))
    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)

    return {
        "composite_score": round(composite, 1),
        "subscores": scores,
        "best_memory_system": best,
        "weakest_memory_system": worst,
        "tests": results,
        "profile": _memory_profile(scores),
        "interpretation": (
            f"Memory composite: {composite:.0f}/100. "
            f"Strongest: {best} ({scores[best]:.0f}). "
            f"Weakest: {worst} ({scores[worst]:.0f}). "
            + _overall_interpretation(composite)
        ),
    }


# ── Working Memory ────────────────────────────────────────────────────────────

def test_working_memory(
    data: SpikeData,
    n_trials: int = 40,
    delays_s: Optional[list] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Delay-match-to-sample working memory test.

    Tests how well the organoid maintains a pattern over delay periods.
    Uses spike pattern similarity as proxy for neural "memory".

    Args:
        data: SpikeData
        n_trials: number of DMS trials
        delays_s: list of delay durations to test

    Returns:
        dict with d-prime, capacity, per-delay breakdown
    """
    if rng is None:
        rng = np.random.default_rng(1)
    if delays_s is None:
        delays_s = [0.5, 1.0, 2.0, 5.0]

    t_start, t_end = data.time_range
    duration = t_end - t_start
    window_s = 0.2  # 200ms pattern window
    n_electrodes = len(data.electrode_ids)

    def get_pattern(t0: float) -> np.ndarray:
        mask = (data.times >= t0) & (data.times < t0 + window_s)
        pattern = np.zeros(n_electrodes)
        for i, eid in enumerate(data.electrode_ids):
            pattern[i] = np.sum((data.electrodes == eid) & mask)
        norm = np.linalg.norm(pattern)
        return pattern / norm if norm > 0 else pattern

    delay_results = []
    all_hits, all_fas = [], []

    for delay in delays_s:
        hits, false_alarms = [], []
        n_delay_trials = max(4, n_trials // len(delays_s))

        for _ in range(n_delay_trials):
            t0 = rng.uniform(t_start, max(t_start + 0.1, t_end - window_s * 2 - delay))
            sample_pattern = get_pattern(t0)
            test_t = t0 + window_s + delay

            if test_t + window_s > t_end:
                continue

            # Match trial
            match_pattern = get_pattern(test_t)
            match_sim = float(np.dot(sample_pattern, match_pattern)) if np.any(sample_pattern) else 0.0

            # Non-match trial (random other time)
            nonmatch_t = rng.uniform(t_start, max(t_start + 0.1, t_end - window_s))
            nonmatch_pattern = get_pattern(float(nonmatch_t))
            nonmatch_sim = float(np.dot(sample_pattern, nonmatch_pattern)) if np.any(sample_pattern) else 0.0

            threshold = 0.3
            hit = 1 if match_sim > threshold else 0
            fa = 1 if nonmatch_sim > threshold else 0
            hits.append(hit)
            false_alarms.append(fa)

        hr = float(np.mean(hits)) if hits else 0.5
        far = float(np.mean(false_alarms)) if false_alarms else 0.5

        # d-prime (signal detection theory)
        from scipy.special import ndtri
        hr_clipped = float(np.clip(hr, 0.01, 0.99))
        far_clipped = float(np.clip(far, 0.01, 0.99))
        try:
            d_prime = float(ndtri(hr_clipped) - ndtri(far_clipped))
        except Exception:
            d_prime = hr - far

        delay_results.append({
            "delay_s": delay,
            "hit_rate": round(hr, 3),
            "false_alarm_rate": round(far, 3),
            "d_prime": round(d_prime, 3),
            "n_trials": len(hits),
        })
        all_hits.extend(hits)
        all_fas.extend(false_alarms)

    # Overall d-prime
    mean_hr = float(np.mean(all_hits)) if all_hits else 0.5
    mean_far = float(np.mean(all_fas)) if all_fas else 0.5
    overall_dprime = max(0.0, mean_hr - mean_far) * 4  # scale to ~0-4

    # Capacity: delay at which d-prime drops to 0.5
    dprime_by_delay = [(r["delay_s"], r["d_prime"]) for r in delay_results]
    capacity_s = delays_s[-1]
    for delay, dp in dprime_by_delay:
        if dp < 0.5:
            capacity_s = delay
            break

    score = min(100.0, overall_dprime * 25)

    return {
        "test": "working_memory",
        "score": round(score, 1),
        "overall_d_prime": round(overall_dprime, 3),
        "working_memory_capacity_s": round(capacity_s, 1),
        "per_delay": delay_results,
        "interpretation": (
            f"Working memory: d'={overall_dprime:.2f}, capacity ~{capacity_s:.1f}s. "
            + ("Strong WM." if overall_dprime > 2 else "Moderate WM." if overall_dprime > 1 else "Limited WM.")
        ),
    }


# ── Short-Term Memory ─────────────────────────────────────────────────────────

def test_short_term_memory(
    data: SpikeData,
    max_span: int = 9,
    n_trials_per_span: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Serial position / digit-span analogue for organoids.

    Presents N sequential patterns, tests if organoid can distinguish
    them in order. Finds span length where accuracy falls to chance.

    Returns:
        dict with memory span, serial position curves, primacy/recency effects
    """
    if rng is None:
        rng = np.random.default_rng(2)

    t_start, t_end = data.time_range
    duration = t_end - t_start
    window_s = 0.1

    span_results = []
    for span in range(1, max_span + 1):
        required_duration = span * window_s * 2
        if required_duration > duration * 0.8:
            break

        trial_accuracies = []
        position_hits = [[] for _ in range(span)]

        for _ in range(n_trials_per_span):
            t0 = rng.uniform(t_start, max(t_start + 0.01, t_end - required_duration))

            # Record N patterns
            patterns = []
            for p in range(span):
                t_p = t0 + p * window_s * 1.5
                mask = (data.times >= t_p) & (data.times < t_p + window_s)
                fr = np.zeros(len(data.electrode_ids))
                for i, eid in enumerate(data.electrode_ids):
                    fr[i] = np.sum((data.electrodes == eid) & mask)
                patterns.append(fr)

            # Test recall: compare each pattern to all others
            correct = 0
            for pos, pat in enumerate(patterns):
                similarities = [
                    float(np.dot(pat, other) / (np.linalg.norm(pat) * np.linalg.norm(other) + 1e-10))
                    for other in patterns
                ]
                predicted_pos = int(np.argmax(similarities))
                hit = int(predicted_pos == pos)
                correct += hit
                position_hits[pos].append(hit)

            trial_accuracies.append(correct / span)

        mean_acc = float(np.mean(trial_accuracies))
        span_results.append({
            "span": span,
            "accuracy": round(mean_acc, 3),
            "passed": mean_acc > 0.5,
            "position_accuracy": [round(float(np.mean(ph)), 3) for ph in position_hits if ph],
        })

    # Find memory span (last span with accuracy > 50%)
    memory_span = 0
    for r in span_results:
        if r["passed"]:
            memory_span = r["span"]

    # Primacy and recency effects
    if span_results and span_results[-1]["position_accuracy"]:
        pos_acc = span_results[-1]["position_accuracy"]
        primacy = pos_acc[0] if pos_acc else 0
        recency = pos_acc[-1] if pos_acc else 0
        primacy_effect = primacy > float(np.mean(pos_acc[1:-1])) if len(pos_acc) > 2 else False
        recency_effect = recency > float(np.mean(pos_acc[:-1])) if len(pos_acc) > 1 else False
    else:
        primacy = recency = 0.0
        primacy_effect = recency_effect = False

    score = min(100.0, memory_span * 100 / max_span * 1.5)

    return {
        "test": "short_term_memory",
        "score": round(score, 1),
        "memory_span": memory_span,
        "span_results": span_results,
        "primacy_effect": primacy_effect,
        "recency_effect": recency_effect,
        "primacy_score": round(primacy, 3),
        "recency_score": round(recency, 3),
        "interpretation": (
            f"STM span: {memory_span} items. "
            + (f"Primacy effect detected. " if primacy_effect else "")
            + (f"Recency effect detected." if recency_effect else "")
        ),
    }


# ── Long-Term Memory ──────────────────────────────────────────────────────────

def test_long_term_memory(
    data: SpikeData,
    n_patterns: int = 5,
    test_intervals_s: Optional[list] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Spaced retention test.

    Each pattern is presented, then tested at multiple time offsets.
    Retention = similarity of response at late vs early test.

    Returns:
        dict with retention curves, forgetting rate, retention ratio
    """
    if rng is None:
        rng = np.random.default_rng(3)
    if test_intervals_s is None:
        test_intervals_s = [0.0, 5.0, 30.0, 60.0]

    t_start, t_end = data.time_range
    duration = t_end - t_start
    window_s = 0.2
    n_electrodes = len(data.electrode_ids)

    def get_fr_vector(t0: float) -> np.ndarray:
        mask = (data.times >= t0) & (data.times < t0 + window_s)
        v = np.zeros(n_electrodes)
        for i, eid in enumerate(data.electrode_ids):
            v[i] = np.sum((data.electrodes == eid) & mask)
        return v

    pattern_results = []

    for p in range(n_patterns):
        t0 = t_start + p * (duration / n_patterns) * rng.uniform(0.8, 1.0)
        reference = get_fr_vector(t0)

        interval_sims = []
        for interval in test_intervals_s:
            t_test = t0 + interval
            if t_test + window_s > t_end:
                t_test = max(t_start, t_end - window_s)
            test_vec = get_fr_vector(t_test)

            ref_norm = np.linalg.norm(reference)
            test_norm = np.linalg.norm(test_vec)
            if ref_norm > 0 and test_norm > 0:
                sim = float(np.dot(reference, test_vec) / (ref_norm * test_norm))
            else:
                sim = 0.0

            interval_sims.append({"interval_s": interval, "similarity": round(sim, 4)})

        pattern_results.append({
            "pattern": p + 1,
            "t_start": round(t0, 3),
            "retention_curve": interval_sims,
        })

    # Aggregate retention across patterns
    retention_by_interval = {}
    for interval in test_intervals_s:
        sims = []
        for pr in pattern_results:
            for rc in pr["retention_curve"]:
                if rc["interval_s"] == interval:
                    sims.append(rc["similarity"])
        retention_by_interval[interval] = round(float(np.mean(sims)), 4) if sims else 0.0

    # Retention ratio: last / first
    vals = list(retention_by_interval.values())
    retention_ratio = float(vals[-1] / vals[0]) if vals[0] > 0 else 0.0

    # Forgetting rate (slope of linear fit)
    intervals_arr = np.array(list(retention_by_interval.keys()), dtype=float)
    sims_arr = np.array(vals, dtype=float)
    if len(intervals_arr) > 1 and np.std(intervals_arr) > 0:
        forgetting_rate = float(np.polyfit(intervals_arr, sims_arr, 1)[0])
    else:
        forgetting_rate = 0.0

    score = min(100.0, retention_ratio * 100)

    return {
        "test": "long_term_memory",
        "score": round(score, 1),
        "retention_ratio": round(retention_ratio, 3),
        "forgetting_rate_per_s": round(forgetting_rate, 5),
        "retention_by_interval": {str(k): v for k, v in retention_by_interval.items()},
        "pattern_results": pattern_results,
        "interpretation": (
            f"LTM retention ratio: {retention_ratio:.2f}. "
            f"Forgetting rate: {forgetting_rate:.4f}/s. "
            + ("Strong retention." if retention_ratio > 0.7
               else "Moderate decay." if retention_ratio > 0.4
               else "Rapid forgetting.")
        ),
    }


# ── Associative Memory ────────────────────────────────────────────────────────

def test_associative_memory(
    data: SpikeData,
    n_pairs: int = 6,
    n_training_reps: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Paired-associate / Hopfield-network-style associative memory test.

    Trains on A→B pairs, then presents A alone and measures
    how much the response resembles B.

    Returns:
        dict with completion scores, association accuracy, pair results
    """
    if rng is None:
        rng = np.random.default_rng(4)

    t_start, t_end = data.time_range
    duration = t_end - t_start
    n_electrodes = len(data.electrode_ids)
    window_s = 0.15

    if duration < n_pairs * window_s * 2 * (n_training_reps + 1):
        window_s = max(0.05, duration / (n_pairs * 2 * (n_training_reps + 2)))

    def get_pattern(t0: float) -> np.ndarray:
        mask = (data.times >= t0) & (data.times < t0 + window_s)
        v = np.zeros(n_electrodes)
        for i, eid in enumerate(data.electrode_ids):
            v[i] = np.sum((data.electrodes == eid) & mask)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # Define A–B time pairs
    pair_times = []
    segment = duration / (n_pairs + 1)
    for i in range(n_pairs):
        t_a = t_start + i * segment + rng.uniform(0, segment * 0.3)
        t_b = t_a + window_s + rng.uniform(0.1, 0.3)
        if t_b + window_s <= t_end:
            pair_times.append((t_a, t_b))

    pair_results = []
    completion_scores = []

    for i, (t_a, t_b) in enumerate(pair_times):
        pattern_a = get_pattern(t_a)
        pattern_b = get_pattern(t_b)

        # "Training": accumulate association (in biology: STDP potentiation)
        # We simulate by building a weighted average
        association_weight = 0.0
        for rep in range(n_training_reps):
            noise_a = pattern_a + rng.normal(0, 0.05, n_electrodes)
            noise_b = pattern_b + rng.normal(0, 0.05, n_electrodes)
            na_norm = np.linalg.norm(noise_a)
            nb_norm = np.linalg.norm(noise_b)
            if na_norm > 0:
                noise_a /= na_norm
            if nb_norm > 0:
                noise_b /= nb_norm
            association_weight += float(np.dot(noise_a, noise_b))
        association_weight /= n_training_reps

        # "Cued recall": present A, does response look like B?
        t_test_a = t_a + rng.uniform(0.5, 1.5)
        if t_test_a + window_s > t_end:
            t_test_a = max(t_start, t_end - window_s)
        response = get_pattern(t_test_a)

        # Completion score: how similar is cued response to B?
        b_norm = np.linalg.norm(pattern_b)
        r_norm = np.linalg.norm(response)
        if b_norm > 0 and r_norm > 0:
            completion = float(np.dot(pattern_b, response) / (b_norm * r_norm))
        else:
            completion = 0.0

        pair_results.append({
            "pair": i + 1,
            "association_weight": round(float(association_weight), 4),
            "completion_score": round(completion, 4),
            "correctly_associated": completion > 0.3,
        })
        completion_scores.append(completion)

    mean_completion = float(np.mean(completion_scores)) if completion_scores else 0.0
    n_correct = sum(1 for r in pair_results if r["correctly_associated"])
    association_accuracy = float(n_correct / len(pair_results)) if pair_results else 0.0

    score = min(100.0, mean_completion * 150 + association_accuracy * 20)

    return {
        "test": "associative_memory",
        "score": round(score, 1),
        "mean_completion_score": round(mean_completion, 4),
        "association_accuracy": round(association_accuracy, 3),
        "n_pairs": len(pair_results),
        "n_correct_associations": n_correct,
        "pair_results": pair_results,
        "interpretation": (
            f"Associative memory: {n_correct}/{len(pair_results)} pairs correctly associated. "
            f"Mean completion: {mean_completion:.3f}. "
            + ("Strong associative binding." if association_accuracy > 0.7
               else "Partial associations." if association_accuracy > 0.4
               else "Weak associative memory.")
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _memory_profile(scores: dict) -> str:
    """Classify the organoid's memory profile."""
    wm = scores.get("working_memory", 0)
    stm = scores.get("short_term_memory", 0)
    ltm = scores.get("long_term_memory", 0)
    am = scores.get("associative_memory", 0)

    if all(s > 60 for s in [wm, stm, ltm, am]):
        return "Balanced high-capacity memory across all systems"
    if wm > 70 and stm > 60:
        return "Strong online processing — good for real-time tasks"
    if ltm > 70 and am > 60:
        return "Long-term consolidation dominant — good for learning tasks"
    if am > 70:
        return "Associative memory dominant — Hopfield-network-like dynamics"
    return "Variable memory profile — task-specific optimization recommended"


def _overall_interpretation(composite: float) -> str:
    if composite >= 80:
        return "Exceptional memory — all systems well-developed."
    elif composite >= 60:
        return "Good memory profile — suitable for learning experiments."
    elif composite >= 40:
        return "Moderate memory — basic retention present."
    elif composite >= 20:
        return "Limited memory — short retention windows only."
    return "Minimal memory retention detected."
