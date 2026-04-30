import math
import time

import numpy as np


def _normalize_bounds(lb, ub, dim):
    low = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.array(lb, dtype=float)
    up = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.array(ub, dtype=float)
    return low, up


def _compute_ranking_signature(x):
    return tuple(np.argsort(x, kind="mergesort").tolist())


def compute_ranking_signature(position_or_keys):
    return _compute_ranking_signature(position_or_keys)


def _schedule_signature_from_dataframe(schedule_df):
    if schedule_df is None or schedule_df.empty:
        return ("empty",)

    ordered = schedule_df.sort_values(
        ["machine_id", "S_lj", "C_lj", "job_id", "voyage", "op_seq"],
        kind="mergesort",
    )
    return tuple(
        (
            int(row.machine_id),
            int(row.job_id),
            int(row.voyage),
            int(row.op_seq),
        )
        for row in ordered.itertuples(index=False)
    )


def compute_decoded_signatures(decoded_schedules):
    return [_schedule_signature_from_dataframe(schedule_df) for schedule_df in decoded_schedules]


def compute_machine_order_signature(decoded_schedule):
    if decoded_schedule is None or decoded_schedule.empty:
        return ("empty",)

    machine_sequences = []
    for machine_id, machine_df in decoded_schedule.groupby("machine_id", sort=True):
        ordered = machine_df.sort_values(["S_lj", "C_lj", "job_id", "voyage", "op_seq"])
        sequence = tuple(
            (int(row.job_id), int(row.voyage), int(row.op_seq))
            for row in ordered.itertuples(index=False)
        )
        machine_sequences.append((int(machine_id), sequence))
    return tuple(machine_sequences)


def _count_inversions(values):
    if len(values) < 2:
        return 0

    mid = len(values) // 2
    left = values[:mid]
    right = values[mid:]
    inversions = _count_inversions(left) + _count_inversions(right)

    i = 0
    j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    values[:] = merged
    return inversions


def compute_kendall_distance(rank_a, rank_b):
    if rank_a is None or rank_b is None or len(rank_a) != len(rank_b):
        return 1.0

    n = len(rank_a)
    if n < 2:
        return 0.0

    pos_b = {item: idx for idx, item in enumerate(rank_b)}
    mapped = [pos_b[item] for item in rank_a]
    inversions = _count_inversions(mapped)
    return inversions / (n * (n - 1) / 2.0)


def _machine_position_map(machine_order_signature):
    position_map = {}
    if machine_order_signature is None or machine_order_signature == ("empty",):
        return position_map

    for machine_id, sequence in machine_order_signature:
        for idx, operation in enumerate(sequence):
            position_map[operation] = (machine_id, idx)
    return position_map


def compute_machine_order_distance(signature_a, signature_b):
    map_a = _machine_position_map(signature_a)
    map_b = _machine_position_map(signature_b)
    common_ops = set(map_a).intersection(map_b)
    if not common_ops:
        return 1.0

    moved = sum(1 for operation in common_ops if map_a[operation] != map_b[operation])
    return moved / len(common_ops)


def _average_pairwise_distance(signatures, distance_func):
    signatures = [sig for sig in signatures if sig is not None]
    if len(signatures) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            total += distance_func(signatures[i], signatures[j])
            count += 1
    return total / max(1, count)


def _resolve_count_threshold(threshold, population_size):
    if threshold <= 1.0:
        return max(1, int(math.ceil(threshold * population_size)))
    return max(1, int(threshold))


def detect_fitness_plateau(best_score_history, stagnation_patience, eps_improve):
    if len(best_score_history) < stagnation_patience + 1:
        return False, None

    old_score = float(best_score_history[-stagnation_patience - 1]["score"])
    current_score = float(best_score_history[-1]["score"])
    improvement = old_score - current_score
    return improvement <= eps_improve, improvement


def detect_structural_stagnation(
    machine_order_signatures,
    ranking_signatures,
    decoded_signatures,
    gbest_decoded_signature,
    population_size,
    machine_family_threshold,
    ranking_collapse_threshold,
    structural_distance_threshold,
    unique_schedule_threshold,
    dup_ratio_threshold,
    use_machine_order_signature=True,
    use_ranking_similarity=True,
):
    unique_machine_families = (
        len(set(machine_order_signatures)) if use_machine_order_signature else population_size
    )
    machine_family_limit = _resolve_count_threshold(machine_family_threshold, population_size)
    avg_machine_distance = (
        _average_pairwise_distance(machine_order_signatures, compute_machine_order_distance)
        if use_machine_order_signature
        else 1.0
    )
    avg_ranking_distance = (
        _average_pairwise_distance(ranking_signatures, compute_kendall_distance)
        if use_ranking_similarity
        else 1.0
    )

    machine_order_collapse = (
        use_machine_order_signature
        and (
            unique_machine_families <= machine_family_limit
            or avg_machine_distance <= structural_distance_threshold
        )
    )
    ranking_collapse = use_ranking_similarity and avg_ranking_distance <= ranking_collapse_threshold

    unique_schedule_count = len(set(sig for sig in decoded_signatures if sig is not None))
    schedule_limit = _resolve_count_threshold(unique_schedule_threshold, population_size)
    schedule_collapse = unique_schedule_count <= schedule_limit

    dup_ratio_to_gbest = 0.0
    if gbest_decoded_signature is not None:
        dup_ratio_to_gbest = float(
            sum(sig == gbest_decoded_signature for sig in decoded_signatures) / max(1, population_size)
        )
    gbest_dup_collapse = dup_ratio_to_gbest >= dup_ratio_threshold

    structural_stagnation = (
        machine_order_collapse
        or ranking_collapse
        or schedule_collapse
        or gbest_dup_collapse
    )
    details = {
        "unique_machine_families": int(unique_machine_families),
        "machine_family_limit": int(machine_family_limit),
        "unique_schedule_count": int(unique_schedule_count),
        "schedule_limit": int(schedule_limit),
        "dup_ratio_to_gbest": float(dup_ratio_to_gbest),
        "avg_machine_order_distance": float(avg_machine_distance),
        "avg_ranking_distance": float(avg_ranking_distance),
        "machine_order_collapse": bool(machine_order_collapse),
        "ranking_collapse": bool(ranking_collapse),
        "schedule_collapse": bool(schedule_collapse),
        "gbest_dup_collapse": bool(gbest_dup_collapse),
        "structural_stagnation": bool(structural_stagnation),
    }
    return structural_stagnation, details


def compute_stagnation_score(
    fitness_plateau_score,
    machine_order_collapse_score,
    ranking_collapse_score,
    schedule_collapse_score,
    gbest_dup_collapse_score,
    stagnation_score_weights=None,
):
    weights = {
        "fitness": 0.4,
        "machine_order": 0.2,
        "ranking": 0.2,
        "schedule": 0.1,
        "gbest_dup": 0.1,
    }
    if stagnation_score_weights:
        weights.update(stagnation_score_weights)

    return (
        weights["fitness"] * fitness_plateau_score
        + weights["machine_order"] * machine_order_collapse_score
        + weights["ranking"] * ranking_collapse_score
        + weights["schedule"] * schedule_collapse_score
        + weights["gbest_dup"] * gbest_dup_collapse_score
    )


def should_activate_ssr(
    fitness_plateau,
    structural_stagnation,
    stagnation_score,
    stagnation_mode="rule",
    stagnation_score_threshold=0.7,
):
    if stagnation_mode == "score":
        return fitness_plateau and stagnation_score >= stagnation_score_threshold
    return fitness_plateau and structural_stagnation


def _evaluate_candidate(x, fobj=None, decoder=None, build_signatures=False):
    if decoder is not None:
        schedule_df, metrics = decoder.decode_from_continuous(x)
        metrics = dict(metrics) if metrics is not None else {}
        if "is_feasible" not in metrics:
            metrics["is_feasible"] = metrics.get("infeasible_reason") in (None, "", False)
        decoded_signature = None
        machine_order_signature = None
        ranking_signature = None
        if build_signatures:
            decoded_signature = _schedule_signature_from_dataframe(schedule_df)
            machine_order_signature = compute_machine_order_signature(schedule_df)
            ranking_signature = _compute_ranking_signature(x)
        return {
            "fitness": float(metrics["total_tardiness"]),
            "schedule_df": schedule_df,
            "metrics": metrics,
            "decoded_signature": decoded_signature,
            "machine_order_signature": machine_order_signature,
            "ranking_signature": ranking_signature,
            "op_priority": None,
        }

    if fobj is None:
        raise ValueError("Either `fobj` or `decoder` must be provided.")

    ranking_signature = _compute_ranking_signature(x) if build_signatures else None
    return {
        "fitness": float(fobj(x)),
        "schedule_df": None,
        "metrics": None,
        "decoded_signature": None,
        "machine_order_signature": None,
        "ranking_signature": ranking_signature,
        "op_priority": None,
    }


def _build_evaluation_metadata(evaluation, position):
    schedule_df = evaluation.get("schedule_df")
    if evaluation.get("decoded_signature") is None:
        evaluation["decoded_signature"] = _schedule_signature_from_dataframe(schedule_df)
    if evaluation.get("machine_order_signature") is None:
        evaluation["machine_order_signature"] = compute_machine_order_signature(schedule_df)
    if evaluation.get("ranking_signature") is None:
        evaluation["ranking_signature"] = _compute_ranking_signature(position)
    if evaluation.get("op_priority") is None:
        evaluation["op_priority"] = _extract_operation_priority_from_schedule(schedule_df)
    return evaluation


def _refresh_population_metadata(pos, evaluations):
    decoded_signatures = []
    machine_order_signatures = []
    ranking_signatures = []
    for idx, evaluation in enumerate(evaluations):
        _build_evaluation_metadata(evaluation, pos[idx])
        decoded_signatures.append(evaluation["decoded_signature"])
        machine_order_signatures.append(evaluation["machine_order_signature"])
        ranking_signatures.append(evaluation["ranking_signature"])
    return decoded_signatures, machine_order_signatures, ranking_signatures


def _update_elite_archive(elite_archive, pos, fitness, evaluations, archive_signatures, elite_size):
    merged = {}
    for item in elite_archive:
        key = item.get("archive_signature", item.get("decoded_signature"))
        best_existing = merged.get(key)
        if best_existing is None or item["fitness"] < best_existing["fitness"]:
            merged[key] = {
                "position": item["position"].copy(),
                "fitness": float(item["fitness"]),
                "decoded_signature": item.get("decoded_signature"),
                "op_priority": item.get("op_priority"),
                "is_feasible": bool(item.get("is_feasible", True)),
                "archive_signature": key,
            }

    for idx in np.argsort(fitness):
        key = archive_signatures[idx]
        metrics = evaluations[idx].get("metrics")
        candidate = {
            "position": pos[idx].copy(),
            "fitness": float(fitness[idx]),
            "decoded_signature": evaluations[idx].get("decoded_signature"),
            "op_priority": evaluations[idx].get("op_priority"),
            "is_feasible": _is_feasible_metrics(metrics, missing_feasibility_is_feasible=True),
            "archive_signature": key,
        }
        best_existing = merged.get(key)
        if best_existing is None or candidate["fitness"] < best_existing["fitness"]:
            merged[key] = candidate

    archive = sorted(merged.values(), key=lambda item: item["fitness"])
    return archive[:elite_size]


def _select_protected_elite_indices(fitness, archive_signatures, elite_size):
    protected_indices = []
    seen_signatures = set()
    for idx in np.argsort(fitness):
        signature = archive_signatures[idx]
        if signature in seen_signatures:
            continue
        protected_indices.append(int(idx))
        seen_signatures.add(signature)
        if len(protected_indices) >= elite_size:
            break
    return protected_indices


def _is_feasible_metrics(metrics, missing_feasibility_is_feasible=False):
    if not metrics:
        return False
    if "is_feasible" not in metrics:
        return bool(missing_feasibility_is_feasible)
    return bool(metrics["is_feasible"])


def _validate_operation_reference(operation_reference, dim):
    if len(operation_reference) != dim:
        return False, f"operation_reference_dim_mismatch:{len(operation_reference)}!={dim}"
    for op in operation_reference:
        if not isinstance(op, tuple) or len(op) != 3:
            return False, "operation_reference_invalid_tuple_format"
    return True, None


def _plain_random_position(lb, ub):
    interval = np.maximum(ub - lb, 1e-12)
    return lb + interval * np.random.rand(lb.size)


def _extract_operation_priority_from_schedule(schedule_df):
    if schedule_df is None or schedule_df.empty:
        return {}

    ordered = schedule_df.sort_values(
        ["S_lj", "C_lj", "machine_id", "job_id", "voyage", "op_seq"],
        kind="mergesort",
    )
    denom = max(1, len(ordered) - 1)
    knowledge = {}
    for rank, row in enumerate(ordered.itertuples(index=False)):
        operation = (int(row.job_id), int(row.voyage), int(row.op_seq))
        knowledge[operation] = rank / denom
    return knowledge


def _select_ssr_replacement_indices(
    fitness,
    metrics_list,
    protected_indices,
    restart_budget,
    decoder_available=True,
    missing_feasibility_is_feasible=False,
):
    if restart_budget <= 0:
        return [], 0, 0

    protected = set(int(idx) for idx in protected_indices)
    worst_order = [int(idx) for idx in np.argsort(fitness)[::-1] if int(idx) not in protected]

    if not decoder_available:
        selected = worst_order[:restart_budget]
        return selected, 0, len(selected)

    infeasible_indices = [
        idx for idx in worst_order
        if not _is_feasible_metrics(
            metrics_list[idx],
            missing_feasibility_is_feasible=missing_feasibility_is_feasible,
        )
    ]
    feasible_indices = [
        idx for idx in worst_order
        if _is_feasible_metrics(
            metrics_list[idx],
            missing_feasibility_is_feasible=missing_feasibility_is_feasible,
        )
    ]

    selected = infeasible_indices[:restart_budget]
    infeasible_replaced = len(selected)

    remaining = restart_budget - len(selected)
    feasible_replaced = 0
    if remaining > 0:
        extra = feasible_indices[:remaining]
        selected.extend(extra)
        feasible_replaced = len(extra)

    return selected, infeasible_replaced, feasible_replaced


def _build_operation_priority_knowledge_from_archive(
    elite_archive,
    decoder,
    operation_reference,
):
    if not elite_archive or decoder is None or not operation_reference:
        return None

    best_fit = min(float(item["fitness"]) for item in elite_archive)
    fit_diffs = np.array(
        [max(0.0, float(item["fitness"]) - best_fit) for item in elite_archive],
        dtype=float,
    )
    fit_scale = max(1.0, float(np.median(fit_diffs)) if fit_diffs.size else 1.0)

    weighted_scores = {operation: 0.0 for operation in operation_reference}
    weighted_squares = {operation: 0.0 for operation in operation_reference}
    total_weights = {operation: 0.0 for operation in operation_reference}
    sample_counts = {operation: 0 for operation in operation_reference}

    for item in elite_archive:
        if not bool(item.get("is_feasible", True)):
            continue

        op_priority = item.get("op_priority")
        if not op_priority:
            continue

        fit_gap = max(0.0, float(item["fitness"]) - best_fit)
        weight = 1.0 / (1.0 + fit_gap / fit_scale)
        for operation, priority_score in op_priority.items():
            if operation not in weighted_scores:
                continue
            priority_score = float(priority_score)
            weighted_scores[operation] += weight * priority_score
            weighted_squares[operation] += weight * priority_score * priority_score
            total_weights[operation] += weight
            sample_counts[operation] += 1

    fallback = np.linspace(0.0, 1.0, num=len(operation_reference), dtype=float)
    priority = np.empty(len(operation_reference), dtype=float)
    confidence = np.zeros(len(operation_reference), dtype=float)
    matched_count = 0

    for pos_idx, operation in enumerate(operation_reference):
        weight = total_weights[operation]
        if weight <= 0.0:
            priority[pos_idx] = fallback[pos_idx]
            continue

        mean_score = weighted_scores[operation] / weight
        second_moment = weighted_squares[operation] / weight
        variance = max(0.0, second_moment - mean_score * mean_score)
        std_score = math.sqrt(variance)

        priority[pos_idx] = mean_score
        # Rank scores live in [0, 1]; 0.5 is a maximally uncertain spread.
        agreement = 1.0 - min(1.0, std_score / 0.5)
        sample_support = min(1.0, sample_counts[operation] / max(1.0, len(elite_archive)))
        confidence[pos_idx] = float(np.clip(agreement * sample_support, 0.0, 1.0))
        matched_count += 1

    if matched_count == 0:
        return None

    signal_ratio = matched_count / max(1, len(operation_reference))
    return {
        "priority": priority,
        "confidence": confidence,
        "signal_ratio": float(signal_ratio),
        "matched_count": int(matched_count),
        "total_count": int(len(operation_reference)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_min": float(np.min(confidence)),
        "confidence_max": float(np.max(confidence)),
    }


def _generate_random_key_agent_from_priority_knowledge(
    priority_knowledge,
    lb,
    ub,
    base_noise_scale=0.08,
    uniform_mix=0.10,
    min_noise_scale=0.01,
    max_confidence=0.85,
):
    if priority_knowledge is None:
        return _plain_random_position(lb, ub)

    priority = np.array(priority_knowledge["priority"], dtype=float).copy()
    confidence = np.array(priority_knowledge["confidence"], dtype=float)
    confidence = np.clip(confidence, 0.0, max_confidence)
    noise_scale = min_noise_scale + base_noise_scale * (1.0 - confidence)
    noisy_scores = priority + np.random.normal(loc=0.0, scale=noise_scale)

    if uniform_mix > 0.0:
        mix_mask = np.random.rand(noisy_scores.size) < uniform_mix
        if np.any(mix_mask):
            noisy_scores[mix_mask] = np.random.rand(int(np.sum(mix_mask)))

    noisy_scores = np.clip(noisy_scores, 0.0, 1.0)
    order = np.argsort(noisy_scores, kind="mergesort")
    normalized = np.empty_like(noisy_scores, dtype=float)
    if noisy_scores.size == 1:
        normalized[0] = 0.5
    else:
        normalized[order] = np.linspace(0.0, 1.0, num=noisy_scores.size, dtype=float)
    return np.clip(lb + (ub - lb) * normalized, lb, ub)


def _build_dimensional_search_guidance(
    pos,
    fitness,
    elite_archive,
    gbest,
    lb,
    ub,
    elite_k,
    min_width_ratio=0.04,
    width_scale=2.5,
):
    """Build cheap dimension-level SSR/DEx guidance from current and historical elites."""
    pos = np.asarray(pos, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    interval = np.maximum(ub - lb, 1e-12)
    elite_k = max(1, min(int(elite_k), pos.shape[0]))

    current_elites = pos[np.argsort(fitness)[:elite_k]]
    archive_elites = [
        np.asarray(item["position"], dtype=float)
        for item in elite_archive[:elite_k]
        if item.get("position") is not None
    ]
    if archive_elites:
        elite_positions = np.vstack([current_elites, np.vstack(archive_elites)])
    else:
        elite_positions = current_elites

    elite_center = np.mean(elite_positions, axis=0)
    elite_std = np.std(elite_positions, axis=0)
    elite_low = np.min(elite_positions, axis=0)
    elite_high = np.max(elite_positions, axis=0)

    x_centered = pos - np.mean(pos, axis=0)
    y_centered = fitness - np.mean(fitness)
    denom = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
    rho = np.divide(
        np.sum(x_centered * y_centered[:, None], axis=0),
        denom,
        out=np.zeros(pos.shape[1], dtype=float),
        where=denom > 1e-12,
    )
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    directional_anchor = np.where(rho > 0.0, elite_low, elite_high)
    corr_strength = np.clip(np.abs(rho), 0.0, 1.0)
    convergence_strength = 1.0 - np.clip(elite_std / interval, 0.0, 1.0)
    confidence = np.clip(0.5 * corr_strength + 0.5 * convergence_strength, 0.0, 1.0)

    center = confidence * directional_anchor + (1.0 - confidence) * elite_center
    radius = np.maximum(width_scale * elite_std, min_width_ratio * interval)
    reduced_lb = np.clip(center - radius, lb, ub)
    reduced_ub = np.clip(center + radius, lb, ub)

    return {
        "center": np.clip(center, lb, ub),
        "reduced_lb": reduced_lb,
        "reduced_ub": reduced_ub,
        "confidence": confidence,
        "rho": rho,
        "gbest": np.clip(np.asarray(gbest, dtype=float), lb, ub),
    }


def _generate_reduced_space_candidate(
    guidance,
    lb,
    ub,
    gbest_pull=0.35,
    uncertain_uniform_ratio=0.20,
):
    sample = guidance["reduced_lb"] + (
        guidance["reduced_ub"] - guidance["reduced_lb"]
    ) * np.random.rand(lb.size)
    if gbest_pull > 0.0:
        pull = np.random.rand(lb.size) < gbest_pull
        sample[pull] = 0.5 * sample[pull] + 0.5 * guidance["gbest"][pull]

    uncertain = guidance["confidence"] < 0.35
    if np.any(uncertain) and uncertain_uniform_ratio > 0.0:
        mix = uncertain & (np.random.rand(lb.size) < uncertain_uniform_ratio)
        sample[mix] = lb[mix] + (ub[mix] - lb[mix]) * np.random.rand(int(np.sum(mix)))

    return np.clip(sample, lb, ub)


def _generate_diversity_exploration_candidate(
    guidance,
    lb,
    ub,
    dim_ratio=0.25,
    opposition_ratio=0.50,
    gaussian_scale=0.025,
):
    interval = np.maximum(ub - lb, 1e-12)
    candidate = guidance["gbest"].copy()
    dim_ratio = float(np.clip(dim_ratio, 0.0, 1.0))
    mask = np.random.rand(lb.size) < dim_ratio
    if not np.any(mask):
        mask[np.random.randint(0, lb.size)] = True

    uniform_mask = mask & (np.random.rand(lb.size) >= opposition_ratio)
    opposite_mask = mask & ~uniform_mask
    candidate[uniform_mask] = lb[uniform_mask] + interval[uniform_mask] * np.random.rand(int(np.sum(uniform_mask)))
    candidate[opposite_mask] = lb[opposite_mask] + ub[opposite_mask] - candidate[opposite_mask]

    jitter = np.random.normal(loc=0.0, scale=gaussian_scale, size=lb.size) * interval
    candidate += jitter
    return np.clip(candidate, lb, ub)


def _apply_inline_search_guidance(
    candidate,
    guidance,
    lb,
    ub,
    activation_prob=0.15,
    confidence_threshold=0.70,
    reduced_dim_ratio=0.20,
    reduced_blend=0.35,
    explore_dim_ratio=0.15,
    opposition_ratio=0.50,
):
    if guidance is None or np.random.rand() >= activation_prob:
        return np.clip(candidate, lb, ub), 0, 0

    guided = np.asarray(candidate, dtype=float).copy()
    interval = np.maximum(ub - lb, 1e-12)
    confidence = guidance["confidence"]
    high_confidence = confidence >= confidence_threshold
    low_confidence = ~high_confidence

    reduced_mask = high_confidence & (np.random.rand(lb.size) < reduced_dim_ratio)
    reduced_count = int(np.sum(reduced_mask))
    if reduced_count > 0:
        reduced_sample = guidance["reduced_lb"][reduced_mask] + (
            guidance["reduced_ub"][reduced_mask] - guidance["reduced_lb"][reduced_mask]
        ) * np.random.rand(reduced_count)
        guided[reduced_mask] = (
            (1.0 - reduced_blend) * guided[reduced_mask]
            + reduced_blend * reduced_sample
        )

    explore_mask = low_confidence & (np.random.rand(lb.size) < explore_dim_ratio)
    explore_count = int(np.sum(explore_mask))
    if explore_count > 0:
        uniform_mask = explore_mask & (np.random.rand(lb.size) >= opposition_ratio)
        opposite_mask = explore_mask & ~uniform_mask
        uniform_count = int(np.sum(uniform_mask))
        if uniform_count > 0:
            guided[uniform_mask] = (
                lb[uniform_mask]
                + interval[uniform_mask] * np.random.rand(uniform_count)
            )
        if np.any(opposite_mask):
            guided[opposite_mask] = (
                lb[opposite_mask] + ub[opposite_mask] - guided[opposite_mask]
            )

    return np.clip(guided, lb, ub), reduced_count, explore_count


def CAOA_SSR(
    N,
    max_iter,
    lb,
    ub,
    dim,
    fobj=None,
    decoder=None,
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    delta=1e-3,
    initial_energy=10.0,
    max_FEs=None,
    initial_pos=None,
    IT=10,
    K=30,
    stagnation_window=None,
    stagnation_patience=None,
    eps_improve=1e-6,
    elite_size=5,
    dup_ratio_threshold=0.6,
    unique_schedule_threshold=0.25,
    machine_family_threshold=0.7,
    ranking_collapse_threshold=0.08,
    structural_distance_threshold=0.25,
    use_machine_order_signature=True,
    use_ranking_similarity=True,
    stagnation_mode="rule",
    stagnation_score_weights=None,
    stagnation_score_threshold=0.7,
    partial_restart_ratio=0.25,
    preserve_default_random_reinit=True,
    ssr_elite_k=None,
    ssr_min_knowledge_signal_ratio=0.8,
    ssr_knowledge_noise_scale=0.08,
    ssr_knowledge_uniform_mix=0.10,
    ssr_knowledge_min_noise_scale=0.01,
    ssr_knowledge_max_confidence=0.85,
    ssr_allow_plateau_activation=True,
    ssr_min_plateau_checks=2,
    ssr_candidate_trials=3,
    ssr_accept_only_improvement=True,
    ssr_commit_requires_gbest_improvement=False,
    ssr_inline_guidance=False,
    ssr_inline_prob=0.15,
    ssr_inline_confidence_threshold=0.70,
    ssr_inline_reduced_dim_ratio=0.20,
    ssr_inline_reduced_blend=0.35,
    ssr_inline_explore_dim_ratio=0.15,
    ssr_inline_reinit_uses_knowledge=True,
    ssr_random_fallback=False,
    ssr_balanced_reinit=True,
    ssr_explore_dim_ratio=0.25,
    ssr_explore_opposition_ratio=0.50,
    ssr_reduction_min_width=0.04,
    ssr_reduction_width_scale=2.5,
    ssr_reduced_gbest_pull=0.35,
    ssr_uncertain_uniform_ratio=0.20,
    ssr_force_mode_quota=False,
    ssr_adaptive_mode=True,
    ssr_escape_after_failed_activations=2,
    ssr_cooldown_checks=1,
    ssr_skip_last_checks=1,
    ssr_escape_reduction_width_multiplier=1.8,
    ssr_escape_noise_multiplier=1.7,
    ssr_escape_dim_ratio=0.45,
    ssr_escape_gbest_pull=0.15,
    ssr_escape_accept_margin_ratio=0.03,
    ssr_force_escape_after_failed_exploit=True,
    ssr_exploit_max_unique_rank_ratio=0.85,
    ssr_exploit_max_machine_family_ratio=0.85,
    ssr_exploit_min_dup_ratio=0.35,
    ssr_exploit_restart_ratio=0.10,
    ssr_exploit_diversity_quota=1,
    ssr_escape_diversity_quota=3,
    missing_feasibility_is_feasible=False,
    return_diagnostics=False,
    verbose=True,
):
    if fobj is None and decoder is None:
        raise ValueError("Either `fobj` or `decoder` must be provided to CAOA_SSR.")

    lb, ub = _normalize_bounds(lb, ub, dim)
    if stagnation_window is None:
        stagnation_window = K
    if stagnation_patience is None:
        stagnation_patience = max(1, int(math.ceil(stagnation_window / max(1, IT))))

    if initial_pos is not None:
        pos = np.clip(np.array(initial_pos, dtype=float).copy(), lb, ub)
    else:
        pos = lb + (ub - lb) * np.random.rand(N, dim)

    energies = np.full(N, initial_energy, dtype=float)
    fitness = np.zeros(N, dtype=float)
    schedule_dfs = [None] * N
    metrics_list = [None] * N
    decoded_signatures = [None] * N
    machine_order_signatures = [None] * N
    ranking_signatures = [None] * N
    evaluations = [None] * N

    fe_counter = 0
    for i in range(N):
        evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
        evaluations[i] = evaluation
        fitness[i] = evaluation["fitness"]
        schedule_dfs[i] = evaluation["schedule_df"]
        metrics_list[i] = evaluation["metrics"]
        decoded_signatures[i] = evaluation["decoded_signature"]
        machine_order_signatures[i] = evaluation["machine_order_signature"]
        ranking_signatures[i] = evaluation["ranking_signature"]
        fe_counter += 1

    best_idx = int(np.argmin(fitness))
    gBestScore = float(fitness[best_idx])
    gBest = pos[best_idx].copy()
    gBestDecodedSignature = decoded_signatures[best_idx]

    best_score_history = []
    best_decoded_signature_history = []
    best_structural_signature_history = []
    elite_archive = []
    cg_curve = []
    avg_curve = []
    diagnostics = []
    plateau_check_streak = 0
    ssr_no_improve_streak = 0
    ssr_cooldown_remaining = 0
    ssr_failure_reference_best = float(gBestScore)
    last_ssr_failed_mode = None
    inline_dimensional_guidance = None
    inline_priority_knowledge = None

    decoder_available = decoder is not None
    operation_reference = list(getattr(decoder, "L_ref", [])) if decoder_available else []
    operation_reference_valid, operation_reference_error = (
        _validate_operation_reference(operation_reference, dim)
        if decoder_available
        else (False, "decoder_unavailable")
    )
    ssr_elite_k = elite_size if ssr_elite_k is None else int(ssr_elite_k)

    cached_check_metrics = {
        "unique_ranking_count": 0,
        "unique_schedule_count": 0,
        "unique_machine_family_count": 0,
        "avg_machine_order_distance": 0.0,
        "avg_ranking_distance": 0.0,
        "dup_ratio_to_gbest": 0.0,
        "feasible_count": int(
            sum(
                _is_feasible_metrics(
                    m,
                    missing_feasibility_is_feasible=missing_feasibility_is_feasible,
                )
                for m in metrics_list
            )
        )
        if decoder_available
        else 0,
        "knowledge_signal_ratio": 0.0,
        "knowledge_confidence_mean": 0.0,
    }

    start_total = time.perf_counter()
    for t in range(max_iter):
        iter_start = time.perf_counter()
        energy_reinit_count = 0
        deterioration_reinit_count = 0
        ssr_replacement_count = 0
        ssr_candidate_attempt_count = 0
        ssr_rejected_candidate_count = 0
        knowledge_replacement_count = 0
        reduced_space_replacement_count = 0
        diversity_replacement_count = 0
        knowledge_candidate_count = 0
        reduced_space_candidate_count = 0
        diversity_candidate_count = 0
        inline_reduced_dim_count = 0
        inline_explore_dim_count = 0
        inline_knowledge_reinit_count = 0
        ssr_fallback_random_count = 0
        ssr_infeasible_replaced = 0
        ssr_worst_feasible_replaced = 0
        knowledge_signal_ratio = 0.0
        knowledge_confidence_mean = 0.0
        matched_operation_count = 0
        total_operation_count = len(operation_reference)
        ssr_triggered = False
        ssr_search_mode = "none"
        ssr_skip_reason = None
        check_this_iter = (t + 1) % IT == 0

        if max_FEs is not None and fe_counter >= max_FEs:
            break

        f_shifted = fitness - np.min(fitness)
        probs = 1.0 / (1.0 + f_shifted)
        probs /= probs.sum()

        for i in range(N):
            if max_FEs is not None and fe_counter >= max_FEs:
                break

            leader_idx = np.random.choice(N, p=probs)
            if i == leader_idx:
                continue

            leader_pos = pos[leader_idx].copy()
            old_pos = pos[i].copy()
            old_fit = float(fitness[i])

            r = np.random.rand(dim)
            new_pos = pos[i] + alpha * (leader_pos - pos[i]) + beta * (1.0 - 2.0 * r)
            new_pos = np.clip(new_pos, lb, ub)
            inline_guided_candidate = False
            if ssr_inline_guidance and inline_dimensional_guidance is not None:
                new_pos, reduced_count, explore_count = _apply_inline_search_guidance(
                    candidate=new_pos,
                    guidance=inline_dimensional_guidance,
                    lb=lb,
                    ub=ub,
                    activation_prob=ssr_inline_prob,
                    confidence_threshold=ssr_inline_confidence_threshold,
                    reduced_dim_ratio=ssr_inline_reduced_dim_ratio,
                    reduced_blend=ssr_inline_reduced_blend,
                    explore_dim_ratio=ssr_inline_explore_dim_ratio,
                    opposition_ratio=ssr_explore_opposition_ratio,
                )
                inline_reduced_dim_count += reduced_count
                inline_explore_dim_count += explore_count
                inline_guided_candidate = (reduced_count + explore_count) > 0

            evaluation = _evaluate_candidate(new_pos, fobj=fobj, decoder=decoder)
            new_fit = evaluation["fitness"]
            fe_counter += 1
            was_reinitialized = False

            if abs(new_fit - old_fit) > delta and new_fit > old_fit:
                if inline_guided_candidate:
                    new_pos = old_pos
                    new_fit = old_fit
                    evaluation = {
                        "fitness": old_fit,
                        "schedule_df": schedule_dfs[i],
                        "metrics": metrics_list[i],
                        "decoded_signature": decoded_signatures[i],
                        "machine_order_signature": machine_order_signatures[i],
                        "ranking_signature": ranking_signatures[i],
                        "op_priority": evaluations[i].get("op_priority") if evaluations[i] else None,
                    }
                elif preserve_default_random_reinit and (max_FEs is None or fe_counter < max_FEs):
                    if (
                        ssr_inline_guidance
                        and ssr_inline_reinit_uses_knowledge
                        and inline_priority_knowledge is not None
                    ):
                        new_pos = _generate_random_key_agent_from_priority_knowledge(
                            priority_knowledge=inline_priority_knowledge,
                            lb=lb,
                            ub=ub,
                            base_noise_scale=ssr_knowledge_noise_scale,
                            uniform_mix=ssr_knowledge_uniform_mix,
                            min_noise_scale=ssr_knowledge_min_noise_scale,
                            max_confidence=ssr_knowledge_max_confidence,
                        )
                        inline_knowledge_reinit_count += 1
                    else:
                        new_pos = _plain_random_position(lb, ub)
                    evaluation = _evaluate_candidate(new_pos, fobj=fobj, decoder=decoder)
                    new_fit = evaluation["fitness"]
                    fe_counter += 1
                    deterioration_reinit_count += 1
                else:
                    new_pos = old_pos
                    new_fit = old_fit
                    evaluation = {
                        "fitness": old_fit,
                        "schedule_df": schedule_dfs[i],
                        "metrics": metrics_list[i],
                        "decoded_signature": decoded_signatures[i],
                        "machine_order_signature": machine_order_signatures[i],
                        "ranking_signature": ranking_signatures[i],
                        "op_priority": evaluations[i].get("op_priority") if evaluations[i] else None,
                    }

            pos[i] = new_pos
            fitness[i] = new_fit
            evaluations[i] = evaluation
            schedule_dfs[i] = evaluation["schedule_df"]
            metrics_list[i] = evaluation["metrics"]
            decoded_signatures[i] = evaluation["decoded_signature"]
            machine_order_signatures[i] = evaluation["machine_order_signature"]
            ranking_signatures[i] = evaluation["ranking_signature"]

            if not was_reinitialized:
                energies[i] -= gamma * np.linalg.norm(new_pos - old_pos)

            if not was_reinitialized and energies[i] <= 0:
                if max_FEs is None or fe_counter < max_FEs:
                    if (
                        ssr_inline_guidance
                        and ssr_inline_reinit_uses_knowledge
                        and inline_priority_knowledge is not None
                    ):
                        pos[i] = _generate_random_key_agent_from_priority_knowledge(
                            priority_knowledge=inline_priority_knowledge,
                            lb=lb,
                            ub=ub,
                            base_noise_scale=ssr_knowledge_noise_scale,
                            uniform_mix=ssr_knowledge_uniform_mix,
                            min_noise_scale=ssr_knowledge_min_noise_scale,
                            max_confidence=ssr_knowledge_max_confidence,
                        )
                        inline_knowledge_reinit_count += 1
                    else:
                        pos[i] = _plain_random_position(lb, ub)
                    energies[i] = initial_energy
                    evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
                    evaluations[i] = evaluation
                    fitness[i] = evaluation["fitness"]
                    schedule_dfs[i] = evaluation["schedule_df"]
                    metrics_list[i] = evaluation["metrics"]
                    decoded_signatures[i] = evaluation["decoded_signature"]
                    machine_order_signatures[i] = evaluation["machine_order_signature"]
                    ranking_signatures[i] = evaluation["ranking_signature"]
                    fe_counter += 1
                    energy_reinit_count += 1

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < gBestScore:
            gBestScore = float(fitness[best_idx])
            gBest = pos[best_idx].copy()
            gBestDecodedSignature = decoded_signatures[best_idx]
        if gBestScore < ssr_failure_reference_best - eps_improve:
            ssr_no_improve_streak = 0
            ssr_failure_reference_best = float(gBestScore)
            last_ssr_failed_mode = None

        stagnation_details = {
            "improvement_window": None,
            "fitness_plateau": False,
            "structural_stagnation": False,
            "machine_order_collapse": False,
            "ranking_collapse": False,
            "schedule_collapse": False,
            "gbest_dup_collapse": False,
            "stagnation_score": 0.0,
            "activation_reason": "skipped_until_it" if not check_this_iter else "not_checked",
            "ssr_search_mode": ssr_search_mode,
            "ssr_skip_reason": ssr_skip_reason,
            "ssr_no_improve_streak": ssr_no_improve_streak,
            "ssr_cooldown_remaining": ssr_cooldown_remaining,
            "knowledge_warning": None,
        }

        if check_this_iter:
            decoded_signatures, machine_order_signatures, ranking_signatures = _refresh_population_metadata(
                pos,
                evaluations,
            )
            gBestDecodedSignature = decoded_signatures[best_idx]
            archive_signatures = [
                machine_sig if machine_sig is not None else decoded_sig
                for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
            ]
            elite_archive = _update_elite_archive(
                elite_archive,
                pos,
                fitness,
                evaluations,
                archive_signatures,
                elite_size,
            )
            if ssr_inline_guidance and ssr_balanced_reinit:
                inline_dimensional_guidance = _build_dimensional_search_guidance(
                    pos=pos,
                    fitness=fitness,
                    elite_archive=elite_archive[: max(1, ssr_elite_k)],
                    gbest=gBest,
                    lb=lb,
                    ub=ub,
                    elite_k=max(1, ssr_elite_k),
                    min_width_ratio=ssr_reduction_min_width,
                    width_scale=ssr_reduction_width_scale,
                )
            if (
                ssr_inline_guidance
                and ssr_inline_reinit_uses_knowledge
                and operation_reference_valid
                and elite_archive
            ):
                inline_priority_knowledge = _build_operation_priority_knowledge_from_archive(
                    elite_archive=elite_archive[: max(1, ssr_elite_k)],
                    decoder=decoder,
                    operation_reference=operation_reference,
                )
                if inline_priority_knowledge is not None:
                    knowledge_signal_ratio = inline_priority_knowledge["signal_ratio"]
                    knowledge_confidence_mean = inline_priority_knowledge["confidence_mean"]
                    matched_operation_count = inline_priority_knowledge["matched_count"]
                    total_operation_count = inline_priority_knowledge["total_count"]
                    if knowledge_signal_ratio < ssr_min_knowledge_signal_ratio:
                        inline_priority_knowledge = None

            best_score_history.append({"iter": t + 1, "score": float(gBestScore)})
            best_decoded_signature_history.append({"iter": t + 1, "signature": gBestDecodedSignature})
            best_structural_signature_history.append(
                {"iter": t + 1, "signature": machine_order_signatures[best_idx]}
            )

            fitness_plateau, improvement_window = detect_fitness_plateau(
                best_score_history,
                stagnation_patience,
                eps_improve,
            )
            if fitness_plateau:
                plateau_check_streak += 1
            else:
                plateau_check_streak = 0
            structural_stagnation, structural_details = detect_structural_stagnation(
                machine_order_signatures=machine_order_signatures,
                ranking_signatures=ranking_signatures,
                decoded_signatures=decoded_signatures,
                gbest_decoded_signature=gBestDecodedSignature,
                population_size=N,
                machine_family_threshold=machine_family_threshold,
                ranking_collapse_threshold=ranking_collapse_threshold,
                structural_distance_threshold=structural_distance_threshold,
                unique_schedule_threshold=unique_schedule_threshold,
                dup_ratio_threshold=dup_ratio_threshold,
                use_machine_order_signature=use_machine_order_signature,
                use_ranking_similarity=use_ranking_similarity,
            )

            stagnation_score = compute_stagnation_score(
                fitness_plateau_score=1.0 if fitness_plateau else 0.0,
                machine_order_collapse_score=1.0 if structural_details["machine_order_collapse"] else 0.0,
                ranking_collapse_score=1.0 if structural_details["ranking_collapse"] else 0.0,
                schedule_collapse_score=1.0 if structural_details["schedule_collapse"] else 0.0,
                gbest_dup_collapse_score=1.0 if structural_details["gbest_dup_collapse"] else 0.0,
                stagnation_score_weights=stagnation_score_weights,
            )
            stagnated = should_activate_ssr(
                fitness_plateau=fitness_plateau,
                structural_stagnation=structural_stagnation,
                stagnation_score=stagnation_score,
                stagnation_mode=stagnation_mode,
                stagnation_score_threshold=stagnation_score_threshold,
            )
            plateau_only_stagnated = (
                ssr_allow_plateau_activation
                and plateau_check_streak >= max(1, int(ssr_min_plateau_checks))
            )
            structural_collapse = bool(
                structural_details["machine_order_collapse"]
                or structural_details["ranking_collapse"]
                or structural_details["schedule_collapse"]
                or structural_details["gbest_dup_collapse"]
            )
            unique_ranking_count = int(len(set(ranking_signatures)))
            exploit_unique_rank_limit = _resolve_count_threshold(
                ssr_exploit_max_unique_rank_ratio,
                N,
            )
            exploit_machine_family_limit = _resolve_count_threshold(
                ssr_exploit_max_machine_family_ratio,
                N,
            )
            rank_population_collapsed = unique_ranking_count <= exploit_unique_rank_limit
            machine_population_collapsed = (
                structural_details["unique_machine_families"] <= exploit_machine_family_limit
            )
            gbest_population_collapsed = (
                structural_details["dup_ratio_to_gbest"] >= ssr_exploit_min_dup_ratio
            )
            strong_structural_collapse = bool(
                structural_collapse
                and (
                    (rank_population_collapsed and machine_population_collapsed)
                    or gbest_population_collapsed
                )
            )
            cooldown_blocked = ssr_cooldown_remaining > 0
            if cooldown_blocked:
                ssr_cooldown_remaining = max(0, ssr_cooldown_remaining - 1)
            remaining_checks = (max_iter - (t + 1)) // max(1, IT)
            tail_blocked = remaining_checks < max(0, int(ssr_skip_last_checks))

            if ssr_adaptive_mode:
                force_escape_after_failed_exploit = bool(
                    ssr_force_escape_after_failed_exploit
                    and last_ssr_failed_mode == "exploit"
                    and fitness_plateau
                    and plateau_only_stagnated
                )
                exploit_ready = bool(
                    stagnated
                    and fitness_plateau
                    and strong_structural_collapse
                    and not force_escape_after_failed_exploit
                )
                escape_ready = bool(
                    fitness_plateau
                    and plateau_only_stagnated
                    and (not strong_structural_collapse or force_escape_after_failed_exploit)
                )
                if ssr_no_improve_streak >= max(1, int(ssr_escape_after_failed_activations)):
                    escape_ready = bool(fitness_plateau and plateau_only_stagnated)
                    exploit_ready = False

                if tail_blocked:
                    stagnated = False
                    ssr_skip_reason = "tail"
                elif cooldown_blocked:
                    stagnated = False
                    ssr_skip_reason = "cooldown"
                elif exploit_ready:
                    stagnated = True
                    ssr_search_mode = "exploit"
                elif escape_ready:
                    stagnated = True
                    ssr_search_mode = "escape"
                else:
                    stagnated = False
            else:
                stagnated = bool(stagnated or plateau_only_stagnated)
                if tail_blocked:
                    stagnated = False
                    ssr_skip_reason = "tail"
                elif cooldown_blocked:
                    stagnated = False
                    ssr_skip_reason = "cooldown"
                elif structural_stagnation:
                    ssr_search_mode = "exploit"
                elif plateau_only_stagnated:
                    ssr_search_mode = "escape"

            activation_parts = []
            if fitness_plateau:
                activation_parts.append("fitness_plateau")
            if structural_details["machine_order_collapse"]:
                activation_parts.append("machine_order_collapse")
            if structural_details["ranking_collapse"]:
                activation_parts.append("ranking_collapse")
            if structural_details["schedule_collapse"]:
                activation_parts.append("schedule_collapse")
            if structural_details["gbest_dup_collapse"]:
                activation_parts.append("gbest_dup_collapse")
            if plateau_only_stagnated and not structural_stagnation:
                activation_parts.append("plateau_only")
            if ssr_search_mode != "none":
                activation_parts.append(f"ssr_{ssr_search_mode}")
            if ssr_skip_reason:
                activation_parts.append(f"skip_{ssr_skip_reason}")

            stagnation_details.update(structural_details)
            stagnation_details.update(
                {
                    "improvement_window": improvement_window,
                    "fitness_plateau": bool(fitness_plateau),
                    "plateau_check_streak": int(plateau_check_streak),
                    "structural_stagnation": bool(structural_stagnation),
                    "stagnation_score": float(stagnation_score),
                    "activation_reason": "+".join(activation_parts) if stagnated else "not_activated",
                    "ssr_search_mode": ssr_search_mode,
                    "ssr_skip_reason": ssr_skip_reason,
                    "ssr_no_improve_streak": int(ssr_no_improve_streak),
                    "ssr_cooldown_remaining": int(ssr_cooldown_remaining),
                    "strong_structural_collapse": bool(strong_structural_collapse),
                    "last_ssr_failed_mode": last_ssr_failed_mode,
                }
            )

            feasible_count_before = int(
                sum(
                    _is_feasible_metrics(
                        m,
                        missing_feasibility_is_feasible=missing_feasibility_is_feasible,
                    )
                    for m in metrics_list
                )
            ) if decoder_available else 0
            infeasible_count_before = N - feasible_count_before if decoder_available else 0

            if stagnated and decoder_available:
                ssr_best_before_activation = float(gBestScore)
                effective_reduction_width_scale = ssr_reduction_width_scale
                effective_reduced_gbest_pull = ssr_reduced_gbest_pull
                effective_explore_dim_ratio = ssr_explore_dim_ratio
                effective_gaussian_scale = max(
                    ssr_knowledge_min_noise_scale,
                    0.25 * ssr_knowledge_noise_scale,
                )
                if ssr_search_mode == "escape":
                    effective_reduction_width_scale *= ssr_escape_reduction_width_multiplier
                    effective_reduced_gbest_pull = ssr_escape_gbest_pull
                    effective_explore_dim_ratio = ssr_escape_dim_ratio
                    effective_gaussian_scale *= ssr_escape_noise_multiplier

                protected_elite_indices = set(
                    _select_protected_elite_indices(
                        fitness=fitness,
                        archive_signatures=archive_signatures,
                        elite_size=elite_size,
                    )
                )
                active_restart_ratio = partial_restart_ratio
                if ssr_search_mode == "exploit":
                    active_restart_ratio = min(partial_restart_ratio, ssr_exploit_restart_ratio)
                restart_budget = int(math.ceil(active_restart_ratio * N))
                replace_indices, ssr_infeasible_replaced, ssr_worst_feasible_replaced = (
                    _select_ssr_replacement_indices(
                        fitness=fitness,
                        metrics_list=metrics_list,
                        protected_indices=protected_elite_indices,
                        restart_budget=restart_budget,
                        decoder_available=decoder_available,
                        missing_feasibility_is_feasible=missing_feasibility_is_feasible,
                    )
                )

                priority_knowledge = None
                if operation_reference_valid:
                    priority_knowledge = _build_operation_priority_knowledge_from_archive(
                        elite_archive=elite_archive[: max(1, ssr_elite_k)],
                        decoder=decoder,
                        operation_reference=operation_reference,
                    )
                    if priority_knowledge is not None:
                        knowledge_signal_ratio = priority_knowledge["signal_ratio"]
                        knowledge_confidence_mean = priority_knowledge["confidence_mean"]
                        matched_operation_count = priority_knowledge["matched_count"]
                        total_operation_count = priority_knowledge["total_count"]
                    if (
                        priority_knowledge is not None
                        and knowledge_signal_ratio < ssr_min_knowledge_signal_ratio
                    ):
                        priority_knowledge = None

                if not operation_reference_valid:
                    stagnation_details["knowledge_warning"] = operation_reference_error
                elif not elite_archive:
                    stagnation_details["knowledge_warning"] = "no_historical_elite_for_knowledge"
                elif priority_knowledge is None:
                    if knowledge_signal_ratio < ssr_min_knowledge_signal_ratio:
                        stagnation_details["knowledge_warning"] = (
                            f"low_knowledge_signal:{knowledge_signal_ratio:.3f}"
                        )
                    else:
                        stagnation_details["knowledge_warning"] = "knowledge_unavailable"

                dimensional_guidance = None
                if ssr_balanced_reinit:
                    dimensional_guidance = _build_dimensional_search_guidance(
                        pos=pos,
                        fitness=fitness,
                        elite_archive=elite_archive[: max(1, ssr_elite_k)],
                        gbest=gBest,
                        lb=lb,
                        ub=ub,
                        elite_k=max(1, ssr_elite_k),
                        min_width_ratio=ssr_reduction_min_width,
                        width_scale=effective_reduction_width_scale,
                    )

                for replace_order, idx in enumerate(replace_indices):
                    if max_FEs is not None and fe_counter >= max_FEs:
                        break
                    best_candidate_pos = None
                    best_candidate_evaluation = None
                    best_candidate_fit = float("inf")
                    best_candidate_source = None
                    best_by_source = {}
                    preferred_source = None
                    diversity_quota = (
                        ssr_escape_diversity_quota
                        if ssr_search_mode == "escape"
                        else ssr_exploit_diversity_quota
                    )
                    if (
                        ssr_balanced_reinit
                        and dimensional_guidance is not None
                        and diversity_replacement_count < max(0, int(diversity_quota))
                    ):
                        preferred_source = "diversity"
                    elif ssr_balanced_reinit and ssr_force_mode_quota:
                        source_cycle = replace_order % 3
                        if source_cycle == 0 and priority_knowledge is not None:
                            preferred_source = "knowledge"
                        elif source_cycle == 1 and dimensional_guidance is not None:
                            preferred_source = "reduced"
                        elif dimensional_guidance is not None:
                            preferred_source = "diversity"

                    for trial_idx in range(max(1, int(ssr_candidate_trials))):
                        if max_FEs is not None and fe_counter >= max_FEs:
                            break
                        candidate_source = None
                        candidate_pos = None
                        if preferred_source is not None:
                            source_order = [preferred_source]
                        elif ssr_search_mode == "escape":
                            source_order = ["diversity", "reduced", "knowledge"]
                            source_order = source_order[trial_idx % 3 :] + source_order[: trial_idx % 3]
                        else:
                            source_order = ["knowledge", "reduced", "diversity"]
                            source_order = source_order[trial_idx % 3 :] + source_order[: trial_idx % 3]
                        if ssr_random_fallback:
                            source_order.append("random")

                        for source_name in source_order:
                            if source_name == "knowledge" and priority_knowledge is not None:
                                candidate_pos = _generate_random_key_agent_from_priority_knowledge(
                                    priority_knowledge=priority_knowledge,
                                    lb=lb,
                                    ub=ub,
                                    base_noise_scale=ssr_knowledge_noise_scale,
                                    uniform_mix=ssr_knowledge_uniform_mix,
                                    min_noise_scale=ssr_knowledge_min_noise_scale,
                                    max_confidence=ssr_knowledge_max_confidence,
                                )
                                candidate_source = "knowledge"
                            elif source_name == "reduced" and dimensional_guidance is not None:
                                candidate_pos = _generate_reduced_space_candidate(
                                    guidance=dimensional_guidance,
                                    lb=lb,
                                    ub=ub,
                                    gbest_pull=effective_reduced_gbest_pull,
                                    uncertain_uniform_ratio=ssr_uncertain_uniform_ratio,
                                )
                                candidate_source = "reduced"
                            elif source_name == "diversity" and dimensional_guidance is not None:
                                candidate_pos = _generate_diversity_exploration_candidate(
                                    guidance=dimensional_guidance,
                                    lb=lb,
                                    ub=ub,
                                    dim_ratio=effective_explore_dim_ratio,
                                    opposition_ratio=ssr_explore_opposition_ratio,
                                    gaussian_scale=effective_gaussian_scale,
                                )
                                candidate_source = "diversity"
                            elif source_name == "random" and ssr_random_fallback:
                                candidate_pos = _plain_random_position(lb, ub)
                                candidate_source = "random"

                            if candidate_pos is not None:
                                break

                        if candidate_pos is None:
                            continue

                        candidate_evaluation = _evaluate_candidate(candidate_pos, fobj=fobj, decoder=decoder)
                        candidate_fit = candidate_evaluation["fitness"]
                        fe_counter += 1
                        ssr_candidate_attempt_count += 1
                        if candidate_source == "knowledge":
                            knowledge_candidate_count += 1
                        elif candidate_source == "reduced":
                            reduced_space_candidate_count += 1
                        elif candidate_source == "diversity":
                            diversity_candidate_count += 1
                        source_best = best_by_source.get(candidate_source)
                        if source_best is None or candidate_fit < source_best[0]:
                            best_by_source[candidate_source] = (
                                candidate_fit,
                                candidate_pos,
                                candidate_evaluation,
                            )
                        if candidate_fit < best_candidate_fit:
                            best_candidate_pos = candidate_pos
                            best_candidate_evaluation = candidate_evaluation
                            best_candidate_fit = candidate_fit
                            best_candidate_source = candidate_source

                    if best_candidate_evaluation is None:
                        continue

                    escape_accept_margin = max(0.0, float(ssr_escape_accept_margin_ratio))
                    if ssr_search_mode == "escape" and "diversity" in best_by_source:
                        diversity_fit, diversity_pos, diversity_evaluation = best_by_source["diversity"]
                        if diversity_fit <= fitness[idx] * (1.0 + escape_accept_margin):
                            best_candidate_fit = diversity_fit
                            best_candidate_pos = diversity_pos
                            best_candidate_evaluation = diversity_evaluation
                            best_candidate_source = "diversity"
                    allow_escape_diversity_accept = (
                        ssr_search_mode == "escape"
                        and best_candidate_source == "diversity"
                        and best_candidate_fit <= fitness[idx] * (1.0 + escape_accept_margin)
                    )
                    if (
                        ssr_accept_only_improvement
                        and best_candidate_fit >= fitness[idx]
                        and not allow_escape_diversity_accept
                    ):
                        ssr_rejected_candidate_count += 1
                        continue
                    if (
                        ssr_commit_requires_gbest_improvement
                        and best_candidate_fit >= gBestScore - eps_improve
                    ):
                        ssr_rejected_candidate_count += 1
                        continue

                    pos[idx] = best_candidate_pos
                    evaluation = best_candidate_evaluation
                    _build_evaluation_metadata(evaluation, pos[idx])
                    if best_candidate_source == "knowledge":
                        knowledge_replacement_count += 1
                    elif best_candidate_source == "reduced":
                        reduced_space_replacement_count += 1
                    elif best_candidate_source == "diversity":
                        diversity_replacement_count += 1
                    else:
                        ssr_fallback_random_count += 1

                    energies[idx] = initial_energy
                    fitness[idx] = evaluation["fitness"]
                    evaluations[idx] = evaluation
                    schedule_dfs[idx] = evaluation["schedule_df"]
                    metrics_list[idx] = evaluation["metrics"]
                    decoded_signatures[idx] = evaluation["decoded_signature"]
                    machine_order_signatures[idx] = evaluation["machine_order_signature"]
                    ranking_signatures[idx] = evaluation["ranking_signature"]
                    ssr_replacement_count += 1

                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < gBestScore:
                    gBestScore = float(fitness[best_idx])
                    gBest = pos[best_idx].copy()
                    gBestDecodedSignature = decoded_signatures[best_idx]

                archive_signatures = [
                    machine_sig if machine_sig is not None else decoded_sig
                    for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
                ]
                elite_archive = _update_elite_archive(
                    elite_archive,
                    pos,
                    fitness,
                    evaluations,
                    archive_signatures,
                    elite_size,
                )
                ssr_triggered = True
                if gBestScore < ssr_best_before_activation - eps_improve:
                    ssr_no_improve_streak = 0
                    ssr_failure_reference_best = float(gBestScore)
                    last_ssr_failed_mode = None
                else:
                    ssr_no_improve_streak += 1
                    ssr_failure_reference_best = min(
                        ssr_failure_reference_best,
                        ssr_best_before_activation,
                    )
                    last_ssr_failed_mode = ssr_search_mode
                ssr_cooldown_remaining = max(
                    ssr_cooldown_remaining,
                    max(0, int(ssr_cooldown_checks)),
                )
                stagnation_details["ssr_no_improve_streak"] = int(ssr_no_improve_streak)
                stagnation_details["ssr_cooldown_remaining"] = int(ssr_cooldown_remaining)
                stagnation_details["last_ssr_failed_mode"] = last_ssr_failed_mode

            feasible_count_after = int(
                sum(
                    _is_feasible_metrics(
                        m,
                        missing_feasibility_is_feasible=missing_feasibility_is_feasible,
                    )
                    for m in metrics_list
                )
            ) if decoder_available else 0
            infeasible_count_after = N - feasible_count_after if decoder_available else 0

            cached_check_metrics = {
                "unique_ranking_count": int(len(set(ranking_signatures))),
                "unique_schedule_count": int(len(set(sig for sig in decoded_signatures if sig is not None))),
                "unique_machine_family_count": int(len(set(sig for sig in machine_order_signatures if sig is not None))),
                "avg_machine_order_distance": float(structural_details["avg_machine_order_distance"]),
                "avg_ranking_distance": float(structural_details["avg_ranking_distance"]),
                "dup_ratio_to_gbest": float(
                    sum(sig == gBestDecodedSignature for sig in decoded_signatures) / max(1, N)
                ),
                "feasible_count": feasible_count_after,
                "knowledge_signal_ratio": float(knowledge_signal_ratio),
                "knowledge_confidence_mean": float(knowledge_confidence_mean),
            }
        else:
            feasible_count_before = cached_check_metrics["feasible_count"]
            infeasible_count_before = N - feasible_count_before if decoder_available else 0
            feasible_count_after = feasible_count_before
            infeasible_count_after = infeasible_count_before

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < gBestScore:
            gBestScore = float(fitness[best_idx])
            gBest = pos[best_idx].copy()
            gBestDecodedSignature = decoded_signatures[best_idx]

        cg_curve.append(gBestScore)
        avg_curve.append(float(np.mean(fitness)))

        iter_time = time.perf_counter() - iter_start
        total_time = time.perf_counter() - start_total
        feasible_count = cached_check_metrics["feasible_count"]
        infeasible_count = N - feasible_count if decoder_available else 0

        log_entry = {
            "iter": t + 1,
            "best_fitness": float(gBestScore),
            "ssr_check_iter": bool(check_this_iter),
            "improvement_window": stagnation_details.get("improvement_window"),
            "fitness_plateau": bool(stagnation_details.get("fitness_plateau", False)),
            "plateau_check_streak": int(stagnation_details.get("plateau_check_streak", plateau_check_streak)),
            "unique_ranking_count": int(cached_check_metrics["unique_ranking_count"]),
            "unique_schedule_count": int(cached_check_metrics["unique_schedule_count"]),
            "unique_machine_order_families": int(cached_check_metrics["unique_machine_family_count"]),
            "avg_pairwise_machine_order_distance": float(cached_check_metrics["avg_machine_order_distance"]),
            "avg_pairwise_ranking_distance": float(cached_check_metrics["avg_ranking_distance"]),
            "dup_ratio_to_gbest": float(cached_check_metrics["dup_ratio_to_gbest"]),
            "feasible_count": int(feasible_count),
            "infeasible_count": int(infeasible_count),
            "machine_order_collapse": bool(stagnation_details.get("machine_order_collapse", False)),
            "ranking_collapse": bool(stagnation_details.get("ranking_collapse", False)),
            "schedule_collapse": bool(stagnation_details.get("schedule_collapse", False)),
            "gbest_dup_collapse": bool(stagnation_details.get("gbest_dup_collapse", False)),
            "structural_stagnation": bool(stagnation_details.get("structural_stagnation", False)),
            "stagnation_score": float(stagnation_details.get("stagnation_score", 0.0)),
            "ssr_activation_reason": stagnation_details.get("activation_reason", "skipped_until_it"),
            "ssr_search_mode": stagnation_details.get("ssr_search_mode", ssr_search_mode),
            "ssr_skip_reason": stagnation_details.get("ssr_skip_reason", ssr_skip_reason),
            "ssr_no_improve_streak": int(stagnation_details.get("ssr_no_improve_streak", ssr_no_improve_streak)),
            "ssr_cooldown_remaining": int(stagnation_details.get("ssr_cooldown_remaining", ssr_cooldown_remaining)),
            "strong_structural_collapse": bool(stagnation_details.get("strong_structural_collapse", False)),
            "last_ssr_failed_mode": stagnation_details.get("last_ssr_failed_mode", last_ssr_failed_mode),
            "reinitialized": int(energy_reinit_count + deterioration_reinit_count + ssr_replacement_count),
            "energy_reinit": int(energy_reinit_count),
            "deterioration_reinit": int(deterioration_reinit_count),
            "ssr_active": bool(ssr_triggered),
            "ssr_replacement_count": int(ssr_replacement_count),
            "ssr_candidate_attempt_count": int(ssr_candidate_attempt_count),
            "ssr_rejected_candidate_count": int(ssr_rejected_candidate_count),
            "knowledge_replacement_count": int(knowledge_replacement_count),
            "reduced_space_replacement_count": int(reduced_space_replacement_count),
            "diversity_replacement_count": int(diversity_replacement_count),
            "knowledge_candidate_count": int(knowledge_candidate_count),
            "reduced_space_candidate_count": int(reduced_space_candidate_count),
            "diversity_candidate_count": int(diversity_candidate_count),
            "inline_reduced_dim_count": int(inline_reduced_dim_count),
            "inline_explore_dim_count": int(inline_explore_dim_count),
            "inline_knowledge_reinit_count": int(inline_knowledge_reinit_count),
            "ssr_infeasible_replaced": int(ssr_infeasible_replaced),
            "ssr_worst_feasible_replaced": int(ssr_worst_feasible_replaced),
            "knowledge_signal_ratio": float(knowledge_signal_ratio),
            "knowledge_confidence_mean": float(knowledge_confidence_mean),
            "matched_operation_count": int(matched_operation_count),
            "total_operation_count": int(total_operation_count),
            "used_knowledge_reinit": bool(knowledge_replacement_count > 0),
            "ssr_fallback_random_count": int(ssr_fallback_random_count),
            "feasible_count_before": int(feasible_count_before),
            "feasible_count_after": int(feasible_count_after),
            "infeasible_count_before": int(infeasible_count_before),
            "infeasible_count_after": int(infeasible_count_after),
            "operation_reference_valid": bool(operation_reference_valid),
            "operation_reference_error": operation_reference_error,
            "operation_reference_len": int(len(operation_reference)),
            "knowledge_warning": stagnation_details.get("knowledge_warning"),
            "fe_counter": int(fe_counter),
            "iter_time_sec": float(iter_time),
            "total_time_sec": float(total_time),
        }
        diagnostics.append(log_entry)

        if verbose:
            print(
                f"Iterasi {t+1}/{max_iter} | "
                f"Best: {gBestScore:.2f} | "
                f"Feasible: {feasible_count} | "
                f"Infeasible: {infeasible_count} | "
                f"UniqueRank: {cached_check_metrics['unique_ranking_count']} | "
                f"MachineFam: {cached_check_metrics['unique_machine_family_count']} | "
                f"SSR: {ssr_triggered} | "
                f"SSRPolicy: {stagnation_details.get('ssr_search_mode', ssr_search_mode)} | "
                f"SSRReplace: {ssr_replacement_count} | "
                f"SSRReject: {ssr_rejected_candidate_count} | "
                f"SSRMode K/R/D: {knowledge_replacement_count}/{reduced_space_replacement_count}/{diversity_replacement_count} | "
                f"SSRCand K/R/D: {knowledge_candidate_count}/{reduced_space_candidate_count}/{diversity_candidate_count} | "
                f"Inline R/D/K: {inline_reduced_dim_count}/{inline_explore_dim_count}/{inline_knowledge_reinit_count} | "
                f"EReinit: {energy_reinit_count} | "
                f"DReinit: {deterioration_reinit_count} | "
                f"KSignal: {knowledge_signal_ratio:.2f} | "
                f"KConf: {knowledge_confidence_mean:.2f} | "
                f"FEs: {fe_counter} | "
                f"Time: {iter_time:.4f}s"
            )

    outputs = (
        gBestScore,
        gBest,
        np.array(cg_curve),
        np.array(avg_curve),
    )
    if return_diagnostics:
        extra = {
            "logs": diagnostics,
            "elite_archive": elite_archive,
            "best_score_history": best_score_history,
            "best_decoded_signature_history": best_decoded_signature_history,
            "best_structural_signature_history": best_structural_signature_history,
        }
        return outputs + (extra,)
    return outputs
