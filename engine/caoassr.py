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


def _compute_continuous_signature(x, decimals=8):
    return tuple(np.round(x, decimals=decimals).tolist())


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


def _evaluate_candidate(x, fobj=None, decoder=None):
    if decoder is not None:
        schedule_df, metrics = decoder.decode_from_continuous(x)
        metrics = dict(metrics) if metrics is not None else {}
        if "is_feasible" not in metrics:
            metrics["is_feasible"] = metrics.get("infeasible_reason") in (None, "", False)
        return {
            "fitness": float(metrics["total_tardiness"]),
            "schedule_df": schedule_df,
            "metrics": metrics,
            "decoded_signature": _schedule_signature_from_dataframe(schedule_df),
            "machine_order_signature": compute_machine_order_signature(schedule_df),
            "ranking_signature": _compute_ranking_signature(x),
            "continuous_signature": _compute_continuous_signature(x),
        }

    if fobj is None:
        raise ValueError("Either `fobj` or `decoder` must be provided.")

    return {
        "fitness": float(fobj(x)),
        "schedule_df": None,
        "metrics": None,
        "decoded_signature": None,
        "machine_order_signature": None,
        "ranking_signature": _compute_ranking_signature(x),
        "continuous_signature": _compute_continuous_signature(x),
    }


def _update_elite_archive(elite_archive, pos, fitness, decoded_signatures, archive_signatures, elite_size):
    merged = {}
    for item in elite_archive:
        key = item.get("archive_signature", item.get("decoded_signature"))
        best_existing = merged.get(key)
        if best_existing is None or item["fitness"] < best_existing["fitness"]:
            merged[key] = {
                "position": item["position"].copy(),
                "fitness": float(item["fitness"]),
                "decoded_signature": item.get("decoded_signature"),
                "archive_signature": key,
            }

    for idx in np.argsort(fitness):
        key = archive_signatures[idx]
        candidate = {
            "position": pos[idx].copy(),
            "fitness": float(fitness[idx]),
            "decoded_signature": decoded_signatures[idx],
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


def _compute_dimensional_statistics(pos):
    mean = np.mean(pos, axis=0)
    std = np.std(pos, axis=0)
    return mean, std


def _safe_pearson_corr(values, fitness):
    values = np.asarray(values, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    if values.size != fitness.size or values.size < 2:
        return 0.0
    if np.allclose(values, values[0]) or np.allclose(fitness, fitness[0]):
        return 0.0
    corr = np.corrcoef(values, fitness)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _classify_correlation(corr, threshold=0.6):
    if corr >= threshold:
        return 1
    if corr <= -threshold:
        return -1
    return 0


def _build_historical_samples_from_archive(elite_archive, fallback_pos, fallback_fitness):
    if elite_archive:
        sample_positions = np.array([item["position"] for item in elite_archive], dtype=float)
        sample_fitness = np.array([float(item["fitness"]) for item in elite_archive], dtype=float)
        if len(sample_positions) >= 2:
            return sample_positions, sample_fitness
    return fallback_pos.copy(), fallback_fitness.copy()


def _apply_search_space_reduction(
    low_dyn,
    up_dyn,
    pos,
    sample_positions,
    sample_fitness,
    convergence_mask,
    corr_threshold=0.6,
):
    new_low = low_dyn.copy()
    new_up = up_dyn.copy()
    dim = pos.shape[1]
    mean_current, _ = _compute_dimensional_statistics(pos)
    correlations = np.zeros(dim, dtype=float)
    correlation_flags = np.zeros(dim, dtype=int)
    reduced_dims = []

    for j in range(dim):
        if not convergence_mask[j]:
            continue

        corr = _safe_pearson_corr(sample_positions[:, j], sample_fitness)
        corr_flag = _classify_correlation(corr, threshold=corr_threshold)
        correlations[j] = corr
        correlation_flags[j] = corr_flag

        values = pos[:, j]
        max_val = float(np.max(values))
        min_val = float(np.min(values))
        # Use the best point from the historical sample to mimic Huang's "best solutions" guidance.
        best_hist_idx = int(np.argmin(sample_fitness))
        best_val = float(sample_positions[best_hist_idx, j])
        span = max(float(up_dyn[j] - low_dyn[j]), 1e-12)
        denom = max(max_val - min_val, 1e-12)
        dis = (max_val - best_val) / denom
        dis = float(np.clip(dis, 0.0, 1.0))
        ri = span * max(dis, 1.0 - dis)

        if corr_flag > 0:
            candidate_up = new_up[j] - ri
            if candidate_up < new_low[j]:
                candidate_up = max(mean_current[j], new_low[j])
            new_up[j] = float(np.clip(candidate_up, new_low[j], up_dyn[j]))
        elif corr_flag < 0:
            candidate_low = new_low[j] + ri
            if candidate_low > new_up[j]:
                candidate_low = min(mean_current[j], new_up[j])
            new_low[j] = float(np.clip(candidate_low, low_dyn[j], new_up[j]))
        else:
            candidate_up = new_up[j] - ri
            candidate_low = new_low[j] + ri
            if candidate_up < candidate_low:
                midpoint = float(mean_current[j])
                candidate_low = min(midpoint, new_up[j])
                candidate_up = max(midpoint, new_low[j])
            new_low[j] = float(np.clip(candidate_low, low_dyn[j], new_up[j]))
            new_up[j] = float(np.clip(candidate_up, new_low[j], up_dyn[j]))

        reduced_dims.append(int(j))

    return new_low, new_up, correlations, correlation_flags, reduced_dims


def _select_feasible_elite_indices(
    fitness,
    metrics_list,
    top_k,
    missing_feasibility_is_feasible=False,
):
    ordered = [int(idx) for idx in np.argsort(fitness)]
    feasible = [
        idx for idx in ordered
        if _is_feasible_metrics(
            metrics_list[idx],
            missing_feasibility_is_feasible=missing_feasibility_is_feasible,
        )
    ]
    return feasible[: max(1, int(top_k))]


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


def _build_priority_knowledge_from_feasible_elites(
    elite_indices,
    schedule_dfs,
    fitness,
    operation_reference,
):
    if not elite_indices or not operation_reference:
        return None, 0.0, 0, len(operation_reference)

    best_fit = min(float(fitness[idx]) for idx in elite_indices)
    weighted_scores = {operation: 0.0 for operation in operation_reference}
    total_weights = {operation: 0.0 for operation in operation_reference}

    for idx in elite_indices:
        op_priority = _extract_operation_priority_from_schedule(schedule_dfs[idx])
        if not op_priority:
            continue
        weight = 1.0 / (1.0 + max(0.0, float(fitness[idx]) - best_fit))
        for operation, priority_score in op_priority.items():
            if operation not in weighted_scores:
                continue
            weighted_scores[operation] += weight * float(priority_score)
            total_weights[operation] += weight

    fallback = np.linspace(0.0, 1.0, num=len(operation_reference), dtype=float)
    knowledge = np.empty(len(operation_reference), dtype=float)
    matched_count = 0
    for pos_idx, operation in enumerate(operation_reference):
        if total_weights[operation] > 0.0:
            knowledge[pos_idx] = weighted_scores[operation] / total_weights[operation]
            matched_count += 1
        else:
            knowledge[pos_idx] = fallback[pos_idx]

    if matched_count == 0:
        return None, 0.0, 0, len(operation_reference)

    signal_ratio = matched_count / max(1, len(operation_reference))
    return knowledge, float(signal_ratio), int(matched_count), int(len(operation_reference))


def _build_priority_knowledge_from_archive(
    elite_archive,
    decoder,
    operation_reference,
):
    if not elite_archive or decoder is None or not operation_reference:
        return None, 0.0, 0, len(operation_reference)

    best_fit = min(float(item["fitness"]) for item in elite_archive)
    weighted_scores = {operation: 0.0 for operation in operation_reference}
    total_weights = {operation: 0.0 for operation in operation_reference}

    for item in elite_archive:
        schedule_df, metrics = decoder.decode_from_continuous(item["position"])
        is_feasible = True
        if metrics is not None and "is_feasible" in metrics:
            is_feasible = bool(metrics["is_feasible"])
        if not is_feasible:
            continue

        op_priority = _extract_operation_priority_from_schedule(schedule_df)
        if not op_priority:
            continue

        weight = 1.0 / (1.0 + max(0.0, float(item["fitness"]) - best_fit))
        for operation, priority_score in op_priority.items():
            if operation not in weighted_scores:
                continue
            weighted_scores[operation] += weight * float(priority_score)
            total_weights[operation] += weight

    fallback = np.linspace(0.0, 1.0, num=len(operation_reference), dtype=float)
    knowledge = np.empty(len(operation_reference), dtype=float)
    matched_count = 0
    for pos_idx, operation in enumerate(operation_reference):
        if total_weights[operation] > 0.0:
            knowledge[pos_idx] = weighted_scores[operation] / total_weights[operation]
            matched_count += 1
        else:
            knowledge[pos_idx] = fallback[pos_idx]

    if matched_count == 0:
        return None, 0.0, 0, len(operation_reference)

    signal_ratio = matched_count / max(1, len(operation_reference))
    return knowledge, float(signal_ratio), int(matched_count), int(len(operation_reference))


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


def _generate_random_key_agent_from_knowledge(
    priority_knowledge,
    low_dyn,
    up_dyn,
    noise_scale=0.08,
    uniform_mix=0.10,
):
    if priority_knowledge is None:
        return _plain_random_position(low_dyn, up_dyn)

    noisy_scores = priority_knowledge.copy()
    if noise_scale > 0.0:
        noisy_scores = noisy_scores + np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=noisy_scores.size,
        )

    if uniform_mix > 0.0:
        mix_mask = np.random.rand(noisy_scores.size) < uniform_mix
        if np.any(mix_mask):
            noisy_scores[mix_mask] = np.random.rand(int(np.sum(mix_mask)))

    order = np.argsort(noisy_scores, kind="mergesort")
    normalized = np.empty_like(noisy_scores, dtype=float)
    if noisy_scores.size == 1:
        normalized[0] = 0.5
    else:
        normalized[order] = np.linspace(0.0, 1.0, num=noisy_scores.size, dtype=float)
    return np.clip(low_dyn + (up_dyn - low_dyn) * normalized, low_dyn, up_dyn)


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
        schedule_df, metrics = decoder.decode_from_continuous(item["position"])
        if metrics is not None and "is_feasible" in metrics and not bool(metrics["is_feasible"]):
            continue

        op_priority = _extract_operation_priority_from_schedule(schedule_df)
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
    low_dyn,
    up_dyn,
    base_noise_scale=0.08,
    uniform_mix=0.10,
    min_noise_scale=0.01,
    max_confidence=0.85,
):
    if priority_knowledge is None:
        return _plain_random_position(low_dyn, up_dyn)

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
    return np.clip(low_dyn + (up_dyn - low_dyn) * normalized, low_dyn, up_dyn)


def _deterministic_probability_gate(probability, *values):
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True

    seed = 0
    for idx, value in enumerate(values):
        seed += (idx + 1) * 2654435761 * (int(value) + 1)
    return ((seed % 10000) / 10000.0) < probability


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
    ssr_use_knowledge_for_default_reinit=True,
    ssr_energy_knowledge_prob=0.15,
    ssr_deterioration_knowledge_prob=0.0,
    ssr_candidate_trials=3,
    ssr_accept_only_improvement=True,
    ssr_random_fallback=False,
    ssr_mode="hybrid",
    dim_convergence_threshold=1e-3,
    ssr_correlation_threshold=0.6,
    missing_feasibility_is_feasible=False,
    return_diagnostics=False,
    verbose=True,
):
    if fobj is None and decoder is None:
        raise ValueError("Either `fobj` or `decoder` must be provided to CAOA_SSR.")
    if ssr_mode not in {"continuous", "knowledge", "hybrid"}:
        raise ValueError("`ssr_mode` must be one of: 'continuous', 'knowledge', or 'hybrid'.")

    lb, ub = _normalize_bounds(lb, ub, dim)
    if stagnation_window is None:
        stagnation_window = K
    if stagnation_patience is None:
        stagnation_patience = max(1, int(math.ceil(stagnation_window / max(1, IT))))

    if initial_pos is not None:
        pos = np.clip(np.array(initial_pos, dtype=float).copy(), lb, ub)
    else:
        pos = lb + (ub - lb) * np.random.rand(N, dim)
    low_dyn = lb.copy()
    up_dyn = ub.copy()

    energies = np.full(N, initial_energy, dtype=float)
    fitness = np.zeros(N, dtype=float)
    schedule_dfs = [None] * N
    metrics_list = [None] * N
    decoded_signatures = [None] * N
    machine_order_signatures = [None] * N
    ranking_signatures = [None] * N
    continuous_signatures = [None] * N

    fe_counter = 0
    for i in range(N):
        evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
        fitness[i] = evaluation["fitness"]
        schedule_dfs[i] = evaluation["schedule_df"]
        metrics_list[i] = evaluation["metrics"]
        decoded_signatures[i] = evaluation["decoded_signature"]
        machine_order_signatures[i] = evaluation["machine_order_signature"]
        ranking_signatures[i] = evaluation["ranking_signature"]
        continuous_signatures[i] = evaluation["continuous_signature"]
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
    priority_knowledge_memory = None

    decoder_available = decoder is not None
    operation_reference = list(getattr(decoder, "L_ref", [])) if decoder_available else []
    operation_reference_valid, operation_reference_error = (
        _validate_operation_reference(operation_reference, dim)
        if decoder_available
        else (False, "decoder_unavailable")
    )
    ssr_elite_k = elite_size if ssr_elite_k is None else int(ssr_elite_k)

    cached_check_metrics = {
        "unique_continuous_count": int(len(set(continuous_signatures))),
        "unique_ranking_count": int(len(set(ranking_signatures))),
        "unique_schedule_count": int(len(set(sig for sig in decoded_signatures if sig is not None))),
        "unique_machine_family_count": 0,
        "avg_machine_order_distance": 0.0,
        "avg_ranking_distance": 0.0,
        "dup_ratio_to_gbest": float(
            sum(sig == gBestDecodedSignature for sig in decoded_signatures) / max(1, N)
        ),
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
        knowledge_energy_reinit_count = 0
        knowledge_deterioration_reinit_count = 0
        ssr_fallback_random_count = 0
        ssr_infeasible_replaced = 0
        ssr_worst_feasible_replaced = 0
        knowledge_signal_ratio = 0.0
        knowledge_confidence_mean = 0.0
        matched_operation_count = 0
        total_operation_count = len(operation_reference)
        ssr_triggered = False
        check_this_iter = (t + 1) % IT == 0
        reduced_dims = []
        converged_dims = []
        convergence_mask = np.zeros(dim, dtype=bool)
        dimension_correlations = np.zeros(dim, dtype=float)
        correlation_flags = np.zeros(dim, dtype=int)

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
            new_pos = np.clip(new_pos, low_dyn, up_dyn)

            evaluation = _evaluate_candidate(new_pos, fobj=fobj, decoder=decoder)
            new_fit = evaluation["fitness"]
            fe_counter += 1
            was_reinitialized = False

            if abs(new_fit - old_fit) > delta and new_fit > old_fit:
                if preserve_default_random_reinit and (max_FEs is None or fe_counter < max_FEs):
                    use_knowledge_default_reinit = (
                        ssr_use_knowledge_for_default_reinit
                        and priority_knowledge_memory is not None
                        and _deterministic_probability_gate(
                            ssr_deterioration_knowledge_prob,
                            t,
                            i,
                            deterioration_reinit_count,
                        )
                    )
                    if use_knowledge_default_reinit:
                        new_pos = _generate_random_key_agent_from_priority_knowledge(
                            priority_knowledge=priority_knowledge_memory,
                            low_dyn=low_dyn,
                            up_dyn=up_dyn,
                            base_noise_scale=ssr_knowledge_noise_scale,
                            uniform_mix=ssr_knowledge_uniform_mix,
                            min_noise_scale=ssr_knowledge_min_noise_scale,
                            max_confidence=ssr_knowledge_max_confidence,
                        )
                        knowledge_deterioration_reinit_count += 1
                    else:
                        new_pos = _plain_random_position(low_dyn, up_dyn)
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
                        "continuous_signature": continuous_signatures[i],
                    }

            pos[i] = new_pos
            fitness[i] = new_fit
            schedule_dfs[i] = evaluation["schedule_df"]
            metrics_list[i] = evaluation["metrics"]
            decoded_signatures[i] = evaluation["decoded_signature"]
            machine_order_signatures[i] = evaluation["machine_order_signature"]
            ranking_signatures[i] = evaluation["ranking_signature"]
            continuous_signatures[i] = evaluation["continuous_signature"]

            if not was_reinitialized:
                energies[i] -= gamma * np.linalg.norm(new_pos - old_pos)

            if not was_reinitialized and energies[i] <= 0:
                if max_FEs is None or fe_counter < max_FEs:
                    use_knowledge_default_reinit = (
                        ssr_use_knowledge_for_default_reinit
                        and priority_knowledge_memory is not None
                        and _deterministic_probability_gate(
                            ssr_energy_knowledge_prob,
                            t,
                            i,
                            energy_reinit_count,
                        )
                    )
                    if use_knowledge_default_reinit:
                        pos[i] = _generate_random_key_agent_from_priority_knowledge(
                            priority_knowledge=priority_knowledge_memory,
                            low_dyn=low_dyn,
                            up_dyn=up_dyn,
                            base_noise_scale=ssr_knowledge_noise_scale,
                            uniform_mix=ssr_knowledge_uniform_mix,
                            min_noise_scale=ssr_knowledge_min_noise_scale,
                            max_confidence=ssr_knowledge_max_confidence,
                        )
                        knowledge_energy_reinit_count += 1
                    else:
                        pos[i] = _plain_random_position(low_dyn, up_dyn)
                    energies[i] = initial_energy
                    evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
                    fitness[i] = evaluation["fitness"]
                    schedule_dfs[i] = evaluation["schedule_df"]
                    metrics_list[i] = evaluation["metrics"]
                    decoded_signatures[i] = evaluation["decoded_signature"]
                    machine_order_signatures[i] = evaluation["machine_order_signature"]
                    ranking_signatures[i] = evaluation["ranking_signature"]
                    continuous_signatures[i] = evaluation["continuous_signature"]
                    fe_counter += 1
                    energy_reinit_count += 1

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
            decoded_signatures,
            archive_signatures,
            elite_size,
        )
        if check_this_iter and decoder_available and operation_reference_valid and elite_archive:
            candidate_knowledge = _build_operation_priority_knowledge_from_archive(
                elite_archive=elite_archive[: max(1, ssr_elite_k)],
                decoder=decoder,
                operation_reference=operation_reference,
            )
            if (
                candidate_knowledge is not None
                and candidate_knowledge["signal_ratio"] >= ssr_min_knowledge_signal_ratio
            ):
                priority_knowledge_memory = candidate_knowledge

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
            "knowledge_warning": None,
        }

        if check_this_iter:
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
            _, std_dim = _compute_dimensional_statistics(pos)
            convergence_mask = std_dim <= dim_convergence_threshold
            converged_dims = np.where(convergence_mask)[0].astype(int).tolist()
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
                ssr_mode in {"knowledge", "hybrid"}
                and ssr_allow_plateau_activation
                and plateau_check_streak >= max(1, int(ssr_min_plateau_checks))
            )
            stagnated = bool(stagnated or plateau_only_stagnated)

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
                activation_parts.append("plateau_only_knowledge")

            stagnation_details.update(structural_details)
            stagnation_details.update(
                {
                    "improvement_window": improvement_window,
                    "fitness_plateau": bool(fitness_plateau),
                    "plateau_check_streak": int(plateau_check_streak),
                    "structural_stagnation": bool(structural_stagnation),
                    "stagnation_score": float(stagnation_score),
                    "activation_reason": "+".join(activation_parts) if stagnated else "not_activated",
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

            can_apply_continuous_ssr = ssr_mode in {"continuous", "hybrid"} and np.any(convergence_mask)
            can_apply_knowledge_ssr = ssr_mode in {"knowledge", "hybrid"} and decoder_available
            if stagnated and (can_apply_continuous_ssr or can_apply_knowledge_ssr):
                if can_apply_continuous_ssr:
                    sample_positions, sample_fitness = _build_historical_samples_from_archive(
                        elite_archive=elite_archive,
                        fallback_pos=pos,
                        fallback_fitness=fitness,
                    )
                    (
                        low_dyn,
                        up_dyn,
                        dimension_correlations,
                        correlation_flags,
                        reduced_dims,
                    ) = _apply_search_space_reduction(
                        low_dyn=low_dyn,
                        up_dyn=up_dyn,
                        pos=pos,
                        sample_positions=sample_positions,
                        sample_fitness=sample_fitness,
                        convergence_mask=convergence_mask,
                        corr_threshold=ssr_correlation_threshold,
                    )
                protected_elite_indices = set(
                    _select_protected_elite_indices(
                        fitness=fitness,
                        archive_signatures=archive_signatures,
                        elite_size=elite_size,
                    )
                )
                restart_budget = int(math.ceil(partial_restart_ratio * N))
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
                if can_apply_knowledge_ssr and operation_reference_valid:
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
                    if priority_knowledge is not None:
                        priority_knowledge_memory = priority_knowledge

                if can_apply_knowledge_ssr and not operation_reference_valid:
                    stagnation_details["knowledge_warning"] = operation_reference_error
                elif can_apply_knowledge_ssr and not elite_archive:
                    stagnation_details["knowledge_warning"] = "no_historical_elite_for_knowledge"
                elif can_apply_knowledge_ssr and priority_knowledge is None:
                    if knowledge_signal_ratio < ssr_min_knowledge_signal_ratio:
                        stagnation_details["knowledge_warning"] = (
                            f"low_knowledge_signal:{knowledge_signal_ratio:.3f}"
                        )
                    else:
                        stagnation_details["knowledge_warning"] = "knowledge_unavailable"

                for idx in replace_indices:
                    if max_FEs is not None and fe_counter >= max_FEs:
                        break
                    best_candidate_pos = None
                    best_candidate_evaluation = None
                    best_candidate_fit = float("inf")
                    used_knowledge_candidate = False

                    for _ in range(max(1, int(ssr_candidate_trials))):
                        if max_FEs is not None and fe_counter >= max_FEs:
                            break
                        if priority_knowledge is not None:
                            candidate_pos = _generate_random_key_agent_from_priority_knowledge(
                                priority_knowledge=priority_knowledge,
                                low_dyn=low_dyn,
                                up_dyn=up_dyn,
                                base_noise_scale=ssr_knowledge_noise_scale,
                                uniform_mix=ssr_knowledge_uniform_mix,
                                min_noise_scale=ssr_knowledge_min_noise_scale,
                                max_confidence=ssr_knowledge_max_confidence,
                            )
                            candidate_used_knowledge = True
                        elif ssr_random_fallback:
                            candidate_pos = _plain_random_position(low_dyn, up_dyn)
                            candidate_used_knowledge = False
                        else:
                            continue

                        candidate_evaluation = _evaluate_candidate(candidate_pos, fobj=fobj, decoder=decoder)
                        candidate_fit = candidate_evaluation["fitness"]
                        fe_counter += 1
                        ssr_candidate_attempt_count += 1
                        if candidate_fit < best_candidate_fit:
                            best_candidate_pos = candidate_pos
                            best_candidate_evaluation = candidate_evaluation
                            best_candidate_fit = candidate_fit
                            used_knowledge_candidate = candidate_used_knowledge

                    if best_candidate_evaluation is None:
                        continue

                    if ssr_accept_only_improvement and best_candidate_fit >= fitness[idx]:
                        ssr_rejected_candidate_count += 1
                        continue

                    pos[idx] = best_candidate_pos
                    evaluation = best_candidate_evaluation
                    if priority_knowledge is not None:
                        knowledge_replacement_count += 1
                    else:
                        ssr_fallback_random_count += 1

                    energies[idx] = initial_energy
                    fitness[idx] = evaluation["fitness"]
                    schedule_dfs[idx] = evaluation["schedule_df"]
                    metrics_list[idx] = evaluation["metrics"]
                    decoded_signatures[idx] = evaluation["decoded_signature"]
                    machine_order_signatures[idx] = evaluation["machine_order_signature"]
                    ranking_signatures[idx] = evaluation["ranking_signature"]
                    continuous_signatures[idx] = evaluation["continuous_signature"]
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
                    decoded_signatures,
                    archive_signatures,
                    elite_size,
                )
                ssr_triggered = True

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
                "unique_continuous_count": int(len(set(continuous_signatures))),
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
                "converged_dimension_count": int(np.sum(convergence_mask)),
                "reduced_dimension_count": int(len(reduced_dims)),
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
            "ssr_mode": ssr_mode,
            "ssr_check_iter": bool(check_this_iter),
            "improvement_window": stagnation_details.get("improvement_window"),
            "fitness_plateau": bool(stagnation_details.get("fitness_plateau", False)),
            "plateau_check_streak": int(stagnation_details.get("plateau_check_streak", plateau_check_streak)),
            "unique_continuous_count": int(cached_check_metrics["unique_continuous_count"]),
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
            "reinitialized": int(energy_reinit_count + deterioration_reinit_count + ssr_replacement_count),
            "energy_reinit": int(energy_reinit_count),
            "deterioration_reinit": int(deterioration_reinit_count),
            "knowledge_energy_reinit": int(knowledge_energy_reinit_count),
            "knowledge_deterioration_reinit": int(knowledge_deterioration_reinit_count),
            "ssr_active": bool(ssr_triggered),
            "ssr_replacement_count": int(ssr_replacement_count),
            "ssr_candidate_attempt_count": int(ssr_candidate_attempt_count),
            "ssr_rejected_candidate_count": int(ssr_rejected_candidate_count),
            "knowledge_replacement_count": int(knowledge_replacement_count),
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
            "converged_dimension_count": int(np.sum(convergence_mask)),
            "reduced_dimension_count": int(len(reduced_dims)),
            "converged_dimensions": converged_dims,
            "reduced_dimensions": reduced_dims,
            "dynamic_bound_span_mean": float(np.mean(up_dyn - low_dyn)),
            "dynamic_bound_span_min": float(np.min(up_dyn - low_dyn)),
            "avg_abs_dimension_correlation": float(np.mean(np.abs(dimension_correlations))) if dimension_correlations.size else 0.0,
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
                f"SSRReplace: {ssr_replacement_count} | "
                f"SSRReject: {ssr_rejected_candidate_count} | "
                f"EReinit: {energy_reinit_count} | "
                f"DReinit: {deterioration_reinit_count} | "
                f"KE: {knowledge_energy_reinit_count} | "
                f"KD: {knowledge_deterioration_reinit_count} | "
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
