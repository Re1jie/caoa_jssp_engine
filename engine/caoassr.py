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
        ["machine_id", "S_lj", "C_lj", "job_id", "voyage", "op_seq"]
    )
    rows = []
    for row in ordered.itertuples(index=False):
        rows.append(
            (
                int(row.machine_id),
                int(row.job_id),
                int(row.voyage),
                int(row.op_seq),
                round(float(row.S_lj), 6),
                round(float(row.C_lj), 6),
            )
        )
    return tuple(rows)


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


def compute_critical_order_signature(decoded_schedule, critical_k=20):
    if decoded_schedule is None or decoded_schedule.empty or critical_k <= 0:
        return ("empty",)

    df = decoded_schedule.copy()
    df["_wait_score"] = df.get("congestion_wait", 0.0) + df.get("tidal_wait", 0.0)

    machine_load = df.groupby("machine_id")["p_lj"].sum()
    bottleneck_machine = int(machine_load.idxmax())
    bottleneck_ops = (
        df[df["machine_id"] == bottleneck_machine]
        .sort_values(["S_lj", "C_lj", "job_id", "voyage", "op_seq"])
        .head(critical_k)
    )

    waiting_ops = (
        df.sort_values(
            ["_wait_score", "p_lj", "S_lj", "job_id", "voyage", "op_seq"],
            ascending=[False, False, True, True, True, True],
        )
        .head(critical_k)
        .sort_values(["machine_id", "S_lj", "C_lj", "job_id", "voyage", "op_seq"])
    )

    def _ops_tuple(frame):
        return tuple(
            (int(row.machine_id), int(row.job_id), int(row.voyage), int(row.op_seq))
            for row in frame.itertuples(index=False)
        )

    return (
        ("bottleneck", bottleneck_machine, _ops_tuple(bottleneck_ops)),
        ("waiting", _ops_tuple(waiting_ops)),
    )


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

    moved = 0
    for operation in common_ops:
        if map_a[operation] != map_b[operation]:
            moved += 1
    return moved / len(common_ops)


def compute_structural_similarity(signature_a, signature_b):
    return 1.0 - compute_machine_order_distance(signature_a, signature_b)


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


def detect_fitness_plateau(best_score_history, current_iter, stagnation_window, eps_improve):
    history_window = [
        item for item in best_score_history
        if current_iter - item["iter"] < stagnation_window
    ]

    if len(history_window) < 2:
        return False, None

    scores = np.array([item["score"] for item in history_window], dtype=float)
    improvement = float(np.max(scores) - np.min(scores))
    return improvement <= eps_improve, improvement


def detect_structural_stagnation(
    machine_order_signatures,
    ranking_signatures,
    population_size,
    machine_family_threshold,
    ranking_collapse_threshold,
    structural_distance_threshold,
    use_machine_order_signature=True,
    use_ranking_similarity=True,
):
    unique_machine_families = len(set(machine_order_signatures)) if use_machine_order_signature else population_size
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
    ranking_collapse = (
        use_ranking_similarity
        and avg_ranking_distance <= ranking_collapse_threshold
    )
    structural_stagnation = machine_order_collapse or ranking_collapse

    details = {
        "unique_machine_families": int(unique_machine_families),
        "machine_family_limit": int(machine_family_limit),
        "avg_machine_order_distance": float(avg_machine_distance),
        "avg_ranking_distance": float(avg_ranking_distance),
        "machine_order_collapse": bool(machine_order_collapse),
        "ranking_collapse": bool(ranking_collapse),
        "structural_stagnation": bool(structural_stagnation),
    }
    return structural_stagnation, details


def compute_stagnation_score(
    fitness_plateau_score,
    machine_order_collapse_score,
    ranking_collapse_score,
    stagnation_score_weights=None,
):
    weights = {
        "fitness": 0.5,
        "machine_order": 0.3,
        "ranking": 0.2,
    }
    if stagnation_score_weights:
        weights.update(stagnation_score_weights)

    return (
        weights["fitness"] * fitness_plateau_score
        + weights["machine_order"] * machine_order_collapse_score
        + weights["ranking"] * ranking_collapse_score
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


def identify_collapsed_dimensions(pos, eps_std):
    stds = np.std(pos, axis=0)
    collapsed_dims = stds < eps_std
    return collapsed_dims, stds


def compute_dimension_correlations(pos, fitness, collapsed_dims):
    correlations = np.zeros(pos.shape[1], dtype=float)
    fitness_std = float(np.std(fitness))
    if fitness_std <= 1e-12:
        return correlations

    fitness_centered = fitness - np.mean(fitness)
    fitness_norm = np.linalg.norm(fitness_centered)
    if fitness_norm <= 1e-12:
        return correlations

    for j in np.where(collapsed_dims)[0]:
        x = pos[:, j]
        x_centered = x - np.mean(x)
        x_norm = np.linalg.norm(x_centered)
        if x_norm <= 1e-12:
            correlations[j] = 0.0
            continue
        correlations[j] = float(np.dot(x_centered, fitness_centered) / (x_norm * fitness_norm))

    return correlations


def apply_ssr(
    low_dyn,
    up_dyn,
    collapsed_dims,
    correlations,
    eta_shrink,
    rho_pos,
    rho_neg,
    min_interval,
    mild_shrink_ratio=0.5,
):
    shrunk_dims = np.zeros_like(collapsed_dims, dtype=bool)
    shrink_directions = np.full(collapsed_dims.shape, "", dtype=object)

    for j in np.where(collapsed_dims)[0]:
        interval = float(up_dyn[j] - low_dyn[j])
        if interval <= min_interval:
            continue

        max_delta = max(0.0, interval - min_interval)
        if max_delta <= 0.0:
            continue

        delta = min(eta_shrink * interval, max_delta)
        if delta <= 0.0:
            continue

        if correlations[j] > rho_pos:
            new_up = max(low_dyn[j] + min_interval, up_dyn[j] - delta)
            if new_up < up_dyn[j]:
                up_dyn[j] = new_up
                shrunk_dims[j] = True
                shrink_directions[j] = "upper"
        elif correlations[j] < -rho_neg:
            new_low = min(up_dyn[j] - min_interval, low_dyn[j] + delta)
            if new_low > low_dyn[j]:
                low_dyn[j] = new_low
                shrunk_dims[j] = True
                shrink_directions[j] = "lower"
        else:
            mild_delta = min(delta * mild_shrink_ratio, 0.5 * max(0.0, interval - min_interval))
            if mild_delta <= 0.0:
                continue
            new_low = low_dyn[j] + mild_delta
            new_up = up_dyn[j] - mild_delta
            if new_up - new_low >= min_interval:
                low_dyn[j] = new_low
                up_dyn[j] = new_up
                shrunk_dims[j] = True
                shrink_directions[j] = "both"

    return low_dyn, up_dyn, shrunk_dims, shrink_directions


def build_refocus_center(elite_archive, dim, M_refocus=3, use_elite_mean=True):
    if not elite_archive:
        return None

    elite_archive = sorted(elite_archive, key=lambda item: item["fitness"])
    top_elites = elite_archive[: max(1, min(M_refocus, len(elite_archive)))]
    elite_positions = np.array([item["position"] for item in top_elites], dtype=float)

    if use_elite_mean or elite_positions.shape[0] == 1:
        return np.mean(elite_positions, axis=0)

    return 0.5 * (np.min(elite_positions, axis=0) + np.max(elite_positions, axis=0))


def reinitialize_in_reduced_space(
    low_dyn,
    up_dyn,
    ref_center=None,
    restart_sigma=0.2,
    uniform_mix=0.15,
):
    interval = np.maximum(up_dyn - low_dyn, 1e-12)
    if ref_center is None:
        return low_dyn + interval * np.random.rand(low_dyn.size)

    local = ref_center + np.random.normal(loc=0.0, scale=restart_sigma * interval, size=low_dyn.size)
    if uniform_mix > 0.0:
        mix_mask = np.random.rand(low_dyn.size) < uniform_mix
        local[mix_mask] = low_dyn[mix_mask] + interval[mix_mask] * np.random.rand(np.sum(mix_mask))
    return np.clip(local, low_dyn, up_dyn)


def _evaluate_candidate(
    x,
    fobj=None,
    decoder=None,
    critical_k=20,
    compute_machine_signature=False,
    compute_critical_signature=False,
):
    if decoder is not None:
        schedule_df, metrics = decoder.decode_from_continuous(x)
        fitness = float(metrics["total_tardiness"])
        decoded_signature = _schedule_signature_from_dataframe(schedule_df)
        return {
            "fitness": fitness,
            "schedule_df": schedule_df,
            "metrics": metrics,
            "decoded_signature": decoded_signature,
            "machine_order_signature": (
                compute_machine_order_signature(schedule_df)
                if compute_machine_signature
                else None
            ),
            "critical_order_signature": (
                compute_critical_order_signature(schedule_df, critical_k=critical_k)
                if compute_critical_signature
                else None
            ),
            "ranking_signature": _compute_ranking_signature(x),
            "continuous_signature": _compute_continuous_signature(x),
        }

    if fobj is None:
        raise ValueError("Either `fobj` or `decoder` must be provided.")

    fitness = float(fobj(x))
    return {
        "fitness": fitness,
        "schedule_df": None,
        "metrics": None,
        "decoded_signature": None,
        "machine_order_signature": None,
        "critical_order_signature": None,
        "ranking_signature": _compute_ranking_signature(x),
        "continuous_signature": _compute_continuous_signature(x),
    }


def _collect_structural_signatures(
    schedule_dfs,
    critical_k=20,
    use_machine_order_signature=True,
    use_critical_order_signature=True,
):
    machine_order_signatures = [None] * len(schedule_dfs)
    critical_order_signatures = [None] * len(schedule_dfs)

    for idx, schedule_df in enumerate(schedule_dfs):
        if use_machine_order_signature:
            machine_order_signatures[idx] = compute_machine_order_signature(schedule_df)
        if use_critical_order_signature:
            critical_order_signatures[idx] = compute_critical_order_signature(
                schedule_df,
                critical_k=critical_k,
            )

    return machine_order_signatures, critical_order_signatures


def _update_elite_archive(
    elite_archive,
    pos,
    fitness,
    decoded_signatures,
    archive_signatures,
    elite_size,
):
    merged = {}
    for item in elite_archive:
        key = item.get("archive_signature", item.get("decoded_signature"))
        best_existing = merged.get(key)
        if best_existing is None or item["fitness"] < best_existing["fitness"]:
            merged[key] = {
                "position": item["position"].copy(),
                "fitness": float(item["fitness"]),
                "decoded_signature": item.get("decoded_signature"),
                "exact_decoded_signature": item.get("exact_decoded_signature"),
                "archive_signature": key,
            }

    order = np.argsort(fitness)
    for idx in order:
        key = archive_signatures[idx]
        best_existing = merged.get(key)
        candidate = {
            "position": pos[idx].copy(),
            "fitness": float(fitness[idx]),
            "decoded_signature": decoded_signatures[idx],
            "exact_decoded_signature": decoded_signatures[idx],
            "archive_signature": key,
        }
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
    eps_improve=1e-6,
    eps_std=1e-3,
    rho_pos=0.15,
    rho_neg=0.15,
    eta_shrink=0.08,
    elite_size=5,
    min_interval=0.05,
    dup_ratio_threshold=0.6,
    unique_schedule_threshold=0.25,
    machine_family_threshold=0.7,
    ranking_collapse_threshold=0.08,
    structural_distance_threshold=0.25,
    critical_k=20,
    use_machine_order_signature=True,
    use_critical_order_signature=True,
    use_ranking_similarity=True,
    stagnation_mode="rule",
    stagnation_score_weights=None,
    stagnation_score_threshold=0.7,
    M_refocus=3,
    use_elite_mean=True,
    partial_restart_ratio=0.25,
    restart_sigma=0.2,
    restart_uniform_mix=0.15,
    mild_shrink_ratio=0.5,
    return_diagnostics=False,
    verbose=True,
):
    if fobj is None and decoder is None:
        raise ValueError("Either `fobj` or `decoder` must be provided to CAOA_SSR.")

    lb, ub = _normalize_bounds(lb, ub, dim)
    if stagnation_window is None:
        stagnation_window = K

    low_dyn = lb.copy()
    up_dyn = ub.copy()

    if initial_pos is not None:
        pos = np.clip(np.array(initial_pos, dtype=float).copy(), low_dyn, up_dyn)
    else:
        pos = low_dyn + (up_dyn - low_dyn) * np.random.rand(N, dim)

    energies = np.full(N, initial_energy, dtype=float)
    fitness = np.zeros(N, dtype=float)
    schedule_dfs = [None] * N
    decoded_signatures = [None] * N
    machine_order_signatures = [None] * N
    critical_order_signatures = [None] * N
    ranking_signatures = [None] * N
    continuous_signatures = [None] * N

    fe_counter = 0
    for i in range(N):
        evaluation = _evaluate_candidate(
            pos[i],
            fobj=fobj,
            decoder=decoder,
            critical_k=critical_k,
        )
        fitness[i] = evaluation["fitness"]
        schedule_dfs[i] = evaluation["schedule_df"]
        decoded_signatures[i] = evaluation["decoded_signature"]
        ranking_signatures[i] = evaluation["ranking_signature"]
        continuous_signatures[i] = evaluation["continuous_signature"]
        fe_counter += 1

    best_idx = int(np.argmin(fitness))
    gBestScore = float(fitness[best_idx])
    gBest = pos[best_idx].copy()
    gBestDecodedSignature = decoded_signatures[best_idx]
    gBestMachineOrderSignature = None

    best_score_history = []
    best_decoded_signature_history = []
    best_structural_signature_history = []
    elite_archive = []
    cg_curve = []
    avg_curve = []
    diagnostics = []
    ref_center = None
    cached_check_metrics = {
        "unique_continuous_count": int(len(set(continuous_signatures))),
        "unique_ranking_count": int(len(set(ranking_signatures))),
        "unique_schedule_count": int(len(set(sig for sig in decoded_signatures if sig is not None))),
        "unique_machine_family_count": 0,
        "unique_critical_signature_count": 0,
        "avg_machine_order_distance": 0.0,
        "avg_ranking_distance": 0.0,
        "dup_ratio_to_gbest": float(
            sum(sig == gBestDecodedSignature for sig in decoded_signatures) / max(1, N)
        ),
        "collapsed_dims_count": 0,
    }

    start_total = time.perf_counter()
    for t in range(max_iter):
        iter_start = time.perf_counter()
        depleted_count = 0
        deterioration_reinit_count = 0
        partial_restart_count = 0
        shrunk_dims_mask = np.zeros(dim, dtype=bool)
        collapsed_dims_mask = np.zeros(dim, dtype=bool)
        ssr_triggered = False
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
            new_pos = np.clip(new_pos, low_dyn, up_dyn)

            evaluation = _evaluate_candidate(
                new_pos,
                fobj=fobj,
                decoder=decoder,
                critical_k=critical_k,
            )
            new_fit = evaluation["fitness"]
            fe_counter += 1

            if abs(new_fit - old_fit) > delta and new_fit > old_fit:
                if max_FEs is None or fe_counter < max_FEs:
                    new_pos = reinitialize_in_reduced_space(
                        low_dyn=low_dyn,
                        up_dyn=up_dyn,
                        ref_center=ref_center,
                        restart_sigma=restart_sigma,
                        uniform_mix=restart_uniform_mix,
                    )
                    evaluation = _evaluate_candidate(
                        new_pos,
                        fobj=fobj,
                        decoder=decoder,
                        critical_k=critical_k,
                    )
                    new_fit = evaluation["fitness"]
                    fe_counter += 1
                    deterioration_reinit_count += 1

            pos[i] = new_pos
            fitness[i] = new_fit
            schedule_dfs[i] = evaluation["schedule_df"]
            decoded_signatures[i] = evaluation["decoded_signature"]
            machine_order_signatures[i] = None
            critical_order_signatures[i] = None
            ranking_signatures[i] = evaluation["ranking_signature"]
            continuous_signatures[i] = evaluation["continuous_signature"]

            dist = np.linalg.norm(new_pos - old_pos)
            energies[i] -= gamma * dist

            if energies[i] <= 0:
                depleted_count += 1
                if max_FEs is None or fe_counter < max_FEs:
                    pos[i] = reinitialize_in_reduced_space(
                        low_dyn=low_dyn,
                        up_dyn=up_dyn,
                        ref_center=ref_center,
                        restart_sigma=restart_sigma,
                        uniform_mix=restart_uniform_mix,
                    )
                    energies[i] = initial_energy
                    evaluation = _evaluate_candidate(
                        pos[i],
                        fobj=fobj,
                        decoder=decoder,
                        critical_k=critical_k,
                    )
                    fitness[i] = evaluation["fitness"]
                    schedule_dfs[i] = evaluation["schedule_df"]
                    decoded_signatures[i] = evaluation["decoded_signature"]
                    machine_order_signatures[i] = None
                    critical_order_signatures[i] = None
                    ranking_signatures[i] = evaluation["ranking_signature"]
                    continuous_signatures[i] = evaluation["continuous_signature"]
                    fe_counter += 1

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < gBestScore:
            gBestScore = float(fitness[best_idx])
            gBest = pos[best_idx].copy()
            gBestDecodedSignature = decoded_signatures[best_idx]
            gBestMachineOrderSignature = None

        archive_signatures = [
            machine_sig if machine_sig is not None else decoded_sig
            for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
        ]
        elite_archive = _update_elite_archive(
            elite_archive=elite_archive,
            pos=pos,
            fitness=fitness,
            decoded_signatures=decoded_signatures,
            archive_signatures=archive_signatures,
            elite_size=elite_size,
        )
        ref_center = build_refocus_center(
            elite_archive=elite_archive,
            dim=dim,
            M_refocus=M_refocus,
            use_elite_mean=use_elite_mean,
        )

        stagnation_details = {
            "improvement_window": None,
            "fitness_plateau": False,
            "structural_stagnation": False,
            "unique_machine_families": cached_check_metrics["unique_machine_family_count"],
            "unique_critical_signatures": cached_check_metrics["unique_critical_signature_count"],
            "avg_machine_order_distance": cached_check_metrics["avg_machine_order_distance"],
            "avg_ranking_distance": cached_check_metrics["avg_ranking_distance"],
            "machine_order_collapse": False,
            "ranking_collapse": False,
            "stagnation_score": 0.0,
            "activation_reason": "skipped_until_it" if not check_this_iter else "not_checked",
        }

        if check_this_iter:
            machine_order_signatures, critical_order_signatures = _collect_structural_signatures(
                schedule_dfs=schedule_dfs,
                critical_k=critical_k,
                use_machine_order_signature=use_machine_order_signature,
                use_critical_order_signature=use_critical_order_signature,
            )
            archive_signatures = [
                machine_sig if machine_sig is not None else decoded_sig
                for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
            ]
            elite_archive = _update_elite_archive(
                elite_archive=elite_archive,
                pos=pos,
                fitness=fitness,
                decoded_signatures=decoded_signatures,
                archive_signatures=archive_signatures,
                elite_size=elite_size,
            )
            ref_center = build_refocus_center(
                elite_archive=elite_archive,
                dim=dim,
                M_refocus=M_refocus,
                use_elite_mean=use_elite_mean,
            )
            gBestMachineOrderSignature = machine_order_signatures[best_idx] if use_machine_order_signature else None
            best_score_history.append({"iter": t + 1, "score": float(gBestScore)})
            best_decoded_signature_history.append(
                {"iter": t + 1, "signature": gBestDecodedSignature}
            )
            best_structural_signature_history.append(
                {"iter": t + 1, "signature": gBestMachineOrderSignature}
            )

            fitness_plateau, improvement_window = detect_fitness_plateau(
                best_score_history=best_score_history,
                current_iter=t + 1,
                stagnation_window=stagnation_window,
                eps_improve=eps_improve,
            )
            structural_stagnation, structural_details = detect_structural_stagnation(
                machine_order_signatures=machine_order_signatures,
                ranking_signatures=ranking_signatures,
                population_size=N,
                machine_family_threshold=machine_family_threshold,
                ranking_collapse_threshold=ranking_collapse_threshold,
                structural_distance_threshold=structural_distance_threshold,
                use_machine_order_signature=use_machine_order_signature,
                use_ranking_similarity=use_ranking_similarity,
            )

            fitness_plateau_score = 1.0 if fitness_plateau else 0.0
            machine_order_collapse_score = 1.0 if structural_details["machine_order_collapse"] else 0.0
            ranking_collapse_score = 1.0 if structural_details["ranking_collapse"] else 0.0
            stagnation_score = compute_stagnation_score(
                fitness_plateau_score=fitness_plateau_score,
                machine_order_collapse_score=machine_order_collapse_score,
                ranking_collapse_score=ranking_collapse_score,
                stagnation_score_weights=stagnation_score_weights,
            )
            stagnated = should_activate_ssr(
                fitness_plateau=fitness_plateau,
                structural_stagnation=structural_stagnation,
                stagnation_score=stagnation_score,
                stagnation_mode=stagnation_mode,
                stagnation_score_threshold=stagnation_score_threshold,
            )

            activation_parts = []
            if fitness_plateau:
                activation_parts.append("fitness_plateau")
            if structural_details["machine_order_collapse"]:
                activation_parts.append("machine_order_collapse")
            if structural_details["ranking_collapse"]:
                activation_parts.append("ranking_collapse")
            stagnation_details = {
                "improvement_window": improvement_window,
                "fitness_plateau": bool(fitness_plateau),
                "structural_stagnation": bool(structural_stagnation),
                "unique_machine_families": structural_details["unique_machine_families"],
                "unique_critical_signatures": len(
                    set(sig for sig in critical_order_signatures if sig is not None)
                ) if use_critical_order_signature else 0,
                "avg_machine_order_distance": structural_details["avg_machine_order_distance"],
                "avg_ranking_distance": structural_details["avg_ranking_distance"],
                "machine_order_collapse": structural_details["machine_order_collapse"],
                "ranking_collapse": structural_details["ranking_collapse"],
                "stagnation_score": float(stagnation_score),
                "activation_reason": "+".join(activation_parts) if stagnated else "not_activated",
            }

            if stagnated:
                collapsed_dims_mask, dim_stds = identify_collapsed_dimensions(pos, eps_std=eps_std)
                correlations = compute_dimension_correlations(pos, fitness, collapsed_dims_mask)
                low_dyn, up_dyn, shrunk_dims_mask, _ = apply_ssr(
                    low_dyn=low_dyn,
                    up_dyn=up_dyn,
                    collapsed_dims=collapsed_dims_mask,
                    correlations=correlations,
                    eta_shrink=eta_shrink,
                    rho_pos=rho_pos,
                    rho_neg=rho_neg,
                    min_interval=min_interval,
                    mild_shrink_ratio=mild_shrink_ratio,
                )

                ref_center = build_refocus_center(
                    elite_archive=elite_archive,
                    dim=dim,
                    M_refocus=M_refocus,
                    use_elite_mean=use_elite_mean,
                )

                protected_elite_indices = set(
                    _select_protected_elite_indices(
                        fitness=fitness,
                        archive_signatures=archive_signatures,
                        elite_size=elite_size,
                    )
                )
                non_elite_indices = [
                    idx for idx in np.argsort(fitness)[::-1]
                    if int(idx) not in protected_elite_indices
                ]
                restart_budget = int(math.ceil(partial_restart_ratio * max(0, len(non_elite_indices))))
                restart_budget = max(0, restart_budget)

                for idx in non_elite_indices[:restart_budget]:
                    if max_FEs is not None and fe_counter >= max_FEs:
                        break
                    pos[idx] = reinitialize_in_reduced_space(
                        low_dyn=low_dyn,
                        up_dyn=up_dyn,
                        ref_center=ref_center,
                        restart_sigma=restart_sigma,
                        uniform_mix=restart_uniform_mix,
                    )
                    energies[idx] = initial_energy
                    evaluation = _evaluate_candidate(
                        pos[idx],
                        fobj=fobj,
                        decoder=decoder,
                        critical_k=critical_k,
                    )
                    fitness[idx] = evaluation["fitness"]
                    schedule_dfs[idx] = evaluation["schedule_df"]
                    decoded_signatures[idx] = evaluation["decoded_signature"]
                    machine_order_signatures[idx] = None
                    critical_order_signatures[idx] = None
                    ranking_signatures[idx] = evaluation["ranking_signature"]
                    continuous_signatures[idx] = evaluation["continuous_signature"]
                    fe_counter += 1
                    partial_restart_count += 1

                machine_order_signatures, critical_order_signatures = _collect_structural_signatures(
                    schedule_dfs=schedule_dfs,
                    critical_k=critical_k,
                    use_machine_order_signature=use_machine_order_signature,
                    use_critical_order_signature=use_critical_order_signature,
                )
                archive_signatures = [
                    machine_sig if machine_sig is not None else decoded_sig
                    for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
                ]

                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < gBestScore:
                    gBestScore = float(fitness[best_idx])
                    gBest = pos[best_idx].copy()
                    gBestDecodedSignature = decoded_signatures[best_idx]
                    gBestMachineOrderSignature = (
                        machine_order_signatures[best_idx] if use_machine_order_signature else None
                    )

                elite_archive = _update_elite_archive(
                    elite_archive=elite_archive,
                    pos=pos,
                    fitness=fitness,
                    decoded_signatures=decoded_signatures,
                    archive_signatures=archive_signatures,
                    elite_size=elite_size,
                )
                ref_center = build_refocus_center(
                    elite_archive=elite_archive,
                    dim=dim,
                    M_refocus=M_refocus,
                    use_elite_mean=use_elite_mean,
                )
                ssr_triggered = True
            else:
                dim_stds = np.std(pos, axis=0)
                collapsed_dims_mask = dim_stds < eps_std

            cached_check_metrics = {
                "unique_continuous_count": int(len(set(continuous_signatures))),
                "unique_ranking_count": int(len(set(ranking_signatures))),
                "unique_schedule_count": int(len(set(sig for sig in decoded_signatures if sig is not None))),
                "unique_machine_family_count": int(len(set(sig for sig in machine_order_signatures if sig is not None))),
                "unique_critical_signature_count": int(
                    len(set(sig for sig in critical_order_signatures if sig is not None))
                ) if use_critical_order_signature else 0,
                "avg_machine_order_distance": float(
                    stagnation_details.get("avg_machine_order_distance", 0.0)
                ),
                "avg_ranking_distance": float(
                    stagnation_details.get("avg_ranking_distance", 0.0)
                ),
                "dup_ratio_to_gbest": float(
                    sum(sig == gBestDecodedSignature for sig in decoded_signatures) / max(1, N)
                ),
                "collapsed_dims_count": int(np.sum(collapsed_dims_mask)),
            }

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < gBestScore:
            gBestScore = float(fitness[best_idx])
            gBest = pos[best_idx].copy()
            gBestDecodedSignature = decoded_signatures[best_idx]
            gBestMachineOrderSignature = None

        cg_curve.append(gBestScore)
        avg_curve.append(float(np.mean(fitness)))

        iter_time = time.perf_counter() - iter_start
        total_time = time.perf_counter() - start_total
        dynamic_interval = up_dyn - low_dyn
        unique_continuous_count = cached_check_metrics["unique_continuous_count"]
        unique_ranking_count = cached_check_metrics["unique_ranking_count"]
        unique_schedule_count = cached_check_metrics["unique_schedule_count"]
        unique_machine_family_count = cached_check_metrics["unique_machine_family_count"]
        unique_critical_signature_count = cached_check_metrics["unique_critical_signature_count"]
        avg_machine_order_distance = cached_check_metrics["avg_machine_order_distance"]
        avg_ranking_distance = cached_check_metrics["avg_ranking_distance"]
        dup_ratio_to_gbest = cached_check_metrics["dup_ratio_to_gbest"]
        collapsed_dims_count = cached_check_metrics["collapsed_dims_count"] if not check_this_iter else int(np.sum(collapsed_dims_mask))

        log_entry = {
            "iter": t + 1,
            "best_fitness": float(gBestScore),
            "ssr_check_iter": bool(check_this_iter),
            "improvement_window": stagnation_details.get("improvement_window"),
            "fitness_plateau": bool(stagnation_details.get("fitness_plateau", False)),
            "unique_continuous_count": int(unique_continuous_count),
            "unique_ranking_count": int(unique_ranking_count),
            "unique_schedule_count": int(unique_schedule_count),
            "unique_machine_order_families": int(unique_machine_family_count),
            "unique_critical_order_signatures": int(unique_critical_signature_count),
            "avg_pairwise_machine_order_distance": float(avg_machine_order_distance),
            "avg_pairwise_ranking_distance": float(avg_ranking_distance),
            "dup_ratio_to_gbest": float(dup_ratio_to_gbest),
            "machine_order_collapse": bool(stagnation_details.get("machine_order_collapse", False)),
            "ranking_collapse": bool(stagnation_details.get("ranking_collapse", False)),
            "structural_stagnation": bool(stagnation_details.get("structural_stagnation", False)),
            "stagnation_score": float(stagnation_details.get("stagnation_score", 0.0)),
            "ssr_activation_reason": stagnation_details.get(
                "activation_reason",
                "skipped_until_it"
            ) if not check_this_iter else stagnation_details.get("activation_reason", "not_checked"),
            "collapsed_dims": int(collapsed_dims_count),
            "shrunk_dims": int(np.sum(shrunk_dims_mask)),
            "avg_dynamic_interval": float(np.mean(dynamic_interval)),
            "reinitialized": int(depleted_count + deterioration_reinit_count + partial_restart_count),
            "energy_reinit": int(depleted_count),
            "deterioration_reinit": int(deterioration_reinit_count),
            "partial_restart": int(partial_restart_count),
            "ssr_active": bool(ssr_triggered),
            "fe_counter": int(fe_counter),
            "iter_time_sec": float(iter_time),
            "total_time_sec": float(total_time),
            "stagnation_improvement_window": stagnation_details.get("improvement_window"),
        }
        diagnostics.append(log_entry)

        if verbose:
            print(
                f"Iterasi {t+1}/{max_iter} | "
                f"Best: {gBestScore:.2f} | "
                f"MachineFam: {unique_machine_family_count} | "
                f"CritSig: {unique_critical_signature_count} | "
                f"UniqueRank: {unique_ranking_count} | "
                f"MachDist: {avg_machine_order_distance:.3f} | "
                f"RankDist: {avg_ranking_distance:.3f} | "
                f"Collapsed: {collapsed_dims_count} | "
                f"Shrunk: {int(np.sum(shrunk_dims_mask))} | "
                f"AvgDynInt: {np.mean(dynamic_interval):.4f} | "
                f"Reinit: {depleted_count + deterioration_reinit_count + partial_restart_count} | "
                f"SSR: {ssr_triggered} | "
                f"Reason: {log_entry['ssr_activation_reason']} | "
                f"FEs: {fe_counter} | "
                f"Iter time: {iter_time:.4f}s | "
                f"Total time: {total_time:.2f}s"
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
            "low_dyn": low_dyn.copy(),
            "up_dyn": up_dyn.copy(),
            "elite_archive": elite_archive,
            "best_score_history": best_score_history,
            "best_decoded_signature_history": best_decoded_signature_history,
            "best_structural_signature_history": best_structural_signature_history,
        }
        return outputs + (extra,)
    return outputs
