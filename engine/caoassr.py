import math
import time

import numpy as np

from engine.kdr import KnowledgeDrivenReinitializer


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

    moved = 0
    for operation in common_ops:
        if map_a[operation] != map_b[operation]:
            moved += 1
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


def detect_fitness_plateau(best_score_history, current_iter, stagnation_window, eps_improve):
    """Return whether best fitness has materially improved in the recent window."""

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
    """Detect population collapse in decoded machine order or random-key ranking."""

    unique_machine_families = (
        len(set(machine_order_signatures))
        if use_machine_order_signature
        else population_size
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
    structural_stagnation = machine_order_collapse or ranking_collapse

    return structural_stagnation, {
        "unique_machine_families": int(unique_machine_families),
        "avg_machine_order_distance": float(avg_machine_distance),
        "avg_ranking_distance": float(avg_ranking_distance),
        "machine_order_collapse": bool(machine_order_collapse),
        "ranking_collapse": bool(ranking_collapse),
        "structural_stagnation": bool(structural_stagnation),
    }


def compute_stagnation_score(
    fitness_plateau_score,
    machine_order_collapse_score,
    ranking_collapse_score,
    stagnation_score_weights=None,
):
    weights = {"fitness": 0.5, "machine_order": 0.3, "ranking": 0.2}
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


def _evaluate_candidate(
    x,
    fobj=None,
    decoder=None,
):
    """Evaluate one position and keep only schedule metadata needed downstream."""

    if decoder is not None:
        schedule_df, metrics = decoder.decode_from_continuous(x)
        return {
            "fitness": float(metrics["total_tardiness"]),
            "schedule_df": schedule_df,
            "metrics": metrics,
            "decoded_signature": _schedule_signature_from_dataframe(schedule_df),
            "ranking_signature": _compute_ranking_signature(x),
        }

    if fobj is None:
        raise ValueError("Either `fobj` or `decoder` must be provided.")

    return {
        "fitness": float(fobj(x)),
        "schedule_df": None,
        "metrics": None,
        "decoded_signature": None,
        "ranking_signature": _compute_ranking_signature(x),
    }


def _update_elite_archive(
    elite_archive,
    pos,
    fitness,
    schedule_dfs,
    metrics_list,
    decoded_signatures,
    archive_signatures,
    elite_size,
):
    """Keep the best structurally distinct elites for KDR."""

    merged = {}
    for item in elite_archive:
        key = item["archive_signature"]
        best_existing = merged.get(key)
        if best_existing is None or item["fitness"] < best_existing["fitness"]:
            merged[key] = {
                "position": item["position"].copy(),
                "fitness": float(item["fitness"]),
                "decoded_signature": item["decoded_signature"],
                "schedule_df": item["schedule_df"],
                "metrics": item["metrics"],
                "archive_signature": key,
            }

    for idx in np.argsort(fitness):
        key = archive_signatures[idx]
        candidate = {
            "position": pos[idx].copy(),
            "fitness": float(fitness[idx]),
            "decoded_signature": decoded_signatures[idx],
            "schedule_df": schedule_dfs[idx],
            "metrics": metrics_list[idx],
            "archive_signature": key,
        }
        best_existing = merged.get(key)
        if best_existing is None or candidate["fitness"] < best_existing["fitness"]:
            merged[key] = candidate

    return sorted(merged.values(), key=lambda item: item["fitness"])[:elite_size]


def _select_protected_elite_indices(fitness, archive_signatures, elite_size):
    """Protect the best unique elites from being replaced."""

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
    elite_size=5,
    machine_family_threshold=0.7,
    ranking_collapse_threshold=0.08,
    structural_distance_threshold=0.25,
    use_machine_order_signature=True,
    use_ranking_similarity=True,
    stagnation_mode="rule",
    stagnation_score_weights=None,
    stagnation_score_threshold=0.7,
    partial_restart_ratio=0.25,
    use_kdr=True,
    kdr_rate=0.45,
    kdr_elite_size=3,
    kdr_noise_scale=0.07,
    kdr_stability_threshold=0.88,
    kdr_uniform_mix=0.30,
    kdr_bottleneck_perturbation=0.45,
    ssr_kdr_rate=None,
    ssr_kdr_elite_size=None,
    ssr_kdr_noise_scale=None,
    ssr_kdr_stability_threshold=None,
    ssr_kdr_uniform_mix=None,
    ssr_kdr_bottleneck_perturbation=None,
    return_diagnostics=False,
    verbose=True,
):
    """CAOA with SSR-triggered, knowledge-driven reinitialization for JSSP."""

    if fobj is None and decoder is None:
        raise ValueError("Either `fobj` or `decoder` must be provided to CAOA_SSR.")

    if stagnation_window is None:
        stagnation_window = K

    low_dyn, up_dyn = _normalize_bounds(lb, ub, dim)
    interval = np.maximum(up_dyn - low_dyn, 1e-12)

    if initial_pos is not None:
        pos = np.clip(np.array(initial_pos, dtype=float).copy(), low_dyn, up_dyn)
    else:
        pos = low_dyn + interval * np.random.rand(N, dim)

    energies = np.full(N, initial_energy, dtype=float)
    fitness = np.zeros(N, dtype=float)
    schedule_dfs = [None] * N
    metrics_list = [None] * N
    decoded_signatures = [None] * N
    ranking_signatures = [None] * N

    fe_counter = 0
    for i in range(N):
        evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
        fitness[i] = evaluation["fitness"]
        schedule_dfs[i] = evaluation["schedule_df"]
        metrics_list[i] = evaluation["metrics"]
        decoded_signatures[i] = evaluation["decoded_signature"]
        ranking_signatures[i] = evaluation["ranking_signature"]
        fe_counter += 1

    best_idx = int(np.argmin(fitness))
    gBestScore = float(fitness[best_idx])
    gBest = pos[best_idx].copy()
    gBestDecodedSignature = decoded_signatures[best_idx]

    elite_archive = []
    best_score_history = []
    cg_curve = []
    avg_curve = []
    diagnostics = []

    def _resolve_ssr_param(value, default):
        return default if value is None else value

    normal_kdr_config = {
        "rate": float(kdr_rate),
        "elite_size": int(kdr_elite_size),
        "noise_scale": float(kdr_noise_scale),
        "stability_threshold": float(kdr_stability_threshold),
        "uniform_mix": float(kdr_uniform_mix),
        "bottleneck_perturbation": float(kdr_bottleneck_perturbation),
    }
    ssr_kdr_config = {
        "rate": float(_resolve_ssr_param(ssr_kdr_rate, max(kdr_rate, 0.8))),
        "elite_size": int(_resolve_ssr_param(ssr_kdr_elite_size, kdr_elite_size)),
        "noise_scale": float(_resolve_ssr_param(ssr_kdr_noise_scale, max(0.5 * kdr_noise_scale, 1e-3))),
        "stability_threshold": float(
            _resolve_ssr_param(ssr_kdr_stability_threshold, max(0.75, kdr_stability_threshold - 0.05))
        ),
        "uniform_mix": float(_resolve_ssr_param(ssr_kdr_uniform_mix, max(0.05, 0.5 * kdr_uniform_mix))),
        "bottleneck_perturbation": float(
            _resolve_ssr_param(ssr_kdr_bottleneck_perturbation, kdr_bottleneck_perturbation)
        ),
    }

    kdr_reinitializers = {}
    if use_kdr and decoder is not None and hasattr(decoder, "L_ref"):
        operation_reference = list(decoder.L_ref)
        kdr_reinitializers["normal"] = KnowledgeDrivenReinitializer(
            operation_reference=operation_reference,
            dim=dim,
            stability_threshold=normal_kdr_config["stability_threshold"],
            noise_scale=normal_kdr_config["noise_scale"],
            uniform_mix=normal_kdr_config["uniform_mix"],
            bottleneck_perturbation=normal_kdr_config["bottleneck_perturbation"],
        )
        kdr_reinitializers["ssr"] = KnowledgeDrivenReinitializer(
            operation_reference=operation_reference,
            dim=dim,
            stability_threshold=ssr_kdr_config["stability_threshold"],
            noise_scale=ssr_kdr_config["noise_scale"],
            uniform_mix=ssr_kdr_config["uniform_mix"],
            bottleneck_perturbation=ssr_kdr_config["bottleneck_perturbation"],
        )

    kdr_context_cache = {}
    machine_order_signatures = [None] * N

    def _uniform_restart(count):
        return low_dyn + interval * np.random.rand(count, dim)

    def _collect_archive_signatures():
        return [
            machine_sig if machine_sig is not None else decoded_sig
            for machine_sig, decoded_sig in zip(machine_order_signatures, decoded_signatures)
        ]

    def _refresh_archive():
        nonlocal elite_archive, kdr_context_cache
        elite_archive = _update_elite_archive(
            elite_archive=elite_archive,
            pos=pos,
            fitness=fitness,
            schedule_dfs=schedule_dfs,
            metrics_list=metrics_list,
            decoded_signatures=decoded_signatures,
            archive_signatures=_collect_archive_signatures(),
            elite_size=elite_size,
        )
        kdr_context_cache = {}

    def _collect_kdr_elites(target_elite_size):
        top_elites = sorted(elite_archive, key=lambda item: item["fitness"])[
            : max(1, min(target_elite_size, len(elite_archive)))
        ]
        elite_schedules = [
            item["schedule_df"]
            for item in top_elites
            if item["schedule_df"] is not None and not item["schedule_df"].empty
        ]
        elite_positions = [
            item["position"]
            for item in top_elites
            if item["schedule_df"] is not None and not item["schedule_df"].empty
        ]
        elite_fitness = [
            float(item["fitness"])
            for item in top_elites
            if item["schedule_df"] is not None and not item["schedule_df"].empty
        ]
        return elite_schedules, elite_positions, elite_fitness

    def _get_kdr_context(mode):
        if mode not in kdr_reinitializers:
            return None
        if mode in kdr_context_cache:
            return kdr_context_cache[mode]

        config = normal_kdr_config if mode == "normal" else ssr_kdr_config
        elite_schedules, elite_positions, elite_fitness = _collect_kdr_elites(config["elite_size"])
        knowledge = kdr_reinitializers[mode].extract_knowledge(
            elite_schedules=elite_schedules,
            elite_positions=elite_positions,
            fitness=elite_fitness,
        )
        kdr_context_cache[mode] = {
            "knowledge": knowledge,
            "elite_schedules": elite_schedules,
            "elite_positions": elite_positions,
            "elite_fitness": elite_fitness,
            "config": config,
        }
        return kdr_context_cache[mode]

    def _sample_restart_positions(candidate_indices, protected_indices, mode="normal"):
        selected = [int(idx) for idx in candidate_indices if int(idx) not in protected_indices]
        if not selected:
            return {}, {
                "used": False,
                "reason": "protected_or_empty",
                "reinitialized": 0,
                "elite_count": 0,
                "stable_pair_count": 0,
                "used_fallback": True,
                "mode": mode,
            }

        if mode not in kdr_reinitializers:
            samples = _uniform_restart(len(selected))
            return (
                {idx: samples[i] for i, idx in enumerate(selected)},
                {
                    "used": False,
                    "reason": "uniform_restart",
                    "reinitialized": len(selected),
                    "elite_count": 0,
                    "stable_pair_count": 0,
                    "used_fallback": True,
                    "mode": mode,
                },
            )

        kdr_context = _get_kdr_context(mode)
        config = kdr_context["config"]
        selected_count = max(1, int(math.ceil(config["rate"] * len(selected))))
        selected = selected[:selected_count]
        replacements, info = kdr_reinitializers[mode].reinitialize_agents(
            population=pos,
            fitness=fitness,
            candidate_indices=selected,
            elite_schedules=kdr_context["elite_schedules"],
            low_dyn=low_dyn,
            up_dyn=up_dyn,
            elite_positions=kdr_context["elite_positions"],
            elite_fitness=kdr_context["elite_fitness"],
            noise_scale=config["noise_scale"],
            uniform_mix=config["uniform_mix"],
            knowledge=kdr_context["knowledge"],
        )
        if replacements:
            info["mode"] = mode
            return replacements, info

        samples = _uniform_restart(len(selected))
        return (
            {idx: samples[i] for i, idx in enumerate(selected)},
            {
                "used": False,
                "reason": "empty_knowledge",
                "reinitialized": len(selected),
                "elite_count": 0,
                "stable_pair_count": 0,
                "used_fallback": True,
                "mode": mode,
            },
        )

    _refresh_archive()
    start_total = time.perf_counter()

    for t in range(max_iter):
        iter_start = time.perf_counter()
        energy_reinit_count = 0
        deterioration_reinit_count = 0
        partial_restart_count = 0
        kdr_reinit_count = 0
        ssr_triggered = False
        check_this_iter = (t + 1) % IT == 0

        protected_indices = set(
            _select_protected_elite_indices(
                fitness=fitness,
                archive_signatures=_collect_archive_signatures(),
                elite_size=elite_size,
            )
        )
        kdr_last_info = {
            "used": False,
            "reason": "not_triggered",
            "reinitialized": 0,
            "elite_count": 0,
            "stable_pair_count": 0,
            "used_fallback": True,
        }

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

            old_pos = pos[i].copy()
            old_fit = float(fitness[i])

            r = np.random.rand(dim)
            new_pos = np.clip(
                pos[i] + alpha * (pos[leader_idx] - pos[i]) + beta * (1.0 - 2.0 * r),
                low_dyn,
                up_dyn,
            )

            evaluation = _evaluate_candidate(new_pos, fobj=fobj, decoder=decoder)
            new_fit = evaluation["fitness"]
            fe_counter += 1

            if abs(new_fit - old_fit) > delta and new_fit > old_fit:
                if i in protected_indices:
                    new_pos = old_pos
                    new_fit = old_fit
                    evaluation = {
                        "fitness": old_fit,
                        "schedule_df": schedule_dfs[i],
                        "metrics": metrics_list[i],
                        "decoded_signature": decoded_signatures[i],
                        "ranking_signature": ranking_signatures[i],
                        }
                elif max_FEs is None or fe_counter < max_FEs:
                    replacements, kdr_last_info = _sample_restart_positions(
                        [i],
                        protected_indices,
                        mode="normal",
                    )
                    if i in replacements:
                        new_pos = replacements[i]
                        evaluation = _evaluate_candidate(new_pos, fobj=fobj, decoder=decoder)
                        new_fit = evaluation["fitness"]
                        fe_counter += 1
                        deterioration_reinit_count += 1
                        kdr_reinit_count += int(kdr_last_info.get("reinitialized", 0))

            pos[i] = new_pos
            fitness[i] = new_fit
            schedule_dfs[i] = evaluation["schedule_df"]
            metrics_list[i] = evaluation["metrics"]
            decoded_signatures[i] = evaluation["decoded_signature"]
            ranking_signatures[i] = evaluation["ranking_signature"]
            machine_order_signatures[i] = None

            energies[i] -= gamma * np.linalg.norm(new_pos - old_pos)
            if energies[i] <= 0:
                if i in protected_indices:
                    energies[i] = initial_energy
                elif max_FEs is None or fe_counter < max_FEs:
                    replacements, kdr_last_info = _sample_restart_positions(
                        [i],
                        protected_indices,
                        mode="normal",
                    )
                    if i in replacements:
                        pos[i] = replacements[i]
                        energies[i] = initial_energy
                        evaluation = _evaluate_candidate(pos[i], fobj=fobj, decoder=decoder)
                        fitness[i] = evaluation["fitness"]
                        schedule_dfs[i] = evaluation["schedule_df"]
                        metrics_list[i] = evaluation["metrics"]
                        decoded_signatures[i] = evaluation["decoded_signature"]
                        ranking_signatures[i] = evaluation["ranking_signature"]
                        machine_order_signatures[i] = None
                        fe_counter += 1
                        energy_reinit_count += 1
                        kdr_reinit_count += int(kdr_last_info.get("reinitialized", 0))

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < gBestScore:
            gBestScore = float(fitness[best_idx])
            gBest = pos[best_idx].copy()
            gBestDecodedSignature = decoded_signatures[best_idx]

        _refresh_archive()

        stagnation_details = {
            "improvement_window": None,
            "fitness_plateau": False,
            "unique_machine_families": 0,
            "avg_machine_order_distance": 0.0,
            "avg_ranking_distance": 0.0,
            "machine_order_collapse": False,
            "ranking_collapse": False,
            "structural_stagnation": False,
            "stagnation_score": 0.0,
            "activation_reason": "skipped_until_it" if not check_this_iter else "not_checked",
        }

        if check_this_iter:
            # Use schedule structure, not continuous coordinates, to detect collapse.
            machine_order_signatures = [
                compute_machine_order_signature(schedule_df)
                if use_machine_order_signature
                else None
                for schedule_df in schedule_dfs
            ]
            _refresh_archive()

            best_score_history.append({"iter": t + 1, "score": float(gBestScore)})
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

            stagnation_score = compute_stagnation_score(
                fitness_plateau_score=1.0 if fitness_plateau else 0.0,
                machine_order_collapse_score=1.0 if structural_details["machine_order_collapse"] else 0.0,
                ranking_collapse_score=1.0 if structural_details["ranking_collapse"] else 0.0,
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
                "unique_machine_families": structural_details["unique_machine_families"],
                "avg_machine_order_distance": structural_details["avg_machine_order_distance"],
                "avg_ranking_distance": structural_details["avg_ranking_distance"],
                "machine_order_collapse": structural_details["machine_order_collapse"],
                "ranking_collapse": structural_details["ranking_collapse"],
                "structural_stagnation": structural_details["structural_stagnation"],
                "stagnation_score": float(stagnation_score),
                "activation_reason": "+".join(activation_parts) if stagnated else "not_activated",
            }

            if stagnated:
                # SSR keeps its role as a trigger for replacing the worst non-elites.
                protected_indices = set(
                    _select_protected_elite_indices(
                        fitness=fitness,
                        archive_signatures=_collect_archive_signatures(),
                        elite_size=elite_size,
                    )
                )
                non_elite_indices = [
                    int(idx)
                    for idx in np.argsort(fitness)[::-1]
                    if int(idx) not in protected_indices
                ]
                restart_budget = int(math.ceil(partial_restart_ratio * max(0, len(non_elite_indices))))
                restart_budget = max(0, restart_budget)
                restart_targets = non_elite_indices[:restart_budget]
                replacements, kdr_last_info = _sample_restart_positions(
                    restart_targets,
                    protected_indices,
                    mode="ssr",
                )

                for idx in restart_targets:
                    if max_FEs is not None and fe_counter >= max_FEs:
                        break
                    if idx not in replacements:
                        continue

                    pos[idx] = replacements[idx]
                    energies[idx] = initial_energy
                    evaluation = _evaluate_candidate(pos[idx], fobj=fobj, decoder=decoder)
                    fitness[idx] = evaluation["fitness"]
                    schedule_dfs[idx] = evaluation["schedule_df"]
                    metrics_list[idx] = evaluation["metrics"]
                    decoded_signatures[idx] = evaluation["decoded_signature"]
                    ranking_signatures[idx] = evaluation["ranking_signature"]
                    machine_order_signatures[idx] = None
                    fe_counter += 1
                    partial_restart_count += 1

                kdr_reinit_count += partial_restart_count
                _refresh_archive()

                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < gBestScore:
                    gBestScore = float(fitness[best_idx])
                    gBest = pos[best_idx].copy()
                    gBestDecodedSignature = decoded_signatures[best_idx]
                ssr_triggered = True

        cg_curve.append(gBestScore)
        avg_curve.append(float(np.mean(fitness)))

        iter_time = time.perf_counter() - iter_start
        total_time = time.perf_counter() - start_total

        log_entry = {
            "iter": t + 1,
            "best_fitness": float(gBestScore),
            "reinitialized": int(energy_reinit_count + deterioration_reinit_count + partial_restart_count),
            "energy_reinit": int(energy_reinit_count),
            "deterioration_reinit": int(deterioration_reinit_count),
            "partial_restart": int(partial_restart_count),
            "kdr_used": bool(kdr_last_info.get("used", False)),
            "kdr_reinitialized": int(kdr_reinit_count),
            "kdr_elite_count": int(kdr_last_info.get("elite_count", 0)),
            "kdr_stable_pairs": int(kdr_last_info.get("stable_pair_count", 0)),
            "kdr_used_fallback": bool(kdr_last_info.get("used_fallback", True)),
            "kdr_reason": kdr_last_info.get("reason", "not_triggered"),
            "kdr_mode": kdr_last_info.get("mode", "none"),
            "ssr_active": bool(ssr_triggered),
            "ssr_check_iter": bool(check_this_iter),
            "ssr_activation_reason": stagnation_details["activation_reason"],
            "fitness_plateau": bool(stagnation_details["fitness_plateau"]),
            "unique_machine_order_families": int(stagnation_details["unique_machine_families"]),
            "avg_pairwise_machine_order_distance": float(stagnation_details["avg_machine_order_distance"]),
            "avg_pairwise_ranking_distance": float(stagnation_details["avg_ranking_distance"]),
            "fe_counter": int(fe_counter),
            "iter_time_sec": float(iter_time),
            "total_time_sec": float(total_time),
        }
        diagnostics.append(log_entry)

        if verbose:
            print(
                f"Iterasi {t+1}/{max_iter} | "
                f"Best: {gBestScore:.2f} | "
                f"MachineFam: {log_entry['unique_machine_order_families']} | "
                f"MachDist: {log_entry['avg_pairwise_machine_order_distance']:.3f} | "
                f"RankDist: {log_entry['avg_pairwise_ranking_distance']:.3f} | "
                f"Reinit: {log_entry['reinitialized']} | "
                f"KDR: {log_entry['kdr_reinitialized']} | "
                f"KDRMode: {log_entry['kdr_mode']} | "
                f"KDRPairs: {log_entry['kdr_stable_pairs']} | "
                f"SSR: {log_entry['ssr_active']} | "
                f"Reason: {log_entry['ssr_activation_reason']} | "
                f"FEs: {fe_counter} | "
                f"Iter time: {iter_time:.4f}s | "
                f"Total time: {total_time:.2f}s"
            )

    outputs = (gBestScore, gBest, np.array(cg_curve), np.array(avg_curve))
    if return_diagnostics:
        extra = {
            "logs": diagnostics,
            "low_dyn": low_dyn.copy(),
            "up_dyn": up_dyn.copy(),
            "elite_archive": elite_archive,
            "kdr_enabled": bool(kdr_reinitializers),
            "normal_kdr_config": normal_kdr_config,
            "ssr_kdr_config": ssr_kdr_config,
            "best_score_history": best_score_history,
        }
        return outputs + (extra,)
    return outputs
