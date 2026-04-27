from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


OperationKey = tuple[int, int, int]


@dataclass
class ScheduleKnowledgeModel:
    """Compact knowledge extracted from elite decoded schedules."""

    operation_scores: dict[OperationKey, float] = field(default_factory=dict)
    pairwise_precedence: dict[tuple[OperationKey, OperationKey], float] = field(default_factory=dict)
    stable_pairs: dict[tuple[OperationKey, OperationKey], float] = field(default_factory=dict)
    stable_predecessors: dict[OperationKey, list[tuple[OperationKey, float]]] = field(default_factory=dict)
    stable_successors: dict[OperationKey, list[tuple[OperationKey, float]]] = field(default_factory=dict)
    bottleneck_scores: dict[OperationKey, float] = field(default_factory=dict)
    elite_count: int = 0
    used_fallback: bool = False


class KnowledgeDrivenReinitializer:
    """Generate new random-key agents from elite schedule structure."""

    def __init__(
        self,
        operation_reference: list[OperationKey],
        dim: int,
        stability_threshold: float = 0.75,
        noise_scale: float = 0.03,
        uniform_mix: float = 0.15,
        bottleneck_perturbation: float = 0.25,
        pair_window: int = 12,
    ) -> None:
        self.operation_reference = [tuple(map(int, op)) for op in operation_reference]
        self.dim = int(dim)
        self.stability_threshold = float(stability_threshold)
        self.noise_scale = float(noise_scale)
        self.uniform_mix = float(uniform_mix)
        self.bottleneck_perturbation = float(bottleneck_perturbation)
        self.pair_window = max(1, int(pair_window))
        self.op_to_dim = {
            op_key: idx for idx, op_key in enumerate(self.operation_reference[: self.dim])
        }

    @staticmethod
    def _normalize_weights(fitness: list[float] | None, count: int) -> np.ndarray:
        if not fitness or len(fitness) != count:
            return np.ones(count, dtype=float) / max(1, count)

        arr = np.array(fitness, dtype=float)
        arr = arr - np.min(arr)
        weights = 1.0 / (1.0 + arr)
        total = float(np.sum(weights))
        if total <= 0.0:
            return np.ones(count, dtype=float) / max(1, count)
        return weights / total

    @staticmethod
    def _build_operation_key(row: Any) -> OperationKey:
        return (int(row.job_id), int(row.voyage), int(row.op_seq))

    def extract_knowledge(
        self,
        elite_schedules: list[pd.DataFrame],
        elite_positions: list[np.ndarray] | None = None,
        fitness: list[float] | None = None,
    ) -> ScheduleKnowledgeModel:
        """Learn operation priority and pairwise precedence from elite schedules."""

        valid_schedules = [
            schedule_df
            for schedule_df in elite_schedules
            if schedule_df is not None and not schedule_df.empty
        ]
        if not valid_schedules:
            return ScheduleKnowledgeModel(used_fallback=True)

        weights = self._normalize_weights(fitness, len(valid_schedules))
        weighted_ranks: dict[OperationKey, float] = {}
        rank_weights: dict[OperationKey, float] = {}
        pair_counts: dict[tuple[OperationKey, OperationKey], float] = {}
        pair_totals: dict[frozenset[OperationKey], float] = {}
        wait_scores: dict[OperationKey, float] = {}

        for schedule_idx, schedule_df in enumerate(valid_schedules):
            weight = float(weights[schedule_idx])
            ordered = schedule_df.sort_values(
                ["S_lj", "C_lj", "machine_id", "job_id", "voyage", "op_seq"]
            ).reset_index(drop=True)

            for rank, row in enumerate(ordered.itertuples(index=False)):
                op_key = self._build_operation_key(row)
                weighted_ranks[op_key] = weighted_ranks.get(op_key, 0.0) + weight * rank
                rank_weights[op_key] = rank_weights.get(op_key, 0.0) + weight
                wait_score = float(
                    getattr(row, "tidal_wait", 0.0) + getattr(row, "congestion_wait", 0.0)
                )
                wait_scores[op_key] = wait_scores.get(op_key, 0.0) + weight * wait_score

            for _, machine_df in ordered.groupby("machine_id", sort=False):
                machine_order = [
                    self._build_operation_key(row)
                    for row in machine_df.itertuples(index=False)
                ]
                for i in range(len(machine_order)):
                    upper = min(len(machine_order), i + 1 + self.pair_window)
                    for j in range(i + 1, upper):
                        op_i = machine_order[i]
                        op_j = machine_order[j]
                        pair_counts[(op_i, op_j)] = pair_counts.get((op_i, op_j), 0.0) + weight
                        pair_totals[frozenset((op_i, op_j))] = (
                            pair_totals.get(frozenset((op_i, op_j)), 0.0) + weight
                        )

        operation_scores = {
            op_key: weighted_ranks[op_key] / max(rank_weights.get(op_key, 1.0), 1e-12)
            for op_key in weighted_ranks
        }

        pairwise_precedence: dict[tuple[OperationKey, OperationKey], float] = {}
        stable_pairs: dict[tuple[OperationKey, OperationKey], float] = {}
        stable_predecessors: dict[OperationKey, list[tuple[OperationKey, float]]] = {}
        stable_successors: dict[OperationKey, list[tuple[OperationKey, float]]] = {}
        for pair_key, forward_weight in pair_counts.items():
            op_i, op_j = pair_key
            total_weight = pair_totals.get(frozenset((op_i, op_j)), 0.0)
            if total_weight <= 0.0:
                continue
            probability = forward_weight / total_weight
            pairwise_precedence[(op_i, op_j)] = probability
            pairwise_precedence[(op_j, op_i)] = 1.0 - probability
            if probability >= self.stability_threshold:
                stable_pairs[(op_i, op_j)] = probability
                stable_successors.setdefault(op_i, []).append((op_j, probability))
                stable_predecessors.setdefault(op_j, []).append((op_i, probability))
            elif (1.0 - probability) >= self.stability_threshold:
                stable_pairs[(op_j, op_i)] = 1.0 - probability
                stable_successors.setdefault(op_j, []).append((op_i, 1.0 - probability))
                stable_predecessors.setdefault(op_i, []).append((op_j, 1.0 - probability))

        bottleneck_scores = {}
        if wait_scores:
            max_wait = max(wait_scores.values())
            if max_wait > 1e-12:
                bottleneck_scores = {
                    op_key: min(1.0, wait / max_wait)
                    for op_key, wait in wait_scores.items()
                }

        return ScheduleKnowledgeModel(
            operation_scores=operation_scores,
            pairwise_precedence=pairwise_precedence,
            stable_pairs=stable_pairs,
            stable_predecessors=stable_predecessors,
            stable_successors=stable_successors,
            bottleneck_scores=bottleneck_scores,
            elite_count=len(valid_schedules),
            used_fallback=False,
        )

    def generate_sequence_from_knowledge(
        self,
        knowledge: ScheduleKnowledgeModel,
        random_state: np.random.Generator | None = None,
    ) -> list[OperationKey]:
        """Sample a soft-biased operation sequence and repair job precedence."""

        rng = random_state if random_state is not None else np.random.default_rng()
        if not knowledge.operation_scores:
            sequence = list(self.operation_reference[: self.dim])
            rng.shuffle(sequence)
            return self._repair_job_precedence(sequence)

        base_scores = {}
        max_rank = max(knowledge.operation_scores.values()) if knowledge.operation_scores else 0.0
        for order_idx, op_key in enumerate(self.operation_reference[: self.dim]):
            fallback_rank = max_rank + float(order_idx)
            score = knowledge.operation_scores.get(op_key, fallback_rank)

            for other_key, probability in knowledge.stable_successors.get(op_key, ()):
                score -= 0.15 * probability
                if other_key not in base_scores:
                    continue

            for other_key, probability in knowledge.stable_predecessors.get(op_key, ()):
                score += 0.15 * probability
                if other_key not in base_scores:
                    continue

            bottleneck = knowledge.bottleneck_scores.get(op_key, 0.0)
            score += rng.normal(0.0, self.noise_scale * (1.0 + self.bottleneck_perturbation * bottleneck))
            base_scores[op_key] = score

        sequence = sorted(base_scores, key=lambda op: (base_scores[op], op[0], op[1], op[2]))
        return self._repair_job_precedence(sequence)

    @staticmethod
    def _repair_job_precedence(sequence: list[OperationKey]) -> list[OperationKey]:
        grouped: dict[tuple[int, int], list[OperationKey]] = {}
        for op_key in sequence:
            grouped.setdefault((op_key[0], op_key[1]), []).append(op_key)

        for job_key in grouped:
            grouped[job_key].sort(key=lambda op: op[2])

        repaired = []
        next_required = {job_key: 0 for job_key in grouped}
        remaining = set(sequence)

        while remaining:
            progressed = False
            for op_key in sequence:
                if op_key not in remaining:
                    continue
                job_key = (op_key[0], op_key[1])
                required_seq = grouped[job_key][next_required[job_key]][2]
                if op_key[2] != required_seq:
                    continue
                repaired.append(op_key)
                remaining.remove(op_key)
                next_required[job_key] += 1
                progressed = True
            if progressed:
                continue

            fallback = min(remaining, key=lambda op: (op[2], op[0], op[1]))
            repaired.append(fallback)
            remaining.remove(fallback)
            next_required[(fallback[0], fallback[1])] += 1

        return repaired

    def sequence_to_random_keys(
        self,
        sequence: list[OperationKey],
        low_dyn: np.ndarray,
        up_dyn: np.ndarray,
        noise_scale: float | None = None,
        uniform_mix: float | None = None,
        bottleneck_scores: dict[OperationKey, float] | None = None,
        random_state: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Convert a priority sequence into continuous random keys."""

        rng = random_state if random_state is not None else np.random.default_rng()
        sigma = self.noise_scale if noise_scale is None else float(noise_scale)
        mix_rate = self.uniform_mix if uniform_mix is None else float(uniform_mix)
        interval = np.maximum(up_dyn - low_dyn, 1e-12)
        x = low_dyn + interval * rng.random(self.dim)

        if not sequence:
            return np.clip(x, low_dyn, up_dyn)

        denom = max(1, len(sequence) - 1)
        for rank, op_key in enumerate(sequence):
            dim_idx = self.op_to_dim.get(op_key)
            if dim_idx is None or dim_idx >= self.dim:
                continue
            base = rank / denom
            local_sigma = sigma * interval[dim_idx]
            if bottleneck_scores:
                local_sigma *= 1.0 + self.bottleneck_perturbation * bottleneck_scores.get(op_key, 0.0)
            x[dim_idx] = low_dyn[dim_idx] + base * interval[dim_idx] + rng.normal(0.0, local_sigma)

        if mix_rate > 0.0:
            mix_mask = rng.random(self.dim) < mix_rate
            x[mix_mask] = low_dyn[mix_mask] + interval[mix_mask] * rng.random(np.sum(mix_mask))

        return np.clip(x, low_dyn, up_dyn)

    def reinitialize_agents(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        candidate_indices: list[int],
        elite_schedules: list[pd.DataFrame],
        low_dyn: np.ndarray,
        up_dyn: np.ndarray,
        elite_positions: list[np.ndarray] | None = None,
        elite_fitness: list[float] | None = None,
        noise_scale: float | None = None,
        uniform_mix: float | None = None,
        knowledge: ScheduleKnowledgeModel | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        """Generate new positions for selected agents using elite schedule knowledge."""

        if not candidate_indices:
            return {}, {
                "used": False,
                "reason": "empty_candidate_indices",
                "reinitialized": 0,
                "elite_count": 0,
                "stable_pair_count": 0,
                "used_fallback": True,
            }

        if knowledge is None:
            knowledge = self.extract_knowledge(
                elite_schedules=elite_schedules,
                elite_positions=elite_positions,
                fitness=elite_fitness,
            )
        if knowledge.used_fallback:
            return {}, {
                "used": False,
                "reason": "empty_knowledge",
                "reinitialized": 0,
                "elite_count": 0,
                "stable_pair_count": 0,
                "used_fallback": True,
            }

        replacements: dict[int, np.ndarray] = {}
        for idx in candidate_indices:
            sequence = self.generate_sequence_from_knowledge(knowledge)
            replacements[int(idx)] = self.sequence_to_random_keys(
                sequence=sequence,
                low_dyn=low_dyn,
                up_dyn=up_dyn,
                noise_scale=noise_scale,
                uniform_mix=uniform_mix,
                bottleneck_scores=knowledge.bottleneck_scores,
            )

        info = {
            "used": True,
            "reason": "knowledge_driven",
            "reinitialized": len(replacements),
            "elite_count": knowledge.elite_count,
            "stable_pair_count": len(knowledge.stable_pairs),
            "used_fallback": False,
            "avg_candidate_fitness": float(np.mean(fitness[candidate_indices])) if len(candidate_indices) > 0 else None,
        }
        return replacements, info
