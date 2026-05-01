import numpy as np
import pandas as pd

from engine.metrics import build_infeasible_metrics, compute_schedule_metrics
from engine.tidal_checker import TidalChecker


class ActiveScheduleDecoder:
    def __init__(
        self,
        df_ops: pd.DataFrame,
        df_machine_master: pd.DataFrame,
        tidal_checker: TidalChecker | None = None,
        df_job_target: pd.DataFrame | None = None,
    ):
        df_ops = df_ops.sort_values(["job_id", "voyage", "op_seq"]).reset_index(drop=True)

        self.n_ops = len(df_ops)

        self.jobs = list(
            df_ops[["job_id", "voyage"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        self._op_data = {}
        self._job_n_ops = {}

        for _, row in df_ops.iterrows():
            j, v, o = int(row["job_id"]), int(row["voyage"]), int(row["op_seq"])
            self._op_data[(j, v, o)] = {
                "machine_id": int(row["machine_id"]),
                "A_lj": float(row["A_lj"]),
                "p_lj": float(row["p_lj"]),
                "TSail_lj": float(row["TSail_lj"]) if pd.notna(row["TSail_lj"]) else 0.0,
                "voyage": int(row["voyage"]),
                "ship_name": row["ship_name"] if "ship_name" in row.index else None,
            }

        for j, v in self.jobs:
            mask = (df_ops["job_id"] == j) & (df_ops["voyage"] == v)
            self._job_n_ops[(j, v)] = int(df_ops[mask]["op_seq"].max()) + 1

        self._machine_capacity = dict(
            zip(
                df_machine_master["machine_id"].astype(int),
                df_machine_master["num_berth"].astype(int),
            )
        )

        self._tidal = tidal_checker

        self._job_target = {}
        if df_job_target is not None:
            for _, row in df_job_target.iterrows():
                key = (int(row["job_id"]), int(row["voyage"]))
                self._job_target[key] = {
                    "target_time": float(row["T_j"]),
                }

        self.L_ref = [
            (int(r.job_id), int(r.voyage), int(r.op_seq))
            for r in df_ops.itertuples()
        ]

        self._first_op = {}
        for j, v, o in self._op_data:
            key = (j, v)
            if key not in self._first_op or o < self._first_op[key]:
                self._first_op[key] = o

    def decode_from_continuous(self, X: np.ndarray):
        priority_map = {
            key: X[i]
            for i, key in enumerate(self.L_ref)
        }

        machine_schedules = {m: [] for m in self._machine_capacity}
        scheduled_ops = {}
        next_op_by_job = {
            job_key: self._first_op[job_key]
            for job_key in self.jobs
        }
        results = []

        while len(scheduled_ops) < self.n_ops:
            eligible_ops = []

            for job_key, op_seq in next_op_by_job.items():
                if op_seq is None:
                    continue

                arrival_h = self._get_release_time(job_key, op_seq, scheduled_ops)
                if arrival_h is None:
                    continue

                job_id, voyage = job_key
                eligible_ops.append(
                    (priority_map[(job_id, voyage, op_seq)], arrival_h, job_id, voyage, op_seq)
                )

            if not eligible_ops:
                schedule_df = self._build_schedule_df(results)
                metrics = build_infeasible_metrics(
                    reason="no_eligible_operation_found",
                )
                return schedule_df, metrics

            _, arrival_h, job_id, voyage, op_seq = min(
                eligible_ops,
                key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
            )

            op_data = self._op_data[(job_id, voyage, op_seq)]
            machine_id = op_data["machine_id"]
            duration_h = op_data["p_lj"]
            ship_name = op_data["ship_name"]

            capacity_ready = self._find_earliest_feasible_slot(
                machine_id=machine_id,
                release_h=arrival_h,
                duration_h=duration_h,
                ship_name=ship_name,
                machine_schedules=machine_schedules,
                use_tidal=False,
            )

            start_h = self._find_earliest_feasible_slot(
                machine_id=machine_id,
                release_h=arrival_h,
                duration_h=duration_h,
                ship_name=ship_name,
                machine_schedules=machine_schedules,
                use_tidal=True,
            )

            if start_h == float("inf"):
                schedule_df = self._build_schedule_df(results)
                metrics = build_infeasible_metrics(
                    reason=(
                        "tidal_window_not_found:"
                        f"machine_id={machine_id},job_id={job_id},voyage={voyage},op_seq={op_seq}"
                    )
                )
                return schedule_df, metrics

            completion_h = start_h + duration_h

            interval = (start_h, completion_h, job_id, voyage, op_seq)
            machine_schedules[machine_id].append(interval)
            machine_schedules[machine_id].sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))

            scheduled_ops[(job_id, voyage, op_seq)] = {
                "A_lj": arrival_h,
                "S_lj": start_h,
                "C_lj": completion_h,
            }

            next_seq = op_seq + 1
            if next_seq < self._job_n_ops[(job_id, voyage)]:
                next_op_by_job[(job_id, voyage)] = next_seq
            else:
                next_op_by_job[(job_id, voyage)] = None

            results.append(
                {
                    "job_id": job_id,
                    "voyage": voyage,
                    "machine_id": machine_id,
                    "op_seq": op_seq,
                    "A_lj": arrival_h,
                    "S_lj": start_h,
                    "C_lj": completion_h,
                    "p_lj": duration_h,
                    "tidal_wait": max(0.0, start_h - capacity_ready),
                    "congestion_wait": max(0.0, capacity_ready - arrival_h),
                }
            )

        schedule_df = self._build_schedule_df(results)
        metrics = self._compute_metrics(schedule_df)
        return schedule_df, metrics

    def fitness(self, X: np.ndarray) -> float:
        _, metrics = self.decode_from_continuous(X)
        return metrics["total_tardiness"]

    def _compute_metrics(self, schedule_df: pd.DataFrame) -> dict:
        return compute_schedule_metrics(schedule_df, self._job_target)

    def _get_release_time(
        self,
        job_key: tuple[int, int],
        op_seq: int,
        scheduled_ops: dict[tuple[int, int, int], dict[str, float]],
    ) -> float | None:
        job_id, voyage = job_key

        if op_seq == self._first_op[job_key]:
            return self._op_data[(job_id, voyage, op_seq)]["A_lj"]

        prev_key = (job_id, voyage, op_seq - 1)
        prev_result = scheduled_ops.get(prev_key)
        if prev_result is None:
            return None

        prev_op = self._op_data[prev_key]
        return prev_result["C_lj"] + prev_op["TSail_lj"]

    def _find_earliest_feasible_slot(
        self,
        machine_id: int,
        release_h: float,
        duration_h: float,
        ship_name: str | None,
        machine_schedules: dict[int, list[tuple[float, float, int, int, int]]],
        use_tidal: bool,
    ) -> float:
        timeline = machine_schedules[machine_id]

        for base_start in self._build_machine_candidates(timeline, release_h):
            candidate_start = base_start

            if use_tidal and self._tidal and self._tidal.has_tidal_constraint(machine_id, ship_name):
                candidate_start = self._tidal.find_next_start(
                    machine_id,
                    candidate_start,
                    duration_h,
                    ship_name=ship_name,
                )
                if candidate_start == float("inf"):
                    return float("inf")

            if self._has_capacity_for_interval(
                timeline=timeline,
                start_h=candidate_start,
                end_h=candidate_start + duration_h,
                capacity=self._machine_capacity[machine_id],
            ):
                return candidate_start

        if use_tidal and self._tidal and self._tidal.has_tidal_constraint(machine_id, ship_name):
            return float("inf")

        return release_h

    @staticmethod
    def _build_machine_candidates(
        timeline: list[tuple[float, float, int, int, int]],
        release_h: float,
    ) -> list[float]:
        candidates = {float(release_h)}
        for _, end_h, _, _, _ in timeline:
            if end_h >= release_h:
                candidates.add(float(end_h))
        return sorted(candidates)

    @staticmethod
    def _has_capacity_for_interval(
        timeline: list[tuple[float, float, int, int, int]],
        start_h: float,
        end_h: float,
        capacity: int,
    ) -> bool:
        if end_h <= start_h:
            return True

        overlap_events = []
        for interval_start, interval_end, _, _, _ in timeline:
            if interval_end <= start_h or interval_start >= end_h:
                continue
            overlap_events.append((max(start_h, interval_start), 1))
            overlap_events.append((min(end_h, interval_end), -1))

        if not overlap_events:
            return True

        overlap_events.sort(key=lambda item: (item[0], item[1]))

        active = 0
        prev_time = start_h

        for event_time, delta in overlap_events:
            if prev_time < event_time and active >= capacity:
                return False
            active += delta
            prev_time = event_time

        if prev_time < end_h and active >= capacity:
            return False

        return True

    @staticmethod
    def _build_schedule_df(results: list[dict]) -> pd.DataFrame:
        columns = [
            "job_id",
            "voyage",
            "machine_id",
            "op_seq",
            "A_lj",
            "S_lj",
            "C_lj",
            "p_lj",
            "tidal_wait",
            "congestion_wait",
        ]
        if not results:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(results, columns=columns).sort_values(
            ["job_id", "voyage", "op_seq"]
        ).reset_index(drop=True)

    def get_dimension(self) -> int:
        return self.n_ops
