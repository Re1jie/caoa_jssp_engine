import heapq
import itertools

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
        lookahead: float = 3.0
    ):
        df_ops = df_ops.sort_values(['job_id', 'voyage', 'op_seq']).reset_index(drop=True)

        self.lookahead = lookahead
        self.n_ops = len(df_ops)

        self.jobs = list(
            df_ops[['job_id','voyage']]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        self._op_data = {}
        self._job_n_ops = {}

        for _, row in df_ops.iterrows():
            j, v, o = int(row['job_id']), int(row['voyage']), int(row['op_seq'])
            self._op_data[(j, v, o)] = {
                'machine_id': int(row['machine_id']),
                'A_lj': float(row['A_lj']),
                'p_lj': float(row['p_lj']),
                'TSail_lj': float(row['TSail_lj']) if pd.notna(row['TSail_lj']) else 0.0,
                'voyage': int(row['voyage']),
                'ship_name': row['ship_name'] if 'ship_name' in row.index else None,
            }

        for j, v in self.jobs:
            mask = (df_ops['job_id']==j) & (df_ops['voyage']==v)
            self._job_n_ops[(j,v)] = df_ops[mask]['op_seq'].max()+1

        self._machine_capacity = dict(zip(
            df_machine_master['machine_id'].astype(int),
            df_machine_master['num_berth'].astype(int)
        ))

        self._tidal = tidal_checker

        self._job_target = {}
        if df_job_target is not None:
            for _, row in df_job_target.iterrows():
                key = (int(row['job_id']), int(row['voyage']))
                self._job_target[key] = {
                    'target_time': float(row['T_j']),
                }

        self.L_ref = [
            (int(r.job_id), int(r.voyage), int(r.op_seq))
            for r in df_ops.itertuples()
        ]

        self._first_op = {}
        for (j, v, o) in self._op_data:
            key = (j, v)
            if key not in self._first_op or o < self._first_op[key]:
                self._first_op[key] = o


    def decode_from_continuous(self, X: np.ndarray):

        # priority mapping
        priority_map = {
            key: X[i]
            for i, key in enumerate(self.L_ref)
        }

        counter = itertools.count()
        events = []
        machine_waiting_pool = {m: [] for m in self._machine_capacity}
        machine_active = {m: 0 for m in self._machine_capacity}

        # seed first ops
        for j, v in self.jobs:
            first_o = self._first_op[(j, v)]
            op = self._op_data[(j, v, first_o)]
            heapq.heappush(
                events,
                (op['A_lj'], 0, next(counter), j, v, first_o, op['A_lj'])
            )

        results = []
        while events:
            t, ev_type, _, job_id, voyage, op_seq, arrival_h = heapq.heappop(events)

            # arrival event
            if ev_type == 0:
                op = self._op_data[(job_id, voyage, op_seq)]
                m = op['machine_id']
                machine_waiting_pool[m].append({
                    'job_id': job_id,
                    'voyage': voyage,
                    'op_seq': op_seq,
                    'A_lj': float(arrival_h),
                })

            # completion event
            else:
                op = self._op_data[(job_id, voyage, op_seq)]
                m = op['machine_id']
                machine_active[m] -= 1
                
                if op_seq + 1 < self._job_n_ops[(job_id, voyage)]:
                    next_A = t + op['TSail_lj']
                    heapq.heappush(
                        events,
                        (next_A, 0, next(counter), job_id, voyage, op_seq + 1, next_A)
                    )

            # GLOBAL ACTIVE DISPATCH
            # DISPATCH UNTIL NO CANDIDATES LEFT

            while True:
                global_candidates = []

                for m_id in self._machine_capacity:

                    if machine_active[m_id] >= self._machine_capacity[m_id]:
                        continue

                    for x in machine_waiting_pool[m_id]:
                        if x['A_lj'] <= t + self.lookahead:
                            global_candidates.append((m_id, x))

                if not global_candidates:
                    break

                m_id, winner = min(
                    global_candidates,
                    key=lambda item: priority_map[
                        (item[1]['job_id'], item[1]['voyage'], item[1]['op_seq'])
                    ]
                )

                # remove
                for i, item in enumerate(machine_waiting_pool[m_id]):
                    if (
                        item['job_id'] == winner['job_id']
                        and item['voyage'] == winner['voyage']
                        and item['op_seq'] == winner['op_seq']
                    ):
                        machine_waiting_pool[m_id].pop(i)
                        break

                w_job = winner['job_id']
                w_v = winner['voyage']
                w_op = winner['op_seq']
                w_arrival = winner['A_lj']

                op_data = self._op_data[(w_job, w_v, w_op)]
                w_p = op_data['p_lj']
                ship_name = op_data['ship_name']

                s = max(t, w_arrival)
                tidal_wait = 0.0

                if self._tidal and self._tidal.has_tidal_constraint(m_id, ship_name):
                    feasible = self._tidal.find_next_start(
                        m_id,
                        s,
                        w_p,
                        ship_name=ship_name,
                    )
                    if feasible == float("inf"):
                        schedule_df = self._build_schedule_df(results)
                        metrics = build_infeasible_metrics(
                            reason=(
                                "tidal_window_not_found:"
                                f"machine_id={m_id},job_id={w_job},voyage={w_v},op_seq={w_op}"
                            )
                        )
                        return schedule_df, metrics

                    tidal_wait = feasible - s
                    s = feasible

                c = s + w_p
                machine_active[m_id] += 1

                results.append({
                    'job_id': w_job,
                    'voyage': w_v,
                    'machine_id': m_id,
                    'op_seq': w_op,
                    'A_lj': w_arrival,
                    'S_lj': s,
                    'C_lj': c,
                    'p_lj': w_p,
                    'tidal_wait': tidal_wait,
                    'congestion_wait': max(0.0, s - w_arrival - tidal_wait)
                })

                heapq.heappush(
                    events,
                    (c, 1, next(counter), w_job, w_v, w_op, w_arrival)
                )

        schedule_df = self._build_schedule_df(results)

        metrics = self._compute_metrics(schedule_df)

        return schedule_df, metrics


    def fitness(self, X: np.ndarray) -> float:
        _, metrics = self.decode_from_continuous(X)
        return metrics['total_tardiness']


    def _compute_metrics(self, schedule_df: pd.DataFrame) -> dict:
        return compute_schedule_metrics(schedule_df, self._job_target)


    @staticmethod
    def _build_schedule_df(results: list[dict]) -> pd.DataFrame:
        columns = [
            'job_id',
            'voyage',
            'machine_id',
            'op_seq',
            'A_lj',
            'S_lj',
            'C_lj',
            'p_lj',
            'tidal_wait',
            'congestion_wait',
        ]
        if not results:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(results, columns=columns).sort_values(
            ['job_id', 'voyage', 'op_seq']
        ).reset_index(drop=True)


    def get_dimension(self) -> int:
        return self.n_ops
