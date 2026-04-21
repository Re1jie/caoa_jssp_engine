import numpy as np
import pandas as pd
import heapq
import itertools
from engine.metrics import compute_schedule_metrics
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
        self.n_voyages = len(self.jobs)

        self._op_data = {}
        self._job_n_ops = {}

        for _, row in df_ops.iterrows():
            j, v, o = int(row['job_id']), int(row['voyage']), int(row['op_seq'])
            self._op_data[(j, v, o)] = {
                'machine_id': int(row['machine_id']),
                'A_lj': float(row['A_lj']),
                'p_lj': float(row['p_lj']),
                'TSail_lj': float(row['TSail_lj']) if pd.notna(row['TSail_lj']) else 0.0,
                'voyage': int(row['voyage'])
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
        min_required_by_job = (
            df_ops.sort_values(['job_id', 'voyage', 'op_seq'])
            .groupby(['job_id', 'voyage'], as_index=False)
            .apply(
                lambda group: pd.Series({
                    'total_processing_time': float(group['p_lj'].sum()),
                    'total_sailing_time': float(group['TSail_lj'].iloc[:-1].sum()),
                }),
                include_groups=False,
            )
        )
        min_required_lookup = {
            (int(row['job_id']), int(row['voyage'])): (
                float(row['total_processing_time']) + float(row['total_sailing_time'])
            )
            for _, row in min_required_by_job.iterrows()
        }
        if df_job_target is not None:
            for _, row in df_job_target.iterrows():
                key = (int(row['job_id']), int(row['voyage']))
                self._job_target[key] = {
                    'target_time': float(row['T_j']),
                    'weight': float(row['w_j']) if 'w_j' in row.index and pd.notna(row['w_j']) else 1.0,
                    'min_required_time': min_required_lookup.get(key, 0.0),
                }

        # Dimensi direduksi menjadi per voyage
        self.V_ref = self.jobs

        self._first_op = {}
        for (j, v, o) in self._op_data:
            key = (j, v)
            if key not in self._first_op or o < self._first_op[key]:
                self._first_op[key] = o


    def decode_from_continuous(self, X: np.ndarray):

        # priority mapping per voyage
        priority_map = {
            key: X[i]
            for i, key in enumerate(self.V_ref)
        }

        counter = itertools.count()
        events = []
        machine_waiting_pool = {m: [] for m in self._machine_capacity}
        machine_active = {m: 0 for m in self._machine_capacity}

        # Seed first ops
        for j, v in self.jobs:
            first_o = self._first_op[(j, v)]
            op = self._op_data[(j, v, first_o)]
            m = op['machine_id']
            # Put in pool immediately to expose to lookahead
            machine_waiting_pool[m].append({
                'job_id': j,
                'voyage': v,
                'op_seq': first_o,
                'A_lj': op['A_lj']
            })
            heapq.heappush(events, (op['A_lj'], 0, next(counter), j, v, first_o))

        results = []
        conflict_resolved_count = 0

        while events:
            t, ev_type, _, job_id, voyage, op_seq = heapq.heappop(events)

            # arrival event
            if ev_type == 0:
                pass

            # completion event
            else:
                op = self._op_data[(job_id, voyage, op_seq)]
                m = op['machine_id']
                machine_active[m] -= 1
                
                if op_seq + 1 < self._job_n_ops[(job_id, voyage)]:
                    next_A = t + op['TSail_lj']
                    next_op = self._op_data[(job_id, voyage, op_seq + 1)]
                    next_m = next_op['machine_id']
                    
                    # Put in pool immediately to expose to lookahead
                    machine_waiting_pool[next_m].append({
                        'job_id': job_id,
                        'voyage': voyage,
                        'op_seq': op_seq + 1,
                        'A_lj': next_A
                    })
                    heapq.heappush(
                        events,
                        (next_A, 0, next(counter), job_id, voyage, op_seq + 1)
                    )

            # Global active dispatch
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

                # Pilih kandidat dengan nilai prioritas voyage terendah
                m_id, winner = min(
                    global_candidates,
                    key=lambda item: priority_map[
                        (item[1]['job_id'], item[1]['voyage'])
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

                conflict_resolved_count += 1

                w_job = winner['job_id']
                w_v = winner['voyage']
                w_op = winner['op_seq']
                w_arrival = winner['A_lj']

                w_p = self._op_data[(w_job, w_v, w_op)]['p_lj']

                s = max(t, w_arrival)
                tidal_wait = 0.0

                if self._tidal and self._tidal.has_tidal_constraint(m_id):
                    feasible = self._tidal.find_next_start(m_id, s, w_p)
                    if feasible != float("inf"):
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
                    (c, 1, next(counter), w_job, w_v, w_op)
                )

        schedule_df = (
            pd.DataFrame(results)
            .sort_values(['job_id', 'voyage', 'op_seq'])
            .reset_index(drop=True)
        )

        metrics = self._compute_metrics(schedule_df)
        metrics['conflict_resolved_count'] = conflict_resolved_count

        return schedule_df, metrics


    def fitness(self, X: np.ndarray) -> float:
        _, metrics = self.decode_from_continuous(X)
        return metrics['weighted_avg_tardiness']


    def _compute_metrics(self, schedule_df: pd.DataFrame) -> dict:
        return compute_schedule_metrics(schedule_df, self._job_target)


    def get_dimension(self) -> int:
        return self.n_voyages
