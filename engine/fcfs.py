import pandas as pd
import numpy as np
import heapq
import itertools
from engine.metrics import compute_schedule_metrics
from engine.tidal_checker import TidalChecker

def run_fcfs_baseline(
    df_ops: pd.DataFrame, 
    df_machine_master: pd.DataFrame, 
    df_job_target: pd.DataFrame,
    tidal_checker=None
) -> tuple[pd.DataFrame, dict]:
    
    jobs = df_ops['job_id'].unique()
    machines = df_machine_master['machine_id'].unique()

    job_ops = {}
    for j in jobs:
        ops = df_ops[df_ops['job_id'] == j].sort_values('op_seq').to_dict('records')
        job_ops[j] = ops

    events = []
    counter = itertools.count()

    # Inisialisasi Event Kedatangan Pertama
    for j in jobs:
        if len(job_ops[j]) > 0:
            first_op = job_ops[j][0]
            heapq.heappush(events, (first_op['A_lj'], 0, next(counter), j, 0))

    machine_queues = {m: [] for m in machines}
    machine_capacity = dict(zip(df_machine_master['machine_id'], df_machine_master['num_berth']))
    machine_active_berths = {m: 0 for m in machines}
    results = []

    # Mesin Event-Driven FCFS dengan Tidal Injection
    while events:
        t, ev_type, _, job_id, op_seq = heapq.heappop(events)
        op = job_ops[job_id][op_seq]
        m = op['machine_id']
        p_lj = op['p_lj']
        tsail = op['TSail_lj']

        if ev_type == 0:  # Kedatangan kapal
            if machine_active_berths.get(m, 0) < machine_capacity.get(m, 1):
                machine_active_berths[m] = machine_active_berths.get(m, 0) + 1
                
                # Definisi waktu awal
                earliest_start = t 
                
                # Cek pasang surut
                if tidal_checker is not None and tidal_checker.has_tidal_constraint(m):
                    feasible_start = tidal_checker.find_next_start(m, earliest_start, p_lj)
                    if feasible_start == float("inf"):
                        s_lj = earliest_start
                        tidal_wait = 0.0
                    else:
                        s_lj = feasible_start
                        tidal_wait = s_lj - earliest_start
                else:
                    s_lj = earliest_start
                    tidal_wait = 0.0

                c_lj = s_lj + p_lj
                results.append({
                    'job_id': job_id, 'voyage': op['voyage'], 'machine_id': m, 'op_seq': op_seq,
                    'A_lj': t, 'S_lj': s_lj, 'C_lj': c_lj, 
                    'tidal_wait': tidal_wait,
                    'congestion_wait': max(0.0, s_lj - t - tidal_wait)
                })
                heapq.heappush(events, (c_lj, 1, next(counter), job_id, op_seq))
            else:
                machine_queues[m].append((t, job_id, op_seq))

        elif ev_type == 1:  # Selesai sandar
            next_op_seq = op_seq + 1
            if next_op_seq < len(job_ops[job_id]):
                next_arrival = t + tsail if not np.isnan(tsail) else t
                heapq.heappush(events, (next_arrival, 0, next(counter), job_id, next_op_seq))

            if machine_queues[m]:
                arr_t, next_job_id, next_job_op_seq = machine_queues[m].pop(0)
                next_p = job_ops[next_job_id][next_job_op_seq]['p_lj']
                
                # Definisi waktu awal antrean
                earliest_start = max(t, arr_t)
                
                # Cek pasang surut
                if tidal_checker is not None and tidal_checker.has_tidal_constraint(m):
                    feasible_start = tidal_checker.find_next_start(m, earliest_start, next_p)
                    if feasible_start == float("inf"):
                        s_lj = earliest_start
                        tidal_wait = 0.0
                    else:
                        s_lj = feasible_start
                        tidal_wait = s_lj - earliest_start
                else:
                    s_lj = earliest_start
                    tidal_wait = 0.0

                c_lj = s_lj + next_p
                
                results.append({
                    'job_id': next_job_id, 'voyage': job_ops[next_job_id][next_job_op_seq]['voyage'], 
                    'machine_id': m, 'op_seq': next_job_op_seq,
                    'A_lj': arr_t, 'S_lj': s_lj, 'C_lj': c_lj, 
                    'tidal_wait': tidal_wait,
                    'congestion_wait': max(0.0, s_lj - arr_t - tidal_wait)
                })
                heapq.heappush(events, (c_lj, 1, next(counter), next_job_id, next_job_op_seq))
            else:
                machine_active_berths[m] -= 1

    schedule_df = pd.DataFrame(results).sort_values(['job_id', 'op_seq']).reset_index(drop=True)
    metrics = _compute_fcfs_metrics(schedule_df, df_job_target)
    
    return schedule_df, metrics

def _compute_fcfs_metrics(schedule_df: pd.DataFrame, df_job_target: pd.DataFrame) -> dict:
    target_dict = {
        (int(row['job_id']), int(row['voyage'])): float(row['T_j'])
        for _, row in df_job_target.iterrows()
    }

    return compute_schedule_metrics(schedule_df, target_dict)
