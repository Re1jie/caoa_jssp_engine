import pandas as pd
import numpy as np
import heapq
import itertools
from engine.metrics import build_infeasible_metrics, compute_schedule_metrics
from engine.tidal_checker import TidalChecker


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
        'TSail_lj',
        'tidal_wait',
        'congestion_wait',
    ]
    if not results:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(results, columns=columns).sort_values(
        ['job_id', 'voyage', 'op_seq']
    ).reset_index(drop=True)

def run_fcfs_baseline(
    df_ops: pd.DataFrame, 
    df_machine_master: pd.DataFrame, 
    df_job_target: pd.DataFrame,
    tidal_checker=None
) -> tuple[pd.DataFrame, dict]:
    
    jobs = list(
        df_ops[['job_id', 'voyage']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    machines = df_machine_master['machine_id'].unique()

    job_ops = {}
    for job_id, voyage in jobs:
        mask = (df_ops['job_id'] == job_id) & (df_ops['voyage'] == voyage)
        ops = df_ops[mask].sort_values('op_seq').to_dict('records')
        job_ops[(job_id, voyage)] = ops

    events = []
    counter = itertools.count()

    # Inisialisasi Event Kedatangan Pertama
    for job_key in jobs:
        if len(job_ops[job_key]) > 0:
            first_op = job_ops[job_key][0]
            heapq.heappush(events, (first_op['A_lj'], 0, next(counter), job_key, 0))

    machine_queues = {m: [] for m in machines}
    machine_capacity = dict(zip(df_machine_master['machine_id'], df_machine_master['num_berth']))
    machine_active_berths = {m: 0 for m in machines}
    results = []

    # Mesin Event-Driven FCFS dengan Tidal Injection
    while events:
        t, ev_type, _, job_key, op_seq = heapq.heappop(events)
        job_id, voyage = job_key
        op = job_ops[job_key][op_seq]
        m = op['machine_id']
        p_lj = op['p_lj']
        tsail = op['TSail_lj']
        ship_name = op.get('ship_name')

        if ev_type == 0:  # Kedatangan kapal
            if machine_active_berths.get(m, 0) < machine_capacity.get(m, 1):
                machine_active_berths[m] = machine_active_berths.get(m, 0) + 1
                
                # Definisi waktu awal
                earliest_start = t 
                
                # Cek pasang surut
                if tidal_checker is not None and tidal_checker.has_tidal_constraint(m, ship_name):
                    feasible_start = tidal_checker.find_next_start(
                        m,
                        earliest_start,
                        p_lj,
                        ship_name=ship_name,
                    )
                    if feasible_start == float("inf"):
                        schedule_df = _build_schedule_df(results)
                        return schedule_df, build_infeasible_metrics(
                            reason=(
                                "tidal_window_not_found:"
                                f"machine_id={m},job_id={job_id},voyage={voyage},op_seq={op_seq}"
                            )
                        )
                    s_lj = feasible_start
                    tidal_wait = s_lj - earliest_start
                else:
                    s_lj = earliest_start
                    tidal_wait = 0.0

                c_lj = s_lj + p_lj
                results.append({
                    'job_id': job_id, 'voyage': voyage, 'machine_id': m, 'op_seq': op_seq,
                    'A_lj': t, 'S_lj': s_lj, 'C_lj': c_lj, 
                    'p_lj': p_lj,
                    'TSail_lj': tsail if not np.isnan(tsail) else 0.0,
                    'tidal_wait': tidal_wait,
                    'congestion_wait': max(0.0, s_lj - t - tidal_wait)
                })
                heapq.heappush(events, (c_lj, 1, next(counter), job_key, op_seq))
            else:
                machine_queues[m].append((t, job_key, op_seq))

        elif ev_type == 1:  # Selesai sandar
            next_op_seq = op_seq + 1
            if next_op_seq < len(job_ops[job_key]):
                next_arrival = t + tsail if not np.isnan(tsail) else t
                heapq.heappush(events, (next_arrival, 0, next(counter), job_key, next_op_seq))

            if machine_queues[m]:
                arr_t, next_job_key, next_job_op_seq = machine_queues[m].pop(0)
                next_job_id, next_voyage = next_job_key
                next_p = job_ops[next_job_key][next_job_op_seq]['p_lj']
                next_ship_name = job_ops[next_job_key][next_job_op_seq].get('ship_name')
                
                # Definisi waktu awal antrean
                earliest_start = max(t, arr_t)
                
                # Cek pasang surut
                if tidal_checker is not None and tidal_checker.has_tidal_constraint(m, next_ship_name):
                    feasible_start = tidal_checker.find_next_start(
                        m,
                        earliest_start,
                        next_p,
                        ship_name=next_ship_name,
                    )
                    if feasible_start == float("inf"):
                        schedule_df = _build_schedule_df(results)
                        return schedule_df, build_infeasible_metrics(
                            reason=(
                                "tidal_window_not_found:"
                                f"machine_id={m},job_id={next_job_id},voyage={next_voyage},op_seq={next_job_op_seq}"
                            )
                        )
                    s_lj = feasible_start
                    tidal_wait = s_lj - earliest_start
                else:
                    s_lj = earliest_start
                    tidal_wait = 0.0

                c_lj = s_lj + next_p

                results.append({
                    'job_id': next_job_id, 'voyage': next_voyage,
                    'machine_id': m, 'op_seq': next_job_op_seq,
                    'A_lj': arr_t, 'S_lj': s_lj, 'C_lj': c_lj, 
                    'p_lj': next_p,
                    'TSail_lj': job_ops[next_job_key][next_job_op_seq]['TSail_lj']
                    if not np.isnan(job_ops[next_job_key][next_job_op_seq]['TSail_lj']) else 0.0,
                    'tidal_wait': tidal_wait,
                    'congestion_wait': max(0.0, s_lj - arr_t - tidal_wait)
                })
                heapq.heappush(events, (c_lj, 1, next(counter), next_job_key, next_job_op_seq))
            else:
                machine_active_berths[m] -= 1

    schedule_df = _build_schedule_df(results)
    metrics = _compute_fcfs_metrics(schedule_df, df_job_target)
    
    return schedule_df, metrics

def _compute_fcfs_metrics(schedule_df: pd.DataFrame, df_job_target: pd.DataFrame) -> dict:
    target_dict = {}
    for _, row in df_job_target.iterrows():
        key = (int(row['job_id']), int(row['voyage']))
        target_dict[key] = {
            'target_time': float(row['T_j']),
        }

    return compute_schedule_metrics(schedule_df, target_dict)
