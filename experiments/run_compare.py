import numpy as np
import pandas as pd
import heapq
import itertools
from engine.decoder import ActiveScheduleDecoder
from engine.caoa import CAOA

# ─── 1. GENERATOR DATA DUMMY ──────────────────────────────────────────────────
def generate_dummy_data(n_jobs, n_ops_per_job, n_machines):
    np.random.seed(2) # Seed dikunci agar hasil replikabel
    
    ops_data = []
    for j in range(n_jobs):
        release_time = np.random.uniform(0, 10) 
        for o in range(n_ops_per_job):
            p_lj = np.random.uniform(5, 15) 
            tsail_lj = np.random.uniform(10, 30) if o < (n_ops_per_job - 1) else 0.0
            
            ops_data.append({
                'job_id': j,
                'op_seq': o,
                'machine_id': np.random.randint(1, n_machines + 1),
                'A_lj': release_time if o == 0 else 0.0,
                'p_lj': p_lj,
                'TSail_lj': tsail_lj
            })
    df_ops = pd.DataFrame(ops_data)

    machine_data = []
    for m in range(1, n_machines + 1):
        machine_data.append({
            'machine_id': m,
            'num_berth': np.random.randint(1, 3) 
        })
    df_machine_master = pd.DataFrame(machine_data)

    target_data = []
    for j in range(n_jobs):
        job_ops = df_ops[df_ops['job_id'] == j]
        estimated_time = job_ops['p_lj'].sum() + job_ops['TSail_lj'].sum() + np.random.uniform(20, 50)
        target_data.append({
            'job_id': j,
            'T_j': estimated_time
        })
    df_job_target = pd.DataFrame(target_data)

    return df_ops, df_machine_master, df_job_target

# ─── 2. ENGINE SIMULATOR FCFS ─────────────────────────────────────────────────
def run_fcfs_baseline(df_ops, df_machine_master, df_job_target):
    jobs = df_ops['job_id'].unique()
    machines = df_machine_master['machine_id'].unique()

    job_ops = {}
    for j in jobs:
        ops = df_ops[df_ops['job_id'] == j].sort_values('op_seq').to_dict('records')
        job_ops[j] = ops

    events = []
    counter = itertools.count()

    # Inisialisasi kedatangan pertama
    for j in jobs:
        first_op = job_ops[j][0]
        heapq.heappush(events, (first_op['A_lj'], 0, next(counter), j, 0))

    machine_queues = {m: [] for m in machines}
    machine_capacity = dict(zip(df_machine_master['machine_id'], df_machine_master['num_berth']))
    machine_active_berths = {m: 0 for m in machines}
    results = []

    while events:
        t, ev_type, _, job_id, op_seq = heapq.heappop(events)
        op = job_ops[job_id][op_seq]
        m = op['machine_id']
        p_lj = op['p_lj']
        tsail = op['TSail_lj']

        if ev_type == 0:  # ARRIVAL
            if machine_active_berths[m] < machine_capacity.get(m, 1):
                machine_active_berths[m] += 1
                s_lj = t
                c_lj = s_lj + p_lj
                results.append({
                    'job_id': job_id, 'machine_id': m, 'op_seq': op_seq,
                    'A_lj': t, 'S_lj': s_lj, 'C_lj': c_lj, 'congestion_wait': 0.0
                })
                heapq.heappush(events, (c_lj, 1, next(counter), job_id, op_seq))
            else:
                machine_queues[m].append((t, job_id, op_seq))

        elif ev_type == 1:  # COMPLETION
            next_op_seq = op_seq + 1
            if next_op_seq < len(job_ops[job_id]):
                next_arrival = t + tsail if not np.isnan(tsail) else t
                heapq.heappush(events, (next_arrival, 0, next(counter), job_id, next_op_seq))

            if machine_queues[m]:
                arr_t, next_job_id, next_job_op_seq = machine_queues[m].pop(0)
                next_p = job_ops[next_job_id][next_job_op_seq]['p_lj']
                
                earliest = max(t, arr_t)
                s_lj = earliest
                c_lj = s_lj + next_p
                congestion_wait = s_lj - arr_t

                results.append({
                    'job_id': next_job_id, 'machine_id': m, 'op_seq': next_job_op_seq,
                    'A_lj': arr_t, 'S_lj': s_lj, 'C_lj': c_lj, 'congestion_wait': congestion_wait
                })
                heapq.heappush(events, (c_lj, 1, next(counter), next_job_id, next_job_op_seq))
            else:
                machine_active_berths[m] -= 1

    res_df = pd.DataFrame(results).sort_values(['job_id', 'op_seq']).reset_index(drop=True)
    
    # Hitung Metrik Evaluasi agar sepadan dengan CAOA
    makespan = res_df['C_lj'].max() - res_df['A_lj'].min()
    total_congestion = res_df['congestion_wait'].sum()
    
    first_ops = res_df.groupby('job_id').first()
    last_ops = res_df.groupby('job_id').last()
    
    tardiness_list = []
    for j in jobs:
        t_j = df_job_target[df_job_target['job_id'] == j]['T_j'].values[0]
        tardiness = max(0.0, last_ops.loc[j, 'C_lj'] - (first_ops.loc[j, 'A_lj'] + t_j))
        tardiness_list.append(tardiness)
        
    return {
        'makespan': makespan,
        'total_congestion': total_congestion,
        'total_tardiness': sum(tardiness_list),
        'avg_tardiness': np.mean(tardiness_list),
        'max_tardiness': max(tardiness_list)
    }

# ─── 3. MAIN EXECUTION & COMPARISON ───────────────────────────────────────────
def main():
    print("Membangun Data Dummy Universal...")
    df_ops, df_machine_master, df_job_target = generate_dummy_data(n_jobs=25, n_ops_per_job=10, n_machines=5)
    
    print("\n[1/2] Menjalankan Evaluasi FCFS Baseline...")
    fcfs_metrics = run_fcfs_baseline(df_ops, df_machine_master, df_job_target)
    
    print("\n[2/2] Menjalankan Crocodile Ambush Optimization Algorithm (CAOA)...")
    decoder = ActiveScheduleDecoder(
        df_ops=df_ops, df_machine_master=df_machine_master,
        tidal_checker=None, df_job_target=df_job_target
    )
    
    def objective_function(X):
        _, metrics = decoder.decode_from_continuous(X)
        return metrics['weighted_avg_tardiness']

    dim = decoder.get_dimension()
    N = 100
    max_iter = 1000
    
    best_score, best_position, _ = CAOA(
        N=N, max_iter=max_iter, lb=0.0, ub=1.0, dim=dim, fobj=objective_function
    )
    
    _, caoa_metrics = decoder.decode_from_continuous(best_position)

    # ─── 4. TABEL PERBANDINGAN KOMPREHENSIF ───────────────────────────────────
    print("\n" + "="*65)
    print(f"{'METRIK EVALUASI':<25} | {'FCFS (BASELINE)':<15} | {'CAOA (OPTIMAL)':<15}")
    print("="*65)
    print(f"{'Makespan':<25} | {fcfs_metrics['makespan']:<15.2f} | {caoa_metrics['makespan']:<15.2f}")
    print(f"{'Total Kongesti (Antrean)':<25} | {fcfs_metrics['total_congestion']:<15.2f} | {caoa_metrics['total_congestion']:<15.2f}")
    print(f"{'Total Tardiness':<25} | {fcfs_metrics['total_tardiness']:<15.2f} | {caoa_metrics['total_tardiness']:<15.2f}")
    print(f"{'Rata-rata Tardiness':<25} | {fcfs_metrics['avg_tardiness']:<15.2f} | {caoa_metrics['avg_tardiness']:<15.2f}")
    print(f"{'Maksimal Tardiness':<25} | {fcfs_metrics['max_tardiness']:<15.2f} | {caoa_metrics['max_tardiness']:<15.2f}")
    print("="*65)

if __name__ == "__main__":
    main()
