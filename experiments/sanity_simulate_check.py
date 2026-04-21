import pandas as pd
import numpy as np
import engine.encoder as encoder
from engine.caoa import CAOA
from preprocessing.data_slicer import slice_data_by_operations
import heapq
import itertools

# --- 1. LOAD DATA ---
df_ops = pd.read_csv("data/processed/jssp_data_sliced.csv")
df_machines = pd.read_csv("data/processed/machine_master.csv")
df_targets = pd.read_csv("data/processed/job_target_time.csv")

# --- 2. PRE-COMPUTATION (Eksekusi 1 kali sebelum iterasi CAOA) ---
num_jobs = df_ops['job_id'].nunique()
num_machines = df_machines['machine_id'].nunique()

# Buat priority_ops_list (Pemetaan array CAOA X ke tuple (job_id, op_seq))
priority_ops_list = []
for _, row in df_ops.iterrows():
    priority_ops_list.append((int(row['job_id']), int(row['op_seq'])))

machines_list = df_machines['machine_id'].unique()

# Buat kapasitas berth per mesin
# machine_berths[machine_id] = jumlah berth
machine_berths = df_machines.set_index('machine_id')['num_berth'].to_dict()

# Hitung jumlah operasi per job untuk tahu kapan operasi terakhir dan pertama
job_max_seq = df_ops.groupby('job_id')['op_seq'].max().to_dict()
job_min_seq = df_ops.groupby('job_id')['op_seq'].min().to_dict()

# Gabungkan target T_j ke job_id
target_dict = df_targets.set_index('job_id')['T_j'].to_dict()

# Bangun dictionary ops_data untuk akses O(1)
ops_data = {}
for _, row in df_ops.iterrows():
    j = int(row['job_id'])
    seq = int(row['op_seq'])
    
    # Ambil A_initial hanya untuk operasi pertama (seq == job_min_seq)
    min_seq = job_min_seq[j]
    a_init = row['A_lj'] if seq == min_seq else 0.0
    
    ops_data[(j, seq)] = {
        'machine': int(row['machine_id']),
        'p': row['p_lj'],
        'tsail': row['TSail_lj'],
        'A_initial': a_init
    }

def evaluate_fitness_priority(X, priority_ops_list, ops_data, job_min_seq, job_max_seq, target_dict, num_jobs, machines_list, machine_berths):
    priority_dict = {priority_ops_list[i]: X[i] for i in range(len(X))}
    
    events = []
    counter = itertools.count()
    
    # Inisialisasi event kedatangan pertama untuk semua job
    for j in range(num_jobs):
        seq = job_min_seq.get(j, 0)
        initial_time = ops_data[(j, seq)]['A_initial']
        prio = priority_dict.get((j, seq), 0)
        heapq.heappush(events, (initial_time, prio, 0, next(counter), j, seq))
        
    machine_queues = {m: [] for m in machines_list}
    machine_active_berths = {m: 0 for m in machines_list}
    
    total_congestion_delay = 0.0
    
    while len(events) > 0:
        t, prio, ev_type, _, job_id, op_seq = heapq.heappop(events)
        op = ops_data[(job_id, op_seq)]
        m = op['machine']
        p_lj = op['p']
        tsail = op['tsail']
        
        if ev_type == 0: # ARRIVAL
            if machine_active_berths[m] < machine_berths.get(m, 1):
                machine_active_berths[m] += 1
                completion_time = t + p_lj
                
                # Push COMPLETION event (prioritas sangat rendah / dijamin jalan sebelum ARRIVAL simultan)
                heapq.heappush(events, (completion_time, -float('inf'), 1, next(counter), job_id, op_seq))
            else:
                machine_queues[m].append((t, job_id, op_seq))
                
        elif ev_type == 1: # COMPLETION
            # 1. Jadwalkan operasi berikutnya untuk kapal yang baru selesai
            next_op_seq = op_seq + 1
            if (job_id, next_op_seq) in ops_data:
                next_arr_time = t + tsail if not np.isnan(tsail) else t
                next_prio = priority_dict.get((job_id, next_op_seq), 0)
                heapq.heappush(events, (next_arr_time, next_prio, 0, next(counter), job_id, next_op_seq))
                
            # 2. Ambil kapal dari antrean yang menunggu di pelabuhan m
            if len(machine_queues[m]) > 0:
                best_idx = 0
                best_prio = float('inf')
                for i, (q_t, q_j, q_seq) in enumerate(machine_queues[m]):
                    p_val = priority_dict.get((q_j, q_seq), 0)
                    if p_val < best_prio:
                        best_prio = p_val
                        best_idx = i
                
                arr_t, next_job_id, next_job_op_seq = machine_queues[m].pop(best_idx)
                
                # Akumulasi antrean (Congestion Delay)
                wait_time = t - arr_t
                total_congestion_delay += wait_time
                
                next_op = ops_data[(next_job_id, next_job_op_seq)]
                
                completion_time = t + next_op['p']
                    
                heapq.heappush(events, (completion_time, -float('inf'), 1, next(counter), next_job_id, next_job_op_seq))
            else:
                machine_active_berths[m] -= 1
                
    return total_congestion_delay

def objective_function(X):
    """
    Fungsi ini yang akan dilempar ke algoritma CAOA.
    Menjembatani vektor kontinu X menjadi fitness score.
    """
    fitness_score = evaluate_fitness_priority(
        X, priority_ops_list, ops_data, job_min_seq, job_max_seq, target_dict, num_jobs, machines_list, machine_berths
    )
    
    return fitness_score

if __name__ == "__main__":
    # Dimensi ruang pencarian adalah total seluruh operasi yang harus dilakukan
    D = len(priority_ops_list) 
    
    # Batas ruang pencarian untuk teknik ROV (Continuous domain)
    # Gunakan rentang yang simetris, misal [-5, 5] atau [-10, 10]
    lb = -5.0
    ub = 5.0
    
    # Parameter Metaheuristik (Mulai dengan ukuran kecil untuk sanity check)
    N_pop = 200
    Max_iter = 1000
    
    print(f"Memulai Optimasi JSSP dengan CAOA...")
    print(f"Total Kapal: {num_jobs}, Total Pelabuhan: {num_machines}, Total Operasi (Dimensi): {D}")
    
    # Eksekusi Solver
    best_fitness, best_X, convergence_curve = CAOA(
        N=N_pop, 
        max_iter=Max_iter, 
        lb=lb, 
        ub=ub, 
        dim=D, 
        fobj=objective_function
    )
    
    print("\n=== HASIL OPTIMASI ===")
    print(f"Total Congestion Delay Terbaik : {best_fitness} Jam")