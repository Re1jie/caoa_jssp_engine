import numpy as np
from utils.data_loader import load_real_jssp_data
from engine.fcfs import run_fcfs_baseline
from engine.decoder_reduction import ActiveScheduleDecoder
from engine.caoa import CAOA
from engine.tidal_checker import TidalChecker

def objective_function(X, decoder):
    _, metrics = decoder.decode_from_continuous(X)
    return metrics['weighted_avg_tardiness']

def main():
    np.random.seed(42)
    
    # Load Data & Init
    df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    tidal_checker = TidalChecker()

    # Baseline FCFS
    _, fcfs_metrics = run_fcfs_baseline(df_ops, df_machine_master, df_job_target, tidal_checker)

    # Optimization CAOA
    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target
    )

    caoa_params = {
        'N': 20, 'max_iter': 200, 'lb': 0.0, 'ub': 1.0, 'dim': decoder.get_dimension(),
        'alpha': 0.98, 'beta': 0.12,
        'gamma': 0.07, 'delta': 1.22,
        'initial_energy': 166.98
    }

    best_score, best_position, cg_curve, avg_curve = CAOA(
        **caoa_params,
        fobj=lambda X: objective_function(X, decoder)
    )

    _, caoa_metrics = decoder.decode_from_continuous(best_position)

    # Reporting
    print(f"\n{'METRIK':<25} | {'FCFS':<15} | {'CAOA':<15}")
    print("-" * 60)
    
    metrics_list = [
        ('Objective (Adj Avg Tard.)', 'weighted_avg_tardiness'),
        ('Makespan (Jam)', 'makespan'),
        ('Total Kongesti (Jam)', 'total_congestion'),
        ('Total Delay Pasang (Jam)', 'total_tidal_delay'),
        ('Total Tardiness Adj (Jam)', 'total_tardiness'),
        ('Rata-rata Tardiness Adj', 'avg_tardiness'),
        ('Maksimal Tardiness Adj', 'max_tardiness'),
        ('Total Tardiness Raw', 'raw_total_tardiness'),
        ('Rata-rata Tardiness Raw', 'raw_avg_tardiness'),
        ('Total Unavoidable Tard.', 'total_unavoidable_tardiness'),
    ]

    for label, key in metrics_list:
        print(f"{label:<25} | {fcfs_metrics.get(key, 0):<15.2f} | {caoa_metrics.get(key, 0):<15.2f}")

if __name__ == "__main__":
    main()
