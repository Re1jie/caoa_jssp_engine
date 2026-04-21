import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils.data_loader import load_real_jssp_data
from engine.fcfs import run_fcfs_baseline
from engine.decoder import ActiveScheduleDecoder
from engine.caoa import CAOA
from engine.tidal_checker import TidalChecker

def objective_function(X, decoder):
    _, metrics = decoder.decode_from_continuous(X)
    return metrics['weighted_avg_tardiness']


def build_voyage_debug_report(schedule_df, df_ops, df_job_target):
    op_summary = (
        df_ops.sort_values(['job_id', 'voyage', 'op_seq'])
        .groupby(['job_id', 'voyage'], as_index=False)
        .apply(
            lambda group: pd.Series({
                'planned_operation_hours': float(group['p_lj'].sum()),
                'planned_sailing_hours': float(group['TSail_lj'].iloc[:-1].sum()),
                'operation_count': int(group['op_seq'].count()),
            }),
            include_groups=False,
        )
    )

    actual_summary = (
        schedule_df.groupby(['job_id', 'voyage'], as_index=False)
        .agg(
            first_arrival_hour=('A_lj', 'min'),
            first_start_hour=('S_lj', 'min'),
            last_completion_hour=('C_lj', 'max'),
            actual_operation_hours=('p_lj', 'sum'),
            total_tidal_wait_hours=('tidal_wait', 'sum'),
            total_congestion_wait_hours=('congestion_wait', 'sum'),
        )
    )

    target_summary = df_job_target.rename(columns={'T_j': 'due_window_hours'}).copy()

    debug_df = (
        actual_summary
        .merge(op_summary, on=['job_id', 'voyage'], how='left')
        .merge(
            target_summary[['job_id', 'voyage', 'due_window_hours']],
            on=['job_id', 'voyage'],
            how='left',
        )
    )

    debug_df['planned_sailing_hours'] = debug_df['planned_sailing_hours'].fillna(0.0)
    debug_df['actual_flow_time_hours'] = (
        debug_df['last_completion_hour'] - debug_df['first_arrival_hour']
    )
    debug_df['waiting_before_first_service_hours'] = (
        debug_df['first_start_hour'] - debug_df['first_arrival_hour']
    ).clip(lower=0.0)
    debug_df['total_wait_hours'] = (
        debug_df['total_tidal_wait_hours'] + debug_df['total_congestion_wait_hours']
    )
    debug_df['due_hour_absolute'] = (
        debug_df['first_arrival_hour'] + debug_df['due_window_hours']
    )
    debug_df['min_required_hours'] = (
        debug_df['planned_operation_hours'] + debug_df['planned_sailing_hours']
    )
    debug_df['slack_hours'] = (
        debug_df['due_window_hours'] - debug_df['min_required_hours']
    )
    debug_df['lateness_hours'] = (
        debug_df['last_completion_hour'] - debug_df['due_hour_absolute']
    )
    debug_df['raw_tardiness_hours'] = debug_df['lateness_hours'].clip(lower=0.0)
    debug_df['unavoidable_tardiness_hours'] = (-debug_df['slack_hours']).clip(lower=0.0)
    debug_df['tardiness_hours'] = (
        debug_df['raw_tardiness_hours'] - debug_df['unavoidable_tardiness_hours']
    ).clip(lower=0.0)
    debug_df['earliness_hours'] = (-debug_df['lateness_hours']).clip(lower=0.0)
    debug_df['slack_to_due_hours'] = (
        debug_df['due_hour_absolute'] - debug_df['last_completion_hour']
    )
    debug_df['operating_vs_due_ratio'] = (
        debug_df['actual_flow_time_hours'] / debug_df['due_window_hours']
    )
    debug_df['processing_vs_due_ratio'] = (
        debug_df['actual_operation_hours'] / debug_df['due_window_hours']
    )
    debug_df['waiting_vs_flow_ratio'] = np.where(
        debug_df['actual_flow_time_hours'] > 0,
        debug_df['total_wait_hours'] / debug_df['actual_flow_time_hours'],
        0.0,
    )
    debug_df['planned_total_work_hours'] = debug_df['min_required_hours']
    debug_df['actual_nonprocessing_hours'] = (
        debug_df['actual_flow_time_hours'] - debug_df['actual_operation_hours']
    )
    debug_df['debug_status'] = np.where(
        debug_df['tardiness_hours'] > 0,
        'LATE',
        'ON_TIME',
    )
    debug_df['target_feasibility'] = np.where(
        debug_df['slack_hours'] >= 0,
        'FEASIBLE_TARGET',
        'INFEASIBLE_TARGET',
    )

    return debug_df.sort_values(
        ['tardiness_hours', 'job_id', 'voyage'],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def save_optimized_results(schedule_df, metrics, best_position, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timetable_path = output_path / "caoa_optimized_timetable.csv"
    metrics_path = output_path / "caoa_optimized_metrics.json"
    position_path = output_path / "caoa_best_position.npy"

    schedule_df.to_csv(timetable_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    np.save(position_path, best_position)

    return timetable_path, metrics_path, position_path


def save_voyage_debug_report(debug_df, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_path = output_path / "caoa_voyage_debug_report.csv"
    summary_path = output_path / "caoa_voyage_debug_summary.json"

    debug_df.to_csv(debug_path, index=False)

    summary = {
        "voyage_count": int(len(debug_df)),
        "late_voyage_count": int((debug_df['tardiness_hours'] > 0).sum()),
        "on_time_voyage_count": int((debug_df['tardiness_hours'] <= 0).sum()),
        "infeasible_target_count": int((debug_df['slack_hours'] < 0).sum()),
        "total_due_window_hours": float(debug_df['due_window_hours'].sum()),
        "total_min_required_hours": float(debug_df['min_required_hours'].sum()),
        "total_actual_flow_time_hours": float(debug_df['actual_flow_time_hours'].sum()),
        "total_actual_operation_hours": float(debug_df['actual_operation_hours'].sum()),
        "total_planned_sailing_hours": float(debug_df['planned_sailing_hours'].sum()),
        "total_wait_hours": float(debug_df['total_wait_hours'].sum()),
        "total_tardiness_hours": float(debug_df['tardiness_hours'].sum()),
        "total_raw_tardiness_hours": float(debug_df['raw_tardiness_hours'].sum()),
        "total_unavoidable_tardiness_hours": float(debug_df['unavoidable_tardiness_hours'].sum()),
        "max_tardiness_hours": float(debug_df['tardiness_hours'].max()),
        "avg_tardiness_hours": float(debug_df['tardiness_hours'].mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    return debug_path, summary_path

def main():
    np.random.seed(42)
    
    # Load Data & Init
    df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    dim = len(df_ops)
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
        'N': 20, 'max_iter': 200, 'lb': 0.0, 'ub': 1.0, 'dim': dim,
        'alpha': 0.9, 'beta': 0.1,
        'gamma': 0.07, 'delta': 1.2,
        'initial_energy': 150
    }

    best_score, best_position, cg_curve, avg_curve = CAOA(
        **caoa_params,
        fobj=lambda X: objective_function(X, decoder)
    )

    caoa_schedule_df, caoa_metrics = decoder.decode_from_continuous(best_position)
    timetable_path, metrics_path, position_path = save_optimized_results(
        caoa_schedule_df,
        caoa_metrics,
        best_position,
    )
    voyage_debug_df = build_voyage_debug_report(
        caoa_schedule_df,
        df_ops,
        df_job_target,
    )
    voyage_debug_path, voyage_summary_path = save_voyage_debug_report(voyage_debug_df)

    # 4. Reporting
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

    print("\nFile hasil optimasi:")
    print(f"- Timetable : {timetable_path}")
    print(f"- Metrics   : {metrics_path}")
    print(f"- Best pos  : {position_path}")
    print(f"- Debug CSV : {voyage_debug_path}")
    print(f"- Debug sum : {voyage_summary_path}")

if __name__ == "__main__":
    main()
