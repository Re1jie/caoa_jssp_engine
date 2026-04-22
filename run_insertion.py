import json
from pathlib import Path
import numpy as np
from engine.caoa import CAOA
from engine.decoder_insertion import ActiveScheduleDecoder
from engine.fcfs import run_fcfs_baseline
from engine.tidal_checker import TidalChecker
from utils.data_loader import load_real_jssp_data

def objective_function(X, decoder):
    _, metrics = decoder.decode_from_continuous(X)
    return metrics['total_tardiness']


def build_voyage_debug_report(schedule_df, df_job_target):
    actual_summary = (
        schedule_df.groupby(['job_id', 'voyage'], as_index=False)
        .agg(
            first_arrival_hour=('A_lj', 'min'),
            last_completion_hour=('C_lj', 'max'),
        )
    )

    target_summary = df_job_target.rename(columns={'T_j': 'due_window_hours'}).copy()

    debug_df = (
        actual_summary
        .merge(
            target_summary[['job_id', 'voyage', 'due_window_hours']],
            on=['job_id', 'voyage'],
            how='left',
        )
    )

    debug_df['due_hour_absolute'] = (
        debug_df['first_arrival_hour'] + debug_df['due_window_hours']
    )
    debug_df['tardiness_hours'] = (
        debug_df['last_completion_hour'] - debug_df['due_hour_absolute']
    ).clip(lower=0.0)
    debug_df['lateness_hours'] = (
        debug_df['last_completion_hour'] - debug_df['due_hour_absolute']
    )
    debug_df['earliness_hours'] = (-debug_df['lateness_hours']).clip(lower=0.0)
    debug_df['debug_status'] = np.where(
        debug_df['tardiness_hours'] > 0,
        'LATE',
        'ON_TIME',
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
        "total_due_window_hours": float(debug_df['due_window_hours'].sum()),
        "total_tardiness_hours": float(debug_df['tardiness_hours'].sum()),
        "max_tardiness_hours": float(debug_df['tardiness_hours'].max()),
        "avg_tardiness_hours": float(debug_df['tardiness_hours'].mean()),
        "max_earliness_hours": float(debug_df['earliness_hours'].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    return debug_path, summary_path


def ensure_feasible(metrics, label: str) -> None:
    if metrics.get('is_feasible', True):
        return
    reason = metrics.get('infeasible_reason', 'unknown')
    penalty = metrics.get('penalty_tardiness', metrics.get('total_tardiness', 0.0))
    raise RuntimeError(
        f"{label} menghasilkan schedule infeasible. "
        f"reason={reason} | penalty_tardiness={penalty}"
    )

def main():
    np.random.seed(42)
    
    # Load Data & Init
    df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    dim = len(df_ops)
    print(f"[Dimensi] {dim}")
    tidal_checker = TidalChecker()

    # Baseline FCFS
    _, fcfs_metrics = run_fcfs_baseline(df_ops, df_machine_master, df_job_target, tidal_checker)
    ensure_feasible(fcfs_metrics, "FCFS baseline")

    # Optimization CAOA
    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target
    )

    caoa_params = {
        'N': 20, 'max_iter': 100, 'lb': 0.0, 'ub': 1.0, 'dim': dim,
        'alpha': 0.9, 'beta': 0.1,
        'gamma': 0.07, 'delta': 1.2,
        'initial_energy': 150
    }

    _, best_position, _, _ = CAOA(
        **caoa_params,
        fobj=lambda X: objective_function(X, decoder)
    )

    caoa_schedule_df, caoa_metrics = decoder.decode_from_continuous(best_position)
    ensure_feasible(caoa_metrics, "CAOA")
    timetable_path, metrics_path, position_path = save_optimized_results(
        caoa_schedule_df,
        caoa_metrics,
        best_position,
    )
    voyage_debug_df = build_voyage_debug_report(
        caoa_schedule_df,
        df_job_target,
    )
    voyage_debug_path, voyage_summary_path = save_voyage_debug_report(voyage_debug_df)

    # 4. Reporting
    print(f"\n{'METRIK':<25} | {'FCFS':<15} | {'CAOA':<15}")
    print("-" * 60)
    
    metrics_list = [
        ('Total Tardiness)', 'total_tardiness'),
        ('Max Tardiness', 'max_tardiness'),
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
