import json
from pathlib import Path
import numpy as np
from engine.caoassr import CAOA_SSR
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

    timetable_path = output_path / "caoassr_optimized_timetable.csv"
    metrics_path = output_path / "caoassr_optimized_metrics.json"
    position_path = output_path / "caoassr_best_position.npy"

    schedule_df.to_csv(timetable_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    np.save(position_path, best_position)

    return timetable_path, metrics_path, position_path


def save_baseline_results(schedule_df, metrics, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timetable_path = output_path / "fcfs_baseline_timetable.csv"
    metrics_path = output_path / "fcfs_baseline_metrics.json"

    schedule_df.to_csv(timetable_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    return timetable_path, metrics_path


def save_voyage_debug_report(debug_df, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_path = output_path / "caoassr_voyage_debug_report.csv"
    summary_path = output_path / "caoassr_voyage_debug_summary.json"

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


def _get_rdk_logs(rdk_info):
    if not isinstance(rdk_info, dict):
        return []
    logs = rdk_info.get("logs")
    if logs is None:
        logs = rdk_info.get("diagnostics", [])
    return logs if isinstance(logs, list) else []


def _sum_log_key(logs, key):
    total = 0
    for item in logs:
        try:
            total += int(item.get(key, 0))
        except Exception:
            pass
    return total


def build_rdk_diagnostic_summary(rdk_info):
    logs = _get_rdk_logs(rdk_info)
    guidance_history = (
        rdk_info.get("rdk_guidance_history", [])
        if isinstance(rdk_info, dict)
        else []
    )

    return {
        "log_count": int(len(logs)),
        "rdk_checks": int(len(guidance_history)),
        "inline_reduced_dim_total": _sum_log_key(logs, "inline_reduced_dim_count"),
        "inline_explore_dim_total": _sum_log_key(logs, "inline_explore_dim_count"),
        "inline_knowledge_reinit_total": _sum_log_key(logs, "inline_knowledge_reinit_count"),
        "rdk_guidance_total": (
            _sum_log_key(logs, "inline_reduced_dim_count")
            + _sum_log_key(logs, "inline_explore_dim_count")
            + _sum_log_key(logs, "inline_knowledge_reinit_count")
        ),
    }


def save_rdk_diagnostics(rdk_info, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logs_path = output_path / "caoassr_rdk_logs.json"
    summary_path = output_path / "caoassr_rdk_summary.json"

    logs = _get_rdk_logs(rdk_info)
    logs_path.write_text(json.dumps(logs, indent=4), encoding="utf-8")
    summary_path.write_text(
        json.dumps(build_rdk_diagnostic_summary(rdk_info), indent=4),
        encoding="utf-8",
    )

    return logs_path, summary_path


def build_schedule_comparison(
    baseline_schedule_df,
    optimized_schedule_df,
    df_job_target,
):
    baseline_voyage_df = build_voyage_debug_report(
        baseline_schedule_df,
        df_job_target,
    ).rename(
        columns={
            "last_completion_hour": "baseline_last_completion_hour",
            "tardiness_hours": "baseline_tardiness_hours",
            "lateness_hours": "baseline_lateness_hours",
            "earliness_hours": "baseline_earliness_hours",
            "debug_status": "baseline_status",
        }
    )
    optimized_voyage_df = build_voyage_debug_report(
        optimized_schedule_df,
        df_job_target,
    ).rename(
        columns={
            "last_completion_hour": "optimized_last_completion_hour",
            "tardiness_hours": "optimized_tardiness_hours",
            "lateness_hours": "optimized_lateness_hours",
            "earliness_hours": "optimized_earliness_hours",
            "debug_status": "optimized_status",
        }
    )

    voyage_comparison_df = baseline_voyage_df.merge(
        optimized_voyage_df[
            [
                "job_id",
                "voyage",
                "optimized_last_completion_hour",
                "optimized_tardiness_hours",
                "optimized_lateness_hours",
                "optimized_earliness_hours",
                "optimized_status",
            ]
        ],
        on=["job_id", "voyage"],
        how="inner",
    )
    voyage_comparison_df["delta_completion_hours"] = (
        voyage_comparison_df["optimized_last_completion_hour"]
        - voyage_comparison_df["baseline_last_completion_hour"]
    )
    voyage_comparison_df["delta_tardiness_hours"] = (
        voyage_comparison_df["optimized_tardiness_hours"]
        - voyage_comparison_df["baseline_tardiness_hours"]
    )
    voyage_comparison_df["delta_lateness_hours"] = (
        voyage_comparison_df["optimized_lateness_hours"]
        - voyage_comparison_df["baseline_lateness_hours"]
    )
    voyage_comparison_df["status_transition"] = (
        voyage_comparison_df["baseline_status"]
        + " -> "
        + voyage_comparison_df["optimized_status"]
    )

    baseline_ops_df = baseline_schedule_df.rename(
        columns={
            "A_lj": "baseline_A_lj",
            "S_lj": "baseline_S_lj",
            "C_lj": "baseline_C_lj",
            "p_lj": "baseline_p_lj",
            "TSail_lj": "baseline_TSail_lj",
            "tidal_wait": "baseline_tidal_wait",
            "congestion_wait": "baseline_congestion_wait",
        }
    )
    optimized_ops_df = optimized_schedule_df.rename(
        columns={
            "A_lj": "optimized_A_lj",
            "S_lj": "optimized_S_lj",
            "C_lj": "optimized_C_lj",
            "p_lj": "optimized_p_lj",
            "TSail_lj": "optimized_TSail_lj",
            "tidal_wait": "optimized_tidal_wait",
            "congestion_wait": "optimized_congestion_wait",
        }
    )

    operation_comparison_df = baseline_ops_df.merge(
        optimized_ops_df,
        on=["job_id", "voyage", "machine_id", "op_seq"],
        how="inner",
    )
    operation_comparison_df["delta_start_hours"] = (
        operation_comparison_df["optimized_S_lj"]
        - operation_comparison_df["baseline_S_lj"]
    )
    operation_comparison_df["delta_completion_hours"] = (
        operation_comparison_df["optimized_C_lj"]
        - operation_comparison_df["baseline_C_lj"]
    )
    operation_comparison_df["delta_tidal_wait_hours"] = (
        operation_comparison_df["optimized_tidal_wait"]
        - operation_comparison_df["baseline_tidal_wait"]
    )
    operation_comparison_df["delta_congestion_wait_hours"] = (
        operation_comparison_df["optimized_congestion_wait"]
        - operation_comparison_df["baseline_congestion_wait"]
    )

    return voyage_comparison_df, operation_comparison_df


def save_schedule_comparison(
    voyage_comparison_df,
    operation_comparison_df,
    output_dir="data/result",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voyage_path = output_path / "fcfs_vs_caoassr_voyage_comparison.csv"
    operation_path = output_path / "fcfs_vs_caoassr_operation_comparison.csv"

    voyage_comparison_df.to_csv(voyage_path, index=False)
    operation_comparison_df.to_csv(operation_path, index=False)

    return voyage_path, operation_path


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
    np.random.seed(451)
    
    # Load Data & Init
    df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    dim = len(df_ops)
    print(f"[Dimensi] {dim}")
    tidal_checker = TidalChecker()

    # Baseline FCFS
    fcfs_schedule_df, fcfs_metrics = run_fcfs_baseline(
        df_ops,
        df_machine_master,
        df_job_target,
        tidal_checker,
    )
    ensure_feasible(fcfs_metrics, "FCFS baseline")
    fcfs_timetable_path, fcfs_metrics_path = save_baseline_results(
        fcfs_schedule_df,
        fcfs_metrics,
    )

    # Optimization CAOA
    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target
    )

    caoa_params = {
        'N': 10, 'max_iter': 100, 'lb': 0.0, 'ub': 1.0, 'dim': dim,
        'alpha': 0.79, 'beta': 0.08,
        'gamma': 0.05, 'delta': 19.48,
        'initial_energy': 10
    }

    it_period = 10
    caoa_rdk_params = {
        'IT': it_period,
        'elite_size': 5,
        'ssr_elite_k': 5,
        'ssr_min_knowledge_signal_ratio': 0.80,
        'ssr_knowledge_noise_scale': 0.12,
        'ssr_knowledge_min_noise_scale': 0.02,
        'ssr_knowledge_max_confidence': 0.85,
        'ssr_knowledge_uniform_mix': 0.20,
        'ssr_inline_guidance': True,
        'ssr_inline_prob': 0.20,
        'ssr_inline_confidence_threshold': 0.70,
        'ssr_inline_reduced_dim_ratio': 0.08,
        'ssr_inline_reduced_blend': 0.25,
        'ssr_inline_explore_dim_ratio': 0.05,
        'ssr_inline_reinit_uses_knowledge': True,
        'ssr_balanced_reinit': True,
        'ssr_explore_opposition_ratio': 0.50,
        'ssr_reduction_min_width': 0.05,
        'ssr_reduction_width_scale': 2.0,
    }

    _, best_position, _, _, rdk_info = CAOA_SSR(
        **caoa_params,
        decoder=decoder,
        return_diagnostics=True,
        **caoa_rdk_params,
    )

    caoa_schedule_df, caoa_metrics = decoder.decode_from_continuous(best_position)
    ensure_feasible(caoa_metrics, "CAOASSR")
    timetable_path, metrics_path, position_path = save_optimized_results(
        caoa_schedule_df,
        caoa_metrics,
        best_position,
    )
    rdk_logs_path, rdk_summary_path = save_rdk_diagnostics(rdk_info)
    voyage_debug_df = build_voyage_debug_report(
        caoa_schedule_df,
        df_job_target,
    )
    voyage_debug_path, voyage_summary_path = save_voyage_debug_report(voyage_debug_df)
    voyage_comparison_df, operation_comparison_df = build_schedule_comparison(
        fcfs_schedule_df,
        caoa_schedule_df,
        df_job_target,
    )
    voyage_comparison_path, operation_comparison_path = save_schedule_comparison(
        voyage_comparison_df,
        operation_comparison_df,
    )

    # 4. Reporting
    print(f"\n{'METRIK':<25} | {'FCFS':<15} | {'CAOASSR':<15}")
    print("-" * 60)
    
    metrics_list = [
        ('Total Tardiness)', 'total_tardiness'),
        ('Max Tardiness', 'max_tardiness'),
    ]

    for label, key in metrics_list:
        print(f"{label:<25} | {fcfs_metrics.get(key, 0):<15.2f} | {caoa_metrics.get(key, 0):<15.2f}")

    print("\nFile hasil optimasi:")
    print(f"- FCFS TT   : {fcfs_timetable_path}")
    print(f"- FCFS Met  : {fcfs_metrics_path}")
    print(f"- Timetable : {timetable_path}")
    print(f"- Metrics   : {metrics_path}")
    print(f"- Best pos  : {position_path}")
    print(f"- RDK logs  : {rdk_logs_path}")
    print(f"- RDK sum   : {rdk_summary_path}")
    print(f"- Debug CSV : {voyage_debug_path}")
    print(f"- Debug sum : {voyage_summary_path}")
    print(f"- Compare V : {voyage_comparison_path}")
    print(f"- Compare O : {operation_comparison_path}")
    rdk_summary = build_rdk_diagnostic_summary(rdk_info)
    print(f"- RDK checks: {rdk_summary['rdk_checks']}")
    print(
        "- RDK R/D/K: "
        f"{rdk_summary['inline_reduced_dim_total']}/"
        f"{rdk_summary['inline_explore_dim_total']}/"
        f"{rdk_summary['inline_knowledge_reinit_total']}"
    )
    print(f"- RDK total : {rdk_summary['rdk_guidance_total']}")

if __name__ == "__main__":
    main()
