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


def _get_ssr_logs(ssr_info):
    if not isinstance(ssr_info, dict):
        return []
    logs = ssr_info.get("logs")
    if logs is None:
        logs = ssr_info.get("diagnostics", [])
    return logs if isinstance(logs, list) else []


def _sum_log_key(logs, key):
    total = 0
    for item in logs:
        try:
            total += int(item.get(key, 0))
        except Exception:
            pass
    return total


def build_ssr_diagnostic_summary(ssr_info):
    logs = _get_ssr_logs(ssr_info)
    best_score_history = (
        ssr_info.get("best_score_history", [])
        if isinstance(ssr_info, dict)
        else []
    )
    activation_reason_counts = {}
    search_mode_counts = {}
    for item in logs:
        reason = item.get("ssr_activation_reason")
        if reason:
            activation_reason_counts[reason] = activation_reason_counts.get(reason, 0) + 1
        mode = item.get("ssr_search_mode")
        if mode:
            search_mode_counts[mode] = search_mode_counts.get(mode, 0) + 1

    return {
        "log_count": int(len(logs)),
        "ssr_checks": int(len(best_score_history)),
        "ssr_active_count": int(
            sum(
                1
                for item in logs
                if bool(item.get("ssr_active", item.get("ssr_triggered", False)))
            )
        ),
        "ssr_replacement_total": _sum_log_key(logs, "ssr_replacement_count"),
        "ssr_candidate_attempt_total": _sum_log_key(logs, "ssr_candidate_attempt_count"),
        "ssr_rejected_candidate_total": _sum_log_key(logs, "ssr_rejected_candidate_count"),
        "knowledge_replacement_total": _sum_log_key(logs, "knowledge_replacement_count"),
        "reduced_space_replacement_total": _sum_log_key(logs, "reduced_space_replacement_count"),
        "diversity_replacement_total": _sum_log_key(logs, "diversity_replacement_count"),
        "inline_reduced_dim_total": _sum_log_key(logs, "inline_reduced_dim_count"),
        "inline_explore_dim_total": _sum_log_key(logs, "inline_explore_dim_count"),
        "inline_knowledge_reinit_total": _sum_log_key(logs, "inline_knowledge_reinit_count"),
        "activation_reason_counts": activation_reason_counts,
        "search_mode_counts": search_mode_counts,
    }


def save_ssr_diagnostics(ssr_info, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logs_path = output_path / "caoassr_ssr_logs.json"
    summary_path = output_path / "caoassr_ssr_summary.json"

    logs = _get_ssr_logs(ssr_info)
    logs_path.write_text(json.dumps(logs, indent=4), encoding="utf-8")
    summary_path.write_text(
        json.dumps(build_ssr_diagnostic_summary(ssr_info), indent=4),
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
    np.random.seed(42)
    
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
        'N': 10, 'max_iter': 200, 'lb': 0.0, 'ub': 1.0, 'dim': dim,
        'alpha': 0.79, 'beta': 0.07,
        'gamma': 0.06, 'delta': 13.12,
        'initial_energy': 10
    }

    it_period = 10
    caoa_ssr_params = {
        'IT': it_period,
        'K': it_period * 3,
        'stagnation_window': it_period * 3,
        'stagnation_patience': 1,
        'eps_improve': 1e-8,
        'elite_size': 5,
        'dup_ratio_threshold': 0.60,
        'unique_schedule_threshold': 0.25,
        'machine_family_threshold': 0.70,
        'ranking_collapse_threshold': 0.08,
        'structural_distance_threshold': 0.25,
        'use_machine_order_signature': True,
        'use_ranking_similarity': True,
        'stagnation_mode': 'rule',
        'partial_restart_ratio': 0.0,
        'ssr_elite_k': 5,
        'ssr_min_knowledge_signal_ratio': 0.80,
        'ssr_knowledge_noise_scale': 0.12,
        'ssr_knowledge_min_noise_scale': 0.02,
        'ssr_knowledge_max_confidence': 0.85,
        'ssr_knowledge_uniform_mix': 0.20,
        'ssr_allow_plateau_activation': True,
        'ssr_min_plateau_checks': 6,
        'ssr_candidate_trials': 3,
        'ssr_accept_only_improvement': True,
        'ssr_commit_requires_gbest_improvement': True,
        'ssr_inline_guidance': True,
        'ssr_inline_prob': 0.20,
        'ssr_inline_confidence_threshold': 0.70,
        'ssr_inline_reduced_dim_ratio': 0.08,
        'ssr_inline_reduced_blend': 0.25,
        'ssr_inline_explore_dim_ratio': 0.05,
        'ssr_inline_reinit_uses_knowledge': True,
        'ssr_random_fallback': False,
        'ssr_balanced_reinit': True,
        'ssr_explore_dim_ratio': 0.30,
        'ssr_explore_opposition_ratio': 0.50,
        'ssr_reduction_min_width': 0.05,
        'ssr_reduction_width_scale': 2.0,
        'ssr_reduced_gbest_pull': 0.40,
        'ssr_uncertain_uniform_ratio': 0.25,
        'ssr_force_mode_quota': False,
        'ssr_adaptive_mode': True,
        'ssr_escape_after_failed_activations': 1,
        'ssr_cooldown_checks': 1,
        'ssr_skip_last_checks': 1,
        'ssr_escape_reduction_width_multiplier': 1.8,
        'ssr_escape_noise_multiplier': 1.7,
        'ssr_escape_dim_ratio': 0.45,
        'ssr_escape_gbest_pull': 0.15,
        'ssr_escape_accept_margin_ratio': 0.02,
        'ssr_force_escape_after_failed_exploit': True,
        'ssr_exploit_max_unique_rank_ratio': 0.80,
        'ssr_exploit_max_machine_family_ratio': 0.80,
        'ssr_exploit_min_dup_ratio': 0.35,
        'ssr_exploit_restart_ratio': 0.0,
        'ssr_exploit_diversity_quota': 0,
        'ssr_escape_diversity_quota': 2,
    }

    _, best_position, _, _, ssr_info = CAOA_SSR(
        **caoa_params,
        decoder=decoder,
        return_diagnostics=True,
        **caoa_ssr_params,
    )

    caoa_schedule_df, caoa_metrics = decoder.decode_from_continuous(best_position)
    ensure_feasible(caoa_metrics, "CAOASSR")
    timetable_path, metrics_path, position_path = save_optimized_results(
        caoa_schedule_df,
        caoa_metrics,
        best_position,
    )
    ssr_logs_path, ssr_summary_path = save_ssr_diagnostics(ssr_info)
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
    print(f"- SSR logs  : {ssr_logs_path}")
    print(f"- SSR sum   : {ssr_summary_path}")
    print(f"- Debug CSV : {voyage_debug_path}")
    print(f"- Debug sum : {voyage_summary_path}")
    print(f"- Compare V : {voyage_comparison_path}")
    print(f"- Compare O : {operation_comparison_path}")
    ssr_summary = build_ssr_diagnostic_summary(ssr_info)
    print(f"- SSR checks: {ssr_summary['ssr_checks']}")
    print(f"- SSR active: {ssr_summary['ssr_active_count']}")
    print(f"- SSR replace: {ssr_summary['ssr_replacement_total']}")
    print(
        "- Inline R/D/K: "
        f"{ssr_summary['inline_reduced_dim_total']}/"
        f"{ssr_summary['inline_explore_dim_total']}/"
        f"{ssr_summary['inline_knowledge_reinit_total']}"
    )

if __name__ == "__main__":
    main()
