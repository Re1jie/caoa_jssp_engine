import json
from pathlib import Path

import numpy as np

from engine.decoder_insertion import ActiveScheduleDecoder
from engine.fcfs import run_fcfs_baseline
from engine.gwo import GWO
from engine.tidal_checker import TidalChecker
from run_insertion import (
    build_schedule_comparison,
    build_voyage_debug_report,
    ensure_feasible,
    objective_function,
    save_baseline_results,
)
from utils.data_loader import load_real_jssp_data


def save_optimized_results(
    schedule_df,
    metrics,
    best_position,
    convergence_curve,
    output_dir="data/result",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timetable_path = output_path / "gwo_optimized_timetable.csv"
    metrics_path = output_path / "gwo_optimized_metrics.json"
    position_path = output_path / "gwo_best_position.npy"
    convergence_path = output_path / "gwo_convergence_curve.npy"
    convergence_json_path = output_path / "gwo_convergence_curve.json"

    schedule_df.to_csv(timetable_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    np.save(position_path, best_position)
    np.save(convergence_path, convergence_curve)
    convergence_json_path.write_text(
        json.dumps([float(value) for value in convergence_curve], indent=4),
        encoding="utf-8",
    )

    return (
        timetable_path,
        metrics_path,
        position_path,
        convergence_path,
        convergence_json_path,
    )


def save_voyage_debug_report(debug_df, output_dir="data/result"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_path = output_path / "gwo_voyage_debug_report.csv"
    summary_path = output_path / "gwo_voyage_debug_summary.json"

    debug_df.to_csv(debug_path, index=False)

    summary = {
        "voyage_count": int(len(debug_df)),
        "late_voyage_count": int((debug_df["tardiness_hours"] > 0).sum()),
        "on_time_voyage_count": int((debug_df["tardiness_hours"] <= 0).sum()),
        "total_due_window_hours": float(debug_df["due_window_hours"].sum()),
        "total_tardiness_hours": float(debug_df["tardiness_hours"].sum()),
        "max_tardiness_hours": float(debug_df["tardiness_hours"].max()),
        "avg_tardiness_hours": float(debug_df["tardiness_hours"].mean()),
        "max_earliness_hours": float(debug_df["earliness_hours"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    return debug_path, summary_path


def save_schedule_comparison(
    voyage_comparison_df,
    operation_comparison_df,
    output_dir="data/result",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voyage_path = output_path / "fcfs_vs_gwo_voyage_comparison.csv"
    operation_path = output_path / "fcfs_vs_gwo_operation_comparison.csv"

    voyage_comparison_df.to_csv(voyage_path, index=False)
    operation_comparison_df.to_csv(operation_path, index=False)

    return voyage_path, operation_path


def main():
    np.random.seed(451)

    df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    dim = len(df_ops)
    print(f"[Dimensi] {dim}")
    tidal_checker = TidalChecker()

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

    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target,
    )

    gwo_params = {
        "pop_size": 10,
        "max_iter": 100,
        "lb": 0.0,
        "ub": 1.0,
        "dim": dim,
    }

    best_score, best_position, convergence_curve = GWO(
        **gwo_params,
        objf=lambda X: objective_function(X, decoder),
    )

    gwo_schedule_df, gwo_metrics = decoder.decode_from_continuous(best_position)
    ensure_feasible(gwo_metrics, "GWO")

    (
        timetable_path,
        metrics_path,
        position_path,
        convergence_path,
        convergence_json_path,
    ) = save_optimized_results(
        gwo_schedule_df,
        gwo_metrics,
        best_position,
        convergence_curve,
    )
    voyage_debug_df = build_voyage_debug_report(
        gwo_schedule_df,
        df_job_target,
    )
    voyage_debug_path, voyage_summary_path = save_voyage_debug_report(voyage_debug_df)
    voyage_comparison_df, operation_comparison_df = build_schedule_comparison(
        fcfs_schedule_df,
        gwo_schedule_df,
        df_job_target,
    )
    voyage_comparison_path, operation_comparison_path = save_schedule_comparison(
        voyage_comparison_df,
        operation_comparison_df,
    )

    print(f"\n{'METRIK':<25} | {'FCFS':<15} | {'GWO':<15}")
    print("-" * 60)

    metrics_list = [
        ("Total Tardiness", "total_tardiness"),
        ("Max Tardiness", "max_tardiness"),
    ]

    for label, key in metrics_list:
        print(
            f"{label:<25} | "
            f"{fcfs_metrics.get(key, 0):<15.2f} | "
            f"{gwo_metrics.get(key, 0):<15.2f}"
        )

    print("\nKonfigurasi GWO:")
    print(f"- pop_size : {gwo_params['pop_size']}")
    print(f"- max_iter : {gwo_params['max_iter']}")
    print(f"- lb/ub    : {gwo_params['lb']} / {gwo_params['ub']}")
    print(f"- best obj : {best_score:.2f}")

    print("\nFile hasil optimasi:")
    print(f"- FCFS TT      : {fcfs_timetable_path}")
    print(f"- FCFS Met     : {fcfs_metrics_path}")
    print(f"- Timetable    : {timetable_path}")
    print(f"- Metrics      : {metrics_path}")
    print(f"- Best pos     : {position_path}")
    print(f"- Curve NPY    : {convergence_path}")
    print(f"- Curve JSON   : {convergence_json_path}")
    print(f"- Debug CSV    : {voyage_debug_path}")
    print(f"- Debug sum    : {voyage_summary_path}")
    print(f"- Compare V    : {voyage_comparison_path}")
    print(f"- Compare O    : {operation_comparison_path}")


if __name__ == "__main__":
    main()
