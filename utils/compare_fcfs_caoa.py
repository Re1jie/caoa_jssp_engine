import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE = Path("data/result/fcfs_baseline_timetable.csv")
DEFAULT_OPTIMIZED = Path("data/result/caoa_optimized_timetable.csv")
DEFAULT_TARGET = Path("data/processed/job_target_time_sliced.csv")
DEFAULT_VOYAGE_OUTPUT = Path("data/result/fcfs_vs_caoa_voyage_comparison.csv")
DEFAULT_OPERATION_OUTPUT = Path("data/result/fcfs_vs_caoa_operation_comparison.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bandingkan timetable FCFS baseline vs CAOA dan tulis CSV "
            "per-voyage serta per-operasi."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path ke timetable baseline FCFS.",
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=DEFAULT_OPTIMIZED,
        help="Path ke timetable hasil optimasi CAOA.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Path ke CSV target voyage.",
    )
    parser.add_argument(
        "--voyage-output",
        type=Path,
        default=DEFAULT_VOYAGE_OUTPUT,
        help="Path output perbandingan per-voyage.",
    )
    parser.add_argument(
        "--operation-output",
        type=Path,
        default=DEFAULT_OPERATION_OUTPUT,
        help="Path output perbandingan per-operasi.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Kolom wajib tidak ditemukan di {source}: {missing_str}")


def build_voyage_debug_report(schedule_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        schedule_df,
        {"job_id", "voyage", "A_lj", "C_lj"},
        Path("<schedule_df>"),
    )
    _require_columns(
        target_df,
        {"job_id", "voyage", "T_j"},
        Path("<target_df>"),
    )

    actual_summary = (
        schedule_df.groupby(["job_id", "voyage"], as_index=False)
        .agg(
            first_arrival_hour=("A_lj", "min"),
            last_completion_hour=("C_lj", "max"),
        )
    )
    target_summary = target_df[["job_id", "voyage", "T_j"]].rename(
        columns={"T_j": "due_window_hours"}
    )
    debug_df = actual_summary.merge(
        target_summary,
        on=["job_id", "voyage"],
        how="left",
    )
    debug_df["due_hour_absolute"] = (
        debug_df["first_arrival_hour"] + debug_df["due_window_hours"]
    )
    debug_df["tardiness_hours"] = (
        debug_df["last_completion_hour"] - debug_df["due_hour_absolute"]
    ).clip(lower=0.0)
    debug_df["lateness_hours"] = (
        debug_df["last_completion_hour"] - debug_df["due_hour_absolute"]
    )
    debug_df["earliness_hours"] = (-debug_df["lateness_hours"]).clip(lower=0.0)
    debug_df["debug_status"] = debug_df["tardiness_hours"].map(
        lambda value: "LATE" if value > 0 else "ON_TIME"
    )
    return debug_df


def build_comparison(
    baseline_schedule_df: pd.DataFrame,
    optimized_schedule_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_voyage_df = build_voyage_debug_report(
        baseline_schedule_df,
        target_df,
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
        target_df,
    ).rename(
        columns={
            "last_completion_hour": "optimized_last_completion_hour",
            "tardiness_hours": "optimized_tardiness_hours",
            "lateness_hours": "optimized_lateness_hours",
            "earliness_hours": "optimized_earliness_hours",
            "debug_status": "optimized_status",
        }
    )

    voyage_df = baseline_voyage_df.merge(
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
    voyage_df["delta_completion_hours"] = (
        voyage_df["optimized_last_completion_hour"]
        - voyage_df["baseline_last_completion_hour"]
    )
    voyage_df["delta_tardiness_hours"] = (
        voyage_df["optimized_tardiness_hours"]
        - voyage_df["baseline_tardiness_hours"]
    )
    voyage_df["delta_lateness_hours"] = (
        voyage_df["optimized_lateness_hours"]
        - voyage_df["baseline_lateness_hours"]
    )
    voyage_df["status_transition"] = (
        voyage_df["baseline_status"] + " -> " + voyage_df["optimized_status"]
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
    operation_df = baseline_ops_df.merge(
        optimized_ops_df,
        on=["job_id", "voyage", "machine_id", "op_seq"],
        how="inner",
    )
    operation_df["delta_start_hours"] = (
        operation_df["optimized_S_lj"] - operation_df["baseline_S_lj"]
    )
    operation_df["delta_completion_hours"] = (
        operation_df["optimized_C_lj"] - operation_df["baseline_C_lj"]
    )
    operation_df["delta_tidal_wait_hours"] = (
        operation_df["optimized_tidal_wait"] - operation_df["baseline_tidal_wait"]
    )
    operation_df["delta_congestion_wait_hours"] = (
        operation_df["optimized_congestion_wait"]
        - operation_df["baseline_congestion_wait"]
    )

    return voyage_df, operation_df


def main() -> None:
    args = parse_args()
    for path in (args.baseline, args.optimized, args.target):
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    baseline_df = pd.read_csv(args.baseline)
    optimized_df = pd.read_csv(args.optimized)
    target_df = pd.read_csv(args.target)

    voyage_df, operation_df = build_comparison(
        baseline_df,
        optimized_df,
        target_df,
    )

    args.voyage_output.parent.mkdir(parents=True, exist_ok=True)
    args.operation_output.parent.mkdir(parents=True, exist_ok=True)
    voyage_df.to_csv(args.voyage_output, index=False)
    operation_df.to_csv(args.operation_output, index=False)

    improved = int((voyage_df["delta_tardiness_hours"] < 0).sum())
    worsened = int((voyage_df["delta_tardiness_hours"] > 0).sum())
    unchanged = int((voyage_df["delta_tardiness_hours"] == 0).sum())

    print("Ringkasan perbandingan FCFS vs CAOA")
    print(f"- Total voyage     : {len(voyage_df)}")
    print(f"- Voyage membaik   : {improved}")
    print(f"- Voyage memburuk  : {worsened}")
    print(f"- Voyage tetap     : {unchanged}")
    print(f"- Output voyage    : {args.voyage_output}")
    print(f"- Output operasi   : {args.operation_output}")


if __name__ == "__main__":
    main()
