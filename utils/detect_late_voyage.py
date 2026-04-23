import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/jssp_data_sliced.csv")
DEFAULT_TARGET = Path("data/processed/job_target_time_sliced.csv")
DEFAULT_OUTPUT = Path("data/processed/job_target_late_voyage_report.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selidiki voyage pada jssp_data.csv yang completion rencananya "
            "melewati due date berdasarkan job_target_time_sliced.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path ke jssp_data.csv.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Path ke CSV target waktu voyage (kolom: job_id, voyage, T_j).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path CSV report yang akan ditulis.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Tampilkan semua voyage, bukan hanya yang LATE.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Jumlah maksimum baris yang ditampilkan di terminal.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Kolom wajib tidak ditemukan di {source}: {missing_str}")


def _prepare_target_df(target_df: pd.DataFrame, source: Path) -> pd.DataFrame:
    required_columns = {"job_id", "voyage", "T_j"}
    _require_columns(target_df, required_columns, source)

    duplicate_targets = target_df[target_df.duplicated(["job_id", "voyage"], keep=False)]
    if not duplicate_targets.empty:
        duplicate_pairs = (
            duplicate_targets[["job_id", "voyage"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            f"Target waktu duplikat di {source}: {sorted(duplicate_pairs)}"
        )

    return target_df[["job_id", "voyage", "T_j"]].rename(
        columns={"T_j": "due_window_hours"}
    )


def build_due_report(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
    source: Path = DEFAULT_INPUT,
    target_source: Path = DEFAULT_TARGET,
) -> pd.DataFrame:
    required_columns = {"job_id", "voyage", "ship_name", "rute", "A_lj", "p_lj"}
    _require_columns(df, required_columns, source)

    work_df = df.copy()
    work_df["planned_completion_hour"] = work_df["A_lj"] + work_df["p_lj"]
    prepared_target_df = _prepare_target_df(target_df, target_source)

    report_df = (
        work_df.groupby(["job_id", "voyage", "ship_name", "rute"], as_index=False)
        .agg(
            first_arrival_hour=("A_lj", "min"),
            last_planned_completion_hour=("planned_completion_hour", "max"),
            operation_count=("op_seq", "count") if "op_seq" in work_df.columns else ("A_lj", "count"),
        )
    )
    report_df = report_df.merge(
        prepared_target_df,
        on=["job_id", "voyage"],
        how="left",
    )

    report_df["due_hour_absolute"] = (
        report_df["first_arrival_hour"] + report_df["due_window_hours"]
    )
    report_df["lateness_hours"] = (
        report_df["last_planned_completion_hour"] - report_df["due_hour_absolute"]
    )
    report_df["tardiness_hours"] = report_df["lateness_hours"].clip(lower=0.0)
    report_df["earliness_hours"] = (-report_df["lateness_hours"]).clip(lower=0.0)
    report_df["actual_flow_time_hours"] = (
        report_df["last_planned_completion_hour"] - report_df["first_arrival_hour"]
    )
    report_df["flow_vs_target_ratio"] = (
        report_df["actual_flow_time_hours"] / report_df["due_window_hours"]
    )
    report_df["debug_status"] = np.select(
        [
            report_df["due_window_hours"].isna(),
            report_df["tardiness_hours"] > 0,
        ],
        ["MISSING_TARGET", "LATE"],
        default="ON_TIME",
    )

    return report_df.sort_values(
        ["debug_status", "tardiness_hours", "job_id", "voyage"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def print_summary(report_df: pd.DataFrame, show_all: bool, limit: int) -> None:
    late_df = report_df[report_df["debug_status"] == "LATE"]
    missing_df = report_df[report_df["debug_status"] == "MISSING_TARGET"]
    late_job_ids = sorted(late_df["job_id"].unique().tolist())

    print("Ringkasan investigasi target voyage")
    print(f"- Total voyage              : {len(report_df)}")
    print(f"- Voyage LATE               : {len(late_df)}")
    print(f"- Voyage ON_TIME            : {(report_df['debug_status'] == 'ON_TIME').sum()}")
    print(f"- Voyage tanpa target T_j   : {len(missing_df)}")
    print(f"- Total tardiness LATE (jam): {late_df['tardiness_hours'].sum():.2f}")
    print(f"- Job dengan voyage LATE    : {late_job_ids}")

    display_df = report_df if show_all else late_df
    if display_df.empty:
        print("\nTidak ada voyage yang melewati due date target.")
        return

    columns = [
        "job_id",
        "voyage",
        "ship_name",
        "rute",
        "first_arrival_hour",
        "last_planned_completion_hour",
        "due_window_hours",
        "due_hour_absolute",
        "tardiness_hours",
        "debug_status",
    ]
    print(f"\nTop {min(limit, len(display_df))} voyage:")
    print(display_df[columns].head(limit).to_string(index=False))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")
    if not args.target.exists():
        raise FileNotFoundError(f"File target tidak ditemukan: {args.target}")

    df = pd.read_csv(args.input)
    target_df = pd.read_csv(args.target)
    report_df = build_due_report(
        df,
        target_df,
        source=args.input,
        target_source=args.target,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.output, index=False)

    print_summary(report_df, show_all=args.show_all, limit=args.limit)
    print(f"\nReport tersimpan di: {args.output}")


if __name__ == "__main__":
    main()
