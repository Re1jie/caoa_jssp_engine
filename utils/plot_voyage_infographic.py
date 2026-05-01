import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("data/result/caoa_voyage_debug_report.csv")
DEFAULT_OUTPUT_DIR = Path("data/result/infographics")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Buat infografis per voyage dari file caoa_voyage_debug_report.csv "
            "untuk membandingkan due date vs realisasi aktual."
        )
    )
    parser.add_argument("--job-id", type=int, required=True, help="ID job yang ingin ditinjau.")
    parser.add_argument("--voyage", type=int, required=True, help="Nomor voyage yang ingin ditinjau.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path ke file CSV debug voyage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder output untuk file PNG infografis.",
    )
    return parser.parse_args()


def load_voyage_row(csv_path: Path, job_id: int, voyage: int) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"File debug report tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "job_id",
        "voyage",
        "first_arrival_hour",
        "last_completion_hour",
        "actual_operation_hours",
        "planned_sailing_hours",
        "total_wait_hours",
        "due_window_hours",
        "due_hour_absolute",
        "tardiness_hours",
        "earliness_hours",
        "debug_status",
        "actual_flow_time_hours",
        "operation_count",
    }
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Kolom wajib tidak ditemukan pada CSV: {missing_str}")

    match = df[(df["job_id"] == job_id) & (df["voyage"] == voyage)]
    if match.empty:
        raise ValueError(
            f"Data untuk job_id={job_id} dan voyage={voyage} tidak ditemukan di {csv_path}"
        )

    return match.iloc[0]


def _status_color(status: str) -> str:
    return "#C0392B" if status == "LATE" else "#1E8449"


def _fmt_hours(value: float) -> str:
    return f"{value:,.1f} jam"


def build_infographic(row: pd.Series, output_path: Path) -> None:
    status = str(row["debug_status"])
    status_color = _status_color(status)
    late_or_early = float(row["tardiness_hours"]) if status == "LATE" else float(row["earliness_hours"])

    arrival = float(row["first_arrival_hour"])
    due_absolute = float(row["due_hour_absolute"])
    completion = float(row["last_completion_hour"])
    due_window = float(row["due_window_hours"])
    actual_flow = float(row["actual_flow_time_hours"])
    actual_operation = float(row["actual_operation_hours"])
    sailing = float(row["planned_sailing_hours"])
    total_wait = float(row["total_wait_hours"])
    tidal_wait = float(row.get("total_tidal_wait_hours", 0.0))
    congestion_wait = float(row.get("total_congestion_wait_hours", 0.0))
    nonprocessing = float(row.get("actual_nonprocessing_hours", max(actual_flow - actual_operation, 0.0)))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(14, 9), facecolor="#F6F3EC")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.25, 1.15], hspace=0.34, wspace=0.2)

    ax_header = fig.add_subplot(gs[0, :])
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_breakdown = fig.add_subplot(gs[2, 0])
    ax_notes = fig.add_subplot(gs[2, 1])

    for ax in [ax_header, ax_timeline, ax_breakdown, ax_notes]:
        ax.set_facecolor("#FFFDF8")

    ax_header.axis("off")
    title = f"Infografis Voyage {int(row['voyage'])} pada Job {int(row['job_id'])}"
    subtitle = (
        f"Due window: {_fmt_hours(due_window)} | "
        f"Flow aktual: {_fmt_hours(actual_flow)} | "
        f"Status: {status}"
    )
    ax_header.text(0.02, 0.82, title, fontsize=24, fontweight="bold", color="#1F2933", transform=ax_header.transAxes)
    ax_header.text(0.02, 0.62, subtitle, fontsize=12.5, color="#52606D", transform=ax_header.transAxes)

    card_specs = [
        ("Due Absolut", _fmt_hours(due_absolute), "#355C7D"),
        ("Completion Aktual", _fmt_hours(completion), "#2A9D8F"),
        ("Selisih", _fmt_hours(late_or_early), status_color),
        ("Jumlah Operasi", f"{int(row['operation_count'])} operasi", "#7A5C3E"),
    ]
    card_x = [0.02, 0.27, 0.52, 0.77]
    for (label, value, color), x0 in zip(card_specs, card_x):
        ax_header.add_patch(
            plt.Rectangle((x0, 0.10), 0.21, 0.32, color=color, alpha=0.11, transform=ax_header.transAxes)
        )
        ax_header.text(x0 + 0.02, 0.30, label, fontsize=11, color="#52606D", transform=ax_header.transAxes)
        ax_header.text(x0 + 0.02, 0.16, value, fontsize=16, fontweight="bold", color=color, transform=ax_header.transAxes)

    timeline_end = max(completion, due_absolute)
    ax_timeline.set_title("Timeline Absolut: arrival vs due date vs completion", loc="left", fontsize=14, fontweight="bold", color="#1F2933")
    ax_timeline.barh(["Target"], [due_absolute - arrival], left=arrival, color="#A7C5BD", height=0.35, label="Window sampai due date")
    ax_timeline.barh(["Aktual"], [completion - arrival], left=arrival, color="#355C7D", height=0.35, label="Flow time aktual")
    ax_timeline.axvline(arrival, color="#7B8794", linestyle="--", linewidth=1.5)
    ax_timeline.axvline(due_absolute, color="#E67E22", linestyle="--", linewidth=2.0)
    ax_timeline.axvline(completion, color=status_color, linestyle="-", linewidth=2.4)
    ax_timeline.text(arrival, 1.18, f"Arrival\n{arrival:.1f}", ha="center", va="bottom", fontsize=10, color="#52606D")
    ax_timeline.text(due_absolute, 1.18, f"Due\n{due_absolute:.1f}", ha="center", va="bottom", fontsize=10, color="#E67E22")
    ax_timeline.text(completion, -0.38, f"Completion\n{completion:.1f}", ha="center", va="top", fontsize=10, color=status_color)
    ax_timeline.set_xlim(arrival - max(due_window * 0.05, 12.0), timeline_end + max(due_window * 0.08, 20.0))
    ax_timeline.grid(axis="x", alpha=0.18)
    ax_timeline.spines[["top", "right", "left"]].set_visible(False)
    ax_timeline.tick_params(axis="y", length=0)
    ax_timeline.set_xlabel("Jam absolut pada horizon schedule")
    ax_timeline.legend(loc="lower right", frameon=False)

    ax_breakdown.set_title("Breakdown Jam Aktual", loc="left", fontsize=14, fontweight="bold", color="#1F2933")
    labels = ["Due Window", "Flow Aktual"]
    operation_color = "#4C78A8"
    sailing_color = "#72B7B2"
    wait_color = "#F58518"
    other_color = "#B279A2"
    ax_breakdown.bar(labels[0], due_window, color="#D9E2EC", width=0.55)
    ax_breakdown.bar(labels[1], actual_operation, color=operation_color, width=0.55, label="Operasi aktual")
    ax_breakdown.bar(labels[1], sailing, bottom=actual_operation, color=sailing_color, width=0.55, label="Sailing planned")
    ax_breakdown.bar(labels[1], total_wait, bottom=actual_operation + sailing, color=wait_color, width=0.55, label="Total waiting")
    remaining = max(nonprocessing - sailing - total_wait, 0.0)
    if remaining > 0:
        ax_breakdown.bar(
            labels[1],
            remaining,
            bottom=actual_operation + sailing + total_wait,
            color=other_color,
            width=0.55,
            label="Non-processing lain",
        )
    ax_breakdown.axhline(due_window, color="#C0392B", linestyle="--", linewidth=1.6)
    ax_breakdown.text(0.03, due_window + max(actual_flow, due_window) * 0.02, f"Due = {_fmt_hours(due_window)}", color="#C0392B", fontsize=10)
    ax_breakdown.grid(axis="y", alpha=0.18)
    ax_breakdown.spines[["top", "right"]].set_visible(False)
    ax_breakdown.set_ylabel("Jam")
    ax_breakdown.legend(frameon=False, loc="upper left")

    ax_notes.axis("off")
    notes = [
        f"Job ID                : {int(row['job_id'])}",
        f"Voyage                : {int(row['voyage'])}",
        f"Status                : {status}",
        f"Arrival pertama       : {_fmt_hours(arrival)}",
        f"Due absolut           : {_fmt_hours(due_absolute)}",
        f"Completion aktual     : {_fmt_hours(completion)}",
        f"Tardiness             : {_fmt_hours(float(row['tardiness_hours']))}",
        f"Earliness             : {_fmt_hours(float(row['earliness_hours']))}",
        f"Jam operasi aktual    : {_fmt_hours(actual_operation)}",
        f"Jam sailing planned   : {_fmt_hours(sailing)}",
        f"Jam tunggu total      : {_fmt_hours(total_wait)}",
        f"  - Delay pasang      : {_fmt_hours(tidal_wait)}",
        f"  - Kongesti          : {_fmt_hours(congestion_wait)}",
        f"Rasio flow vs due     : {float(row['operating_vs_due_ratio']):.2f}x",
        f"Rasio proses vs due   : {float(row['processing_vs_due_ratio']):.2f}x",
        f"Rasio wait vs flow    : {float(row['waiting_vs_flow_ratio']):.2%}",
    ]
    ax_notes.text(
        0.02,
        0.98,
        "Ringkasan Detail",
        fontsize=14,
        fontweight="bold",
        color="#1F2933",
        va="top",
        transform=ax_notes.transAxes,
    )
    ax_notes.text(
        0.02,
        0.88,
        "\n".join(notes),
        fontsize=11.5,
        color="#334E68",
        va="top",
        family="DejaVu Sans Mono",
        linespacing=1.45,
        transform=ax_notes.transAxes,
    )

    fig.text(
        0.02,
        0.015,
        "Sumber data: data/result/caoa_voyage_debug_report.csv",
        fontsize=9.5,
        color="#7B8794",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    row = load_voyage_row(args.input, args.job_id, args.voyage)

    output_path = args.output_dir / f"job_{args.job_id}_voyage_{args.voyage}_infographic.png"
    build_infographic(row, output_path)
    print(f"Infografis berhasil dibuat: {output_path}")


if __name__ == "__main__":
    main()
