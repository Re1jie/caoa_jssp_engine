import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import NormalDist
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scipy.stats import mannwhitneyu as scipy_mannwhitneyu
    from scipy.stats import wilcoxon as scipy_wilcoxon
except Exception:
    scipy_mannwhitneyu = None
    scipy_wilcoxon = None

from engine.caoa import CAOA
from engine.caoassr import CAOA_SSR
from engine.decoder_insertion import ActiveScheduleDecoder
from engine.gwo import GWO
from engine.tidal_checker import TidalChecker
from utils.data_loader import load_real_jssp_data


ALGORITHM_NAMES = ("CAOA", "CAOASSR", "GWO")
NORMAL = NormalDist()


@dataclass
class RunResult:
    seed: int
    algorithm_seed: int
    algorithm: str
    total_tardiness: float
    max_tardiness: float
    runtime_seconds: float
    feasible: bool
    convergence_path: str
    error: str = ""


def objective_function(x: np.ndarray, decoder: ActiveScheduleDecoder) -> float:
    _, metrics = decoder.decode_from_continuous(x)
    return float(metrics["total_tardiness"])


def make_decoder(
    df_ops: Any,
    df_machine_master: Any,
    df_job_target: Any,
    tidal_checker: TidalChecker,
) -> ActiveScheduleDecoder:
    return ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target,
    )


def ensure_feasible(metrics: Dict[str, Any], label: str) -> None:
    if metrics.get("is_feasible", True):
        return
    reason = metrics.get("infeasible_reason", "unknown")
    raise RuntimeError(f"{label} produced an infeasible schedule: {reason}")


def vargha_delaney_a12_minimization(
    treatment_values: Iterable[float],
    control_values: Iterable[float],
) -> float:
    """
    Vargha-Delaney A12 for minimization.

    A12 = P(treatment < control) + 0.5 * P(treatment == control).
    A12 > 0.5 means the treatment tends to produce lower tardiness.
    """
    treatment = list(map(float, treatment_values))
    control = list(map(float, control_values))
    if not treatment or not control:
        return float("nan")

    better = 0.0
    for t in treatment:
        for c in control:
            if t < c:
                better += 1.0
            elif t == c:
                better += 0.5
    return better / (len(treatment) * len(control))


def a12_magnitude(a12: float) -> str:
    if math.isnan(a12):
        return "nan"
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def two_sided_normal_pvalue(z_value: float) -> float:
    return float(2.0 * (1.0 - NORMAL.cdf(abs(z_value))))


def mann_whitney_u_test(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float, str]:
    if scipy_mannwhitneyu is not None:
        result = scipy_mannwhitneyu(x_values, y_values, alternative="two-sided")
        return float(result.statistic), float(result.pvalue), "scipy"

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    n_x = len(x_values)
    n_y = len(y_values)
    if n_x < 1 or n_y < 1:
        return float("nan"), float("nan"), "normal_approx"

    combined = np.concatenate([x_values, y_values])
    ranks = average_ranks(combined)
    rank_sum_x = float(np.sum(ranks[:n_x]))
    u_stat = rank_sum_x - n_x * (n_x + 1) / 2.0

    _, tie_counts = np.unique(combined, return_counts=True)
    n_total = n_x + n_y
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    variance = n_x * n_y / 12.0 * (
        (n_total + 1) - tie_term / (n_total * (n_total - 1))
    )
    if variance <= 0:
        return u_stat, 1.0, "normal_approx"

    mean_u = n_x * n_y / 2.0
    z_value = (u_stat - mean_u) / math.sqrt(variance)
    return u_stat, two_sided_normal_pvalue(z_value), "normal_approx"


def wilcoxon_signed_rank_test(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float, str]:
    if scipy_wilcoxon is not None:
        try:
            result = scipy_wilcoxon(x_values, y_values, alternative="two-sided")
            return float(result.statistic), float(result.pvalue), "scipy"
        except ValueError:
            differences = np.asarray(x_values, dtype=float) - np.asarray(y_values, dtype=float)
            if len(differences) and np.all(differences == 0):
                return 0.0, 1.0, "scipy"
            return float("nan"), float("nan"), "scipy"

    differences = np.asarray(x_values, dtype=float) - np.asarray(y_values, dtype=float)
    differences = differences[differences != 0]
    n = len(differences)
    if n == 0:
        return 0.0, 1.0, "normal_approx"

    ranks = average_ranks(np.abs(differences))
    w_plus = float(np.sum(ranks[differences > 0]))
    w_minus = float(np.sum(ranks[differences < 0]))
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    _, tie_counts = np.unique(np.abs(differences), return_counts=True)
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if variance <= 0:
        return statistic, 1.0, "normal_approx"

    z_value = (w_plus - mean_w) / math.sqrt(variance)
    return statistic, two_sided_normal_pvalue(z_value), "normal_approx"


def describe(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def build_caoa_params(dim: int, max_iter: int, population_size: int, max_fes: int | None) -> Dict[str, Any]:
    return {
        "N": population_size,
        "max_iter": max_iter,
        "lb": 0.0,
        "ub": 1.0,
        "dim": dim,
        "alpha": 0.79,
        "beta": 0.07,
        "gamma": 0.06,
        "delta": 13.12,
        "initial_energy": 10,
        "max_FEs": max_fes,
    }


def run_algorithm_once(
    algorithm: str,
    seed: int,
    algorithm_seed: int,
    initial_pos: np.ndarray,
    df_ops: Any,
    df_machine_master: Any,
    df_job_target: Any,
    tidal_checker: TidalChecker,
    max_iter: int,
    population_size: int,
    max_fes: int | None,
    output_dir: Path,
    verbose_algorithms: bool,
) -> RunResult:
    decoder = make_decoder(df_ops, df_machine_master, df_job_target, tidal_checker)
    dim = len(df_ops)
    np.random.seed(algorithm_seed)
    start = time.perf_counter()

    try:
        stream = None if verbose_algorithms else io.StringIO()
        with contextlib.redirect_stdout(stream) if stream is not None else contextlib.nullcontext():
            if algorithm == "CAOA":
                result = CAOA(
                    **build_caoa_params(dim, max_iter, population_size, max_fes),
                    fobj=lambda x: objective_function(x, decoder),
                    initial_pos=initial_pos,
                )
            elif algorithm == "CAOASSR":
                result = CAOA_SSR(
                    **build_caoa_params(dim, max_iter, population_size, max_fes),
                    decoder=decoder,
                    initial_pos=initial_pos,
                    IT=10,
                    elite_size=5,
                    ssr_elite_k=5,
                    ssr_inline_guidance=True,
                    ssr_inline_prob=0.20,
                    ssr_inline_reduced_dim_ratio=0.08,
                    ssr_inline_reduced_blend=0.25,
                    ssr_inline_explore_dim_ratio=0.05,
                    ssr_inline_reinit_uses_knowledge=True,
                    ssr_balanced_reinit=True,
                    ssr_explore_opposition_ratio=0.50,
                    ssr_reduction_min_width=0.05,
                    ssr_reduction_width_scale=2.0,
                    verbose=verbose_algorithms,
                )
            elif algorithm == "GWO":
                result = GWO(
                    objf=lambda x: objective_function(x, decoder),
                    lb=0.0,
                    ub=1.0,
                    dim=dim,
                    pop_size=population_size,
                    max_iter=max_iter,
                    max_FEs=max_fes,
                    initial_pos=initial_pos,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        best_position = result[1]
        convergence_curve = np.asarray(result[2], dtype=float)
        schedule_df, metrics = decoder.decode_from_continuous(best_position)
        ensure_feasible(metrics, algorithm)
        runtime = time.perf_counter() - start

        curve_dir = output_dir / "convergence_runs" / algorithm.lower()
        curve_dir.mkdir(parents=True, exist_ok=True)
        curve_path = curve_dir / f"seed_{seed}.npy"
        np.save(curve_path, convergence_curve)

        return RunResult(
            seed=seed,
            algorithm_seed=algorithm_seed,
            algorithm=algorithm,
            total_tardiness=float(metrics["total_tardiness"]),
            max_tardiness=float(metrics.get("max_tardiness", float("nan"))),
            runtime_seconds=float(runtime),
            feasible=bool(metrics.get("is_feasible", True)),
            convergence_path=str(curve_path),
        )
    except Exception as exc:
        return RunResult(
            seed=seed,
            algorithm_seed=algorithm_seed,
            algorithm=algorithm,
            total_tardiness=float("nan"),
            max_tardiness=float("nan"),
            runtime_seconds=float(time.perf_counter() - start),
            feasible=False,
            convergence_path="",
            error=repr(exc),
        )


def pad_curve(curve: np.ndarray, length: int) -> np.ndarray:
    if curve.size == 0:
        return np.full(length, np.nan)
    if curve.size >= length:
        return curve[:length]
    return np.pad(curve, (0, length - curve.size), mode="edge")


def save_convergence_summary(output_dir: Path, algorithms: List[str], results: List[RunResult]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    summary_dir = output_dir / "convergence_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for algorithm in algorithms:
        curves = []
        for result in results:
            if result.algorithm == algorithm and result.convergence_path:
                curves.append(np.load(result.convergence_path))
        if not curves:
            continue

        max_len = max(len(curve) for curve in curves)
        matrix = np.vstack([pad_curve(curve, max_len) for curve in curves])
        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(max_len)

        mean_path = summary_dir / f"{algorithm.lower()}_mean_convergence.npy"
        std_path = summary_dir / f"{algorithm.lower()}_std_convergence.npy"
        all_path = summary_dir / f"{algorithm.lower()}_all_convergence.npy"
        np.save(mean_path, mean_curve)
        np.save(std_path, std_curve)
        np.save(all_path, matrix)

        paths[f"{algorithm.lower()}_mean_convergence"] = str(mean_path)
        paths[f"{algorithm.lower()}_std_convergence"] = str(std_path)
        paths[f"{algorithm.lower()}_all_convergence"] = str(all_path)

    return paths


def try_save_convergence_plot(output_dir: Path) -> str:
    try:
        from utils.plot_convergence import plot_comparison

        plot_comparison(output_dir)
    except Exception as exc:
        print(f"Warning: convergence plot was not created ({exc})")
        return ""

    plot_path = output_dir / "convergence_mean_comparison.png"
    return str(plot_path) if plot_path.exists() else ""


def pairwise_tests(results_df: pd.DataFrame, algorithms: List[str], alpha: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid_df = results_df[(results_df["feasible"]) & (results_df["error"] == "")]

    for i, treatment in enumerate(algorithms):
        for control in algorithms[i + 1 :]:
            treatment_df = valid_df[valid_df["algorithm"] == treatment].sort_values("seed")
            control_df = valid_df[valid_df["algorithm"] == control].sort_values("seed")
            treatment_values = treatment_df["total_tardiness"].to_numpy(dtype=float)
            control_values = control_df["total_tardiness"].to_numpy(dtype=float)

            merged = treatment_df[["seed", "total_tardiness"]].merge(
                control_df[["seed", "total_tardiness"]],
                on="seed",
                suffixes=("_treatment", "_control"),
            )
            paired_treatment = merged["total_tardiness_treatment"].to_numpy(dtype=float)
            paired_control = merged["total_tardiness_control"].to_numpy(dtype=float)
            deltas = paired_treatment - paired_control

            if len(treatment_values) > 1 and len(control_values) > 1:
                rank_sum_u, rank_sum_p, rank_sum_method = mann_whitney_u_test(
                    treatment_values,
                    control_values,
                )
            else:
                rank_sum_u = float("nan")
                rank_sum_p = float("nan")
                rank_sum_method = "not_enough_data"

            if len(paired_treatment) > 0:
                signed_stat, signed_p, signed_method = wilcoxon_signed_rank_test(
                    paired_treatment,
                    paired_control,
                )
            else:
                signed_stat = float("nan")
                signed_p = float("nan")
                signed_method = "not_enough_data"

            a12 = vargha_delaney_a12_minimization(treatment_values, control_values)
            rows.append(
                {
                    "treatment": treatment,
                    "control": control,
                    "n_treatment": int(len(treatment_values)),
                    "n_control": int(len(control_values)),
                    "common_seed_count": int(len(merged)),
                    "treatment_better_seed_count": int(np.sum(deltas < 0)) if len(deltas) else 0,
                    "tie_seed_count": int(np.sum(deltas == 0)) if len(deltas) else 0,
                    "treatment_worse_seed_count": int(np.sum(deltas > 0)) if len(deltas) else 0,
                    "mean_delta_treatment_minus_control": float(np.mean(deltas)) if len(deltas) else float("nan"),
                    "median_delta_treatment_minus_control": float(np.median(deltas)) if len(deltas) else float("nan"),
                    "mann_whitney_u": rank_sum_u,
                    "rank_sum_p_value": rank_sum_p,
                    "rank_sum_method": rank_sum_method,
                    "rank_sum_significant_alpha": bool(rank_sum_p < alpha) if np.isfinite(rank_sum_p) else False,
                    "wilcoxon_signed_rank_stat": signed_stat,
                    "wilcoxon_signed_rank_p_value": signed_p,
                    "wilcoxon_signed_rank_method": signed_method,
                    "signed_rank_significant_alpha": bool(signed_p < alpha) if np.isfinite(signed_p) else False,
                    "a12_treatment_better": float(a12),
                    "a12_magnitude": a12_magnitude(a12),
                    "interpretation": (
                        f"A12 > 0.5 favors {treatment}; lower total tardiness is better."
                    ),
                }
            )
    return rows


def write_report(path: Path, summary: Dict[str, Any], pairwise_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Nonparametric Statistical Test Report",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Objective: `{summary['config']['objective']}`",
        f"Runs per algorithm: `{summary['config']['runs']}`",
        f"Seeds: `{summary['config']['seeds'][0]}` to `{summary['config']['seeds'][-1]}`",
        "",
        "## Method",
        "",
        "- Wilcoxon rank-sum / Mann-Whitney U tests whether two independent samples differ in distribution/location.",
        "- Wilcoxon signed-rank is reported as an additional paired test because every algorithm uses the same controlled seed index.",
        "- Vargha-Delaney A12 reports stochastic superiority for minimization: values above 0.5 favor the treatment algorithm.",
        "",
        "## Descriptive Statistics",
        "",
        "| Algorithm | n | mean | std | median | min | max | q1 | q3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for algorithm, desc in summary["descriptive"].items():
        lines.append(
            f"| {algorithm} | {desc['n']} | {desc['mean']:.6f} | {desc['std']:.6f} | "
            f"{desc['median']:.6f} | {desc['min']:.6f} | {desc['max']:.6f} | "
            f"{desc['q1']:.6f} | {desc['q3']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Tests",
            "",
            "| Treatment | Control | rank-sum p | signed-rank p | A12 | magnitude | mean delta |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['treatment']} | {row['control']} | "
            f"{row['rank_sum_p_value']:.10f} | {row['wilcoxon_signed_rank_p_value']:.10f} | "
            f"{row['a12_treatment_better']:.6f} | {row['a12_magnitude']} | "
            f"{row['mean_delta_treatment_minus_control']:.6f} |"
        )

    lines.extend(["", "Negative mean delta favors the treatment algorithm."])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_algorithms(value: str) -> List[str]:
    algorithms = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = sorted(set(algorithms) - set(ALGORITHM_NAMES))
    if unknown:
        raise ValueError(f"Unknown algorithms: {', '.join(unknown)}")
    if len(algorithms) < 2:
        raise ValueError("Use at least two algorithms.")
    return algorithms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--max-fes", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--data-dir", type=str, default="data/processed/")
    parser.add_argument("--output-root", type=str, default="data/results/nonparametric")
    parser.add_argument("--algorithms", type=str, default="CAOA,CAOASSR,GWO")
    parser.add_argument("--verbose-algorithms", action="store_true")
    args = parser.parse_args()

    algorithms = parse_algorithms(args.algorithms)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"nonparametric_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    config = {
        "runs": args.runs,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "max_iter": args.max_iter,
        "population_size": args.population_size,
        "max_fes": args.max_fes,
        "alpha": args.alpha,
        "algorithms": algorithms,
        "data_dir": args.data_dir,
        "objective": "minimize total_tardiness",
        "seed_policy": "same initial population per seed; deterministic algorithm seed per algorithm",
    }

    print("Loading JSSP data...")
    df_ops, df_machine_master, df_job_target = load_real_jssp_data(args.data_dir)
    dim = len(df_ops)
    tidal_checker = TidalChecker()
    print(f"Dimension: {dim}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Runs per algorithm: {args.runs}")

    results: List[RunResult] = []
    algorithm_offsets = {algorithm: (index + 1) * 100_000 for index, algorithm in enumerate(ALGORITHM_NAMES)}

    for run_index, seed in enumerate(seeds, start=1):
        print(f"\nSeed {seed} ({run_index}/{args.runs})")
        initial_rng = np.random.default_rng(seed)
        initial_pos = initial_rng.uniform(0.0, 1.0, size=(args.population_size, dim))

        for algorithm in algorithms:
            algorithm_seed = seed + algorithm_offsets[algorithm]
            result = run_algorithm_once(
                algorithm=algorithm,
                seed=seed,
                algorithm_seed=algorithm_seed,
                initial_pos=initial_pos,
                df_ops=df_ops,
                df_machine_master=df_machine_master,
                df_job_target=df_job_target,
                tidal_checker=tidal_checker,
                max_iter=args.max_iter,
                population_size=args.population_size,
                max_fes=args.max_fes,
                output_dir=output_dir,
                verbose_algorithms=args.verbose_algorithms,
            )
            results.append(result)
            status = "OK" if result.feasible and not result.error else "ERR"
            print(
                f"  [{status}] {algorithm:<7} TT={result.total_tardiness:.2f} "
                f"MaxT={result.max_tardiness:.2f} time={result.runtime_seconds:.2f}s"
            )
            if result.error:
                print(f"        {result.error}")

        pd.DataFrame([asdict(r) for r in results]).to_csv(
            output_dir / "raw_results_partial.csv",
            index=False,
        )

    raw_df = pd.DataFrame([asdict(r) for r in results])
    raw_csv = output_dir / "raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    paired_rows = []
    for seed in seeds:
        row: Dict[str, Any] = {"seed": seed}
        for algorithm in algorithms:
            match = raw_df[(raw_df["seed"] == seed) & (raw_df["algorithm"] == algorithm)]
            row[f"{algorithm.lower()}_total_tardiness"] = (
                float(match.iloc[0]["total_tardiness"]) if not match.empty else float("nan")
            )
        paired_rows.append(row)
    paired_csv = output_dir / "paired_by_seed_results.csv"
    pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)

    convergence_files = save_convergence_summary(output_dir, algorithms, results)
    plot_path = try_save_convergence_plot(output_dir)
    pairwise_rows = pairwise_tests(raw_df, algorithms, args.alpha)
    pairwise_csv = output_dir / "pairwise_nonparametric_tests.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)

    descriptive = {}
    for algorithm in algorithms:
        values = raw_df[
            (raw_df["algorithm"] == algorithm)
            & (raw_df["feasible"])
            & (raw_df["error"] == "")
        ]["total_tardiness"]
        descriptive[algorithm] = describe(values)

    output_files = {
        "raw_results_csv": str(raw_csv),
        "paired_by_seed_csv": str(paired_csv),
        "pairwise_tests_csv": str(pairwise_csv),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(output_dir / "report.md"),
        **convergence_files,
    }
    if plot_path:
        output_files["convergence_plot_png"] = plot_path
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "descriptive": descriptive,
        "pairwise_tests": pairwise_rows,
        "errors": [asdict(r) for r in results if r.error or not r.feasible],
        "files": output_files,
    }

    summary_json = output_dir / "summary.json"
    report_md = output_dir / "report.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_md, summary, pairwise_rows)

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"Raw results: {raw_csv}")
    print(f"Pairwise tests: {pairwise_csv}")
    print(f"Report: {report_md}")

    print("\nKey pairwise results:")
    for row in pairwise_rows:
        print(
            f"  {row['treatment']} vs {row['control']}: "
            f"rank-sum p={row['rank_sum_p_value']:.6f}, "
            f"signed-rank p={row['wilcoxon_signed_rank_p_value']:.6f}, "
            f"A12={row['a12_treatment_better']:.4f} ({row['a12_magnitude']})"
        )


if __name__ == "__main__":
    main()
