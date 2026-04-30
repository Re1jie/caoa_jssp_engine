from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import statistics as stats
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import ranksums, wilcoxon
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scipy is required for Wilcoxon tests. Install it with: pip install scipy"
    ) from exc

from engine.caoa import CAOA
from engine.caoassr import CAOA_SSR
from engine.decoder_insertion import ActiveScheduleDecoder
from engine.fcfs import run_fcfs_baseline
from engine.tidal_checker import TidalChecker
from utils.data_loader import load_real_jssp_data


# -----------------------------
# Core objective / validation
# -----------------------------

def objective_function(x: np.ndarray, decoder: ActiveScheduleDecoder) -> float:
    _, metrics = decoder.decode_from_continuous(x)
    return float(metrics["total_tardiness"])


def ensure_feasible(metrics: Dict[str, Any], label: str) -> None:
    if metrics.get("is_feasible", True):
        return
    reason = metrics.get("infeasible_reason", "unknown")
    penalty = metrics.get("penalty_tardiness", metrics.get("total_tardiness", 0.0))
    raise RuntimeError(
        f"{label} produced an infeasible schedule. "
        f"reason={reason} | penalty_tardiness={penalty}"
    )


# -----------------------------
# Statistics
# -----------------------------

def vargha_delaney_a12_minimization(
    treatment_values: Iterable[float],
    control_values: Iterable[float],
) -> float:
    """
    A12 for minimization problems.

    Returns probability that treatment is better than control:
        P(treatment < control) + 0.5 * P(treatment == control)

    For this experiment, treatment = CAOASSR and control = CAOA.
    A12 > 0.5 means CAOASSR tends to produce lower total tardiness than CAOA.
    """
    treatment = list(map(float, treatment_values))
    control = list(map(float, control_values))
    if not treatment or not control:
        return float("nan")

    better = 0.0
    total = len(treatment) * len(control)
    for t in treatment:
        for c in control:
            if t < c:
                better += 1.0
            elif t == c:
                better += 0.5
    return better / total


def a12_magnitude(a12: float) -> str:
    """
    Common Vargha-Delaney thresholds after transforming away from 0.5.
    Thresholds are often used as rough guidelines, not hard proof.
    """
    if math.isnan(a12):
        return "nan"
    delta = abs(a12 - 0.5)
    if delta < 0.06:
        return "negligible"
    if delta < 0.14:
        return "small"
    if delta < 0.21:
        return "medium"
    return "large"


def describe(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: float("nan") for k in ["n", "mean", "std", "median", "min", "max", "q1", "q3"]}
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


# -----------------------------
# Experiment records
# -----------------------------

@dataclass
class RunResult:
    seed: int
    algorithm: str
    total_tardiness: float
    max_tardiness: float
    runtime_seconds: float
    feasible: bool
    fe_count: Optional[int] = None
    ssr_checks: Optional[int] = None
    ssr_activated_count: Optional[int] = None
    ssr_replacement_total: Optional[int] = None
    inline_reduced_dim_total: Optional[int] = None
    inline_explore_dim_total: Optional[int] = None
    inline_knowledge_reinit_total: Optional[int] = None
    error: Optional[str] = None


# -----------------------------
# Parameter builders
# -----------------------------

def build_caoa_params(dim: int, max_iter: int, population_size: int) -> Dict[str, Any]:
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
    }


def build_caoassr_params(it_period: int = 10) -> Dict[str, Any]:
    return {
        "IT": it_period,
        "K": it_period * 3,
        "stagnation_window": it_period * 3,
        "stagnation_patience": 1,
        "eps_improve": 1e-8,
        "elite_size": 5,
        "dup_ratio_threshold": 0.60,
        "unique_schedule_threshold": 0.25,
        "machine_family_threshold": 0.70,
        "ranking_collapse_threshold": 0.08,
        "structural_distance_threshold": 0.25,
        "use_machine_order_signature": True,
        "use_ranking_similarity": True,
        "stagnation_mode": "rule",
        "partial_restart_ratio": 0.0,
        "ssr_elite_k": 5,
        "ssr_min_knowledge_signal_ratio": 0.80,
        "ssr_knowledge_noise_scale": 0.12,
        "ssr_knowledge_min_noise_scale": 0.02,
        "ssr_knowledge_max_confidence": 0.85,
        "ssr_knowledge_uniform_mix": 0.20,
        "ssr_allow_plateau_activation": True,
        "ssr_min_plateau_checks": 6,
        "ssr_candidate_trials": 3,
        "ssr_accept_only_improvement": True,
        "ssr_commit_requires_gbest_improvement": True,
        "ssr_inline_guidance": True,
        "ssr_inline_prob": 0.20,
        "ssr_inline_confidence_threshold": 0.70,
        "ssr_inline_reduced_dim_ratio": 0.08,
        "ssr_inline_reduced_blend": 0.25,
        "ssr_inline_explore_dim_ratio": 0.05,
        "ssr_inline_reinit_uses_knowledge": True,
        "ssr_random_fallback": False,
        "ssr_balanced_reinit": True,
        "ssr_explore_dim_ratio": 0.30,
        "ssr_explore_opposition_ratio": 0.50,
        "ssr_reduction_min_width": 0.05,
        "ssr_reduction_width_scale": 2.0,
        "ssr_reduced_gbest_pull": 0.40,
        "ssr_uncertain_uniform_ratio": 0.25,
        "ssr_force_mode_quota": False,
        "ssr_adaptive_mode": True,
        "ssr_escape_after_failed_activations": 1,
        "ssr_cooldown_checks": 1,
        "ssr_skip_last_checks": 1,
        "ssr_escape_reduction_width_multiplier": 1.8,
        "ssr_escape_noise_multiplier": 1.7,
        "ssr_escape_dim_ratio": 0.45,
        "ssr_escape_gbest_pull": 0.15,
        "ssr_escape_accept_margin_ratio": 0.02,
        "ssr_force_escape_after_failed_exploit": True,
        "ssr_exploit_max_unique_rank_ratio": 0.80,
        "ssr_exploit_max_machine_family_ratio": 0.80,
        "ssr_exploit_min_dup_ratio": 0.35,
        "ssr_exploit_restart_ratio": 0.0,
        "ssr_exploit_diversity_quota": 0,
        "ssr_escape_diversity_quota": 2,
    }


# -----------------------------
# Algorithm runners
# -----------------------------

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


def run_caoa_once(
    seed: int,
    df_ops: Any,
    df_machine_master: Any,
    df_job_target: Any,
    tidal_checker: TidalChecker,
    max_iter: int,
    population_size: int,
    verbose: bool,
) -> RunResult:
    np.random.seed(seed)
    decoder = make_decoder(df_ops, df_machine_master, df_job_target, tidal_checker)
    dim = len(df_ops)
    params = build_caoa_params(dim=dim, max_iter=max_iter, population_size=population_size)

    start = time.perf_counter()
    try:
        # CAOA in previous project versions commonly returns:
        #   best_score, best_position, cg_curve, avg_curve
        # This unpacking is intentionally tolerant to extra values.
        with contextlib.redirect_stdout(io.StringIO()):
            result = CAOA(
                **params,
                fobj=lambda x: objective_function(x, decoder),
            )
        best_position = result[1]
        schedule_df, metrics = decoder.decode_from_continuous(best_position)
        ensure_feasible(metrics, "CAOA")
        runtime = time.perf_counter() - start
        return RunResult(
            seed=seed,
            algorithm="CAOA",
            total_tardiness=float(metrics["total_tardiness"]),
            max_tardiness=float(metrics.get("max_tardiness", float("nan"))),
            runtime_seconds=float(runtime),
            feasible=bool(metrics.get("is_feasible", True)),
            fe_count=None,
        )
    except Exception as exc:
        runtime = time.perf_counter() - start
        return RunResult(
            seed=seed,
            algorithm="CAOA",
            total_tardiness=float("nan"),
            max_tardiness=float("nan"),
            runtime_seconds=float(runtime),
            feasible=False,
            error=repr(exc),
        )


def _sum_diag_key(diagnostics: List[Dict[str, Any]], key: str) -> int:
    total = 0
    for item in diagnostics or []:
        value = item.get(key, 0)
        try:
            total += int(value)
        except Exception:
            pass
    return total


def _get_ssr_logs(ssr_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(ssr_info, dict):
        return []
    logs = ssr_info.get("logs")
    if logs is None:
        logs = ssr_info.get("diagnostics", [])
    return logs if isinstance(logs, list) else []


def _is_ssr_active_log(item: Dict[str, Any]) -> bool:
    return bool(item.get("ssr_active", item.get("ssr_triggered", False)))


def run_caoassr_once(
    seed: int,
    df_ops: Any,
    df_machine_master: Any,
    df_job_target: Any,
    tidal_checker: TidalChecker,
    max_iter: int,
    population_size: int,
    verbose: bool,
) -> RunResult:
    np.random.seed(seed)
    decoder = make_decoder(df_ops, df_machine_master, df_job_target, tidal_checker)
    dim = len(df_ops)
    caoa_params = build_caoa_params(dim=dim, max_iter=max_iter, population_size=population_size)
    ssr_params = build_caoassr_params(it_period=10)

    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = CAOA_SSR(
                **caoa_params,
                decoder=decoder,
                return_diagnostics=True,
                verbose=False,
                **ssr_params,
            )
        # Expected:
        #   best_score, best_position, cg_curve, avg_curve, ssr_info
        best_position = result[1]
        ssr_info = result[4] if len(result) > 4 else {}
        schedule_df, metrics = decoder.decode_from_continuous(best_position)
        ensure_feasible(metrics, "CAOASSR")
        runtime = time.perf_counter() - start

        diagnostics = _get_ssr_logs(ssr_info)
        best_score_history = ssr_info.get("best_score_history", []) if isinstance(ssr_info, dict) else []
        ssr_activated_count = sum(1 for item in diagnostics if _is_ssr_active_log(item))

        return RunResult(
            seed=seed,
            algorithm="CAOASSR",
            total_tardiness=float(metrics["total_tardiness"]),
            max_tardiness=float(metrics.get("max_tardiness", float("nan"))),
            runtime_seconds=float(runtime),
            feasible=bool(metrics.get("is_feasible", True)),
            ssr_checks=len(best_score_history),
            ssr_activated_count=int(ssr_activated_count),
            ssr_replacement_total=_sum_diag_key(diagnostics, "ssr_replacement_count"),
            inline_reduced_dim_total=_sum_diag_key(diagnostics, "inline_reduced_dim_count"),
            inline_explore_dim_total=_sum_diag_key(diagnostics, "inline_explore_dim_count"),
            inline_knowledge_reinit_total=_sum_diag_key(diagnostics, "inline_knowledge_reinit_count"),
        )
    except Exception as exc:
        runtime = time.perf_counter() - start
        return RunResult(
            seed=seed,
            algorithm="CAOASSR",
            total_tardiness=float("nan"),
            max_tardiness=float("nan"),
            runtime_seconds=float(runtime),
            feasible=False,
            error=repr(exc),
        )


# -----------------------------
# Output writers
# -----------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# CAOA vs CAOASSR Statistical Test Report")
    lines.append("")
    lines.append(f"Generated at: `{summary['generated_at']}`")
    lines.append(f"Runs per algorithm: `{summary['config']['runs']}`")
    lines.append(f"Max iterations: `{summary['config']['max_iter']}`")
    lines.append(f"Population size: `{summary['config']['population_size']}`")
    lines.append("")
    lines.append("## Descriptive statistics: total tardiness")
    lines.append("")
    lines.append("| Algorithm | n | mean | std | median | min | max | q1 | q3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for alg in ["CAOA", "CAOASSR"]:
        d = summary["descriptive"][alg]["total_tardiness"]
        lines.append(
            f"| {alg} | {d['n']} | {d['mean']:.6f} | {d['std']:.6f} | "
            f"{d['median']:.6f} | {d['min']:.6f} | {d['max']:.6f} | "
            f"{d['q1']:.6f} | {d['q3']:.6f} |"
        )
    lines.append("")
    lines.append("## Pairwise by-seed comparison")
    pair = summary["paired_comparison"]
    lines.append("")
    lines.append(f"- CAOASSR better than CAOA: `{pair['caoassr_better_count']}` seeds")
    lines.append(f"- CAOASSR equal to CAOA: `{pair['tie_count']}` seeds")
    lines.append(f"- CAOASSR worse than CAOA: `{pair['caoassr_worse_count']}` seeds")
    lines.append(f"- Mean delta `CAOASSR - CAOA`: `{pair['mean_delta_caoassr_minus_caoa']:.6f}`")
    lines.append(f"- Median delta `CAOASSR - CAOA`: `{pair['median_delta_caoassr_minus_caoa']:.6f}`")
    lines.append("")
    lines.append("Negative delta means CAOASSR is better because total tardiness is minimized.")
    lines.append("")
    lines.append("## Wilcoxon rank-sum test")
    wrs = summary["tests"]["wilcoxon_rank_sum"]
    lines.append("")
    lines.append(f"- Statistic: `{wrs['statistic']:.6f}`")
    lines.append(f"- p-value, two-sided: `{wrs['p_value_two_sided']:.10f}`")
    lines.append("- Null hypothesis: the two independent samples come from distributions with equal location.")
    lines.append("")
    lines.append("## Additional paired Wilcoxon signed-rank test")
    wsr = summary["tests"]["wilcoxon_signed_rank_paired_extra"]
    lines.append("")
    lines.append(f"- Statistic: `{wsr['statistic']:.6f}`")
    lines.append(f"- p-value, two-sided: `{wsr['p_value_two_sided']:.10f}`")
    lines.append("- This is included because the experiment uses matched seeds.")
    lines.append("")
    lines.append("## Vargha-Delaney A12")
    a12 = summary["effect_size"]["vargha_delaney_a12"]
    lines.append("")
    lines.append(f"- A12_CAOASSR_better: `{a12['a12_caoassr_better']:.6f}`")
    lines.append(f"- Magnitude: `{a12['magnitude']}`")
    lines.append("- Interpretation: `A12 > 0.5` means CAOASSR tends to produce lower total tardiness than CAOA.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, value in summary["files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    results: List[RunResult],
    config: Dict[str, Any],
    output_files: Dict[str, str],
) -> Dict[str, Any]:
    valid = [r for r in results if r.error is None and r.feasible]
    by_alg: Dict[str, List[RunResult]] = {
        "CAOA": [r for r in valid if r.algorithm == "CAOA"],
        "CAOASSR": [r for r in valid if r.algorithm == "CAOASSR"],
    }

    caoa_tt = [r.total_tardiness for r in by_alg["CAOA"]]
    ssr_tt = [r.total_tardiness for r in by_alg["CAOASSR"]]

    # Paired rows only where both algorithms succeeded for the same seed.
    caoa_by_seed = {r.seed: r for r in by_alg["CAOA"]}
    ssr_by_seed = {r.seed: r for r in by_alg["CAOASSR"]}
    common_seeds = sorted(set(caoa_by_seed).intersection(ssr_by_seed))
    deltas = [ssr_by_seed[s].total_tardiness - caoa_by_seed[s].total_tardiness for s in common_seeds]

    if len(caoa_tt) < 2 or len(ssr_tt) < 2:
        rank_sum_stat = float("nan")
        rank_sum_p = float("nan")
    else:
        rank_sum = ranksums(ssr_tt, caoa_tt)  # treatment first
        rank_sum_stat = float(rank_sum.statistic)
        rank_sum_p = float(rank_sum.pvalue)

    # Signed-rank can fail if all deltas are exactly zero.
    try:
        signed = wilcoxon(ssr_tt[: len(caoa_tt)], caoa_tt[: len(ssr_tt)], alternative="two-sided")
        signed_stat = float(signed.statistic)
        signed_p = float(signed.pvalue)
    except Exception:
        if deltas and all(d == 0 for d in deltas):
            signed_stat = 0.0
            signed_p = 1.0
        else:
            signed_stat = float("nan")
            signed_p = float("nan")

    a12 = vargha_delaney_a12_minimization(ssr_tt, caoa_tt)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "descriptive": {
            "CAOA": {
                "total_tardiness": describe(caoa_tt),
                "max_tardiness": describe([r.max_tardiness for r in by_alg["CAOA"]]),
                "runtime_seconds": describe([r.runtime_seconds for r in by_alg["CAOA"]]),
            },
            "CAOASSR": {
                "total_tardiness": describe(ssr_tt),
                "max_tardiness": describe([r.max_tardiness for r in by_alg["CAOASSR"]]),
                "runtime_seconds": describe([r.runtime_seconds for r in by_alg["CAOASSR"]]),
                "ssr_checks": describe([r.ssr_checks or 0 for r in by_alg["CAOASSR"]]),
                "ssr_activated_count": describe([r.ssr_activated_count or 0 for r in by_alg["CAOASSR"]]),
                "ssr_replacement_total": describe([r.ssr_replacement_total or 0 for r in by_alg["CAOASSR"]]),
                "inline_reduced_dim_total": describe([r.inline_reduced_dim_total or 0 for r in by_alg["CAOASSR"]]),
                "inline_explore_dim_total": describe([r.inline_explore_dim_total or 0 for r in by_alg["CAOASSR"]]),
                "inline_knowledge_reinit_total": describe([r.inline_knowledge_reinit_total or 0 for r in by_alg["CAOASSR"]]),
            },
        },
        "paired_comparison": {
            "common_seed_count": len(common_seeds),
            "caoassr_better_count": int(sum(1 for d in deltas if d < 0)),
            "tie_count": int(sum(1 for d in deltas if d == 0)),
            "caoassr_worse_count": int(sum(1 for d in deltas if d > 0)),
            "mean_delta_caoassr_minus_caoa": float(np.mean(deltas)) if deltas else float("nan"),
            "median_delta_caoassr_minus_caoa": float(np.median(deltas)) if deltas else float("nan"),
            "min_delta_caoassr_minus_caoa": float(np.min(deltas)) if deltas else float("nan"),
            "max_delta_caoassr_minus_caoa": float(np.max(deltas)) if deltas else float("nan"),
        },
        "tests": {
            "wilcoxon_rank_sum": {
                "statistic": rank_sum_stat,
                "p_value_two_sided": rank_sum_p,
                "alternative": "two-sided",
                "treatment_first": "CAOASSR",
                "control_second": "CAOA",
            },
            "wilcoxon_signed_rank_paired_extra": {
                "statistic": signed_stat,
                "p_value_two_sided": signed_p,
                "alternative": "two-sided",
                "note": "Extra paired diagnostic because seeds are matched.",
            },
        },
        "effect_size": {
            "vargha_delaney_a12": {
                "a12_caoassr_better": float(a12),
                "magnitude": a12_magnitude(a12),
                "interpretation": "A12 > 0.5 favors CAOASSR for minimization.",
            }
        },
        "errors": [asdict(r) for r in results if r.error is not None or not r.feasible],
        "files": output_files,
    }


# -----------------------------
# Main experiment
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="data/processed/")
    parser.add_argument("--output-root", type=str, default="data/result/statistical_tests")
    parser.add_argument("--verbose-algorithms", action="store_true")
    parser.add_argument(
        "--order",
        choices=["caoa-first", "caoassr-first", "alternate"],
        default="alternate",
        help="Run order per seed. Alternating reduces systematic thermal/runtime bias.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"caoa_vs_caoassr_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "runs": args.runs,
        "seed_start": args.seed_start,
        "seeds": list(range(args.seed_start, args.seed_start + args.runs)),
        "max_iter": args.max_iter,
        "population_size": args.population_size,
        "data_dir": args.data_dir,
        "order": args.order,
        "objective": "minimize total_tardiness",
    }

    print("Loading data...")
    df_ops, df_machine_master, df_job_target = load_real_jssp_data(args.data_dir)
    dim = len(df_ops)
    print(f"[Dimensi] {dim}")
    tidal_checker = TidalChecker()

    print("Running FCFS baseline once for reference...")
    fcfs_schedule_df, fcfs_metrics = run_fcfs_baseline(
        df_ops,
        df_machine_master,
        df_job_target,
        tidal_checker,
    )
    ensure_feasible(fcfs_metrics, "FCFS baseline")
    (output_dir / "fcfs_metrics.json").write_text(
        json.dumps(fcfs_metrics, indent=2), encoding="utf-8"
    )

    results: List[RunResult] = []
    seeds = list(range(args.seed_start, args.seed_start + args.runs))

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n=== Seed {seed} ({run_idx}/{args.runs}) ===")

        run_caoa = lambda: run_caoa_once(
            seed=seed,
            df_ops=df_ops,
            df_machine_master=df_machine_master,
            df_job_target=df_job_target,
            tidal_checker=tidal_checker,
            max_iter=args.max_iter,
            population_size=args.population_size,
            verbose=args.verbose_algorithms,
        )
        run_ssr = lambda: run_caoassr_once(
            seed=seed,
            df_ops=df_ops,
            df_machine_master=df_machine_master,
            df_job_target=df_job_target,
            tidal_checker=tidal_checker,
            max_iter=args.max_iter,
            population_size=args.population_size,
            verbose=args.verbose_algorithms,
        )

        if args.order == "caoa-first":
            pair = [run_caoa(), run_ssr()]
        elif args.order == "caoassr-first":
            pair = [run_ssr(), run_caoa()]
        else:
            pair = [run_caoa(), run_ssr()] if run_idx % 2 == 1 else [run_ssr(), run_caoa()]

        for result in pair:
            results.append(result)
            status = "OK" if result.error is None and result.feasible else "ERR"
            print(
                f"[{status}] {result.algorithm:<7} | "
                f"TT={result.total_tardiness:.2f} | "
                f"MaxT={result.max_tardiness:.2f} | "
                f"time={result.runtime_seconds:.2f}s"
            )
            if result.error:
                print(f"    error={result.error}")

        # checkpoint after every seed, so partial results survive interruption
        raw_rows = [asdict(r) for r in results]
        write_csv(output_dir / "raw_results_partial.csv", raw_rows)

    raw_rows = [asdict(r) for r in results]
    raw_csv = output_dir / "raw_results.csv"
    write_csv(raw_csv, raw_rows)

    # Build by-seed wide table
    by_seed_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        caoa = next((r for r in results if r.seed == seed and r.algorithm == "CAOA"), None)
        ssr = next((r for r in results if r.seed == seed and r.algorithm == "CAOASSR"), None)
        row = {
            "seed": seed,
            "caoa_total_tardiness": caoa.total_tardiness if caoa else float("nan"),
            "caoassr_total_tardiness": ssr.total_tardiness if ssr else float("nan"),
            "delta_caoassr_minus_caoa": (
                ssr.total_tardiness - caoa.total_tardiness if caoa and ssr else float("nan")
            ),
            "caoa_max_tardiness": caoa.max_tardiness if caoa else float("nan"),
            "caoassr_max_tardiness": ssr.max_tardiness if ssr else float("nan"),
            "caoa_runtime_seconds": caoa.runtime_seconds if caoa else float("nan"),
            "caoassr_runtime_seconds": ssr.runtime_seconds if ssr else float("nan"),
            "caoassr_ssr_checks": ssr.ssr_checks if ssr else None,
            "caoassr_ssr_activated_count": ssr.ssr_activated_count if ssr else None,
            "caoassr_ssr_replacement_total": ssr.ssr_replacement_total if ssr else None,
            "caoassr_inline_reduced_dim_total": ssr.inline_reduced_dim_total if ssr else None,
            "caoassr_inline_explore_dim_total": ssr.inline_explore_dim_total if ssr else None,
            "caoassr_inline_knowledge_reinit_total": ssr.inline_knowledge_reinit_total if ssr else None,
        }
        by_seed_rows.append(row)

    by_seed_csv = output_dir / "paired_by_seed_results.csv"
    write_csv(by_seed_csv, by_seed_rows)

    output_files = {
        "raw_results_csv": str(raw_csv),
        "paired_by_seed_csv": str(by_seed_csv),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(output_dir / "report.md"),
        "fcfs_metrics_json": str(output_dir / "fcfs_metrics.json"),
    }

    summary = build_summary(results=results, config=config, output_files=output_files)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md = output_dir / "report.md"
    write_markdown_report(report_md, summary)

    print("\n=== DONE ===")
    print(f"Output directory: {output_dir}")
    print(f"Raw results      : {raw_csv}")
    print(f"Paired results   : {by_seed_csv}")
    print(f"Summary JSON     : {summary_json}")
    print(f"Report Markdown  : {report_md}")

    wrs = summary["tests"]["wilcoxon_rank_sum"]
    a12 = summary["effect_size"]["vargha_delaney_a12"]
    pair = summary["paired_comparison"]
    print("\nKey results:")
    print(f"Rank-sum p-value          : {wrs['p_value_two_sided']:.10f}")
    print(f"A12_CAOASSR_better        : {a12['a12_caoassr_better']:.6f} ({a12['magnitude']})")
    print(f"Mean delta CAOASSR - CAOA : {pair['mean_delta_caoassr_minus_caoa']:.6f}")
    print("Negative delta favors CAOASSR.")


if __name__ == "__main__":
    main()
