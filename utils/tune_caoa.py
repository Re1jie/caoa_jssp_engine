import contextlib
import io
import json
import os
import time

import numpy as np
import optuna

from engine.caoa import CAOA
from engine.decoder import ActiveScheduleDecoder
from engine.tidal_checker import TidalChecker
from utils.data_loader import load_real_jssp_data


MAX_FES = 2500
MAX_ITER = 5000
N_TRIALS = 200
TIMEOUT = 28800
N_REPEATS = 3
QUICK_MAX_FES = 1200
QUICK_MAX_ITER = 2500
QUICK_N_TRIALS = 20
QUICK_TIMEOUT = 1800
QUICK_N_REPEATS = 2
OUTPUT_FILE = "experiments/caoa_best_params.json"
QUICK_OUTPUT_FILE = "experiments/caoa_quick_best_params.json"
DEFAULT_DATA_DIR = "data/processed/"
BASE_SEED = 20260421

decoder = None
dim = 0


def objective_function(X, active_decoder):
    _, metrics = active_decoder.decode_from_continuous(X)
    return metrics["weighted_avg_tardiness"]


def suggest_caoa_params(trial: optuna.Trial) -> dict:
    # Lanskap tardiness saat ini terlihat datar; karena itu ruang pencarian
    # digeser untuk mencari keseimbangan eksplorasi-ekspolitasi yang lebih halus.
    population = trial.suggest_int("population", 12, 32, step=4)
    alpha = trial.suggest_float("alpha", 0.05, 0.80)
    beta = trial.suggest_float("beta", 0.01, 0.40)
    gamma = trial.suggest_float("gamma", 1e-3, 0.20, log=True)
    delta = trial.suggest_float("delta", 1e-4, 1.0, log=True)
    initial_energy = trial.suggest_float("initial_energy", 5.0, 80.0, log=True)

    return {
        "N": population,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "initial_energy": initial_energy,
    }


def run_single_seed(params: dict, seed: int, max_fes: int, max_iter: int) -> tuple[float, int]:
    np.random.seed(seed)
    initial_pos = np.random.uniform(0.0, 1.0, (params["N"], dim))

    with contextlib.redirect_stdout(io.StringIO()):
        best_fitness, _, cg_curve, _ = CAOA(
            N=params["N"],
            max_iter=max_iter,
            lb=0.0,
            ub=1.0,
            dim=dim,
            fobj=lambda X: objective_function(X, decoder),
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            delta=params["delta"],
            initial_energy=params["initial_energy"],
            max_FEs=max_fes,
            initial_pos=initial_pos,
        )

    return float(best_fitness), int(len(cg_curve))


def objective(trial: optuna.Trial, max_fes: int, max_iter: int, n_repeats: int) -> float:
    params = suggest_caoa_params(trial)

    scores = []
    effective_iters = []

    try:
        for repeat_idx in range(n_repeats):
            seed = BASE_SEED + (trial.number * 100) + repeat_idx
            score, used_iters = run_single_seed(params, seed, max_fes, max_iter)
            scores.append(score)
            effective_iters.append(used_iters)

            running_mean = float(np.mean(scores))
            trial.report(running_mean, step=repeat_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"Trial {trial.number} gagal: {exc}")
        return float("inf")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))

    trial.set_user_attr("scores", scores)
    trial.set_user_attr("score_std", std_score)
    trial.set_user_attr("mean_effective_iters", float(np.mean(effective_iters)))
    trial.set_user_attr("max_FEs", max_fes)
    trial.set_user_attr("max_iter", max_iter)
    trial.set_user_attr("n_repeats", n_repeats)

    return mean_score


def build_study() -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=BASE_SEED,
        multivariate=True,
        group=True,
        n_startup_trials=25,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=1,
    )

    return optuna.create_study(
        study_name="caoa_jssp_tuning",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )


def save_best_result(
    study: optuna.Study,
    output_file: str,
    max_fes: int,
    max_iter: int,
    n_repeats: int,
    mode: str,
) -> None:
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update(
        {
            "objective": "weighted_avg_tardiness_adjusted",
            "weights": {"w_j_default": 1.0},
            "mode": mode,
            "max_FEs": max_fes,
            "max_iter": max_iter,
            "n_repeats": n_repeats,
            "best_mean_score": float(best_trial.value),
            "best_score_std": float(best_trial.user_attrs.get("score_std", 0.0)),
            "data_dir": DEFAULT_DATA_DIR,
        }
    )

    os.makedirs("experiments", exist_ok=True)
    with open(output_file, "w") as file_obj:
        json.dump(best_params, file_obj, indent=4)


def print_study_summary(study: optuna.Study, elapsed: float) -> None:
    trial = study.best_trial

    print(f"Best mean fitness : {trial.value:.6f}")
    print(f"Best std fitness  : {trial.user_attrs.get('score_std', 0.0):.6f}")
    print(f"Completed trials  : {len(study.trials)}")
    print(f"Duration          : {elapsed:.2f} detik")

    print("\nParameter terbaik:")
    for key, value in trial.params.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.6f}")
        else:
            print(f"- {key}: {value}")

    scores = trial.user_attrs.get("scores", [])
    if scores:
        scores_fmt = ", ".join(f"{score:.6f}" for score in scores)
        print(f"\nSkor per repeat   : [{scores_fmt}]")


def parse_mode() -> str:
    mode = os.environ.get("CAOA_TUNING_MODE", "full").strip().lower()
    if mode not in {"full", "quick"}:
        raise ValueError(
            "CAOA_TUNING_MODE harus bernilai 'full' atau 'quick'."
        )
    return mode


def get_tuning_config(mode: str) -> dict:
    if mode == "quick":
        return {
            "mode": "quick",
            "max_fes": QUICK_MAX_FES,
            "max_iter": QUICK_MAX_ITER,
            "n_trials": QUICK_N_TRIALS,
            "timeout": QUICK_TIMEOUT,
            "n_repeats": QUICK_N_REPEATS,
            "output_file": QUICK_OUTPUT_FILE,
        }

    return {
        "mode": "full",
        "max_fes": MAX_FES,
        "max_iter": MAX_ITER,
        "n_trials": N_TRIALS,
        "timeout": TIMEOUT,
        "n_repeats": N_REPEATS,
        "output_file": OUTPUT_FILE,
    }


def main():
    global decoder, dim

    try:
        df_ops, df_machine_master, df_job_target = load_real_jssp_data(DEFAULT_DATA_DIR)
    except Exception as exc:
        print(f"Gagal memuat data: {exc}")
        return

    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=TidalChecker(),
        df_job_target=df_job_target,
    )
    dim = decoder.get_dimension()
    mode = parse_mode()
    config = get_tuning_config(mode)

    print("Memulai tuning CAOA")
    print(f"- Mode                    : {config['mode']}")
    print(f"- Dimensi                : {dim}")
    print(f"- Objective              : weighted_avg_tardiness_adjusted")
    print(f"- Max FEs per run        : {config['max_fes']}")
    print(f"- Repeat per trial       : {config['n_repeats']}")
    print(f"- Trial budget           : {config['n_trials']}")
    print(f"- Time budget            : {config['timeout']} detik")

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = build_study()

    start_time = time.time()
    try:
        study.optimize(
            lambda trial: objective(
                trial,
                max_fes=config["max_fes"],
                max_iter=config["max_iter"],
                n_repeats=config["n_repeats"],
            ),
            n_trials=config["n_trials"],
            timeout=config["timeout"],
        )
    except KeyboardInterrupt:
        print("Tuning dihentikan manual.")

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("Tidak ada trial yang selesai.")
        return

    elapsed = time.time() - start_time
    print_study_summary(study, elapsed)
    save_best_result(
        study,
        output_file=config["output_file"],
        max_fes=config["max_fes"],
        max_iter=config["max_iter"],
        n_repeats=config["n_repeats"],
        mode=config["mode"],
    )
    print(f"\nParameter terbaik disimpan ke '{config['output_file']}'")


if __name__ == "__main__":
    main()
