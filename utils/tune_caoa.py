import contextlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
import optuna

from engine.caoa import CAOA
from engine.decoder import ActiveScheduleDecoder
from engine.tidal_checker import TidalChecker
from utils.data_loader import load_real_jssp_data


MAX_FES = 2500
MAX_ITER = 5000
N_TRIALS = 200
N_REPEATS = 3
TIMEOUT = 28800

BASE_SEED = 20260421
DATA_DIR = "data/processed/"
OUTPUT_FILE = "experiments/caoa_best_params.json"
STORAGE_FILE = "experiments/caoa_tuning.sqlite3"
STUDY_NAME = "caoa_jssp_tuning"

POPULATION = 20
INITIAL_ENERGY = 10.0
PROGRESS_INTERVAL = 5

decoder = None
dim = 0


def init_decoder() -> None:
    global decoder, dim

    df_ops, df_machine_master, df_job_target = load_real_jssp_data(DATA_DIR)
    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=TidalChecker(),
        df_job_target=df_job_target,
    )
    dim = decoder.get_dimension()


def fitness(x: np.ndarray) -> float:
    _, metrics = decoder.decode_from_continuous(x)
    return metrics["total_tardiness"]


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 0.10, 0.90, step=0.01),
        "beta": trial.suggest_float("beta", 0.01, 0.25, step=0.01),
        "gamma": trial.suggest_float("gamma", 0.01, 0.10, step=0.01),
        "delta": trial.suggest_float("delta", 0.05, 20.00, step=0.01),
    }


def run_caoa(params: dict, seed: int) -> tuple[float, int]:
    np.random.seed(seed)
    initial_pos = np.random.uniform(0.0, 1.0, (POPULATION, dim))

    with contextlib.redirect_stdout(io.StringIO()):
        best_fitness, _, cg_curve, _ = CAOA(
            N=POPULATION,
            max_iter=MAX_ITER,
            lb=0.0,
            ub=1.0,
            dim=dim,
            fobj=fitness,
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            delta=params["delta"],
            initial_energy=INITIAL_ENERGY,
            max_FEs=MAX_FES,
            initial_pos=initial_pos,
        )

    return float(best_fitness), int(len(cg_curve))


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    scores = []
    used_iters = []

    for repeat_idx in range(N_REPEATS):
        seed = BASE_SEED + trial.number * 100 + repeat_idx
        score, n_iters = run_caoa(params, seed)
        scores.append(score)
        used_iters.append(n_iters)

        trial.report(float(np.mean(scores)), step=repeat_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("scores", scores)
    trial.set_user_attr("score_std", float(np.std(scores)))
    trial.set_user_attr("mean_effective_iters", float(np.mean(used_iters)))
    return float(np.mean(scores))


def storage_url() -> str:
    return f"sqlite:///{STORAGE_FILE}"


def load_study(seed: int = BASE_SEED) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=25)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)
    storage = optuna.storages.RDBStorage(
        url=storage_url(),
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    return optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def worker(worker_id: int, n_trials: int, start_time: float) -> int:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        init_decoder()
    study = load_study(seed=BASE_SEED + worker_id)
    remaining_time = max(1, TIMEOUT - int(time.time() - start_time))
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=remaining_time,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    return worker_id


def split_trials(n_trials: int, n_workers: int) -> list[int]:
    base = n_trials // n_workers
    extra = n_trials % n_workers
    return [base + int(idx < extra) for idx in range(n_workers)]


def worker_count() -> int:
    requested = int(os.environ.get("CAOA_TUNING_WORKERS", os.cpu_count() or 1))
    return max(1, min(requested, N_TRIALS))


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "--:--:--"

    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def trial_counts(study: optuna.Study) -> tuple[int, int, float | None]:
    states = optuna.trial.TrialState
    trials = study.trials
    finished = sum(
        trial.state in {states.COMPLETE, states.PRUNED, states.FAIL}
        for trial in trials
    )
    completed = sum(trial.state == states.COMPLETE for trial in trials)
    complete_values = [
        trial.value for trial in trials
        if trial.state == states.COMPLETE and trial.value is not None
    ]
    best = min(complete_values) if complete_values else None
    return finished, completed, best


def print_progress(start_time: float) -> None:
    study = load_study()
    finished, completed, best = trial_counts(study)
    elapsed = time.time() - start_time
    ratio = min(1.0, finished / N_TRIALS)
    filled = int(30 * ratio)
    eta = None if finished == 0 else elapsed * (N_TRIALS - finished) / finished
    best_text = "--" if best is None else f"{best:.4f}"
    bar = "#" * filled + "-" * (30 - filled)

    print(
        f"\r[{bar}] {finished:>3}/{N_TRIALS} "
        f"({ratio * 100:5.1f}%) | complete: {completed:>3} | "
        f"best: {best_text} | elapsed: {format_duration(elapsed)} | "
        f"ETA: {format_duration(eta)}",
        end="",
        flush=True,
    )


def save_best(study: optuna.Study) -> None:
    best = study.best_trial
    result = {
        **best.params,
        "N": POPULATION,
        "initial_energy": INITIAL_ENERGY,
        "objective": "total_tardiness",
        "max_FEs": MAX_FES,
        "max_iter": MAX_ITER,
        "n_repeats": N_REPEATS,
        "best_mean_score": float(best.value),
        "best_score_std": float(best.user_attrs.get("score_std", 0.0)),
        "study_name": STUDY_NAME,
        "storage": STORAGE_FILE,
        "data_dir": DATA_DIR,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as file_obj:
        json.dump(result, file_obj, indent=4)


def print_summary(study: optuna.Study, elapsed: float) -> None:
    best = study.best_trial
    _, completed, _ = trial_counts(study)

    print(f"Best mean fitness : {best.value:.6f}")
    print(f"Best std fitness  : {best.user_attrs.get('score_std', 0.0):.6f}")
    print(f"Completed trials  : {completed}")
    print(f"Duration          : {elapsed:.2f} detik")
    print("\nParameter terbaik:")
    for key, value in best.params.items():
        print(f"- {key}: {value:.6f}")

    scores = best.user_attrs.get("scores", [])
    if scores:
        print("\nSkor per repeat   : [" + ", ".join(f"{v:.6f}" for v in scores) + "]")


def main() -> None:
    os.makedirs("experiments", exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    init_decoder()
    study = load_study()
    existing_finished, _, _ = trial_counts(study)
    remaining_trials = max(0, N_TRIALS - existing_finished)
    n_workers = min(worker_count(), max(1, remaining_trials))
    trial_splits = [
        count for count in split_trials(remaining_trials, n_workers)
        if count
    ]

    print("Memulai tuning CAOA")
    print(f"- Mode             : full")
    print(f"- Dimensi          : {dim}")
    print(f"- Objective        : total_tardiness")
    print(f"- Parameter tuning : alpha, beta, gamma, delta")
    print(f"- Fixed N          : {POPULATION}")
    print(f"- Fixed energy     : {INITIAL_ENERGY}")
    print(f"- Max FEs per run  : {MAX_FES}")
    print(f"- Repeat per trial : {N_REPEATS}")
    print(f"- Trial budget     : {N_TRIALS}")
    print(f"- Time budget      : {TIMEOUT} detik")
    print(f"- Existing trials  : {existing_finished}")
    print(f"- Remaining trials : {remaining_trials}")
    print(f"- Workers          : {len(trial_splits)}")
    print(f"- SQLite storage   : {STORAGE_FILE}")

    if remaining_trials == 0:
        print_progress(time.time())
        print()
        if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            print_summary(study, 0.0)
            save_best(study)
            print(f"\nParameter terbaik disimpan ke '{OUTPUT_FILE}'")
        return

    start_time = time.time()
    try:
        with ProcessPoolExecutor(max_workers=len(trial_splits)) as executor:
            futures = [
                executor.submit(worker, idx, count, start_time)
                for idx, count in enumerate(trial_splits, start=1)
            ]
            while futures:
                done, pending = wait(futures, timeout=PROGRESS_INTERVAL)
                print_progress(start_time)
                for future in done:
                    future.result()
                futures = list(pending)
    except KeyboardInterrupt:
        print("\nTuning dihentikan manual.")

    print()
    elapsed = time.time() - start_time
    study = load_study()
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print("Tidak ada trial yang selesai.")
        return

    print_summary(study, elapsed)
    save_best(study)
    print(f"\nParameter terbaik disimpan ke '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
