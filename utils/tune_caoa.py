import numpy as np
import optuna
import time
import json
import os
import sys

from engine.decoder import ActiveScheduleDecoder
from engine.caoa import CAOA
from utils.data_loader import load_real_jssp_data
from engine.tidal_checker import TidalChecker

STATIC_INITIAL_ENERGY = 100.0
N_POP = 20
MAX_FES = 2000
MAX_ITER = 2000
N_TRIALS = 500
TIMEOUT = 28800
OUTPUT_FILE = "experiments/caoa_best_params.json"

def objective_function(X, decoder):
    _, metrics = decoder.decode_from_continuous(X)

    penalty_tardiness = 1000.0 * metrics['total_tardiness']
    penalty_congestion = 500.0 * metrics['total_congestion']
    penalty_tidal = 1.0 * metrics['total_tidal_delay']

    return penalty_tardiness + penalty_congestion + penalty_tidal

def objective(trial):
    global decoder, dim

    alpha = trial.suggest_float('alpha', 0.1, 1.5)
    beta = trial.suggest_float('beta', 0.1, 1.0)
    gamma = trial.suggest_float('gamma', 0.01, 0.5)
    delta = trial.suggest_float('delta', 0.1, 5.0)

    initial_pos = np.random.uniform(0.0, 1.0, (N_POP, dim))
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        best_fitness, _, _ = CAOA(
            N=N_POP,
            max_iter=MAX_ITER,
            lb=0.0,
            ub=1.0,
            dim=dim,
            fobj=lambda X: objective_function(X, decoder),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            initial_energy=STATIC_INITIAL_ENERGY,
            max_FEs=MAX_FES,
            initial_pos=initial_pos
        )
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Trial failed with error: {e}")
        return float('inf')
    finally:
        sys.stdout = old_stdout
        
    return best_fitness

def main():
    global decoder, dim

    try:
        df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    except Exception as e:
        print(f"Gagal memuat data: {e}")
        return

    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=TidalChecker(),
        df_job_target=df_job_target,
    )
    dim = decoder.get_dimension()

    print(f"Memulai tuning CAOA untuk dimensi {dim}")
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(study_name="caoa_jssp_tuning", direction='minimize')
    start_time = time.time()
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        print("Tuning dihentikan manual.")

    if not study.trials:
        print("Tidak ada trial yang selesai.")
        return

    trial = study.best_trial
    print(f"Best fitness: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"{key}: {value:.4f}")
    print(f"Durasi: {time.time() - start_time:.2f} detik")

    os.makedirs("experiments", exist_ok=True)
    best_params = {**trial.params, "initial_energy": STATIC_INITIAL_ENERGY}

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Parameter terbaik disimpan ke '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
