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

def objective(trial):
    # Mengambil global variables untuk menghindari pemuatan ulang setiap trial
    global decoder, dim
    
    # 1. Definisikan rentang (search space) parameter CAOA
    alpha = trial.suggest_float('alpha', 0.1, 1.5)
    beta = trial.suggest_float('beta', 0.1, 1.0)
    gamma = trial.suggest_float('gamma', 0.01, 0.5)
    delta = trial.suggest_float('delta', 0.1, 5.0)
    initial_energy = trial.suggest_float('initial_energy', 50.0, 200.0)
    
    def fobj(X):
        _, metrics = decoder.decode_from_continuous(X)
        return (0.7 * metrics['total_tardiness']) + (0.3 * metrics['total_congestion'])
    
    # 2. Definisikan parameter masalah (Ukuran Standar untuk Optima Overnight)
    N_pop = 20
    max_FEs = 2000 
    max_iter = 2000
    
    # Posisi awal identik per trial tidak wajib, tapi kita tetap berikan secara acak
    initial_pos = np.random.uniform(0.0, 1.0, (N_pop, dim))
    
    # Untuk menghilangkan output gBestScore per iterasi dari CAOA,
    # kita tangkap stdout sementara ke /dev/null (Optional)
    # Ini membuat log Optuna lebih bersih.
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # 3. Eksekusi CAOA dengan konfigurasi tuning
        best_fitness, _, _ = CAOA(
            N=N_pop,
            max_iter=max_iter,
            lb=0.0,
            ub=1.0,
            dim=dim,
            fobj=fobj,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            initial_energy=initial_energy,
            max_FEs=max_FEs,
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
    print("=" * 70)
    print(" BAYESIAN OPTIMIZATION: TUNING CAOA PARAMETERS WITH OPTUNA")
    print("=" * 70)
    
    global decoder, dim
    print("\n[INIT] Memuat data JSSP riil...")
    try:
        df_ops, df_machine_master, df_job_target = load_real_jssp_data("data/processed/")
    except Exception as e:
        print(f"\n[CRITICAL] Gagal memuat data: {e}")
        return
        
    print("[INIT] Membangunkan Tidal Checker...")
    tidal_checker = TidalChecker()

    decoder = ActiveScheduleDecoder(
        df_ops=df_ops,
        df_machine_master=df_machine_master,
        tidal_checker=tidal_checker,
        df_job_target=df_job_target,
    )
    dim = decoder.get_dimension()
    
    print(f"[INFO] Dimensi (Operations): {dim}")
    print("\n[CONFIG] Memulai Optuna Studi (MODE OVERNIGHT)...")
    print("   N_pop per trial     : 20")
    print("   max_FEs per trial   : 1000 (standard convergence)")
    print("   Total n_trials      : s.d. 300 trial")
    print("   Timeout Limit       : 8 Jam (28800 detik)")
    print("\n[PROSES] Tuning dimulai...\n")
    
    # Tambahkan stream log Optuna explicitly
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    study_name = "caoa_jssp_tuning"
    study = optuna.create_study(study_name=study_name, direction='minimize')
    
    start_time = time.time()
    try:
        # Menambahkan batas waktu maksimal 8 jam (28800 detik)
        # Jika Anda bangun sebelum 8 jam, Anda bisa tekan Ctrl+C.
        study.optimize(objective, n_trials=500, timeout=28800)
    except KeyboardInterrupt:
        print("\nPenelusuran dihentikan manual oleh pengguna (KeyboardInterrupt).")
    
    print("\n" + "=" * 70)
    print(" HASIL TUNING OPTUNA")
    print("=" * 70)
    
    if len(study.trials) > 0:
        print(f"Waktu Pencarian: {time.time() - start_time:.2f} detik")
        print("Trial terbaik:")
        trial = study.best_trial
        print(f"  Nilai Fitness: {trial.value:.4f}")
        print("  Parameter Terbaik:")
        for key, value in trial.params.items():
            print(f"    {key:<15}: {value:.4f}")
            
        # Simpan ke file log
        os.makedirs("experiments", exist_ok=True)
        output_file = "experiments/caoa_best_params.json"
        
        with open(output_file, 'w') as f:
            json.dump(trial.params, f, indent=4)
        print(f"\n[SUKSES] Parameter terbaik telah disimpan ke '{output_file}'")
    else:
        print("[WARNING] Tidak ada trial yang selesai.")

if __name__ == "__main__":
    main()
