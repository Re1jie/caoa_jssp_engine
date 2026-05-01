"""
==========================================================================
 UJI NON-PARAMETRIK: CAOA vs GWO
 Wilcoxon Rank-Sum Test & Vargha-Delaney A12 Effect Size
==========================================================================
 Tujuan  : Membuktikan bahwa perbedaan kinerja antara CAOA dan GWO
           signifikan secara statistik, bukan karena kebetulan acak.
 Metode  : 1. Wilcoxon Rank-Sum (Mann-Whitney U) Test
           2. Vargha-Delaney A12 Effect Size
 Setup   : Data JSSP identik, N_pop & max_iter identik untuk kedua algoritma.
==========================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

from engine.decoder import ActiveScheduleDecoder
from engine.caoa import CAOA
from engine.gwo import GWO
from utils.data_loader import load_real_jssp_data
from engine.tidal_checker import TidalChecker


# ─── VARGHA-DELANEY A12 EFFECT SIZE ──────────────────────────────────────────
def vargha_delaney_A12(group_A: np.ndarray, group_B: np.ndarray) -> float:
    """
    Menghitung Vargha-Delaney A12 effect size (measure of stochastic superiority).
    
    Interpretasi (Vargha & Delaney, 2000):
        A12 ≈ 0.50  → Negligible (tidak ada perbedaan)
        A12 ≈ 0.56  → Small effect
        A12 ≈ 0.64  → Medium effect
        A12 ≈ 0.71+ → Large effect
    
    A12 > 0.5 berarti group_A cenderung menghasilkan nilai LEBIH BESAR dari group_B.
    Untuk problem minimasi, A12 < 0.5 berarti group_A LEBIH BAIK (nilainya lebih kecil).
    """
    m, n = len(group_A), len(group_B)
    
    # Hitung jumlah pasangan di mana A > B dan A == B
    more = 0.0
    equal = 0.0
    for a in group_A:
        for b in group_B:
            if a > b:
                more += 1
            elif a == b:
                equal += 1
    
    A12 = (more + 0.5 * equal) / (m * n)
    return A12


def interpret_A12(A12_value: float) -> str:
    """Interpretasi level magnitude effect size A12."""
    # Kita lihat distance dari 0.5 (negligible)
    dist = abs(A12_value - 0.5)
    if dist < 0.06:
        return "Negligible"
    elif dist < 0.14:
        return "Small"
    elif dist < 0.21:
        return "Medium"
    else:
        return "Large"


# ─── MAIN EXPERIMENT ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" UJI NON-PARAMETRIK: CAOA vs GWO")
    print(" Wilcoxon Rank-Sum Test & Vargha-Delaney A12 Effect Size")
    print("=" * 70)

    # ─── 1. Muat Data (Identik untuk Kedua Algoritma) ────────────────────────
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

    # ─── 2. Definisikan Fungsi Objektif (Sama untuk Kedua Algoritma) ─────────
    def objective_function(X):
        _, metrics = decoder.decode_from_continuous(X)
        return metrics['weighted_avg_tardiness']

    # ─── 3. Parameter Eksperimen ─────────────────────────────────────────────
    N_pop     = 10        # Ukuran populasi (identik)
    max_FEs   = 2500     # Maksimum fungsi evaluasi (BENCHMARKING TERFAIR)
    max_iter  = 2000      # Batas atas loop iterasi (teori)
    N_runs    = 30        # Jumlah independent run (standar statistik)
    alpha_sig = 0.05      # Level signifikansi

    print(f"\n[CONFIG] Parameter Eksperimen:")
    print(f"   N_pop     = {N_pop}")
    print(f"   max_FEs   = {max_FEs} (Apple-to-apple evaluasi fungsi)")
    print(f"   N_runs    = {N_runs} (independent runs)")
    print(f"   alpha     = {alpha_sig}")
    print(f"   dim       = {dim}")

    # ─── 4. Eksekusi N_runs untuk Kedua Algoritma ────────────────────────────
    caoa_fitness_results = []
    gwo_fitness_results  = []

    for run in range(1, N_runs + 1):
        print(f"\n{'─'*50}")
        print(f" RUN {run}/{N_runs}")
        print(f"{'─'*50}")

        # Inisialisasi populasi awal yang identik (fair start)
        initial_pos = np.random.uniform(0.0, 1.0, (N_pop, dim))

        # --- CAOA ---
        print(f"  [CAOA] Mengeksekusi...")
        caoa_best, _, _ = CAOA(
            N=N_pop,
            max_iter=max_iter,
            lb=0.0,
            ub=1.0,
            dim=dim,
            fobj=objective_function,
            max_FEs=max_FEs,
            initial_pos=initial_pos
        )
        caoa_fitness_results.append(caoa_best)
        print(f"  [CAOA] Best Fitness = {caoa_best:.4f}")

        # --- GWO ---
        print(f"  [GWO]  Mengeksekusi...")
        gwo_best, _, _ = GWO(
            objf=objective_function,
            lb=0.0,
            ub=1.0,
            dim=dim,
            pop_size=N_pop,
            max_iter=max_iter,
            max_FEs=max_FEs,
            initial_pos=initial_pos
        )
        gwo_fitness_results.append(gwo_best)
        print(f"  [GWO]  Best Fitness = {gwo_best:.4f}")

    caoa_arr = np.array(caoa_fitness_results)
    gwo_arr  = np.array(gwo_fitness_results)

    # ─── 5. Statistik Deskriptif ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" STATISTIK DESKRIPTIF (N = {})".format(N_runs))
    print("=" * 70)
    desc_header = f"{'Metrik':<20} | {'CAOA':<20} | {'GWO':<20}"
    print(desc_header)
    print("-" * len(desc_header))
    print(f"{'Mean':<20} | {np.mean(caoa_arr):<20.4f} | {np.mean(gwo_arr):<20.4f}")
    print(f"{'Std Dev':<20} | {np.std(caoa_arr, ddof=1):<20.4f} | {np.std(gwo_arr, ddof=1):<20.4f}")
    print(f"{'Median':<20} | {np.median(caoa_arr):<20.4f} | {np.median(gwo_arr):<20.4f}")
    print(f"{'Min':<20} | {np.min(caoa_arr):<20.4f} | {np.min(gwo_arr):<20.4f}")
    print(f"{'Max':<20} | {np.max(caoa_arr):<20.4f} | {np.max(gwo_arr):<20.4f}")

    # ─── 6. Wilcoxon Rank-Sum Test (Mann-Whitney U) ──────────────────────────
    print("\n" + "=" * 70)
    print(" WILCOXON RANK-SUM TEST (Mann-Whitney U)")
    print("=" * 70)
    print(f" H₀: Tidak ada perbedaan distribusi fitness antara CAOA dan GWO")
    print(f" H₁: Terdapat perbedaan distribusi fitness antara CAOA dan GWO")
    print(f" α  = {alpha_sig}")
    print("-" * 70)

    stat_U, p_value = stats.mannwhitneyu(caoa_arr, gwo_arr, alternative='two-sided')
    
    print(f" U-Statistic  = {stat_U:.4f}")
    print(f" p-value       = {p_value:.6f}")
    
    if p_value < alpha_sig:
        print(f"\n ✅ KEPUTUSAN: H₀ DITOLAK (p = {p_value:.6f} < α = {alpha_sig})")
        print(f"    → Terdapat perbedaan SIGNIFIKAN secara statistik antara CAOA dan GWO.")
    else:
        print(f"\n ❌ KEPUTUSAN: H₀ DITERIMA / GAGAL DITOLAK (p = {p_value:.6f} ≥ α = {alpha_sig})")
        print(f"    → Tidak cukup bukti untuk menyatakan perbedaan signifikan.")

    # ─── 7. Vargha-Delaney A12 Effect Size ───────────────────────────────────
    print("\n" + "=" * 70)
    print(" VARGHA-DELANEY A12 EFFECT SIZE")
    print("=" * 70)

    # A12(CAOA, GWO): probabilitas CAOA > GWO
    # Untuk minimasi, A12 < 0.5 → CAOA lebih baik (nilai fitness CAOA lebih kecil)
    A12 = vargha_delaney_A12(caoa_arr, gwo_arr)
    magnitude = interpret_A12(A12)

    print(f" A12(CAOA, GWO)  = {A12:.4f}")
    print(f" Magnitude       = {magnitude}")
    print("-" * 70)
    print(f" Interpretasi (Problem Minimasi):")
    if A12 < 0.5:
        print(f"   A12 = {A12:.4f} < 0.5 → CAOA LEBIH BAIK daripada GWO")
        print(f"   (Fitness CAOA cenderung LEBIH KECIL / lebih optimal)")
    elif A12 > 0.5:
        print(f"   A12 = {A12:.4f} > 0.5 → GWO LEBIH BAIK daripada CAOA")
        print(f"   (Fitness GWO cenderung LEBIH KECIL / lebih optimal)")
    else:
        print(f"   A12 = {A12:.4f} ≈ 0.5 → Tidak ada perbedaan stochastic")
    print(f"   Effect Size   = {magnitude}")

    # ─── 8. Rangkuman Akhir ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RANGKUMAN HASIL")
    print("=" * 70)

    winner_mean = "CAOA" if np.mean(caoa_arr) < np.mean(gwo_arr) else "GWO"
    improvement = abs(np.mean(caoa_arr) - np.mean(gwo_arr)) / max(np.mean(gwo_arr), 1e-10) * 100

    summary_data = {
        'Algoritma': ['CAOA', 'GWO'],
        'Mean Fitness': [np.mean(caoa_arr), np.mean(gwo_arr)],
        'Std Dev': [np.std(caoa_arr, ddof=1), np.std(gwo_arr, ddof=1)],
        'Median': [np.median(caoa_arr), np.median(gwo_arr)],
        'Min': [np.min(caoa_arr), np.min(gwo_arr)],
        'Max': [np.max(caoa_arr), np.max(gwo_arr)],
    }
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    print(f"\n Pemenang berdasarkan Mean Fitness : {winner_mean}")
    print(f" Improvement                       : {improvement:.2f}%")
    print(f" Wilcoxon p-value                  : {p_value:.6f} ({'Signifikan' if p_value < alpha_sig else 'Tidak Signifikan'})")
    print(f" Vargha-Delaney A12                : {A12:.4f} ({magnitude})")

    # ─── 9. Simpan Hasil ke CSV ──────────────────────────────────────────────
    output_path = "experiments/nonparametric_results.csv"
    df_raw = pd.DataFrame({
        'Run': list(range(1, N_runs + 1)),
        'CAOA_Fitness': caoa_fitness_results,
        'GWO_Fitness': gwo_fitness_results,
    })
    df_raw.to_csv(output_path, index=False)
    print(f"\n Data mentah per-run disimpan ke: {output_path}")

    # Simpan ringkasan statistik
    stats_output_path = "experiments/nonparametric_summary.csv"
    df_stats = pd.DataFrame({
        'Metric': ['N_pop', 'max_FEs', 'N_runs', 'alpha',
                   'CAOA_mean', 'CAOA_std', 'CAOA_median', 'CAOA_min', 'CAOA_max',
                   'GWO_mean', 'GWO_std', 'GWO_median', 'GWO_min', 'GWO_max',
                   'U_statistic', 'p_value', 'H0_rejected',
                   'A12', 'A12_magnitude', 'Winner'],
        'Value': [N_pop, max_FEs, N_runs, alpha_sig,
                  np.mean(caoa_arr), np.std(caoa_arr, ddof=1), np.median(caoa_arr), np.min(caoa_arr), np.max(caoa_arr),
                  np.mean(gwo_arr), np.std(gwo_arr, ddof=1), np.median(gwo_arr), np.min(gwo_arr), np.max(gwo_arr),
                  stat_U, p_value, p_value < alpha_sig,
                  A12, magnitude, winner_mean],
    })
    df_stats.to_csv(stats_output_path, index=False)
    print(f" Ringkasan statistik disimpan ke  : {stats_output_path}")

    print("\n" + "=" * 70)
    print(" EKSPERIMEN SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()
