import numpy as np
import time

def GWO(objf, lb, ub, dim, pop_size=30, max_iter=100, max_FEs=None, initial_pos=None):
    # Inisialisasi posisi dan skor untuk Alpha, Beta, dan Delta
    # Diinisialisasi dengan tak terhingga karena kita asumsikan problem minimasi
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf") 
    
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    
    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    # Inisialisasi matriks posisi awal populasi serigala secara acak
    if initial_pos is not None:
        positions = initial_pos.copy()
    else:
        if isinstance(lb, list) or isinstance(lb, np.ndarray):
            positions = np.zeros((pop_size, dim))
            for i in range(dim):
                positions[:, i] = np.random.uniform(0, 1, pop_size) * (ub[i] - lb[i]) + lb[i]
        else:
            positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb

    # Array untuk menyimpan kurva konvergensi
    cg_curve = []
    fe_counter = 0

    # Loop Utama Iterasi
    start_total = time.perf_counter()
    for l in range(max_iter):
        if max_FEs is not None and fe_counter >= max_FEs:
            break
        iter_start = time.perf_counter()

        # Evaluasi Fitness dan Pembaruan Pemimpin
        for i in range(pop_size):
            if max_FEs is not None and fe_counter >= max_FEs:
                break
                
            # Batasi posisi serigala agar tidak keluar dari batas (lb, ub)
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            # Hitung nilai objektif
            fitness = objf(positions[i, :])
            fe_counter += 1
            
            # Pembaruan ketat hierarki: Alpha > Beta > Delta
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i, :].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i, :].copy()

        # Parameter 'a' menurun secara linier dari 2 hingga 0
        a = 2 - l * (2 / max_iter)

        # Pembaruan Posisi Serigala (Vektorisasi tingkat dimensi untuk efisiensi)
        for i in range(pop_size):
            # Vektor r1 dan r2 adalah angka acak [0, 1] 
            r1_alpha = np.random.rand(dim)
            r2_alpha = np.random.rand(dim)
            
            # Kalkulasi Tarikan Alpha
            A1 = 2 * a * r1_alpha - a
            C1 = 2 * r2_alpha
            D_alpha = np.abs(C1 * alpha_pos - positions[i, :])
            X1 = alpha_pos - A1 * D_alpha
            
            # Kalkulasi Tarikan Beta
            r1_beta = np.random.rand(dim)
            r2_beta = np.random.rand(dim)
            A2 = 2 * a * r1_beta - a
            C2 = 2 * r2_beta
            D_beta = np.abs(C2 * beta_pos - positions[i, :])
            X2 = beta_pos - A2 * D_beta
            
            # Kalkulasi Tarikan Delta
            r1_delta = np.random.rand(dim)
            r2_delta = np.random.rand(dim)
            A3 = 2 * a * r1_delta - a
            C3 = 2 * r2_delta
            D_delta = np.abs(C3 * delta_pos - positions[i, :])
            X3 = delta_pos - A3 * D_delta
            
            # Pembaruan posisi akhir: Rata-rata dari tarikan X1, X2, dan X3
            positions[i, :] = (X1 + X2 + X3) / 3.0

        # Simpan skor terbaik pada iterasi ini
        cg_curve.append(alpha_score)

        iter_time = time.perf_counter() - iter_start
        total_time = time.perf_counter() - start_total

        # Tampilkan stats per iterasi
        if (l + 1) % 1 == 0 or l == 0:
            print(
                f"Iterasi {l+1}/{max_iter} | "
                f"gBestScore: {alpha_score:.2f} | "
                f"FEs: {fe_counter} | "
                f"Iter time: {iter_time:.2f}s | "
                f"Total time: {total_time:.2f}s"
            )

    # Memetakan output sesuai permintaan: gBestScore, gBest, cg_curve
    gBestScore = alpha_score
    gBest = alpha_pos
    
    return gBestScore, gBest, np.array(cg_curve)