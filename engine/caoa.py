import time

import numpy as np

def CAOA(N, max_iter, lb, ub, dim, fobj, alpha=0.3, beta=0.1, gamma=0.1, delta=1e-3, initial_energy=10.0, max_FEs=None, initial_pos=None):
    # 1. Inisialisasi Populasi dan Energi (Persamaan 2 & 3)
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub, dtype=float)
    
    if initial_pos is not None:
        pos = initial_pos.copy()
    else:
        pos = lb + (ub - lb) * np.random.rand(N, dim)
        
    energies = np.full(N, initial_energy, dtype=float)
    
    fe_counter = 0
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = fobj(pos[i])
        fe_counter += 1
    
    # Track Global Best
    best_idx = np.argmin(fitness)
    gBestScore = fitness[best_idx]
    gBest = pos[best_idx].copy()
    cg_curve = []
    avg_curve = []
    
    # 2. Main Loop
    start_total = time.perf_counter()
    for t in range(max_iter):
        iter_start = time.perf_counter()
        depleted_count = 0

        if max_FEs is not None and fe_counter >= max_FEs:
            break
            
        # Seleksi Titik Penyergapan / Probabilitas Pemimpin (Persamaan 6)
        # Catatan: fitness digeser ke 0 untuk mencegah error Division by Zero / angka negatif
        f_shifted = fitness - np.min(fitness) 
        probs = 1.0 / (1.0 + f_shifted)
        probs /= probs.sum()
        
        for i in range(N):
            if max_FEs is not None and fe_counter >= max_FEs:
                break
                
            # Pilih pemimpin secara stokastik
            leader_idx = np.random.choice(N, p=probs)
            if i == leader_idx:
                continue
                
            leader_pos = pos[leader_idx].copy()
            old_pos = pos[i].copy()
            old_fit = fitness[i]
            
            # Mekanisme Berburu (Persamaan 9)
            r = np.random.rand(dim)
            new_pos = pos[i] + alpha * (leader_pos - pos[i]) + beta * (1.0 - 2.0 * r)
            new_pos = np.clip(new_pos, lb, ub)
            new_fit = fobj(new_pos)
            fe_counter += 1
            
            # Waktu Serangan / Adaptasi jika solusi memburuk tajam (Persamaan 7 & 8)
            if abs(new_fit - old_fit) > delta and new_fit > old_fit:
                if max_FEs is None or fe_counter < max_FEs:
                    new_pos = lb + (ub - lb) * np.random.rand(dim)
                    new_pos = np.clip(new_pos, lb, ub)
                    new_fit = fobj(new_pos)
                    fe_counter += 1
            
            # Update posisi buaya
            pos[i] = new_pos
            fitness[i] = new_fit
            
            # Peluruhan Energi akibat pergerakan (Persamaan 4 & 5)
            dist = np.linalg.norm(new_pos - old_pos)
            energies[i] -= gamma * dist
            
            # Pemulihan Energi & Reinisialisasi Posisi jika kelelahan
            if energies[i] <= 0:
                depleted_count += 1
                if max_FEs is None or fe_counter < max_FEs:
                    pos[i] = lb + (ub - lb) * np.random.rand(dim)
                    pos[i] = np.clip(pos[i], lb, ub)
                    energies[i] = initial_energy
                    fitness[i] = fobj(pos[i])
                    fe_counter += 1
        
        # Update Global Best (Persamaan 12)
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < gBestScore:
            gBestScore = fitness[min_idx]
            gBest = pos[min_idx].copy()
            
        cg_curve.append(gBestScore)
        avg_curve.append(np.mean(fitness))  # Lacak rata-rata populasi

        iter_time = time.perf_counter() - iter_start
        total_time = time.perf_counter() - start_total
        
        # Tampilkan stats per iterasi
        if (t + 1) % 1 == 0 or t == 0:
            print(
                f"Iterasi {t+1}/{max_iter} | "
                f"Populasi: {N} | "
                f"gBest: {gBestScore:.2f} | "
                f"Rata-rata: {np.mean(fitness):.2f} | "
                f"FEs: {fe_counter} | "
                f"Energy depleted: {depleted_count} | "
                f"Iter time: {iter_time:.4f}s | "
                f"Total time: {total_time:.2f}s"
            )
        
    return gBestScore, gBest, np.array(cg_curve), np.array(avg_curve)
