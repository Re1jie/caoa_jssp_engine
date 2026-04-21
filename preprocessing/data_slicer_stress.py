import pandas as pd
import numpy as np
import os

def slice_stress_test_region_time(
    data_path: str = "data/processed/jssp_data.csv", 
    region: str = "Barat", 
    max_layers: int = 8, 
    out_data_path: str = "data/processed/jssp_data_sliced.csv",
    out_target_path: str = "data/processed/job_target_time_sliced.csv"
):
    print(f"Memulai STRESS-TEST Slicing untuk wilayah: {region} | Siklus: {max_layers}")
    
    # 1. Ekstraksi Data Operasi Utama
    df = pd.read_csv(data_path)
    
    if 'wilayah_kapal' not in df.columns:
        raise ValueError("[CRITICAL ERROR] Kolom 'wilayah_kapal' tidak ada di jssp_data.csv. Perbaiki data_transformer.py!")

    # 2. Isolasi Wilayah (Spatial Cut) & Waktu (Temporal Cut)
    sliced_df = df[(df['wilayah_kapal'] == region) & (df['layer_id'] < max_layers)].copy()
    
    if sliced_df.empty:
        raise ValueError(f"Tidak ada kapal yang beroperasi di wilayah {region} dengan kondisi tersebut.")

    # 3. RE-INDEXING (Wajib untuk Matrix Decoder CAOA)
    unique_jobs = sliced_df['job_id'].unique()
    remap_dict = {old_id: new_id for new_id, old_id in enumerate(unique_jobs)}
    sliced_df['job_id'] = sliced_df['job_id'].map(remap_dict)

    # 4. INJEKSI BIG BANG & TARGET DINAMIS (Stress-Test Logic)
    np.random.seed(42) # Agar eksperimen bisa direplikasi dengan hasil yang sama
    new_targets = []
    
    print("Menyuntikkan kompresi waktu (Big Bang) dan menghitung target dinamis...")
    for j in sliced_df['job_id'].unique():
        job_mask = sliced_df['job_id'] == j
        
        # [A] Kompresi Waktu Kedatangan
        # Ekstrak waktu mulai asli kapal ini
        original_start = sliced_df.loc[job_mask, 'A_lj'].min()
        
        # Paksa kapal ini tiba di pelabuhan pertamanya dalam rentang 24 jam pertama simulasi
        new_start = np.random.uniform(0, 24.0)
        
        # Geser seluruh sisa jadwal kapal ini secara merata agar jarak tempuh/layan tidak rusak
        time_shift = new_start - original_start
        sliced_df.loc[job_mask, 'A_lj'] += time_shift
        
        # [B] Kalkulasi Target Dinamis (T_j)
        # T_j = (Total Waktu Sandar + Total Waktu Layar) + Buffer X%
        total_p = sliced_df.loc[job_mask, 'p_lj'].sum()
        total_sail = sliced_df.loc[job_mask, 'TSail_lj'].sum()
        
        realistic_T_j = (total_p + total_sail) * 1.0 
        
        new_targets.append({
            'job_id': j,
            'route': 'stress_test',
            'T_j': realistic_T_j
        })

    # 5. Penataan Ulang Struktur Data
    # Urutkan ulang berdasarkan job_id dan kedatangan baru yang sudah terkompresi
    sliced_df = sliced_df.sort_values(by=['job_id', 'A_lj']).reset_index(drop=True)
    sliced_df['op_seq'] = sliced_df.groupby('job_id').cumcount()

    # 6. Pembuatan DataFrame Target Baru
    sliced_target = pd.DataFrame(new_targets)

    # 7. Pembersihan dan Ekspor
    sliced_df = sliced_df.drop(columns=['wilayah_kapal'])
    
    sliced_df.to_csv(out_data_path, index=False)
    sliced_target.to_csv(out_target_path, index=False)
    
    print(f"✅ Slicing & Stress-Test Selesai!")
    print(f"Total Kapal: {len(unique_jobs)}")
    print(f"Total Operasi (Dimensi CAOA): {len(sliced_df)}")
    print(f"Output tersimpan di:\n - {out_data_path}\n - {out_target_path}")

if __name__ == "__main__":
    # Eksekusi isolasi untuk wilayah yang diinginkan
    slice_stress_test_region_time(region="Timur", max_layers=5)