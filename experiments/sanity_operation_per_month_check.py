import pandas as pd
import numpy as np

def run_sanity_check():
    # Load dataset
    df = pd.read_csv("data/processed/jssp_data.csv")
    
    # n hari = n * 24 jam = m jam
    batas_jam = 10 * 24
    
    # Memfilter operasi yang terjadi dalam waktu 30 hari pertama
    # Kita menggunakan A_lj (waktu kedatangan) sebagai patokan
    bulan_pertama_df = df[(df['A_lj'] >= 0) & (df['A_lj'] <= batas_jam)]
    
    total_ops = len(bulan_pertama_df)
    unique_jobs_count = bulan_pertama_df['job_id'].nunique()
    
    ops_per_job = bulan_pertama_df.groupby('job_id').size()
    max_ops_per_job = ops_per_job.max() if not ops_per_job.empty else 0
    
    hari = batas_jam // 24
    print(f"=== HASIL SANITY CHECK (HARI 1 - {hari}) ===")
    print(f"Total Operasi keseluruhan: {total_ops} operasi")
    print(f"Total Kapal yang beroperasi: {unique_jobs_count} kapal")
    print(f"Maksimal operasi yang dijalankan 1 kapal: {max_ops_per_job} operasi")

if __name__ == "__main__":
    run_sanity_check()
