import pandas as pd
import numpy as np

def slice_data_by_region_and_time(
    data_path: str = "data/processed/jssp_data.csv", 
    target_path: str = "data/processed/job_target_time.csv",
    region: str = "Timur", 
    max_layers: int = 10, 
    out_data_path: str = "data/processed/jssp_data_sliced.csv",
    out_target_path: str = "data/processed/job_target_time_sliced.csv"
):
    df = pd.read_csv(data_path)
    df_target = pd.read_csv(target_path)
    
    if 'wilayah_kapal' not in df.columns:
        raise ValueError("Kolom 'wilayah_kapal' tidak ditemukan pada jssp_data.csv.")

    sliced_df = df[(df['wilayah_kapal'] == region) & (df['layer_id'] < max_layers)].copy()
    
    if sliced_df.empty:
        raise ValueError(f"Data kosong untuk wilayah {region} dengan layer < {max_layers}.")

    original_jobs = sliced_df['job_id'].unique()
    remap_dict = {old: new for new, old in enumerate(original_jobs)}
    
    sliced_df['job_id'] = sliced_df['job_id'].map(remap_dict)
    sliced_df = sliced_df.sort_values(by=['job_id', 'A_lj']).reset_index(drop=True)
    sliced_df['op_seq'] = sliced_df.groupby('job_id').cumcount()
    sliced_df = sliced_df.drop(columns=['wilayah_kapal'])

    sliced_target = df_target[df_target['job_id'].isin(original_jobs)].copy()
    sliced_target['job_id'] = sliced_target['job_id'].map(remap_dict)
    sliced_target = sliced_target.sort_values(by=['job_id']).reset_index(drop=True)

    sliced_df.to_csv(out_data_path, index=False)
    sliced_target.to_csv(out_target_path, index=False)
    
    print(f"Slicing Region: {region} | Jobs: {len(original_jobs)} | Operations: {len(sliced_df)}")

if __name__ == "__main__":
    slice_data_by_region_and_time(region="Timur", max_layers=999)