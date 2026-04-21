import pandas as pd
import os

def load_real_jssp_data(data_dir: str = "data/processed/") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "ops": os.path.join(data_dir, "jssp_data_sliced.csv"),
        "machine": os.path.join(data_dir, "machine_master.csv"),
        "target": os.path.join(data_dir, "job_target_time_sliced.csv")
    }

    for path in files.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df_ops = pd.read_csv(
        files["ops"],
        usecols=['job_id','voyage','op_seq','machine_id','A_lj','p_lj','TSail_lj']
    )
    df_ops['TSail_lj'] = df_ops['TSail_lj'].fillna(0.0)
    df_ops = df_ops.astype({
        'job_id': int, 'voyage': int, 'op_seq': int, 'machine_id': int, 
        'A_lj': float, 'p_lj': float, 'TSail_lj': float
    }).sort_values(['job_id', 'voyage', 'op_seq']).reset_index(drop=True)

    df_machine_raw = pd.read_csv(files["machine"])
    machine_cols = ['machine_id', 'num_berth']
    if 'PELABUHAN_LOGIS' in df_machine_raw.columns:
        machine_cols.append('PELABUHAN_LOGIS')
        
    df_machine = df_machine_raw[machine_cols].astype({'machine_id': int, 'num_berth': int})

    df_target = pd.read_csv(files["target"], usecols=['job_id', 'voyage', 'T_j'])
    df_target = df_target.astype({'job_id': int, 'voyage': int, 'T_j': float})

    missing_targets = set(df_ops[['job_id', 'voyage']]) - set(df_target[['job_id', 'voyage']])
    if missing_targets:
        raise ValueError(f"Job IDs kehilangan target waktu (T_j): {missing_targets}")

    return df_ops, df_machine, df_target

if __name__ == "__main__":
    ops, machines, targets = load_real_jssp_data("data/processed/")