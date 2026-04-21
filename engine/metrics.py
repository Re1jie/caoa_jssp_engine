import numpy as np
import pandas as pd

def compute_schedule_metrics(
    schedule_df: pd.DataFrame,
    job_target: dict[tuple[int, int], float] | None = None,
) -> dict:
    metrics = {
        'total_congestion': float(schedule_df['congestion_wait'].sum()),
        'total_tidal_delay': float(schedule_df['tidal_wait'].sum()),
        'total_wait': float((schedule_df['S_lj'] - schedule_df['A_lj']).clip(lower=0).sum()),
        'makespan': float(schedule_df['C_lj'].max() - schedule_df['A_lj'].min()),
        'total_tardiness': 0.0,
        'avg_tardiness': 0.0,
        'max_tardiness': 0.0,
    }

    if not job_target:
        return metrics

    tardiness_list = []
    grouped = schedule_df.sort_values(
        ['job_id', 'voyage', 'op_seq']
    ).groupby(['job_id', 'voyage'])

    for (job_id, voyage), group in grouped:
        first_arrival = group['A_lj'].min()
        last_completion = group['C_lj'].max()

        total_wait = group['congestion_wait'].sum() + group['tidal_wait'].sum()
        ideal_completion = last_completion - total_wait

        due = first_arrival + job_target.get((job_id, voyage), 0.0)
        true_due = max(due, ideal_completion)
        tardiness = max(0.0, last_completion - true_due)

        tardiness_list.append(tardiness)

    if tardiness_list:
        metrics['total_tardiness'] = float(sum(tardiness_list))
        metrics['avg_tardiness'] = float(np.mean(tardiness_list))
        metrics['max_tardiness'] = float(max(tardiness_list))

    return metrics
