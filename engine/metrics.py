import numpy as np
import pandas as pd

def compute_schedule_metrics(
    schedule_df: pd.DataFrame,
    job_target: dict[tuple[int, int], float | dict[str, float]] | None = None,
) -> dict:
    metrics = {
        'total_congestion': float(schedule_df['congestion_wait'].sum()),
        'total_tidal_delay': float(schedule_df['tidal_wait'].sum()),
        'total_wait': float((schedule_df['S_lj'] - schedule_df['A_lj']).clip(lower=0).sum()),
        'makespan': float(schedule_df['C_lj'].max() - schedule_df['A_lj'].min()),
        'total_tardiness': 0.0,
        'avg_tardiness': 0.0,
        'max_tardiness': 0.0,
        'weighted_total_tardiness': 0.0,
        'weighted_avg_tardiness': 0.0,
    }

    if not job_target:
        return metrics

    tardiness_list = []
    weighted_tardiness_list = []
    grouped = {
        key: group
        for key, group in schedule_df.sort_values(
        ['job_id', 'voyage', 'op_seq']
    ).groupby(['job_id', 'voyage'])
    }

    schedule_keys = set(grouped.keys())
    target_keys = set(job_target.keys())

    missing_in_schedule = target_keys - schedule_keys
    extra_in_schedule = schedule_keys - target_keys

    if missing_in_schedule:
        missing_preview = sorted(missing_in_schedule)[:5]
        raise ValueError(
            f"Voyage target tidak muncul di schedule: {missing_preview}"
        )

    if extra_in_schedule:
        extra_preview = sorted(extra_in_schedule)[:5]
        raise ValueError(
            f"Schedule mengandung voyage tanpa target: {extra_preview}"
        )

    for (job_id, voyage), target_info in job_target.items():
        group = grouped[(job_id, voyage)]
        first_arrival = float(group['A_lj'].min())
        last_completion = group['C_lj'].max()

        if isinstance(target_info, dict):
            target_time = float(
                target_info.get('target_time', target_info.get('due_time', 0.0))
            )
            weight = float(target_info.get('weight', 1.0))
        else:
            target_time = float(target_info)
            weight = 1.0

        due = first_arrival + target_time
        tardiness = max(0.0, last_completion - due)

        tardiness_list.append(tardiness)
        weighted_tardiness_list.append(weight * tardiness)

    if tardiness_list:
        metrics['total_tardiness'] = float(sum(tardiness_list))
        metrics['avg_tardiness'] = float(sum(tardiness_list) / len(target_keys))
        metrics['max_tardiness'] = float(max(tardiness_list))
        metrics['weighted_total_tardiness'] = float(sum(weighted_tardiness_list))
        metrics['weighted_avg_tardiness'] = float(
            sum(weighted_tardiness_list) / len(target_keys)
        )

    return metrics
