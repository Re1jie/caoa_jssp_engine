import numpy as np
import pandas as pd

INFEASIBLE_TARDINESS_PENALTY = 1e12


def build_infeasible_metrics(
    reason: str,
    penalty: float = INFEASIBLE_TARDINESS_PENALTY,
) -> dict:
    return {
        'total_tardiness': float(penalty),
        'max_tardiness': float(penalty),
        'late_voyage_count': 0,
        'is_feasible': False,
        'infeasible_reason': reason,
        'penalty_tardiness': float(penalty),
    }


def compute_schedule_metrics(
    schedule_df: pd.DataFrame,
    job_target: dict[tuple[int, int], float | dict[str, float]] | None = None,
) -> dict:
    metrics = {
        'total_tardiness': 0.0,
        'max_tardiness': 0.0,
        'late_voyage_count': 0,
        'is_feasible': True,
        'infeasible_reason': None,
        'penalty_tardiness': 0.0,
    }

    if not job_target:
        return metrics

    tardiness_list = []
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
            target_time = float(target_info.get('target_time', target_info.get('due_time', 0.0)))
        else:
            target_time = float(target_info)

        due = first_arrival + target_time
        tardiness = max(0.0, last_completion - due)

        tardiness_list.append(tardiness)

    if tardiness_list:
        metrics['total_tardiness'] = float(sum(tardiness_list))
        metrics['max_tardiness'] = float(max(tardiness_list))
        metrics['late_voyage_count'] = int(sum(t > 0 for t in tardiness_list))

    return metrics
