# CAOA vs CAOASSR Statistical Test Report

Generated at: `2026-04-30T20:31:02`
Runs per algorithm: `30`
Max iterations: `100`
Population size: `20`

## Descriptive statistics: total tardiness

| Algorithm | n | mean | std | median | min | max | q1 | q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAOA | 30 | 3585.666667 | 12.262456 | 3581.000000 | 3575.000000 | 3623.000000 | 3581.000000 | 3586.000000 |
| CAOASSR | 30 | 3579.833333 | 3.141308 | 3580.000000 | 3575.000000 | 3586.000000 | 3577.250000 | 3581.000000 |

## Pairwise by-seed comparison

- CAOASSR better than CAOA: `19` seeds
- CAOASSR equal to CAOA: `3` seeds
- CAOASSR worse than CAOA: `8` seeds
- Mean delta `CAOASSR - CAOA`: `-5.833333`
- Median delta `CAOASSR - CAOA`: `-3.500000`

Negative delta means CAOASSR is better because total tardiness is minimized.

## Wilcoxon rank-sum test

- Statistic: `-2.594667`
- p-value, two-sided: `0.0094682697`
- Null hypothesis: the two independent samples come from distributions with equal location.

## Additional paired Wilcoxon signed-rank test

- Statistic: `79.500000`
- p-value, two-sided: `0.0084179982`
- This is included because the experiment uses matched seeds.

## Vargha-Delaney A12

- A12_CAOASSR_better: `0.695000`
- Magnitude: `medium`
- Interpretation: `A12 > 0.5` means CAOASSR tends to produce lower total tardiness than CAOA.

## Files

- raw_results_csv: `data/result/statistical_tests/caoa_vs_caoassr_20260430_184733/raw_results.csv`
- paired_by_seed_csv: `data/result/statistical_tests/caoa_vs_caoassr_20260430_184733/paired_by_seed_results.csv`
- summary_json: `data/result/statistical_tests/caoa_vs_caoassr_20260430_184733/summary.json`
- report_md: `data/result/statistical_tests/caoa_vs_caoassr_20260430_184733/report.md`
- fcfs_metrics_json: `data/result/statistical_tests/caoa_vs_caoassr_20260430_184733/fcfs_metrics.json`
