# Nonparametric Statistical Test Report

Generated at: `2026-05-03T02:09:40`
Objective: `minimize total_tardiness`
Runs per algorithm: `30`
Seeds: `1` to `30`

## Method

- Wilcoxon rank-sum / Mann-Whitney U tests whether two independent samples differ in distribution/location.
- Wilcoxon signed-rank is reported as an additional paired test because every algorithm uses the same controlled seed index.
- Vargha-Delaney A12 reports stochastic superiority for minimization: values above 0.5 favor the treatment algorithm.

## Descriptive Statistics

| Algorithm | n | mean | std | median | min | max | q1 | q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAOA | 30 | 5760.600000 | 5.957319 | 5758.000000 | 5756.000000 | 5781.000000 | 5756.500000 | 5760.000000 |
| CAOASSR | 30 | 5756.233333 | 0.626062 | 5756.000000 | 5756.000000 | 5758.000000 | 5756.000000 | 5756.000000 |
| GWO | 30 | 5769.266667 | 13.211628 | 5764.000000 | 5756.000000 | 5800.000000 | 5756.500000 | 5781.000000 |

## Pairwise Tests

| Treatment | Control | rank-sum p | signed-rank p | A12 | magnitude | mean delta |
|---|---|---:|---:|---:|---|---:|
| CAOA | CAOASSR | 0.0000008284 | 0.0000614755 | 0.166111 | large | 4.366667 |
| CAOA | GWO | 0.0308062419 | 0.0097648044 | 0.660556 | medium | -8.666667 |
| CAOASSR | GWO | 0.0000004319 | 0.0000354543 | 0.843889 | large | -13.033333 |

Negative mean delta favors the treatment algorithm.