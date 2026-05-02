# Nonparametric Statistical Test Report

Generated at: `2026-05-02T19:36:54`
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
| CAOA | 30 | 5763.500000 | 7.384747 | 5760.000000 | 5756.000000 | 5783.000000 | 5758.000000 | 5766.000000 |
| CAOASSR | 30 | 5757.466667 | 2.956388 | 5756.500000 | 5756.000000 | 5772.000000 | 5756.000000 | 5758.000000 |
| GWO | 30 | 5773.833333 | 14.241957 | 5773.000000 | 5756.000000 | 5801.000000 | 5760.750000 | 5784.500000 |

## Pairwise Tests

| Treatment | Control | rank-sum p | signed-rank p | A12 | magnitude | mean delta |
|---|---|---:|---:|---:|---|---:|
| CAOA | CAOASSR | 0.0000011923 | 0.0000262960 | 0.145556 | large | 6.033333 |
| CAOA | GWO | 0.0104368484 | 0.0020559635 | 0.692222 | medium | -10.333333 |
| CAOASSR | GWO | 0.0000011033 | 0.0000343977 | 0.858889 | large | -16.366667 |

Negative mean delta favors the treatment algorithm.