# Nonparametric Statistical Test Report

Generated at: `2026-05-03T04:41:29`
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
| CAOA | 30 | 5779.333333 | 11.757121 | 5782.000000 | 5756.000000 | 5801.000000 | 5771.250000 | 5785.500000 |
| CAOASSR | 30 | 5761.366667 | 3.079222 | 5760.500000 | 5756.000000 | 5768.000000 | 5759.000000 | 5764.000000 |
| GWO | 30 | 5778.833333 | 14.790685 | 5781.000000 | 5756.000000 | 5817.000000 | 5768.000000 | 5784.000000 |

## Pairwise Tests

| Treatment | Control | rank-sum p | signed-rank p | A12 | magnitude | mean delta |
|---|---|---:|---:|---:|---|---:|
| CAOA | CAOASSR | 0.0000000220 | 0.0000050325 | 0.080000 | large | 17.966667 |
| CAOA | GWO | 0.6783515856 | 0.5959930982 | 0.468333 | negligible | 0.500000 |
| CAOASSR | GWO | 0.0000030040 | 0.0000184684 | 0.850556 | large | -17.466667 |

Negative mean delta favors the treatment algorithm.