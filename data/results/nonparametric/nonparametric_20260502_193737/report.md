# Nonparametric Statistical Test Report

Generated at: `2026-05-02T23:09:35`
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
| CAOA | 30 | 5757.233333 | 2.160513 | 5756.000000 | 5756.000000 | 5764.000000 | 5756.000000 | 5758.000000 |
| CAOASSR | 30 | 5756.300000 | 0.702213 | 5756.000000 | 5756.000000 | 5758.000000 | 5756.000000 | 5756.000000 |
| GWO | 30 | 5765.066667 | 11.750079 | 5758.000000 | 5756.000000 | 5792.000000 | 5756.000000 | 5773.000000 |

## Pairwise Tests

| Treatment | Control | rank-sum p | signed-rank p | A12 | magnitude | mean delta |
|---|---|---:|---:|---:|---|---:|
| CAOA | CAOASSR | 0.0492288699 | 0.0134662248 | 0.385000 | small | 0.933333 |
| CAOA | GWO | 0.0117656942 | 0.0043687315 | 0.674444 | medium | -7.833333 |
| CAOASSR | GWO | 0.0002143782 | 0.0003280071 | 0.740556 | large | -8.766667 |

Negative mean delta favors the treatment algorithm.