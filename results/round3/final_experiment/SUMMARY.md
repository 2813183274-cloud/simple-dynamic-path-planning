# P3-05 final results

No parameter search. Three seeds per PPO arm; DWA is a single deterministic baseline.

| Split | Method | Success | Static collision | Dynamic collision | Timeout |
|---|---|---:|---:|---:|---:|
| same_distribution | base-seed1 | 179/240 | 48 | 13 | 0 |
| same_distribution | base-seed2 | 156/240 | 53 | 31 | 0 |
| same_distribution | base-seed3 | 186/240 | 32 | 22 | 0 |
| same_distribution | instant-seed1 | 190/240 | 31 | 19 | 0 |
| same_distribution | instant-seed2 | 184/240 | 37 | 19 | 0 |
| same_distribution | instant-seed3 | 186/240 | 38 | 16 | 0 |
| same_distribution | predictive-seed1 | 185/240 | 38 | 17 | 0 |
| same_distribution | predictive-seed2 | 168/240 | 45 | 27 | 0 |
| same_distribution | predictive-seed3 | 180/240 | 44 | 16 | 0 |
| same_distribution | DWA | 239/240 | 0 | 0 | 1 |
| challenge | base-seed1 | 70/120 | 29 | 21 | 0 |
| challenge | base-seed2 | 62/120 | 43 | 15 | 0 |
| challenge | base-seed3 | 77/120 | 27 | 16 | 0 |
| challenge | instant-seed1 | 80/120 | 18 | 22 | 0 |
| challenge | instant-seed2 | 68/120 | 31 | 21 | 0 |
| challenge | instant-seed3 | 82/120 | 21 | 17 | 0 |
| challenge | predictive-seed1 | 67/120 | 32 | 21 | 0 |
| challenge | predictive-seed2 | 61/120 | 39 | 20 | 0 |
| challenge | predictive-seed3 | 84/120 | 15 | 21 | 0 |
| challenge | DWA | 118/120 | 0 | 0 | 2 |

## Seed means and sample SD

- same_distribution base: 72.36% +/- 6.54 pp.
- same_distribution instant: 77.78% +/- 1.27 pp.
- same_distribution predictive: 74.03% +/- 3.64 pp.
- challenge base: 58.06% +/- 6.25 pp.
- challenge instant: 63.89% +/- 6.31 pp.
- challenge predictive: 58.89% +/- 9.94 pp.

## Paired hierarchical bootstrap (95%)

- same_distribution, predictive minus instant: -3.75 pp [-8.89, 0.84] (PRIMARY)
- same_distribution, predictive minus base: 1.67 pp [-4.58, 7.08]
- same_distribution, instant minus base: 5.42 pp [-1.11, 12.92]
- same_distribution, predictive minus DWA: -25.56 pp [-30.97, -20.00]
- challenge, predictive minus instant: -5.00 pp [-13.33, 4.44]
- challenge, predictive minus base: 0.83 pp [-5.83, 9.17]
- challenge, instant minus base: 5.83 pp [-1.11, 12.50]
- challenge, predictive minus DWA: -39.44 pp [-50.56, -28.06]

Only three seeds: uncertain intervals, secondary comparisons not multiplicity adjusted.
Success-conditional efficiency/time uses different surviving subsets. Do not compare shaped returns across arms.
Braking delay excludes censored nonresponders; their counts are reported separately.
All completed trajectories passed kinematic, clearance and reward consistency checks.
