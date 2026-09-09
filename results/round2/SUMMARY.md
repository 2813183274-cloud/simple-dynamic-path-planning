# Round-two sealed evaluation

C/D: three matched training seeds; DWA: deterministic baseline.
Means and sample SD below measure seed variation; episode Wilson intervals are in each summary.json.
Three seeds give imprecise uncertainty estimates; no significance or convergence claim is implied.

| Split | Method | Success | Static collisions | Dynamic collisions |
|---|---|---:|---:|---:|
| same_distribution | C-seed1 | 167/240 | 63 | 10 |
| same_distribution | C-seed2 | 175/240 | 47 | 18 |
| same_distribution | C-seed3 | 159/240 | 71 | 10 |
| same_distribution | D-seed1 | 172/240 | 58 | 10 |
| same_distribution | D-seed2 | 148/240 | 62 | 30 |
| same_distribution | D-seed3 | 169/240 | 51 | 20 |
| same_distribution | A-seed1 | 143/240 | 80 | 17 |
| same_distribution | B-seed1 | 153/240 | 71 | 16 |
| same_distribution | round0-seed1 | 154/240 | 60 | 26 |
| same_distribution | DWA | 240/240 | 0 | 0 |
| challenge | C-seed1 | 77/120 | 24 | 19 |
| challenge | C-seed2 | 82/120 | 20 | 18 |
| challenge | C-seed3 | 75/120 | 25 | 20 |
| challenge | D-seed1 | 77/120 | 24 | 19 |
| challenge | D-seed2 | 61/120 | 40 | 19 |
| challenge | D-seed3 | 77/120 | 24 | 19 |
| challenge | A-seed1 | 62/120 | 33 | 25 |
| challenge | B-seed1 | 68/120 | 26 | 26 |
| challenge | round0-seed1 | 69/120 | 33 | 18 |
| challenge | DWA | 120/120 | 0 | 0 |

## Seed-level success statistics

- same_distribution C: 69.58% ± 3.33 percentage points (sample SD); seed t 95% [61.30, 77.86]%.
- same_distribution D: 67.92% ± 5.45 percentage points (sample SD); seed t 95% [54.38, 81.45]%.
- challenge C: 65.00% ± 3.00 percentage points (sample SD); seed t 95% [57.54, 72.46]%.
- challenge D: 59.72% ± 7.70 percentage points (sample SD); seed t 95% [40.60, 78.85]%.

## Paired success differences

- same_distribution, D minus C: -1.67 pp; paired hierarchical bootstrap 95% [-12.22, 6.95] pp.
- same_distribution, C minus DWA: -30.42 pp; paired hierarchical bootstrap 95% [-36.25, -24.58] pp.
- same_distribution, D minus DWA: -32.08 pp; paired hierarchical bootstrap 95% [-39.58, -25.56] pp.
- challenge, D minus C: -5.28 pp; paired hierarchical bootstrap 95% [-20.00, 7.22] pp.
- challenge, C minus DWA: -35.00 pp; paired hierarchical bootstrap 95% [-42.22, -27.50] pp.
- challenge, D minus DWA: -40.28 pp; paired hierarchical bootstrap 95% [-50.83, -31.39] pp.

A/B/round0 are single-seed supplementary ablations, not robust multi-seed evidence.
All PPO policies run in the same D motion environment with the observation encoding they were trained on.
DWA knows the fixed radii and kinematic constants, but consumes no hidden map, pose, future or witness information.
Path efficiency is conditional on success; different successful subsets are not a paired efficiency comparison.
Timing is local CPU wall-clock inference, not a real-time deployment benchmark.
