# P4 final results

Pre-registered speed mapping comparison; no post-test tuning.

| Split | Method | Success | Static collision | Dynamic collision | Timeout | Stagnant timeout |
|---|---|---:|---:|---:|---:|---:|
| same_distribution | clip-seed1 | 167/240 | 53 | 20 | 0 | 0 |
| same_distribution | clip-seed2 | 160/240 | 54 | 26 | 0 | 0 |
| same_distribution | clip-seed3 | 173/240 | 50 | 17 | 0 | 0 |
| same_distribution | speed_tanh-seed1 | 183/240 | 35 | 22 | 0 | 0 |
| same_distribution | speed_tanh-seed2 | 175/240 | 44 | 21 | 0 | 0 |
| same_distribution | speed_tanh-seed3 | 166/240 | 50 | 24 | 0 | 0 |
| same_distribution | DWA | 237/240 | 0 | 0 | 3 | 3 |
| challenge | clip-seed1 | 78/120 | 22 | 20 | 0 | 0 |
| challenge | clip-seed2 | 68/120 | 35 | 17 | 0 | 0 |
| challenge | clip-seed3 | 81/120 | 20 | 19 | 0 | 0 |
| challenge | speed_tanh-seed1 | 78/120 | 22 | 20 | 0 | 0 |
| challenge | speed_tanh-seed2 | 72/120 | 28 | 20 | 0 | 0 |
| challenge | speed_tanh-seed3 | 73/120 | 27 | 20 | 0 | 0 |
| challenge | DWA | 119/120 | 0 | 0 | 1 | 1 |

## Seed success means

- same_distribution clip: 69.44% +/- 2.71 pp (seed sample SD).
- same_distribution speed_tanh: 72.78% +/- 3.54 pp (seed sample SD).
- challenge clip: 63.06% +/- 5.67 pp (seed sample SD).
- challenge speed_tanh: 61.94% +/- 2.68 pp (seed sample SD).

## Paired hierarchical bootstrap (95%)

- same_distribution, speed_tanh minus clip: 3.33 pp [-3.33, 10.00] (PRIMARY)
- same_distribution, speed_tanh minus DWA: -25.97 pp [-31.25, -20.28]
- same_distribution, clip minus DWA: -29.31 pp [-34.72, -24.17]
- challenge, speed_tanh minus clip: -1.11 pp [-8.33, 7.22]
- challenge, speed_tanh minus DWA: -37.22 pp [-43.61, -30.56]
- challenge, clip minus DWA: -36.11 pp [-44.44, -28.33]

Only three training seeds; secondary comparisons are exploratory and not multiplicity-adjusted.
Success-conditional efficiency/time compares different surviving episode sets.
Braking delay is responder-only; censored nonresponses are separately reported.
All completed trajectories passed motion, clearance, and unchanged-reward audits.
