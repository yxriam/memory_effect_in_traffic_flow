# Experiment Results

Generated with:

```powershell
python 02_memory_flow_metrics.py
python 03_memory_driver_correlation.py
python 04a_resample_sensitivity_select.py --sensitivity-max-files-per-mode 100000 --gap-sample-files-per-mode 100000 --max-gaps-per-mode 10000000
python 05_summarize_results.py
```

## Memory Flow

Output folder: `outputs_clean/02_memory_flow/`

- valid users: 181
- grid size: 0.0001 degrees
- consecutive duplicate grid cells: collapsed
- median first-order entropy `H(next | current)`: 0.5000 bits
- median second-order entropy `H(next | previous, current)`: 0.1243 bits
- median entropy reduction: 0.3464 bits
- median entropy reduction ratio: 74.25%
- users with positive entropy reduction: 180 / 181
- global pooled first-order entropy: 1.7400 bits
- global pooled second-order entropy: 0.6774 bits
- global pooled entropy reduction: 1.0626 bits

Interpretation: adding the previous state usually reduces uncertainty in the
next flow direction. This supports the paper narrative that mobility flows have
memory effects that are hidden by a first-order representation.

The two-step return result is a limitation/caution rather than the main claim:

- median empirical two-step return: 0.01855
- median first-order expected two-step return: 0.04605
- median second-order expected two-step return: 0.01855

In this setup, the first-order model overestimates two-step returns. The paper
should therefore emphasize entropy/path-dependence rather than claiming that
second-order return is higher.

## Correlation Analysis

Output folder: `outputs_clean/03_correlation_analysis/`

This analysis was rerun on the current cleaned cache. It computes user-level
trajectory descriptors and relates them to memory-flow metrics with Pearson,
Spearman, Kendall, and mutual information.

Strongest reproducible patterns:

- two-step return feature matches the memory-flow empirical two-step return
  by construction;
- unique-state ratio is strongly negatively correlated with first-order
  entropy: Pearson -0.9438, Spearman -0.9694, Kendall -0.8770;
- second-order history reuse is strongly positively correlated with
  second-order entropy: Spearman 0.9645, Kendall 0.8762;
- dwell ratio and return rates are positively associated with observed
  two-step return structure.

Figure: `outputs_clean/figures/fig_geolife_correlation_heatmaps.png`

## Mode Memory

Output folder: `outputs_clean/04_mode_memory/`

Resampling sensitivity output: `outputs_clean/04_resample_sensitivity/`

The sensitivity script tested 1, 2, 5, and 10 second intervals on the available
mode files. It selected the
smallest candidate satisfying native sampling coverage, segment retention, and
finite-kernel thresholds. The selected interval was 5 seconds.

| candidate sec | median native-gap coverage | segment coverage | eligible |
|---:|---:|---:|---|
| 1 | 18.32% | 1.000 | no |
| 2 | 79.49% | 0.999 | no |
| 5 | 96.02% | 0.994 | yes |
| 10 | 98.31% | 0.988 | yes |

| mode | segments | median Hurst | ACF alpha | kernel alpha |
|---|---:|---:|---:|---:|
| airplane | 13 | 0.9892 | 0.4383 | 1.004 |
| bike | 767 | 0.9065 | 1.216 | 1.007 |
| boat | 3 | 0.9388 | 0.2926 | 0.1464 |
| bus | 1113 | 0.8665 | 0.949 | 1.549 |
| car | 696 | 0.9554 | 1.084 | 1.107 |
| subway | 338 | 0.9616 | 0.6253 | 0.5807 |
| taxi | 220 | 0.9367 | 1.29 | 0.6333 |
| train | 127 | 0.9840 | 0.4856 | 0.7084 |
| walk | 1489 | 0.8876 | 0.7413 | 1.496 |

Modes with very few segments, especially boat and airplane, should be presented cautiously.
motorcycle and run (2–4 users each) are not part of the mode set.

Values above are from the 2026-10-14 re-run on the official GeoLife 1.3 download
(`trips_by_mode_data` rebuilt with `../controls/build_trips_by_mode.py`). Compared with the
earlier run only the kernel exponents moved (e.g. bus 1.542 → 1.549, subway 0.610 → 0.581);
Hurst exponents, segment counts and the memory-flow entropies are unchanged.
