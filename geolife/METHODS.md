# Methods and Interpretation Notes

## Core Claim

The revised experiment should be described as memory-flow analysis, not as a
claim that a second-order Markov predictor is always more accurate in held-out
prediction.

The defensible claim is:

- first-order flow models estimate `P(next | current)`;
- second-order memory-flow models estimate `P(next | previous, current)`;
- if the second distribution has lower conditional entropy or a different
  return-flow structure, then the current location alone does not fully explain
  the observed mobility flow;
- this quantifies memory effects and helps explain why path history matters.

## Experiment Order

1. `00_check_environment.py`
   Checks packages and raw-data paths.

2. `01_preprocess_no_leakage.py`
   Reads raw GeoLife `.plt` files and writes `cache_clean/*.npz`.
   The parser requires valid `YYYY-MM-DD HH:MM:SS` fields and valid latitude /
   longitude ranges, so metadata rows are not treated as coordinates.
   Trips are stored separately to avoid artificial transitions between trips.

3. `02_memory_flow_metrics.py`
   Reconstructs each complete trip from the cached train/test pieces and uses
   the complete observed trip as a flow path. Consecutive duplicate grid cells
   are collapsed by default so that the result is closer to a path through
   spatial states rather than a dwell-time model.

   Main quantities:

   - first-order entropy: `H(next | current)`;
   - second-order entropy: `H(next | previous, current)`;
   - entropy reduction: `H(next | current) - H(next | previous, current)`;
   - empirical two-step return rate: observed `A -> B -> A` rate;
   - first-order expected return: return rate expected from `P(next | current)`;
   - second-order expected return: return rate under
     `P(next | previous, current)`.

4. `03_memory_driver_correlation.py`
   Computes user-level trajectory descriptors from the clean cache and relates
   them to memory-flow metrics using Pearson, Spearman, Kendall, and mutual
   information. This is the current reproducible replacement for the old
   correlation heatmap.

5. `04_mode_hurst_memory.py`
   Uses labeled transportation-mode trajectories to estimate Hurst statistics,
   vector velocity autocorrelation, and memory-kernel decay.

6. `05_summarize_results.py`
   Produces a compact report for the whole experiment.

## What This Does Not Claim

The main result does not require retraining on held-out splits and does not
require second-order holdout accuracy to exceed first-order accuracy. A
second-order model has many more histories, so it can be sparse in prediction
even while it is the correct descriptive representation for memory-dependent
flow.

The optional holdout and RNN/LSTM scripts are kept for transparency, but they
should be cited only as diagnostics or limitations, not as the central evidence.

## Cautions for Writing

The entropy reduction in `02_memory_flow_metrics.py` is measured on the full
observed flow network. It is therefore a descriptive memory statistic, not an
out-of-sample prediction score. Its magnitude depends on spatial grid size,
trajectory sampling, and the decision to collapse consecutive duplicate cells.

The two-step return statistic should be written as a difference between the
return structure implied by first-order flow and the return structure observed
under second-order histories. In this GeoLife run, the first-order model
overestimates two-step returns after repeat-cell collapse, so the paper should
not claim that second-order return is always higher. The stronger and cleaner
result is that second-order histories reduce flow uncertainty for almost all
users.
