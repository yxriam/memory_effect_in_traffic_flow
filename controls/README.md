# controls/ — apparent vs genuine memory

The analyses behind the paper's second version ("apparent and genuine memory"). Each script removes one
known artefact that can create or hide memory, or describes what is left after the three are removed:

| artefact / question | control | scripts |
|---|---|---|
| estimation bias (sparse histories, stays) | stratified permutation null, run-length compression | `geolife_robust.py`, taxi notebook (`../taxi/`) |
| state compression (binned speed, missing acceleration) | state augmentation, coarse-grained AR(1)/AR(2) surrogates | `highd_augment.py`, `highd_fine.py`, `highd_fine3.py`, `coarse_model.py`, `accel_acf.py`, `memory_by_mode.py` |
| biased continuous diagnostics (R/S Hurst, GLE kernel) | AR(1) surrogates, fitting-window sensitivity | `hurst_control.py`, `kernel_window.py`, `highd_augment.py` |
| what remains: persistence, path memory, trip-level memory | regressions, per-mode CMI, origin CMI | `row_persistence.py`, `row_regression.py`, `row_stops.py`, `memory_by_mode.py`, `origin_memory.py`, `taxi_origin.py` |

Units: the CMI tables built on `geolife_robust.py`'s estimators (`geolife_robust_per_user`, `memory_by_mode_path`,
`origin_memory_*`, `taxi_origin_*`) are in **bits**; the held-out log-likelihood gains of the highD scripts,
`coarse_model` and `memory_by_mode_speed` are in **nats** (the paper converts them to bits, × 1/ln 2).

## Layout

```
controls/
├── *.py                 analyses (run from any folder: python controls/<script>.py)
├── taxi_pick.json       300 taxi files sorted by file name; taxi_origin.py pools the first 100 usable ones
├── results/             every table the scripts write (committed)
├── figures/             plotting scripts  → figures/output/ (committed)
└── work/                large intermediate caches, e.g. highd_states.pkl (git-ignored, rebuilt on demand)
```

Data locations come from `../common/paths.py` (see `../local_paths.example.json`); nothing is hard-coded.

## Scripts → outputs

| script | reads | writes (`results/`) |
|---|---|---|
| `geolife_robust.py` | GeoLife cache (`geolife_cache`) | `geolife_robust_per_user.csv` — plug-in / null / excess CMI, third order, held-out back-off gain; 4 grids × de-jitter, 50 permutations (≈ 25–40 min on 2 cores) |
| `hurst_control.py` | `geolife_trips_by_mode`, `../geolife/04_mode_hurst_memory.py` | `hurst_control.csv` — R/S on data, shuffled, AR(1) surrogate, increments; DFA |
| `kernel_window.py` | `../geolife/outputs_clean/04_mode_memory/mode_kernel_curves.csv` | `geolife_kernel_window_sensitivity.csv` |
| `memory_by_mode.py` | `geolife_trips_by_mode` | `memory_by_mode_speed.csv`, `memory_by_mode_path.csv` |
| `row_persistence.py` | `geolife_trips_by_mode` | `row_persistence.csv` — τ_c, stops/km, interpolated share per segment |
| `row_regression.py` | `row_persistence.csv` | `row_regression.csv`, `row_regression_split.csv` |
| `row_stops.py` | `row_persistence.csv` | `row_stops_spearman.csv`, `row_mode_medians.csv` |
| `origin_memory.py` | GeoLife cache, `taxi_raw` | `origin_memory_geolife.csv` (+ per-vehicle `origin_memory_taxi.csv`) |
| `taxi_origin.py [grid]` | `taxi_raw`, `taxi_pick.json` | `taxi_origin_<grid>.csv` (0.002° ≈ 200 m, 0.005° ≈ 500 m) |
| `highd_augment.py` | `highd_tracks` | `augment_markov.csv`, `vacf_summary.json`, `vacf_kernels.npz`; caches `work/highd_states.pkl` |
| `highd_fine.py / highd_fine3.py` | `work/highd_states.pkl` | `fine_resolution.csv` / `fine_third_order.csv` |
| `coarse_model.py` | `work/highd_states.pkl` | `coarse_model.csv`, `coarse_model_params.json` (AR fits, surrogates) |
| `accel_acf.py` | `work/highd_states.pkl` | `highd_accel_acf.csv` |
| `build_trips_by_mode.py [Data] [out]` | `geolife_raw` (labels.txt) | rebuilds `trips_by_mode_data` into `work/` (never overwrites your copy) |

**Taxi origin memory.** The committed `taxi_origin_*.csv` came from the first 100 files of `taxi_pick.json`
(Taxi_106 … Taxi_76698), all of which are usable. The script pools the first 100 usable files in list order and
writes their names to `taxi_origin_<grid>_vehicles.txt`, so on a full copy of the data it picks exactly these
files; re-running it on them reproduces both tables digit for digit (86 413 steps at 0.002°, checked 2026-10-19).

`results/geolife_rerun_comparison.csv` records, item by item, the GeoLife numbers before and after the
re-run on the official GeoLife 1.3 download (2026-10-13/14). The small changes it shows in the Hurst surrogates
and in the bootstrap intervals came from seeds based on Python's `hash()` and from bootstrap draws that followed
the row order of a multiprocessing run; both are fixed since 2026-10-19 (see `../EXPERIMENT_LOG.md`), so every
script here now gives the same numbers on every machine.

### The coarse-graining surrogates (`coarse_model.py`)

`x = v − (segment mean)`; one AR(1) (`ρ`) and one AR(2) (`φ1, φ2`) fitted by least squares on all
vehicles pooled; noise scale = residual standard deviation of each segment; Gaussian innovations; 60-step
burn-in; one surrogate per vehicle with the vehicle's length and mean speed (seed 5); train/validation split
by vehicle (seed 1). Fitted values are in `results/coarse_model_params.json`.

## Figures (`figures/`)

| script | figure | inputs |
|---|---|---|
| `fig_geolife_bias.py` | `fig_geolife_bias.pdf` — plug-in vs null vs excess CMI, resolution, held-out gain | `geolife_robust_per_user.csv` |
| `fig_highd_aug.py` | `fig_highd_augment.pdf` — state augmentation, VACF, kernel | `augment_markov.csv`, `fine_resolution.csv`, `vacf_kernels.npz`, `vacf_summary.json` |
| `fig_coarse_model.py` | `fig_coarse_model.pdf` — gain vs bins for data / AR(1) / AR(2), acceleration ACF | `coarse_model*.{csv,json}`, `highd_accel_acf.csv` |
| `fig_hurst.py` | `fig_geolife_hurst_control.pdf` | `hurst_control.csv` |
| `fig_three_layers.py` | `fig_three_layers.pdf` — speed memory, persistence vs stops, path memory, origin | `memory_by_mode_*.csv`, `row_persistence.csv`, `origin_memory_geolife.csv`, `taxi_origin_0.002.csv` |
| `fig_taxi_gridmode.py` | `fig_taxi_gridmode.pdf` — per-trajectory vs global grid | `../taxi/memory_*.csv` |
| `fig_geolife_kernel.py` | `geolife_kernel.png` — kernel decay by mode (supplement) | `../geolife/outputs_clean/04_mode_memory/`, `geolife_kernel_window_sensitivity.csv` |

Every figure except `fig_highd_augment.pdf` regenerates pixel-identically from the committed tables.
`vacf_kernels.npz` was not kept from the original run, so `fig_highd_aug.py` needs one run of
`highd_augment.py` first (the committed PDF is from that original run).
