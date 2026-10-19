# GeoLife Memory-Flow Reproducible Pipeline

This folder is the cleaned, reproducible experiment folder for the revised
paper narrative:

1. memory effects matter in human mobility flows;
2. first-order flow loses part of that path dependence;
3. second-order memory flow quantifies the missing dependence without claiming
   that a held-out second-order Markov predictor must be more accurate.

The main experiment is descriptive and uses full observed flows, following the
logic of Rosvall-style memory networks. Prediction scripts are kept only as
optional diagnostics and are not part of the main paper result.

## Data Layout

After cleanup, this folder is expected to contain the raw data:

```text
geolife_clean_pipeline/
  data_raw/
    Data/                  # original GeoLife user folders
    trips_by_mode_data/    # transportation-mode .plt files
```

The scripts also fall back to the old root-level paths if the raw data is moved
back there.

## Run

From the `geolife/` folder (PowerShell on Windows; on Linux/macOS run the equivalent `python` commands below):

```powershell
.\run_full_pipeline.ps1
```

Equivalent explicit commands:

```powershell
python 00_check_environment.py
python 01_preprocess_no_leakage.py
python 02_memory_flow_metrics.py
python 03_memory_driver_correlation.py
python 04a_resample_sensitivity_select.py --sensitivity-max-files-per-mode 100000 --gap-sample-files-per-mode 100000 --max-gaps-per-mode 10000000
python 05_summarize_results.py
```

Quick smoke audit:

```powershell
python 99_audit_pipeline.py
```

## Main Files

- `00_check_environment.py`: checks the Python environment and data paths.
- `01_preprocess_no_leakage.py`: parses GeoLife `.plt` files, validates
  datetime rows, rejects impossible coordinates, keeps trips separate, and
  writes clean per-user caches.
- `02b_global_numpy.py`: recomputes the pooled GLOBAL row and summary of step 02 with numpy
  (the dictionary version of step 02 needs more than 7 GB for the 16.5 M pooled transitions).
- `02_memory_flow_metrics.py`: builds full-flow first- and second-order memory
  metrics from the clean caches.
- `03_memory_driver_correlation.py`: computes GeoLife user-level driver
  features and Pearson/Spearman/Kendall/mutual-information correlations with
  memory-flow metrics.
- `04a_resample_sensitivity_select.py`: tests 1/2/5/10 second resampling
  intervals, records sensitivity outputs, selects the smallest stable interval,
  and reruns the final mode-level memory analysis with that interval.
- `04_mode_hurst_memory.py`: computes transportation-mode Hurst, velocity
  autocorrelation, and memory-kernel summaries.
- `05_summarize_results.py`: writes `outputs_clean/summary_report.txt`.
- `06_plot_5s_new_old_comparison.py`: plots the new-vs-old 5 second
  Hurst and memory-kernel comparison in the original visual style.
- `99_audit_pipeline.py`: compiles scripts and runs a small end-to-end smoke
  test.
- `90_optional_markov_holdout_eval.py`: optional prediction diagnostic.
- `91_optional_rnn_lstm_residual.py`: optional neural prediction diagnostic.

## Outputs

- `cache_clean/`: clean per-user trajectory caches.
- `outputs_clean/02_memory_flow/`: first-order vs second-order memory-flow
  metrics.
- `outputs_clean/03_correlation_analysis/`: user-level memory-driver features
  and correlation tables.
- `outputs_clean/04_resample_sensitivity/`: resampling interval sensitivity
  results and the selected interval metadata.
- `outputs_clean/04_mode_memory/`: mode-level long-memory summaries.
- `outputs_clean/04_mode_memory_compare_5s/`: new-vs-old 5 second
  comparison tables and original-style figures.
- `outputs_clean/summary_report.txt`: compact final summary.

See `METHODS.md` for the paper-facing interpretation and cautions.
