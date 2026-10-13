# GeoLife raw data (not distributed)

```
geolife/data_raw/
  Data/                  # the 182 user folders 000 … 181 from "Geolife Trajectories 1.3"
  trips_by_mode_data/    # optional: transportation-mode .plt files used by 04_mode_hurst_memory.py
```

- Download: https://www.microsoft.com/en-us/download/details.aspx?id=52367 (Microsoft Research licence, non-commercial use).
- `01_preprocess_no_leakage.py` writes per-user caches to `geolife/cache_clean/` (git-ignored, ≈ 80 MB).
- All results of the pipeline are already committed under `geolife/outputs_clean/`.
