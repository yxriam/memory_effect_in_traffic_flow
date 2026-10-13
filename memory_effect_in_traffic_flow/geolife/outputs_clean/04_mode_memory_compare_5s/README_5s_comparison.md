# New vs Old Data, 5s Mode-Memory Comparison

Both datasets were rerun with the same cleaned script and the same settings:

```powershell
--resample-sec 5 --max-lag 200 --fit-min-lag 2 --fit-max-lag 80
```

The new data directory is:

```text
geolife_clean_pipeline/data_raw/trips_by_mode_data
```

The old data directory is:

```text
legacy_unused_files_20260528/root_legacy/trips_by_mode/trips_by_mode/data
```

## Main Result

Most modes are identical between the new and old folders under the cleaned 5s
pipeline. The differences are concentrated in bus and car.

| mode | new segments | old segments | new H | old H | new kernel alpha | old kernel alpha |
|---|---:|---:|---:|---:|---:|---:|
| airplane | 13 | 13 | 0.9892 | 0.9892 | 1.0041 | 1.0041 |
| bike | 767 | 767 | 0.9065 | 0.9065 | 1.0106 | 1.0106 |
| boat | 3 | 3 | 0.9388 | 0.9388 | 0.1464 | 0.1464 |
| bus | 1113 | 1188 | 0.8665 | 0.9088 | 1.5418 | 1.1662 |
| car | 696 | 436 | 0.9554 | 0.9425 | 1.0854 | 1.1338 |
| motorcycle | 0 | 0 | NA | NA | NA | NA |
| run | 1 | 1 | 0.8782 | 0.8782 | 0.3738 | 0.3738 |
| subway | 338 | 338 | 0.9616 | 0.9616 | 0.6100 | 0.6100 |
| taxi | 220 | 220 | 0.9367 | 0.9367 | 0.6670 | 0.6670 |
| train | 127 | 127 | 0.9840 | 0.9840 | 0.7085 | 0.7085 |
| walk | 1489 | 1489 | 0.8876 | 0.8876 | 1.4836 | 1.4836 |

## Figures

- `fig_hurst_5s_new_original_style.png`
- `fig_hurst_5s_old_original_style.png`
- `fig_hurst_5s_new_vs_old_original_style.png`
- `fig_memory_kernel_5s_new_original_style.png`
- `fig_memory_kernel_5s_old_original_style.png`
- `fig_memory_kernel_5s_new_vs_old_original_style.png`

