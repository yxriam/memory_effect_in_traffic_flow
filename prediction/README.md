# prediction/ — can the memory be used?

Held-out prediction (smoothed n-gram chains, LSTM, Transformer, history-length ablation) and memory networks
(map equation, official Infomap) on the same token sequences for taxi, GeoLife and highD.
All cross-entropies are in **bits per step**.

## Layout

```
prediction/
├── export_data.py          raw data / caches → data/<name>.npz + .json (fixed train / val / test splits)
├── lm.py                   n-gram library: add-α, interpolated back-off, Witten–Bell, modified Kneser–Ney
├── run_ch5.py              pooled and per-unit n-gram runs            → results/ngram/ch5_pooled_*.csv, ch5_perunit_*.csv
├── ch5_learning_curve.py   KN orders 1–4 vs amount of training data  → results/ngram/ch5_learning_curve_*.csv
├── unit_cache_mix2.py      KN + same-unit model + recent-cell cache  → results/ngram/ch5_mix_*.csv
├── neural_ch5.py           LSTM / Transformer, 128-step context      → results/neural/neural_*.json
├── neural_hist.py          LSTM that sees only the last k symbols    → results/neural/hist/hist_*.json
├── summarise_neural.py     JSON → results/neural/ch5_neural_summary.csv, results/ngram/ch5_hist_ablation.csv
├── mapeq.py                two-level map equation for first-order and memory (state) networks
├── ch6_networks.py         first-order vs memory network + Markov surrogate → results/ngram/ch6_<dataset>.csv, infomap_input/*.net
├── ch6_extra.py            codelength decomposition (entropy vs overlap) → results/ngram/ch6_decomp_*.csv
├── export_surrogates.py    surrogate networks for Infomap              → infomap_input/*_surrogate_*.net
├── run_infomap.py          official Infomap on infomap_input/*.net     → results/infomap/*.clu, infomap_summary.csv
├── figures/                fig_ch5.py, fig_ch6.py, fig_memnet_paper.py → figures/output/
├── scripts/                Windows .bat launchers for the GPU and Infomap runs
├── data/                   *.npz (git-ignored, rebuilt by export_data.py) · *.json split metadata (committed)
├── infomap_input/          exported networks (git-ignored, rebuilt by ch6_networks.py / export_surrogates.py)
└── results/                ngram/ · neural/ (hist/ = ablation) · infomap/
```

Trained model weights (`*.pt`) are not committed.

## Order of a full re-run

```bash
# 1. token data (highD needs controls/work/highd_states.pkl from controls/highd_augment.py)
python prediction/export_data.py geolife_bj
python prediction/export_data.py geolife          # geolife_100m, used by the per-unit GeoLife run
python prediction/export_data.py highd            # highd_v20 and highd_v100
python prediction/export_data.py taxi             # taxi_traj20 (reads taxi_raw from common/paths.py)

# 2. n-gram models and learning curves (CPU)
python prediction/run_ch5.py pooled <dataset>     # taxi_traj20, geolife_bj500m, highd_v20, highd_v100
python prediction/run_ch5.py unit taxi_traj20      # per-unit runs (also geolife_100m)
python prediction/ch5_learning_curve.py taxi_traj20 geolife_bj500m highd_v20
python prediction/unit_cache_mix2.py <dataset>

# 3. neural models (GPU): double-click scripts\run_neural_all.bat, then scripts\run_hist_all.bat
python prediction/summarise_neural.py

# 4. memory networks
python prediction/ch6_networks.py taxi            # also geolife, highd
python prediction/ch6_extra.py highd geolife taxi
python prediction/export_surrogates.py
scripts\run_infomap.bat                           # pip install infomap, then run_infomap.py

# 5. figures
python prediction/figures/fig_ch5.py              # fig_ch5_ngram  (paper: prediction figure)
python prediction/figures/fig_ch6.py              # fig_ch6_memory_network (thesis chapter 6)
python prediction/figures/fig_memnet_paper.py     # fig_memnet_paper (paper: memory-network figure)
```

The `.bat` files use the Python in `scripts\local_python.bat` (copy `local_python.example.bat`, git-ignored)
and fall back to `python` on the PATH. The reference runs used Python 3.8.18, PyTorch 1.10.0 + CUDA 10.2
on a GeForce MX250 (`results/neural/env_check.txt`).

Runs listed in `results/neural/hist/SKIP.txt` (`<dataset> <k> <seed>`) were pruned from the queue and exit at once;
their logs only contain that message.

## Checked

`fig_ch5.py`, `fig_ch6.py` and `fig_memnet_paper.py` regenerate the committed figures pixel-identically from
`results/`, and `summarise_neural.py` reproduces both summary tables from the JSON files.
