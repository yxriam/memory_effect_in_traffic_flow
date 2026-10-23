# airports/ — BTS DB1B origin–destination paths

Code behind the airport part of the paper: the return-flow Sankey diagrams (Fig. 3), the
RNN-vs-LSTM comparison and the memory analysis with the same controls as the trajectory data (steps 4–5). Everything runs from the `airports/` folder with relative paths.

| Step | File | What it does |
|---|---|---|
| 0 | `airports_sequences_split.py` | DB1B `db1b.public.YYYYMM.asc` → `data/sequences_split/YYYYMM/YYYYMM_<n>.txt` (one line per itinerary, grouped by number of airports n) |
| 1 | `markov_chain.py` | `MarkovChain(order)`: transition counts, sampled `transition()`, arg-max `most_likely()`, `predict()` rollouts |
| 2 | `city_markov_sankey.py` | maps airports to cities (`airport_dict.json`, airports within 50 km share a city), fits 1st/2nd-order city chains, optional held-out accuracy, 1000-sample rollouts through a hub, Sankey JSON + HTML per order — **Fig. 3 top (via Chicago)** |
| 3 | `lstm_rnn_predict.ipynb` | trains a 5-layer RNN and LSTM on 3-airport itineraries, 1000-step rollouts with 2-/3-step return rates, next airport after DEN for every prefix, paired Sankey (`compare_sankey.html`) — **Fig. 3 bottom (via DEN)** |
| 4 | `airport_memory.py` | memory measured with the same controls as the trajectory data: CMI of the previous airport, of the third airport back and of the trip origin beyond the last three (stratified-permutation null), held-out cross-entropy of back-off chains of order 1–4, a return-only model and a chain whose context is (origin, current airport) → `results/airport_memory.csv` (2026-10-23; data via `common/paths.py` key `airports_seq`). The rollout return rates of step 3 measure the cycle a greedy rollout falls into, not memory in the data, and are no longer used |
| 5 | `fig_airline_memory.py` | figure from `results/airport_memory_cache.json` and `../controls/results/heading_state_taxi_200m_device.csv` → `figures/fig_airline_memory.png` (report, 2 × 2) and `figures/fig_airline_memory_slide.png` (slide) |

Support files: `model/LSTM.py`, `model/RNN.py` (embedding → recurrent layers → linear head on the last step),
`dataloader/Dataset.py` (`AirportDataset`: sliding windows of airport indices), `airport_dict.json` (IATA → city).

```bash
cd airports
python airports_sequences_split.py data/original_data/db1b.public.202409.asc --out-dir data/sequences_split
python airports_sequences_split.py data/original_data/db1b.public.202412.asc --out-dir data/sequences_split
python city_markov_sankey.py --train data/sequences_split/202409/*.txt --hub 芝加哥 \
       --origins 纽约 亚特兰大 旧金山 西雅图 --n-rollout 1000 --out data/sankey
jupyter lab lstm_rnn_predict.ipynb          # needs torch, plotly, pyecharts
```

City names in `airport_dict.json` look like `芝加哥 Chicago`; `--hub` / `--origins` accept either part as long as it is unique.

Dependencies beyond the core: `torch`, `pyecharts`, `plotly` (`pip install -r ../requirements-full.txt`).

> History: the original research folder also contained mixed Markov+LSTM experiments, per-length LSTM training
> scripts and a Huffman/geo-encoding exploration. None of them feed the paper, several depended on modules that no
> longer exist, so they were removed when the repository was assembled; the paper's airport figures come from the
> four files above.
