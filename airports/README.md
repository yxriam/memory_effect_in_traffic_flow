# airports/ — BTS DB1B origin–destination paths (legacy research code)

This module contains the original scripts behind the airport part of the paper
(Sankey return-flow diagrams, 1st/2nd/3rd-order Markov chains, RNN vs LSTM rollouts).
They are kept as they were used, with only the machine-specific absolute paths replaced by
relative ones (`../data/…`, `./output/…`). They are **not** wrapped in the cache-first notebook
structure used by `highD/` and `taxi/`, and some of them expect intermediate files produced by
earlier scripts.

| Step | Script | Produces |
|---|---|---|
| 1 | `pre_process/airports_sequences_split.py` | `data/sequences_split/<YYYYMM>/<YYYYMM>_<n>.txt` — airport-code paths grouped by length n |
| 2 | `markov_chain/AirportsMarkovChain.py` · `CityMarkovChain.py` | 1st/2nd/3rd-order chains (airport / 50-km city clusters), held-out accuracy, 1000-step rollouts |
| 3 | `markov_chain/airports_return_rate.py` | 2-step / 3-step return rates of the rollouts |
| 4 | `markov_chain/sankey_data.py` · `airports_sankey.py` · `airports_sankey_excel.py` | Sankey data (`sankey_data.json`), pyecharts Sankey HTML, per-airport counts (xlsx) |
| 5 | `lstm_markov/n_sequences_lstm_train.py` → `n_sequences_lstm_test.py` | LSTM per path length (`output/lstm_emb16_hid128_nl2/`) |
| 5' | `lstm_markov/n_ordered_markov_chain_train_test.py` | n-th-order Markov baselines (`output/markov/markov<n>.pkl`) |
| 6 | `lstm_markov/mixed_train_test.py` | Markov + LSTM mixture with tuned α |
| 7 | `lstm_markov/lstm_rnn_predict.py` / **`lstm_rnn_predict.ipynb`** | RNN vs LSTM (seq 3), 1000-step rollouts, return rates, paired Sankey (`compare_sankey.html`) — the figures used in the paper |

`lstm_markov/airport_dict.json` maps IATA codes to city names (used for the 50-km city aggregation).
`lstm_markov/model/` and `dataloader/` hold the tiny `LSTMModel` / `RNNModel` / `AirportDataset` classes;
`n_ordered_markov_chain.py` provides `create_markov_chain(order)`.

Dependencies beyond the core: `torch`, `scikit-learn`, `pyecharts`, `plotly`, `chardet`, `tensorboard`, `openpyxl`
(`pip install -r ../requirements-full.txt`).
