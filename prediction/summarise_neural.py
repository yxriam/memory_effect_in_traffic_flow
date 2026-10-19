"""Collect the neural runs into two tables (no training, reads the JSON files only).

  results/neural/neural_<dataset>_<model>.json      -> results/neural/ch5_neural_summary.csv
  results/neural/hist/hist_<dataset>_k<k>_s<seed>.json -> results/ngram/ch5_hist_ablation.csv

Only the default runs (no --tag suffix) go into ch5_neural_summary.csv; the extra 128-step LSTM seeds
(neural_taxi_traj20_lstm_s1/_s2.json) are kept as JSON next to it.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import json, re
import pandas as pd

DATASETS = ['taxi_traj20', 'geolife_bj500m', 'highd_v20', 'highd_v100']
rows = []
for ds in DATASETS:
    for model in ('lstm', 'transformer'):
        f = P.PRED_NEURAL / f'neural_{ds}_{model}.json'
        if not f.exists():
            continue
        d = json.load(open(f))
        rows.append(dict(dataset=ds, model=model, test_bits=d['test']['bits'], acc1=d['test']['acc1'], acc5=d['test']['acc5'],
                         n_test=d['test']['n'], val_bits=d['val_bits'], epochs=len(d['history']), n_params=d['n_params']))
pd.DataFrame(rows).to_csv(P.out(P.PRED_NEURAL / 'ch5_neural_summary.csv'), index=False)

rows = []
for f in (P.PRED_NEURAL / 'hist').glob('hist_*.json'):
    d = json.load(open(f))
    rows.append(dict(dataset=d['dataset'], hist=d['hist'], seed=d['seed'], val_bits=d['val_bits'], test_bits=d['test']['bits'],
                     acc1=d['test']['acc1'], acc5=d['test']['acc5'], n_test=d['test']['n'], epochs=len(d['history'])))
H = pd.DataFrame(rows).sort_values(['dataset', 'hist', 'seed'])
H.to_csv(P.out(P.PRED_NGRAM / 'ch5_hist_ablation.csv'), index=False)
print(H[['dataset', 'hist', 'seed', 'test_bits']].round(3).to_string(index=False))
