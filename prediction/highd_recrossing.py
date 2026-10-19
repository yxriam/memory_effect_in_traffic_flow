"""highD bin-edge recrossing (paper, memory-network section): after the 20-bin speed state changes from bin j to
bin i, how often is the next state j again (return), i (stay) or a third bin? Compared with the return probability
P(j | i) of the pooled first-order chain. Uses all vehicles of data/highd_v20.npz, as ch6_networks.py does.
Paper: returns 43 %, first-order chain 6 %, third bin < 1 %.   (2026-10-19: 42.7 %, 6.1 %, 0.86 %)

    python prediction/highd_recrossing.py      -> results/ngram/highd_recrossing.json
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import json
import numpy as np

d = np.load(P.PRED_DATA / 'highd_v20.npz')
t, off = d['tokens'].astype(np.int64), d['offsets']
seqs = [t[off[i]:off[i + 1]] for i in range(len(off) - 1)]
V = int(t.max()) + 1
M = np.zeros((V, V))
for s in seqs:
    if len(s) >= 2:
        np.add.at(M, (s[:-1], s[1:]), 1)
M /= M.sum(1, keepdims=True)                      # pooled first-order chain P(next | current)
a = np.concatenate([s[:-2] for s in seqs if len(s) >= 3])
b = np.concatenate([s[1:-1] for s in seqs if len(s) >= 3])
c = np.concatenate([s[2:] for s in seqs if len(s) >= 3])
m = a != b                                         # the speed has just crossed a bin edge
res = dict(bin_changes=int(m.sum()),
           p_return=float(np.mean(c[m] == a[m])),
           p_stay=float(np.mean(c[m] == b[m])),
           p_third_bin=float(np.mean((c[m] != a[m]) & (c[m] != b[m]))),
           p_return_first_order_chain=float(np.mean(M[b[m], a[m]])))
for k, v in res.items():
    print(f'{k:28s} {v:.4f}' if isinstance(v, float) else f'{k:28s} {v}')
json.dump(res, open(P.out(P.PRED_NGRAM / 'highd_recrossing.json'), 'w'), indent=1)
