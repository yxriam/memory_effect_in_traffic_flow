"""Decompose the gap between smoothed n-gram and LSTM: trajectory/user identity vs within-piece repetition.
Components (per-token probabilities on the same val/test targets as all chapter 5 models):
  pool : pooled KN order 2 (N=3)            unit : KN order 2 fitted on the same unit's training pieces (V global)
  c1, c2, c3 : unigram / bigram / trigram caches of the piece's own history (empirical, no smoothing)
  r10, r50   : exponentially decaying unigram caches (half-life 10 and 50 steps)
Mixture weights by EM on validation, applied to test.   usage: python unit_cache_mix2.py <dataset>"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as _P
import sys, math, numpy as np, pandas as pd
from collections import Counter, defaultdict
import lm
name = sys.argv[1]; N = 3
d = np.load(_P.PRED_DATA / f'{name}.npz'); tok, off, part, unit = d['tokens'], d['offsets'], d['part'], d['unit']; V = int(d['vocab_size'])
seqs = [tok[off[i]:off[i + 1]].tolist() for i in range(len(part))]
by = defaultdict(lambda: [[], [], []])
for s, p, u in zip(seqs, part, unit):
    if p < 3: by[int(u)][p].append(s)
pool = lm.NGram(N, V, 'kn').fit([s for s, p in zip(seqs, part) if p == 0]); print('pool fitted', flush=True)
use_unit = name != 'highd_v20' and name != 'highd_v100'
P = {1: [], 2: []}; dec = {10: 0.5 ** (1 / 10), 50: 0.5 ** (1 / 50)}
for u, (tr, va, te) in by.items():
    if not va and not te: continue
    um = lm.NGram(N, V, 'kn').fit(tr) if (use_unit and tr) else None
    for which, pcs in ((1, va), (2, te)):
        for s in pcs:
            c1 = Counter(); c2 = defaultdict(Counter); c3 = defaultdict(Counter); r = {10: defaultdict(float), 50: defaultdict(float)}; rt = {10: 0.0, 50: 0.0}; t0 = {10: 0, 50: 0}
            for t in range(1, len(s)):
                h = tuple(s[max(0, t - N + 1):t]); w = s[t]
                pp = pool.prob(h, w); pu = um.prob(h, w) if um else pp
                x = s[t - 1]; c1[x] += 1
                if t >= 2: c2[s[t - 2]][x] += 1
                if t >= 3: c3[(s[t - 3], s[t - 2])][x] += 1
                for hl in (10, 50):        # lazy decay: weights stored as dec^-(t-t0); rescaled before overflow
                    if t - t0[hl] > 2000:
                        f = dec[hl] ** (t - t0[hl]); r[hl] = defaultdict(float, {k: v * f for k, v in r[hl].items()}); t0[hl] = t
                    rt[hl] = rt[hl] * dec[hl] + 1.0; r[hl][x] = r[hl].get(x, 0.0) + dec[hl] ** (-(t - t0[hl]))
                q1 = c1[w] / t
                g = c2.get(x); q2 = g[w] / sum(g.values()) if g else 0.0
                g = c3.get(tuple(s[t - 2:t])) if t >= 2 else None; q3 = g[w] / sum(g.values()) if g else 0.0
                qr = [r[hl].get(w, 0.0) * dec[hl] ** (t - t0[hl]) / rt[hl] for hl in (10, 50)]
                P[which].append((pp, pu, q1, q2, q3, qr[0], qr[1]))
print('probs done', len(P[1]), len(P[2]), flush=True)
A = {w: np.array(P[w]) for w in (1, 2)}
def fit(cols):
    X = A[1][:, cols]; lam = np.full(len(cols), 1 / len(cols))
    for _ in range(300): num = X * lam; lam = (num / num.sum(1, keepdims=True)).mean(0)
    return lam
bits = lambda cols, lam, w: -np.log2((A[w][:, cols] * lam).sum(1)).mean()
out = []
cfgs = [('pool', [0]), ('pool+unit', [0, 1]), ('pool+cache', [0, 2, 3, 4, 5, 6]), ('pool+unit+cache', [0, 1, 2, 3, 4, 5, 6])]
for label, cols in cfgs:
    if not use_unit and 1 in cols: continue
    lam = fit(cols) if len(cols) > 1 else np.array([1.0])
    rr = dict(dataset=name, model=label, val_bits=bits(cols, lam, 1), test_bits=bits(cols, lam, 2), weights=' '.join(f'{v:.3f}' for v in lam), n_test=len(A[2]))
    print(rr, flush=True); out.append(rr)
pd.DataFrame(out).to_csv(_P.out(_P.PRED_NGRAM / f'ch5_mix_{name}.csv'), index=False)
