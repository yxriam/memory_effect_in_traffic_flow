"""Chapter 5: smoothing and variable-order Markov predictors, pooled and per-unit settings."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, json, time, numpy as np, pandas as pd
import lm
D = str(P.PRED_DATA); R = str(P.PRED_NGRAM)
import os; os.makedirs(R, exist_ok=True)
BETAS = [0.3, 1, 3, 10, 30, 100, 300]; NMAX = 6

def load(name):
    d = np.load(f'{D}/{name}.npz'); t, off, part, unit = d['tokens'], d['offsets'], d['part'], d['unit']
    seqs = [t[off[i]:off[i + 1]].tolist() for i in range(len(part))]
    return seqs, part, unit, int(d['vocab_size'])

def configs():
    yield ('unigram', 1, None)
    for k in (1, 2, 3): yield ('addalpha', k + 1, None)
    for N in (2, 3, 4):
        for b in BETAS: yield ('jm', N, b)
    for N in range(2, NMAX + 1): yield ('wb', N, None)
    for N in range(2, NMAX + 1): yield ('kn', N, None)

def mk(full, kind, N, b):
    if kind == 'unigram': return lm.view(full, 1, 'wb')
    return lm.view(full, N, kind, beta=b or 1.0, alpha=0.5)

def pooled(name):
    t0 = time.time(); seqs, part, unit, V = load(name)
    tr = [s for s, p in zip(seqs, part) if p == 0]; va = [s for s, p in zip(seqs, part) if p == 1]; te = [s for s, p in zip(seqs, part) if p == 2]
    full = lm.NGram(NMAX, V, 'wb').fit(tr); print(name, 'fit', round(time.time() - t0), 's', flush=True)
    rows = []
    for kind, N, b in configs():
        m = mk(full, kind, N, b)
        rv = lm.evaluate(m, va, topk=False); rt = lm.evaluate(m, te, topk=False)
        rows.append(dict(dataset=name, kind=kind, N=N, beta=b, val_bits=rv['bits'], test_bits=rt['bits'], n_test=rt['n']))
        print(rows[-1], round(time.time() - t0), flush=True)
    R0 = pd.DataFrame(rows)
    best = R0.loc[R0.groupby('kind').val_bits.idxmin()]
    out = []
    for _, r in best.iterrows():
        m = mk(full, r.kind, int(r.N), None if pd.isna(r.beta) else r.beta)
        e = lm.evaluate(m, te, topk=True); out.append(dict(r.to_dict(), **{'acc1': e['acc1'], 'acc5': e['acc5']}))
        print('BEST', out[-1], flush=True)
    R0.to_csv(f'{R}/ch5_pooled_grid_{name}.csv', index=False); pd.DataFrame(out).to_csv(f'{R}/ch5_pooled_best_{name}.csv', index=False)

def per_unit(name, units_max=None):
    seqs, part, unit, _ = load(name)
    by = {}
    for s, p, u in zip(seqs, part, unit):
        if p < 3: by.setdefault(u, [[], [], []])[p].append(s)
    cfg = [('addalpha', 2, None), ('addalpha', 3, None)] + [('jm', N, b) for N in (2, 3, 4) for b in BETAS] + \
          [('wb', N, None) for N in (2, 3, 4, 5)] + [('kn', N, None) for N in (2, 3, 4, 5)] + [('unigram', 1, None)]
    TOPK = {('addalpha', 2, None), ('jm', 3, 10), ('wb', 3, None), ('kn', 3, None)}
    out = f'{R}/ch5_perunit_{name}.csv'
    done = set(pd.read_csv(out).unit) if os.path.exists(out) else set()
    for i, (u, (tr, va, te)) in enumerate(sorted(by.items())):
        if units_max and i >= units_max: break
        if not tr or not te or int(u) in done: continue
        V = len(set(x for s in tr + va + te for x in s))
        full = lm.NGram(5, V, 'wb').fit(tr); rows = []
        for kind, N, b in cfg:
            m = mk(full, kind, N, b)
            rv = lm.evaluate(m, va, topk=False) if va else dict(bits=np.nan, n=0)
            rt = lm.evaluate(m, te, topk=(kind, N, b) in TOPK)
            rows.append(dict(dataset=name, unit=int(u), kind=kind, N=N, beta=b, V=V, val_bits=rv['bits'], n_val=rv['n'],
                             test_bits=rt['bits'], n_test=rt['n'], acc1=rt.get('acc1', np.nan), acc5=rt.get('acc5', np.nan)))
        pd.DataFrame(rows).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(name, i, u, flush=True)

if __name__ == '__main__':
    mode, name = sys.argv[1], sys.argv[2]
    pooled(name) if mode == 'pooled' else per_unit(name)
