"""Chapter 6: first-order vs second-order (memory) networks on a global grid, map-equation communities.
For every dataset: sequences -> first-order network (physical nodes) and memory network (state nodes i->j),
both with empirical link flows. Partitions by an Infomap-style optimiser (mapeq.optimise_ml).
Control: a first-order Markov surrogate with the same starts and lengths, analysed identically, shows how much
apparent memory structure finite sampling alone creates (cf. chapter 4)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.CONTROLS))  # geolife_robust.py
import sys, os, glob, json, time, numpy as np, pandas as pd
import mapeq
R = str(P.PRED_NGRAM); X = str(P.INFOMAP_INPUT); os.makedirs(R, exist_ok=True); os.makedirs(X, exist_ok=True)

def taxi_sequences(g=0.01):
    files = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), '*'))); LA, LO, ID = [], [], []
    for k, f in enumerate(files):
        try: d = pd.read_csv(f, header=None, usecols=[2, 3])
        except Exception: continue
        LO.append(d[2].to_numpy(float)); LA.append(d[3].to_numpy(float)); ID.append(np.full(len(d), k))
    lo, la, idv = np.concatenate(LO), np.concatenate(LA), np.concatenate(ID)
    b = [np.percentile(la, .5), np.percentile(la, 99.5), np.percentile(lo, .5), np.percentile(lo, 99.5)]
    seqs = []
    for k in np.unique(idv):
        m = idv == k; x, y = lo[m], la[m]; ok = (y >= b[0]) & (y <= b[1]) & (x >= b[2]) & (x <= b[3])
        c = (np.floor(y[ok] / g).astype(np.int64) * 100000 + np.floor(x[ok] / g).astype(np.int64))
        if len(c) == 0: continue
        c = c[np.r_[True, c[1:] != c[:-1]]]
        if len(c) >= 3: seqs.append(c)
    return seqs

def npz_sequences(name, parts=(0, 1, 2)):
    d = np.load(P.PRED_DATA / f'{name}.npz'); t, off, part = d['tokens'], d['offsets'], d['part']
    return [t[off[i]:off[i + 1]].astype(np.int64) for i in range(len(part)) if part[i] in parts and off[i + 1] - off[i] >= 3]

def build(seqs):
    allc = np.unique(np.concatenate(seqs)); enc = lambda s: np.searchsorted(allc, s)
    S = [enc(s) for s in seqs]
    a = np.concatenate([s[:-1] for s in S]); b = np.concatenate([s[1:] for s in S])
    k1, w1 = np.unique(a * len(allc) + b, return_counts=True)
    first = mapeq.Net(k1 // len(allc), k1 % len(allc), w1 / w1.sum(), np.arange(len(allc)))
    x = np.concatenate([s[:-2] for s in S if len(s) >= 3]); y = np.concatenate([s[1:-1] for s in S if len(s) >= 3]); z = np.concatenate([s[2:] for s in S if len(s) >= 3])
    sid_from = x * len(allc) + y; sid_to = y * len(allc) + z
    states, inv = np.unique(np.r_[sid_from, sid_to], return_inverse=True); n2 = len(x)
    k2, w2 = np.unique(inv[:n2] * len(states) + inv[n2:], return_counts=True)
    mem = mapeq.Net(k2 // len(states), k2 % len(states), w2 / w2.sum(), states % len(allc))
    return S, first, mem, states, len(allc)

def surrogate(S, V, rng):
    """First-order Markov surrogate: same first symbol and length, transitions from the pooled P(j|i)."""
    a = np.concatenate([s[:-1] for s in S]); b = np.concatenate([s[1:] for s in S])
    order = np.argsort(a, kind='stable'); a_s, b_s = a[order], b[order]
    start = np.searchsorted(a_s, np.arange(V)); end = np.searchsorted(a_s, np.arange(V), 'right')
    out = []
    for s in S:
        x = [s[0]]
        for _ in range(len(s) - 1):
            i = x[-1]
            if end[i] == start[i]: break
            x.append(b_s[rng.integers(start[i], end[i])])
        out.append(np.array(x))
    return [s for s in out if len(s) >= 3]

def entropy_rates(S):
    import geolife_robust as gr
    a = np.concatenate([s[:-2] for s in S if len(s) >= 3]); b = np.concatenate([s[1:-1] for s in S if len(s) >= 3]); c = np.concatenate([s[2:] for s in S if len(s) >= 3])
    K = int(max(a.max(), b.max(), c.max())) + 1
    h1 = gr.cond_H(b, c, K); h2 = gr.cond_H(np.unique(a * K + b, return_inverse=True)[1].astype(np.int64), c, K)
    return h1, h2

def analyse(label, seqs, trials=2, seed=0):
    t0 = time.time(); S, first, mem, states, V = build(seqs)
    L1f = first.one_module(); Lf, mf = mapeq.optimise_ml(first, trials=trials, seed=seed)
    L1m = mem.one_module(); Lm, mm = mapeq.optimise_ml(mem, trials=trials, seed=seed)
    phys = mem.phys; key = np.unique(np.c_[phys, mm], axis=0)
    nmod_per_phys = np.bincount(key[:, 0], minlength=V)
    pphys = np.bincount(phys, weights=mem.p, minlength=V)
    multi = nmod_per_phys >= 2
    h1, h2 = entropy_rates(S)
    r = dict(dataset=label, n_seq=len(S), n_steps=int(sum(len(s) - 1 for s in S)), phys_nodes=V, first_links=sum(len(d) for d in first.out),
             state_nodes=mem.n, mem_links=sum(len(d) for d in mem.out), H1=h1, H2=h2,
             first_L1=L1f, first_L=Lf, first_modules=int(mf.max() + 1), first_saving=(L1f - Lf) / L1f,
             mem_L1=L1m, mem_L=Lm, mem_modules=int(mm.max() + 1), mem_saving=(L1m - Lm) / L1m,
             overlap_frac_nodes=float((multi & (pphys > 0)).sum() / max((pphys > 0).sum(), 1)),
             overlap_frac_flow=float(pphys[multi].sum() / pphys.sum()), modules_per_phys_flowavg=float((nmod_per_phys * pphys).sum() / pphys.sum()),
             sec=time.time() - t0)
    return r, (S, first, mem, states, V, mf, mm)

def export_infomap(tag, first, mem, states, V):
    with open(f'{X}/{tag}_first.net', 'w') as f:
        f.write(f'*Vertices {first.n}\n'); f.write(''.join(f'{i + 1} "{i}"\n' for i in range(first.n)))
        f.write('*Arcs\n'); f.write(''.join(f'{a + 1} {b + 1} {x:.10g}\n' for a in range(first.n) for b, x in first.out[a].items()))
    with open(f'{X}/{tag}_memory.net', 'w') as f:
        f.write(f'*Vertices {V}\n'); f.write(''.join(f'{i + 1} "{i}"\n' for i in range(V)))
        f.write('*States\n'); f.write(''.join(f'{s + 1} {mem.phys[s] + 1}\n' for s in range(mem.n)))
        f.write('*Links\n'); f.write(''.join(f'{a + 1} {b + 1} {x:.10g}\n' for a in range(mem.n) for b, x in mem.out[a].items()))

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'taxi': seqs = taxi_sequences()
    elif which == 'geolife': seqs = npz_sequences('geolife_bj500m')
    elif which == 'highd': seqs = npz_sequences('highd_v20', parts=(0, 1, 2, 3))
    r, (S, first, mem, states, V, mf, mm) = analyse(which, seqs); print(r, flush=True)
    export_infomap(which, first, mem, states, V)
    np.savez_compressed(f'{R}/ch6_partitions_{which}.npz', first_mod=mf, mem_mod=mm, mem_phys=mem.phys, mem_p=mem.p, first_p=first.p)
    rows = [r]
    rng = np.random.default_rng(1)
    for rep in range(2):
        rs, _ = analyse(which + f'_markov_surrogate{rep}', surrogate(S, V, rng), seed=rep); print(rs, flush=True); rows.append(rs)
    pd.DataFrame(rows).to_csv(f'{R}/ch6_{which}.csv', index=False)
