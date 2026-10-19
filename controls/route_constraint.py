"""P1-6  Path memory and route constraint.

Two checks on the per-mode path memory of memory_by_mode.py (GeoLife mode-labelled trips, segments split at gaps
> 30 s, 100 m cells, repeats collapsed, no de-jitter):
(i)  Equal samples: 12 users per mode and the same number of trigrams per user (the first N_TRI trigrams of the
     user's trips in time order; users with fewer are not drawn), 10 random draws of users. Absolute excess CMI (bits)
     and relative excess (/ H(S_{t+1}|S_t)) per mode, stratified permutation null with 30 permutations as in the paper.
(ii) Branching: for every cell b visited as the middle of >= 20 trigrams of one user and mode, the local excess
     CMI I_b = [H(C|B=b) - H(C|A,B=b)] minus its mean over 30 within-cell permutations of A, the branching degree
     (number of distinct successors) and the successor entropy H(C|B=b). Weighted least squares (weight = visits):
     I_b ~ mode, and I_b ~ log2(branching) + mode (car = reference); cluster bootstrap over users (200 draws) for
     intervals.

    python controls/route_constraint.py   -> results/route_equal_sample.csv, route_cells.csv, route_branching_wls.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
import geolife_robust as gr
import memory_by_mode as mm

MODES = mm.MODES
N_USERS, N_TRI, N_DRAWS, NPERM, MIN_VISITS = 12, 1000, 10, 30, 20


def user_paths(args):
    mode, f = args
    _, user, paths, _ = mm.load(args)
    return mode, user, paths


def first_trigrams(paths, n):
    out, tot = [], 0
    for s in paths:
        if tot >= n: break
        if len(s) < 3: continue
        take = min(len(s), n - tot + 2)
        out.append(s[:take]); tot += take - 2
    return out if tot >= n else None


def cmi_paths(paths, seed):
    a, b, c = gr.ngrams(paths, 3)
    vocab = np.unique(np.concatenate(paths)); K = len(vocab); enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c = enc(a), enc(b), enc(c)
    real, nullm, p, h1, h2 = gr.cmi_with_null(a, b, c, K, NPERM, np.random.default_rng(seed))
    return dict(n_trigrams=len(a), H1=h1, excess=real - nullm, rel_excess=(real - nullm) / h1 if h1 > 0 else np.nan, p=p)


def local_cmi(paths, seed):
    """Per-cell plug-in CMI, its within-cell permutation mean, visits, branching degree and successor entropy."""
    a, b, c = gr.ngrams(paths, 3)
    vocab = np.unique(np.concatenate(paths)); K = len(vocab); enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c = enc(a), enc(b), enc(c)
    def per_cell(a_):
        # H(C|B=b) - H(C|A,B=b) for every b, from joint counts
        nb = np.bincount(b, minlength=K).astype(float)
        kbc, nbc = np.unique(b * K + c, return_counts=True)
        Hc_b = np.zeros(K); np.add.at(Hc_b, kbc // K, -nbc * np.log2(nbc / nb[kbc // K]))
        ab = np.unique(a_ * K + b, return_inverse=True)[1].astype(np.int64)
        nab = np.bincount(ab).astype(float)
        kabc, nabc = np.unique(ab * K + c, return_counts=True)
        b_of_ab = np.zeros(len(nab), np.int64); b_of_ab[ab] = b
        Hc_ab = np.zeros(K); np.add.at(Hc_ab, b_of_ab[kabc // K], -nabc * np.log2(nabc / nab[kabc // K]))
        with np.errstate(invalid='ignore', divide='ignore'):
            return (Hc_b - Hc_ab) / nb, nb, Hc_b / nb
    real, nb, Hsucc = per_cell(a)
    rng = np.random.default_rng(seed); order = np.argsort(b, kind='stable')
    null = np.mean([per_cell(gr.strat_shuffle(a, b, rng, order))[0] for _ in range(NPERM)], axis=0)
    deg = np.bincount(np.unique(b * K + c) // K, minlength=K)
    keep = nb >= MIN_VISITS
    return pd.DataFrame(dict(cell=np.flatnonzero(keep), visits=nb[keep], local_plugin=real[keep], local_null=null[keep],
                             local_excess=(real - null)[keep], branching=deg[keep], succ_entropy=Hsucc[keep]))


def wls(df, terms):
    X = [np.ones(len(df))]; names = ['const']
    for t in terms:
        if t == 'mode':
            for m in MODES:
                if m != 'car': X.append((df['mode'] == m).astype(float).to_numpy()); names.append('mode_' + m)
        else:
            X.append(df[t].to_numpy(float)); names.append(t)
    X = np.column_stack(X); w = df.visits.to_numpy(float); sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], df.local_excess.to_numpy() * sw, rcond=None)[0]
    return pd.Series(beta, index=names)


if __name__ == '__main__':
    jobs = [(m, f) for m in MODES for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), m, 'plt', '*.plt')))]
    with Pool(2) as pool:
        U = [r for r in pool.imap(user_paths, jobs)]
    # (i) equal samples
    rows = []
    for mode in MODES:
        cand = [(u, first_trigrams(p, N_TRI)) for m, u, p in U if m == mode]
        cand = [(u, p) for u, p in cand if p is not None]
        for d in range(N_DRAWS):
            rng = np.random.default_rng(zlib.crc32(f'{mode}|{d}'.encode()))
            pick = rng.choice(len(cand), min(N_USERS, len(cand)), replace=False)
            for j in pick:
                u, p = cand[j]
                r = cmi_paths(p, zlib.crc32(f'{mode}|{u}|{d}'.encode()))
                rows.append(dict(mode=mode, draw=d, user=u, eligible_users=len(cand), **r))
        print(mode, 'eligible users', len(cand), flush=True)
    E = pd.DataFrame(rows); E.to_csv(P.out(P.CONTROLS_RESULTS / 'route_equal_sample.csv'), index=False)
    S = E.groupby(['mode', 'draw']).agg(abs_excess=('excess', 'median'), rel_excess=('rel_excess', 'median'), sig=('p', lambda x: (x < .05).mean()))
    print(S.groupby('mode').agg(['mean', 'min', 'max']).round(4))
    # (ii) branching
    cells = []
    for mode, u, p in U:
        a, _, _ = gr.ngrams(p, 3)
        if len(a) < 200: continue
        c = local_cmi(p, zlib.crc32(f'{mode}|{u}|cells'.encode())); c['mode'] = mode; c['user'] = u
        cells.append(c)
    C = pd.concat(cells, ignore_index=True)
    C['log2_branching'] = np.log2(C.branching)
    C.to_csv(P.out(P.CONTROLS_RESULTS / 'route_cells.csv'), index=False)
    fits = {'mode only': ['mode'], 'branching + mode': ['log2_branching', 'mode'], 'successor entropy + mode': ['succ_entropy', 'mode']}
    est = {k: wls(C, t) for k, t in fits.items()}
    users = C.user + '|' + C['mode']
    uu = users.unique(); boot = {k: [] for k in fits}
    rng = np.random.default_rng(7)
    grp = {g: d for g, d in C.groupby(users)}
    for _ in range(200):
        Cb = pd.concat([grp[g] for g in rng.choice(uu, len(uu))], ignore_index=True)
        for k, t in fits.items(): boot[k].append(wls(Cb, t))
    out = []
    for k in fits:
        B = pd.DataFrame(boot[k])
        for term, v in est[k].items():
            out.append(dict(model=k, term=term, coef=v, lo=B[term].quantile(.025), hi=B[term].quantile(.975)))
    O = pd.DataFrame(out); O.to_csv(P.out(P.CONTROLS_RESULTS / 'route_branching_wls.csv'), index=False)
    print(O.round(4).to_string(index=False))
    print(C.groupby('mode').agg(cells=('cell', 'size'), branching=('branching', 'median'), local_excess=('local_excess', 'median')).round(4))
