"""Genuine memory by transportation mode (GeoLife mode-labelled trajectories, segments split at gaps > 30 s).
(A) Path memory: 100 m grid (0.001 deg) cells from native GPS fixes, repeats collapsed (optionally de-jittered);
    per-user bias-corrected CMI I(S_{t+1};S_{t-1}|S_t) with stratified permutation null (30 perms), relative to H(S_{t+1}|S_t).
(B) Speed memory: 5 s speed; held-out gain from adding v_{t-1} under coarse (10-bin) and fine (50-bin) current speed,
    and after adding acceleration a_t = (v_t - v_{t-1})/dt binned in 10 quantiles. Target v_{t+1} in 10 bins. Split by user."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as _P
import importlib.util, glob, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location('m4', str(_P.GEOLIFE_PIPELINE / '04_mode_hurst_memory.py')); m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
import geolife_robust as gr
MODES = ['train', 'subway', 'car', 'taxi', 'bus', 'bike', 'walk']
GRID = 0.001; DT = 5.0

def load(args):
    mode, f = args
    raw = m4.read_plt(m4.Path(f)); user = os.path.basename(f)[:3]
    paths, speeds = [], []
    for seg in m4.split_by_gap(raw, 30.0, 30):
        st = gr.to_states(seg.lat.to_numpy(), seg.lon.to_numpy(), GRID)
        if len(st) >= 3: paths.append(st)
        rs = m4.resample_segment(seg, DT)
        if rs is None: continue
        _, _, sp = m4.velocity(rs, DT)
        if len(sp) >= 20: speeds.append(sp)
    return mode, user, paths, speeds

def path_memory(seqs, dj, rng):
    if dj: seqs = [gr.dejitter(s) for s in seqs]; seqs = [s for s in seqs if len(s) >= 3]
    a, b, c = gr.ngrams(seqs, 3)
    if len(a) < 200: return None
    vocab = np.unique(np.concatenate(seqs)); K = len(vocab); enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c = enc(a), enc(b), enc(c)
    real, nullm, p, h1, h2 = gr.cmi_with_null(a, b, c, K, 30, rng)
    return dict(n_trigrams=len(a), H1=h1, cmi_plugin=real, cmi_null=nullm, cmi_excess=real - nullm, p=p, rel_excess=(real - nullm) / h1 if h1 > 0 else np.nan)

def speed_memory(train, val):
    allv = np.concatenate(train)
    e10 = np.quantile(allv, np.linspace(0, 1, 11)); e50 = np.quantile(allv, np.linspace(0, 1, 51))
    acc = np.concatenate([np.diff(s) / DT for s in train]); ea = np.quantile(acc, np.linspace(0, 1, 11))
    for e in (e10, e50, ea): e[0], e[-1] = -np.inf, np.inf
    q = lambda x, e: np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2).astype(np.int64)
    def steps(seqs):
        cols = {k: [] for k in ['y', 'v10', 'v50', 'p10', 'a10']}
        for s in seqs:
            if len(s) < 4: continue
            t = np.arange(1, len(s) - 1)
            cols['y'].append(q(s[t + 1], e10)); cols['v10'].append(q(s[t], e10)); cols['v50'].append(q(s[t], e50))
            cols['p10'].append(q(s[t - 1], e10)); cols['a10'].append(q((s[t] - s[t - 1]) / DT, ea))
        return {k: np.concatenate(v) for k, v in cols.items()}
    T, V = steps(train), steps(val)
    radix = dict(v10=10, v50=50, p10=10, a10=10)
    def ll(cols, alpha=0.5):
        k = lambda d: sum(d[c] * int(np.prod([radix[x] for x in cols[i + 1:]])) for i, c in enumerate(cols))
        kt, kv = k(T), k(V)
        kj, nj = np.unique(kt * 10 + T['y'], return_counts=True); kc, nc = np.unique(kt, return_counts=True)
        look = lambda keys, cnt, qq: np.where(keys[np.clip(np.searchsorted(keys, qq), 0, len(keys) - 1)] == qq, cnt[np.clip(np.searchsorted(keys, qq), 0, len(keys) - 1)], 0)
        p = (look(kj, nj, kv * 10 + V['y']) + alpha) / (look(kc, nc, kv) + alpha * 10)
        return float(np.log(p).mean())
    r = dict(V1c=ll(['v10']), V2c=ll(['v10', 'p10']), V1f=ll(['v50']), V2f=ll(['v50', 'p10']), VAf=ll(['v50', 'a10']), VA2f=ll(['v50', 'a10', 'p10']))
    r.update(gain_coarse=r['V2c'] - r['V1c'], gain_fine=r['V2f'] - r['V1f'], gain_after_a=r['VA2f'] - r['VAf'], n_val=len(V['y']), n_train=len(T['y']))
    return r

if __name__ == '__main__':
    jobs = [(m, f) for m in MODES for f in sorted(glob.glob(os.path.join(str(_P.GEOLIFE_TRIPS_BY_MODE), m, 'plt', '*.plt')))]
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with Pool(2) as p: res = list(p.imap_unordered(load, jobs))
    P, S = [], []
    for mode, user, paths, speeds in res:
        for dj in (0, 1):
            r = path_memory(paths, dj, np.random.default_rng(int(user) * 7 + dj))
            if r: P.append(dict(mode=mode, user=user, dejitter=dj, **r))
    P = pd.DataFrame(P); P.to_csv(_P.out(_P.CONTROLS_RESULTS / 'memory_by_mode_path.csv'), index=False)
    pd.set_option('display.width', 250)
    print(P.groupby(['dejitter', 'mode']).agg(users=('user', 'size'), H1=('H1', 'median'), plugin=('cmi_plugin', 'median'), excess=('cmi_excess', 'median'),
          rel=('rel_excess', 'median'), sig=('p', lambda x: (x < .05).mean())).round(3))
    for mode in MODES:
        users = sorted({u for m, u, _, sp in res if m == mode and sp})
        for split in range(5):
            rng = np.random.default_rng(split); perm = rng.permutation(users); ntr = max(1, int(round(0.7 * len(perm))))
            tr = [s for m, u, _, sp in res if m == mode and u in perm[:ntr] for s in sp]
            va = [s for m, u, _, sp in res if m == mode and u in perm[ntr:] for s in sp]
            if not tr or not va: continue
            S.append(dict(mode=mode, split=split, n_users=len(users), **speed_memory(tr, va)))
    S = pd.DataFrame(S); S.to_csv(_P.out(_P.CONTROLS_RESULTS / 'memory_by_mode_speed.csv'), index=False)
    print(S.groupby('mode')[['n_users', 'n_val', 'V1c', 'V1f', 'gain_coarse', 'gain_fine', 'gain_after_a']].agg(['mean', 'std']).round(4))
