"""
highD supplement: does the apparent non-Markovian memory of the longitudinal-velocity process vanish
once the state is augmented with acceleration and car-following interaction variables?

Common sampling grid: frames with frame % 5 == 0 (dt = 0.2 s), so that every vehicle present in a frame
can be matched to its leader in the same frame.

Per vehicle and sampled frame t:
  v_t   signed longitudinal velocity, central difference of x at 25 Hz            (same as main analysis)
  a_t   backward acceleration  (v_raw[t] - v_raw[t-5]) / 0.2 s                      (uses only the past)
  dv_t  leader speed minus own speed, in driving direction (leader = nearest vehicle ahead, same direction,
        |dy| < 1.8 m)
  g_t   centre-to-centre gap to that leader (m); 'no leader' is its own category
Target: v_{t+1} discretised into 20 equal-frequency bins (train-only quantiles).
Contexts compared (all evaluated on the same steps):
  V1 = (v_t)            V2 = (v_t, v_{t-1})            V3 = (v_t, v_{t-1}, v_{t-2})
  VA1 = (v_t, a_t)      VA2 = (v_t, a_t, v_{t-1})
  VAI1 = (v_t, a_t, dv_t, g_t)                         VAI2 = (v_t, a_t, dv_t, g_t, v_{t-1})
The same with |v| (speed magnitude) for V1..V3.
Also: VACF/GLE kernel on long tracks only (>= 24 s), at 0.2 s and at 0.04 s.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, sys, json, time
import numpy as np
import pandas as pd

DATA = str(P.HIGHD_TRACKS)
DT_RAW, DS = 0.04, 5
DT = DT_RAW * DS
RES = str(P.CONTROLS_RESULTS)   # tables and kernels (committed)
WORK = str(P.CONTROLS_WORK)     # highd_states.pkl, raw_long.npy (large, git-ignored)
os.makedirs(RES, exist_ok=True); os.makedirs(WORK, exist_ok=True)


def process_file(path):
    df = pd.read_csv(path, usecols=['frame', 'id', 'x', 'y'], dtype={'frame': np.int32, 'id': np.int32, 'x': np.float64, 'y': np.float32})
    df.sort_values(['id', 'frame'], inplace=True)
    rows = []
    raw_v = []          # 25 Hz velocity series for long tracks (VACF at 0.04 s)
    for vid, g in df.groupby('id', sort=False):
        if len(g) < 50:
            continue
        fr = g.frame.to_numpy(); x = g.x.to_numpy(); y = g.y.to_numpy()
        v = np.full(len(x), np.nan)
        v[1:-1] = (x[2:] - x[:-2]) / (2 * DT_RAW)
        a = np.full(len(x), np.nan)
        a[6:] = (v[6:] - v[1:-5]) / DT
        d = np.sign(np.nanmedian(v))
        keep = (fr % DS == 0) & np.isfinite(v)
        rows.append(pd.DataFrame({'frame': fr[keep], 'id': vid, 'x': x[keep], 'y': y[keep], 'v': v[keep], 'a': a[keep], 'dir': d}))
        vv = v[1:-1]
        if len(vv) * DT_RAW >= 24.0:
            raw_v.append(vv.astype(np.float32))
    s = pd.concat(rows, ignore_index=True)
    # ---- leader search within each frame
    base = s[['frame', 'id', 'x', 'y', 'v', 'dir']]
    frames = np.sort(base.frame.unique())
    found = []
    for k in range(0, len(frames), 400):               # chunk over frames to bound memory
        fr_chunk = frames[k:k + 400]
        b = base[base.frame.isin(fr_chunk)]
        m = b.merge(b, on=['frame', 'dir'], suffixes=('', '_l'))
        m = m[(m.id != m.id_l) & (np.abs(m.y - m.y_l) < 1.8)]
        m = m.assign(gap=(m.x_l - m.x) * m.dir)
        m = m[m.gap > 0]
        if len(m):
            m = m.loc[m.groupby(['frame', 'id']).gap.idxmin()]
            m = m.assign(dv=(m.v_l - m.v) * m.dir)
            found.append(m[['frame', 'id', 'gap', 'dv']])
    m = pd.concat(found) if found else pd.DataFrame(columns=['frame', 'id', 'gap', 'dv'])
    s = s.merge(m, on=['frame', 'id'], how='left')
    s.sort_values(['id', 'frame'], inplace=True)
    s['rec'] = os.path.basename(path)[:2]
    return s, raw_v


def load_all():
    files = sorted(glob.glob(os.path.join(DATA, '*_tracks.csv')))
    parts, raws = [], []
    t0 = time.time()
    for f in files:
        s, rv = process_file(f)
        parts.append(s); raws += rv
        print(f'{os.path.basename(f)} rows={len(s)} leader_frac={s.gap.notna().mean():.3f} [{time.time()-t0:.0f}s]', flush=True)
    return pd.concat(parts, ignore_index=True), raws


# ------------------------------------------------------------------ Markov comparison
def qbins(x, edges):
    return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2).astype(np.int64)


def qedges(x, n):
    e = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, n + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def build_steps(S, ids_set, edges, speed=False):
    """Return dict of int arrays aligned on steps t (needs t-2..t+1 consecutive frames)."""
    sub = S[S.uid.isin(ids_set)]
    vv = np.abs(sub.v.to_numpy()) if speed else sub.v.to_numpy()
    vb = qbins(vv, edges['v'])
    ab = qbins(sub.a.to_numpy(), edges['a'])
    dvb = qbins(sub.dv.fillna(0).to_numpy(), edges['dv'])
    gb = np.where(sub.gap.isna(), len(edges['g']) - 1, qbins(sub.gap.fillna(0).to_numpy(), edges['g']))
    uid = sub.uid.to_numpy(); fr = sub.frame.to_numpy(); afin = np.isfinite(sub.a.to_numpy())
    n = len(sub)
    ok = np.zeros(n, bool)
    # step at index i means current t=i, history i-1,i-2, target i+1
    i = np.arange(2, n - 1)
    ok[i] = (uid[i - 2] == uid[i + 1]) & (fr[i + 1] - fr[i - 2] == 3 * DS) & afin[i]
    i = np.where(ok)[0]
    return dict(y=vb[i + 1], v0=vb[i], v1=vb[i - 1], v2=vb[i - 2], a=ab[i], dv=dvb[i], g=gb[i])


def ctx_key(st, cols, radix):
    k = np.zeros(len(st['y']), np.int64)
    for c in cols:
        k = k * radix[c] + st[c]
    return k


def eval_model(tr, va, cols, radix, nv, alpha=0.5):
    ktr = ctx_key(tr, cols, radix); kva = ctx_key(va, cols, radix)
    joint = ktr * nv + tr['y']
    kj, nj = np.unique(joint, return_counts=True)
    kc, nc = np.unique(ktr, return_counts=True)
    def look(keys, cnt, q):
        idx = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        return np.where(keys[idx] == q, cnt[idx], 0).astype(float)
    def ll(k, y):
        p = (look(kj, nj, k * nv + y) + alpha) / (look(kc, nc, k) + alpha * nv)
        return np.log(p).mean()
    return ll(ktr, tr['y']), ll(kva, va['y']), len(kc)


MODELS = {
    'V1': ['v0'], 'V2': ['v0', 'v1'], 'V3': ['v0', 'v1', 'v2'],
    'VA1': ['v0', 'a'], 'VA2': ['v0', 'a', 'v1'],
    'VAI1': ['v0', 'a', 'dv', 'g'], 'VAI2': ['v0', 'a', 'dv', 'g', 'v1'],
}


def markov_block(S, n_splits=3, speed=False, models=MODELS):
    uids = S.uid.unique()
    res = []
    for sp in range(n_splits):
        rng = np.random.default_rng(100 + sp)
        perm = rng.permutation(uids)
        ntr = int(0.8 * len(perm))
        tr_ids, va_ids = set(perm[:ntr]), set(perm[ntr:])
        T = S[S.uid.isin(tr_ids)]
        vv = np.abs(T.v.to_numpy()) if speed else T.v.to_numpy()
        edges = dict(v=qedges(vv, 20), a=qedges(T.a.to_numpy(), 7), dv=qedges(T.dv.to_numpy(), 5), g=qedges(T.gap.to_numpy(), 4))
        # g edges has 5 entries -> 4 bins (0..3) ; 'no leader' -> 4
        edges['g'] = np.append(edges['g'], np.inf)
        radix = dict(v0=20, v1=20, v2=20, a=7, dv=5, g=5)
        tr = build_steps(S, tr_ids, edges, speed); va = build_steps(S, va_ids, edges, speed)
        for name, cols in models.items():
            ltr, lva, nctx = eval_model(tr, va, cols, radix, 20)
            res.append(dict(split=sp, model=name, train_ll=ltr, val_ll=lva, n_ctx=nctx, n_train=len(tr['y']), n_val=len(va['y'])))
        print(f'split {sp} speed={speed}:', {r['model']: round(r['val_ll'], 4) for r in res if r['split'] == sp}, flush=True)
    return pd.DataFrame(res)


# ------------------------------------------------------------------ VACF / kernel
def vacf(seqs, max_lag):
    acc = np.zeros(max_lag); cnt = np.zeros(max_lag)
    for v in seqs:
        u = v.astype(np.float64) - v.mean()
        n = len(u)
        F = np.fft.rfft(u, 2 * n)
        ac = np.fft.irfft(F * np.conj(F))[:n]
        m = min(n, max_lag)
        acc[:m] += ac[:m]; cnt[:m] += np.arange(n, n - m, -1)
    C = acc / np.maximum(cnt, 1)
    return C / C[0], C[0]


def kernel(C, dt):
    N = len(C); K = np.zeros(N - 1)
    Cavg = 0.5 * (C[:-1] + C[1:]); L = -np.diff(C) / dt ** 2
    for m in range(N - 1):
        K[m] = (L[m] - np.dot(K[:m][::-1], Cavg[1:m + 1])) / Cavg[0]
    return K


def kernel_summary(C, dt, label):
    K = kernel(C, dt)
    tau = float(np.argmax(C < np.exp(-1)) * dt) if np.any(C < np.exp(-1)) else float('nan')
    g_inst = K[0] * dt
    Ts = [0.4, 1, 2, 4, 6, 7.6]
    frac = {}
    for T in Ts:
        m = int(round(T / dt))
        gm = K[1:m + 1].sum() * dt
        frac[f'{T:g}'] = float(gm / (g_inst + gm))
    return dict(label=label, dt=dt, tau_C=tau, K0=float(K[0]), K1=float(K[1]), K1_over_K0=float(K[1] / K[0]),
                plateau_mean=float(K[int(1 / dt):].mean()), mem_frac_vs_T=frac,
                C_zero_cross=float(np.argmax(C < 0) * dt) if np.any(C < 0) else None), K


if __name__ == '__main__':
    t0 = time.time()
    cache = os.path.join(WORK, 'highd_states.pkl')
    if os.path.exists(cache):
        S = pd.read_pickle(cache); raws = list(np.load(os.path.join(WORK, 'raw_long.npy'), allow_pickle=True))
    else:
        S, raws = load_all()
        S['uid'] = S.rec.astype(int) * 100000 + S.id
        S.to_pickle(cache)
        np.save(os.path.join(WORK, 'raw_long.npy'), np.array(raws, dtype=object), allow_pickle=True)
    print('vehicles', S.uid.nunique(), 'rows', len(S), 'leader frac', round(S.gap.notna().mean(), 3), flush=True)

    r = markov_block(S, speed=False); r['state'] = 'signed_v'
    r2 = markov_block(S, speed=True, models={k: MODELS[k] for k in ['V1', 'V2', 'V3']}); r2['state'] = 'speed'
    R = pd.concat([r, r2]); R.to_csv(os.path.join(RES, 'augment_markov.csv'), index=False)
    print(R.groupby(['state', 'model'])[['train_ll', 'val_ll', 'n_ctx']].mean().round(4), flush=True)

    # VACF on the 0.2 s grid: per-vehicle sequences of v, long tracks only
    out = {}
    seqs02 = [g.v.to_numpy() for _, g in S.groupby('uid') if len(g) * DT >= 24.0]
    seqs02_all = [g.v.to_numpy() for _, g in S.groupby('uid') if len(g) >= 10]
    C, c0 = vacf(seqs02_all, 40); out['dt0.2_all'], K_all = kernel_summary(C, DT, 'all tracks, dt=0.2')
    C2, c02 = vacf(seqs02, 40);   out['dt0.2_long'], K_long = kernel_summary(C2, DT, 'tracks>=24 s, dt=0.2')
    C3, c03 = vacf(raws, 200);    out['dt0.04_long'], K_raw = kernel_summary(C3, DT_RAW, 'tracks>=24 s, dt=0.04')
    out['n_long_tracks'] = len(seqs02); out['n_all_tracks'] = len(seqs02_all)
    np.savez(os.path.join(RES, 'vacf_kernels.npz'), C_all=C, C_long=C2, C_raw=C3, K_all=K_all, K_long=K_long, K_raw=K_raw)
    json.dump(out, open(os.path.join(RES, 'vacf_summary.json'), 'w'), indent=2)
    print(json.dumps(out, indent=1))
    print('done', time.time() - t0)
