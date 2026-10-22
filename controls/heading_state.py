"""P3-1  Is second-order path memory only the missing velocity of the grid state?

After run-length compression the previous cell S_{t-1}, given the current cell S_t, tells mainly from which direction,
how fast and through which side the trajectory entered S_t: a kinematic state, like the acceleration in highD. Here the
grid state is augmented with a kinematic label A_t measured from the raw fixes at the entry into S_t, and we ask how
much of the second-order gain survives:
  heading  bearing from the latest earlier fix at least D metres away to the entry fix, 8 sectors (+ 'unknown')
  side     position of the entry fix inside the cell, 3 x 3
  speed    distance / time over the same displacement (taxi), or the distance from the previous fix (GeoLife cache,
           which has no timestamps; GeoLife fixes are 1-5 s apart), tertiles of the training part (+ 'unknown')
Two measures per unit (bits):
  1. held-out log-likelihood with interpolated back-off (as in the paper, beta = 3 and 10, add-1 unigram base):
       G2 = LL[S, S-] - LL[S]               second-order gain (the paper's path memory)
       GA = LL[S, A] - LL[S]                gain of the kinematic label
       R  = LL[S, A, S-] - LL[S, A]         residual second-order gain given the label   (chain S -> S,A -> S,A,S-)
     for A = heading, heading + side, heading + side + speed.
     Second version (added 10-22 after P3-9 showed that one beta for every level favours the longer chain, and the
     taxi R turned negative at beta = 3): G2_tuned, GA_*_tuned, R_*_tuned with each model's strengths chosen level by
     level from GRID on an inner split (last 20 % of every training sequence), refitted on the training part.
  2. excess CMI with the stratified permutation null: E0 = I(S+; S- | S), Eh = I(S+; S- | S, heading); the same with
     the heading shuffled within cells (keeps the number of strata, removes the kinematic link) as a power control.
Units: GeoLife users from the cleaned cache (train / test parts as in the paper), 10 m and 100 m; taxi: all vehicles
pooled on a global 0.002 deg grid (~200 m), sequences split at gaps > 5 min, test = vehicles with crc32(file) % 5 == 0.
Seeds: crc32('<dataset>|<grid>|<unit>'). Each unit's result is written atomically to
controls/work/heading_state/<dataset>_<grid>/<unit>.json and skipped on a restart; the parsed taxi table is cached
in controls/work/heading_state/taxi_fixes.pkl.

    python controls/heading_state.py geolife 0.001     -> results/heading_state_geolife_100m.csv
    python controls/heading_state.py geolife 0.0001    -> results/heading_state_geolife_10m.csv
    python controls/heading_state.py taxi 0.002        -> results/heading_state_taxi_200m.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, json, os, sys, time, zlib
import numpy as np, pandas as pd
import geolife_robust as gr

NPERM = 20
BETAS = (3.0, 10.0)
WORK = P.CONTROLS_WORK / 'heading_state'
REF_DIST = {0.0001: 10.0, 0.001: 30.0, 0.002: 50.0}          # metres: minimum displacement for the heading
LOOKBACK = 20                                                 # fixes searched back for that displacement


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = str(path) + '.tmp'
    json.dump(obj, open(tmp, 'w')); os.replace(tmp, path)


def runs(lat, lon, g, t=None, ref=30.0):
    """Cells of one trip after run-length compression and the kinematic label at each run's entry fix.
    Returns cell ids (int64), heading (0-7, 8 = unknown), side (0-8), raw speed value (nan = unknown)."""
    la = np.floor(lat / g).astype(np.int64); lo = np.floor(lon / g).astype(np.int64)
    key = la * 10_000_000 + lo
    n = len(key)
    if n == 0:
        return [np.zeros(0, np.int64)] * 3 + [np.zeros(0)]
    start = np.r_[True, key[1:] != key[:-1]]; e = np.where(start)[0]
    side = (np.minimum((lat[e] / g - la[e]) * 3, 2.999).astype(int) * 3 + np.minimum((lon[e] / g - lo[e]) * 3, 2.999).astype(int))
    head = np.full(len(e), 8, np.int64); spd = np.full(len(e), np.nan)
    found = np.zeros(len(e), bool); cosl = np.cos(np.radians(np.nanmean(lat)))
    for k in range(1, LOOKBACK + 1):
        j = e - k; ok = (j >= 0) & ~found
        if not ok.any():
            break
        jj = np.where(ok, j, 0)
        dx = (lon[e] - lon[jj]) * cosl * 111320.0; dy = (lat[e] - lat[jj]) * 110540.0; d = np.hypot(dx, dy)
        hit = ok & (d >= ref)
        if hit.any():
            brg = (np.degrees(np.arctan2(dx[hit], dy[hit])) + 360.0) % 360.0
            head[hit] = ((brg + 22.5) // 45).astype(np.int64) % 8
            if t is not None:
                dt = t[e[hit]] - t[jj[hit]]; spd[hit] = np.where(dt > 0, d[hit] / np.maximum(dt, 1e-9), np.nan)
            found |= hit
    if t is None:                                              # GeoLife cache: no time; distance from the previous fix
        j = e - 1; ok = j >= 0; jj = np.where(ok, j, 0)
        d1 = np.hypot((lon[e] - lon[jj]) * cosl * 111320.0, (lat[e] - lat[jj]) * 110540.0)
        spd = np.where(ok & (head < 8), d1, np.nan)
    return key[e], head, side, spd


def trigram_table(seqs):
    """seqs: list of (cells, head, side, spd) per trip -> columns of all trigrams (S-, S, S+) with the label at S."""
    cols = {k: [] for k in ('m', 's', 'p', 'h', 'sd', 'v')}
    for c, h, sd, v in seqs:
        if len(c) < 3:
            continue
        cols['m'].append(c[:-2]); cols['s'].append(c[1:-1]); cols['p'].append(c[2:])
        cols['h'].append(h[1:-1]); cols['sd'].append(sd[1:-1]); cols['v'].append(v[1:-1])
    return {k: (np.concatenate(x) if x else np.zeros(0)) for k, x in cols.items()}


def combine(*ids):
    """Factorise a tuple of non-negative int arrays into one id array (no overflow)."""
    out = np.zeros(len(ids[0]), np.int64)
    for a in ids:
        a = np.unique(a, return_inverse=True)[1].astype(np.int64)
        out = np.unique(out * (int(a.max()) + 1 if len(a) else 1) + a, return_inverse=True)[1].astype(np.int64)
    return out


def bo_ll(y, chain, K, tr, beta):
    """Interpolated back-off: P_0 = add-1 unigram, P_k = (n(ctx_k, y) + beta P_{k-1}) / (n(ctx_k) + beta).
    y, chain: arrays over train + test rows (contexts already factorised jointly); tr: boolean train mask. Bits/step."""
    te = ~tr
    uni = np.bincount(y[tr], minlength=K).astype(float); Pp = (uni[y[te]] + 1.0) / (uni.sum() + K)
    for ctx in chain:
        kj, nj = np.unique(ctx[tr] * K + y[tr], return_counts=True); kc, nc = np.unique(ctx[tr], return_counts=True)
        Pp = (gr.lookup(kj, nj, ctx[te] * K + y[te]) + beta * Pp) / (gr.lookup(kc, nc, ctx[te]) + beta)
    return float(np.log2(Pp).mean())


def heldout(T, tr):
    """T: trigram table over train + test rows; tr: train mask. Returns the gains for the three labels."""
    vocab = np.unique(np.concatenate([T['m'], T['s'], T['p']])); K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    m, s, y = enc(T['m']), enc(T['s']), enc(T['p'])
    edges = np.nanquantile(T['v'][tr], [1 / 3, 2 / 3]) if np.isfinite(T['v'][tr]).any() else np.array([np.inf, np.inf])
    vb = np.where(np.isfinite(T['v']), np.searchsorted(edges, np.nan_to_num(T['v'], nan=0.0)), 3)
    labels = {'heading': combine(T['h']), 'heading+side': combine(T['h'], T['sd']), 'heading+side+speed': combine(T['h'], T['sd'], vb)}
    S = combine(s); SM = combine(s, m)
    out = dict(n_train=int(tr.sum()), n_test=int((~tr).sum()), share_heading_known=float((T['h'] < 8).mean()))
    for beta in BETAS:
        b = f'_b{beta:g}'
        l1 = bo_ll(y, [S], K, tr, beta); l2 = bo_ll(y, [S, SM], K, tr, beta)
        out['G2' + b] = l2 - l1
        for name, A in labels.items():
            SA = combine(s, A); SAM = combine(s, A, m)
            la = bo_ll(y, [S, SA], K, tr, beta); la2 = bo_ll(y, [S, SA, SAM], K, tr, beta)
            out[f'GA_{name}{b}'] = la - l1; out[f'R_{name}{b}'] = la2 - la
    return out


GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def bo_ll_multi(y, chain, K, tr, betas):
    """bo_ll with one beta per level of the chain."""
    te = ~tr
    uni = np.bincount(y[tr], minlength=K).astype(float); Pp = (uni[y[te]] + 1.0) / (uni.sum() + K)
    for ctx, beta in zip(chain, betas):
        kj, nj = np.unique(ctx[tr] * K + y[tr], return_counts=True); kc, nc = np.unique(ctx[tr], return_counts=True)
        Pp = (gr.lookup(kj, nj, ctx[te] * K + y[te]) + beta * Pp) / (gr.lookup(kc, nc, ctx[te]) + beta)
    return float(np.log2(Pp).mean())


def chains(T, tr):
    """Context chains of the models compared (ids over train + test rows of table T)."""
    vocab = np.unique(np.concatenate([T['m'], T['s'], T['p']])); K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    m, s, y = enc(T['m']), enc(T['s']), enc(T['p'])
    edges = np.nanquantile(T['v'][tr], [1 / 3, 2 / 3]) if np.isfinite(T['v'][tr]).any() else np.array([np.inf, np.inf])
    vb = np.where(np.isfinite(T['v']), np.searchsorted(edges, np.nan_to_num(T['v'], nan=0.0)), 3)
    labels = {'heading': (T['h'],), 'heading+side': (T['h'], T['sd']), 'heading+side+speed': (T['h'], T['sd'], vb)}
    C = {'M1': [combine(s)], 'M2': [combine(s), combine(s, m)]}
    for name, A in labels.items():
        C[f'MA_{name}'] = [combine(s), combine(s, *A)]
        C[f'MA2_{name}'] = [combine(s), combine(s, *A), combine(s, *A, m)]
    return y, K, C


def heldout_tuned(Tin, trin, T, tr):
    """Each model's back-off strengths chosen level by level (grid GRID) on the inner split (Tin, trin: inner-train
    rows vs validation rows, both inside the training part), then refitted on the whole training part and scored on
    the test rows. Removes the advantage a fixed beta gives to the longer chain (P3-9). Bits per test step."""
    yi, Ki, Ci = chains(Tin, trin); y, K, C = chains(T, tr)
    ll, bet = {}, {}
    for name, ch in Ci.items():
        b = []
        for lev in range(len(ch)):
            b.append(max(GRID, key=lambda x: bo_ll_multi(yi, ch[:lev + 1], Ki, trin, b + [x])))
        bet[name] = b; ll[name] = bo_ll_multi(y, C[name], K, tr, b)
    out = dict(G2_tuned=ll['M2'] - ll['M1'], betas_M2='/'.join(f'{x:g}' for x in bet['M2']))
    for lab in ('heading', 'heading+side', 'heading+side+speed'):
        out[f'GA_{lab}_tuned'] = ll[f'MA_{lab}'] - ll['M1']; out[f'R_{lab}_tuned'] = ll[f'MA2_{lab}'] - ll[f'MA_{lab}']
        out[f'betas_MA2_{lab}'] = '/'.join(f'{x:g}' for x in bet[f'MA2_{lab}'])
    return out


def inner_split(seqs, frac=0.8):
    """Split each training sequence (tuple of arrays from runs()) in time: first 80 % inner-train, rest validation."""
    a, b = [], []
    for r in seqs:
        n = len(r[0]); cut = int(round(n * frac))
        if cut >= 3 and n - cut >= 3:
            a.append(tuple(x[:cut] for x in r)); b.append(tuple(x[cut:] for x in r))
        else:
            a.append(r)
    return a, b


def stack(seqs_a, seqs_b):
    Ta, Tb = trigram_table(seqs_a), trigram_table(seqs_b)
    return {k: np.r_[Ta[k], Tb[k]] for k in Ta}, np.r_[np.ones(len(Ta['p']), bool), np.zeros(len(Tb['p']), bool)]


def cmi_part(T, seed):
    vocab = np.unique(np.concatenate([T['m'], T['s'], T['p']])); K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    m, s, y = enc(T['m']), enc(T['s']), enc(T['p']); h = T['h'].astype(np.int64)
    rng = np.random.default_rng(seed)
    r0, n0, p0, h1, _ = gr.cmi_with_null(m, s, y, K, NPERM, rng)
    r1, n1, p1, _, _ = gr.cmi_with_null(m, gr.pair_id(s, h, 9), y, K, NPERM, rng)
    hs = gr.strat_shuffle(h, s, rng, np.argsort(s, kind='stable'))         # heading shuffled within cells
    r2, n2, p2, _, _ = gr.cmi_with_null(m, gr.pair_id(s, hs, 9), y, K, NPERM, rng)
    return dict(n_trigrams=len(y), H1=h1, E0=r0 - n0, p0=p0, Eh=r1 - n1, p_h=p1, Eh_shuffled=r2 - n2, p_h_shuffled=p2)


# ---------------------------------------------------------------- GeoLife (cleaned cache, per user)
def geolife_user(path, g):
    d = np.load(path, allow_pickle=True); ref = REF_DIST[g]
    full, train, test = [], [], []
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        tla, tlo, sla, slo = [np.asarray(x, float) for x in (tla, tlo, sla, slo)]
        m1 = np.isfinite(tla) & np.isfinite(tlo); m2 = np.isfinite(sla) & np.isfinite(slo)
        tla, tlo, sla, slo = tla[m1], tlo[m1], sla[m2], slo[m2]
        full.append(runs(np.r_[tla, sla], np.r_[tlo, slo], g, ref=ref))
        train.append(runs(tla, tlo, g, ref=ref)); test.append(runs(sla, slo, g, ref=ref))
    return full, train, test


def run_geolife(g):
    """Per user two units, each written atomically and skipped on a restart: geolife_<grid>/<user>.json (fixed beta
    held-out part + CMI part) and geolife_<grid>_tuned/<user>.json (held-out part with tuned betas)."""
    lab = {0.0001: '10m', 0.001: '100m'}[g]; wd = WORK / f'geolife_{lab}'; wt = WORK / f'geolife_{lab}_tuned'
    files = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz')))
    t0 = time.time()
    for i, f in enumerate(files):
        user = os.path.basename(f)[:3]; out = wd / f'{user}.json'; out_t = wt / f'{user}.json'
        if out.exists() and out_t.exists():
            continue
        full, train, test = geolife_user(f, g)
        Ttr = trigram_table(train); Tte = trigram_table(test)
        ok = len(Tte['p']) >= 50 and len(Ttr['p']) >= 200
        if not out.exists():
            Tf = trigram_table(full); res = dict(user=user)
            if ok:
                T = {k: np.r_[Ttr[k], Tte[k]] for k in Ttr}; tr = np.r_[np.ones(len(Ttr['p']), bool), np.zeros(len(Tte['p']), bool)]
                res.update(heldout(T, tr))
            if len(Tf['p']) >= 200:
                res.update(cmi_part(Tf, zlib.crc32(f'geolife|{g}|{user}'.encode())))
            atomic_json(out, res)
        res_t = dict(user=user)
        if ok:
            a, b = inner_split(train); Tin, trin = stack(a, b)
            if trin.sum() >= 150 and (~trin).sum() >= 30:
                T, tr = stack(train, test); res_t.update(heldout_tuned(Tin, trin, T, tr))
        atomic_json(out_t, res_t)
        print(f'{lab} {i + 1}/{len(files)} user {user} [{time.time() - t0:.0f}s]', flush=True)
    R = pd.DataFrame([{**json.load(open(wd / f'{os.path.basename(f)[:3]}.json')), **json.load(open(wt / f'{os.path.basename(f)[:3]}.json'))}
                      for f in files])
    R.to_csv(P.out(P.CONTROLS_RESULTS / f'heading_state_geolife_{lab}.csv'), index=False)
    return R


# ---------------------------------------------------------------- taxi (all vehicles pooled)
def taxi_fixes():
    cache = WORK / 'taxi_fixes.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    parts = []
    for f in sorted(glob.glob(os.path.join(str(P.TAXI_RAW), 'Taxi_*'))):
        d = pd.read_csv(f, header=None, names=['id', 't', 'lon', 'lat', 'v', 'h', 's'], skipinitialspace=True, usecols=[1, 2, 3])
        d['t'] = pd.to_datetime(d.t, errors='coerce'); d = d.dropna()
        d = d[(d.lon > 120) & (d.lon < 123) & (d.lat > 30) & (d.lat < 32.5)].sort_values('t').drop_duplicates('t')
        d['file'] = os.path.basename(f); parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    D['ts'] = (D.t - pd.Timestamp('2007-02-20')).dt.total_seconds()
    WORK.mkdir(parents=True, exist_ok=True); tmp = str(cache) + '.tmp'; D.to_pickle(tmp); os.replace(tmp, cache)
    return D


def run_taxi(g):
    """Two units, each written atomically and skipped on a restart: the held-out part and the CMI part."""
    wd = WORK / f'taxi_{g:g}'; wd.mkdir(parents=True, exist_ok=True)
    f_ho, f_cmi, f_t = wd / 'heldout.json', wd / 'cmi.json', wd / 'heldout_tuned.json'
    if not (f_ho.exists() and f_cmi.exists() and f_t.exists()):
        D = taxi_fixes(); seqs_tr, seqs_te, seqs = [], [], []
        for f, d in D.groupby('file', sort=True):
            lat, lon, ts = d.lat.to_numpy(), d.lon.to_numpy(), d.ts.to_numpy()
            cut = np.r_[0, np.where(np.diff(ts) > 300)[0] + 1, len(ts)]
            test = zlib.crc32(f.encode()) % 5 == 0
            for a, b in zip(cut[:-1], cut[1:]):
                r = runs(lat[a:b], lon[a:b], g, t=ts[a:b], ref=REF_DIST[g])
                seqs.append(r); (seqs_te if test else seqs_tr).append(r)
        nveh = int(D.file.nunique()); del D
        if not f_ho.exists():
            Ttr, Tte = trigram_table(seqs_tr), trigram_table(seqs_te)
            T = {k: np.r_[Ttr[k], Tte[k]] for k in Ttr}; tr = np.r_[np.ones(len(Ttr['p']), bool), np.zeros(len(Tte['p']), bool)]
            atomic_json(f_ho, dict(unit='all vehicles', vehicles=nveh, **heldout(T, tr))); print('taxi held-out part done', flush=True)
        if not f_cmi.exists():
            atomic_json(f_cmi, cmi_part(trigram_table(seqs), zlib.crc32(f'taxi|{g}|pooled'.encode())))
        if not f_t.exists():
            a, b = inner_split(seqs_tr); Tin, trin = stack(a, b); T, tr = stack(seqs_tr, seqs_te)
            atomic_json(f_t, heldout_tuned(Tin, trin, T, tr)); print('taxi tuned held-out part done', flush=True)
    R = pd.DataFrame([{**json.load(open(f_ho)), **json.load(open(f_cmi)), **json.load(open(f_t))}])
    R.to_csv(P.out(P.CONTROLS_RESULTS / f'heading_state_taxi_{round(g * 1e5):d}m.csv'), index=False)
    return R


if __name__ == '__main__':
    ds, g = sys.argv[1], float(sys.argv[2])
    R = run_geolife(g) if ds == 'geolife' else run_taxi(g)
    pd.set_option('display.width', 250)
    num = R.select_dtypes('number')
    print(num.median().round(5).to_string() if len(R) > 1 else num.T.round(5).to_string())
