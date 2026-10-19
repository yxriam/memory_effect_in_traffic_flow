# NOTE: only the GeoLife part of this script is used in the paper; the taxi test is taxi_origin.py.
"""Purposive (origin) memory: given the last two (or three) cells, does the coarse trip ORIGIN still predict the next cell?
I(S_{t+1}; O | S_{t-1}, S_t) and I(S_{t+1}; O | S_{t-2}, S_{t-1}, S_t), bias-corrected by permuting O within the
conditioning strata; compared with the third-order residual I(S_{t+1}; S_{t-2} | S_{t-1}, S_t) on the same steps.
Only steps at least K cells after the trip start are used, so the origin is not trivially local.
GeoLife: 100 m cells, trips = .plt files, origin = first fix on a 1 km grid.
Taxi (302 sampled vehicles): per-trajectory 20x20 grid (as in the main analysis), trips separated by dwells of
>= 3 consecutive fixes (~3 min) in one cell, origin = trip start on the 5x5 coarse grid."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, sys, zlib, numpy as np, pandas as pd
from multiprocessing import Pool
import geolife_robust as gr
NPERM = 30

def enc_pairs(*cols):
    key = np.zeros(len(cols[0]), np.int64)
    for c in cols:
        u, c2 = np.unique(c, return_inverse=True)
        key = np.unique(key * (c2.max() + 1) + c2, return_inverse=True)[1].astype(np.int64)
    return key

def cmi(target, hist, cond, rng):
    K = int(target.max()) + 1
    hc = gr.cond_H(cond, target, K); hf = gr.cond_H(enc_pairs(cond, hist), target, K)
    real = hc - hf
    order = np.argsort(cond, kind='stable'); null = []
    for _ in range(NPERM):
        hp = gr.strat_shuffle(hist, cond, rng, order)
        null.append(hc - gr.cond_H(enc_pairs(cond, hp), target, K))
    null = np.array(null)
    return real - null.mean(), (1 + np.sum(null >= real)) / (NPERM + 1), hc

def steps_from_trips(trips, K0):
    rows = []
    for cells, origin in trips:
        n = len(cells)
        for t in range(max(3, K0), n - 1):
            rows.append((cells[t - 2], cells[t - 1], cells[t], cells[t + 1], origin))
    return np.array(rows, dtype=np.int64) if rows else None

def analyse(unit, trips, K0, seed):
    X = steps_from_trips(trips, K0)
    if X is None or len(X) < 300: return None
    rng = np.random.default_rng(seed)
    a, b, c, y, o = [np.unique(X[:, i], return_inverse=True)[1].astype(np.int64) for i in range(5)]
    cond2 = enc_pairs(b, c); cond3 = enc_pairs(a, b, c)
    i3, p3, h2 = cmi(y, a, cond2, rng)
    io, po, _ = cmi(y, o, cond2, rng)
    io3, po3, h3 = cmi(y, o, cond3, rng)
    return dict(unit=unit, n_steps=len(X), n_trips=len(trips), H_given2=h2, cmi3_excess=i3, p3=p3,
                origin_excess=io, p_origin=po, origin_beyond3_excess=io3, p_origin3=po3)

def geolife_user(path):
    full, _, _ = gr.load_user(path, 0.001, False)
    d = np.load(path, allow_pickle=True); trips = []
    j = 0
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        lat = np.concatenate([np.asarray(tla, float), np.asarray(sla, float)]); lon = np.concatenate([np.asarray(tlo, float), np.asarray(slo, float)])
        m = np.isfinite(lat) & np.isfinite(lon); lat, lon = lat[m], lon[m]
        if len(lat) < 3: continue
        cells = gr.to_states(lat, lon, 0.001)
        if len(cells) < 15: continue
        origin = int(np.floor(lat[0] / 0.01) * 100000 + np.floor(lon[0] / 0.01))
        trips.append((cells, origin))
    return analyse('gl' + os.path.basename(path)[:3], trips, 10, int(os.path.basename(path)[:3]))

def taxi_vehicle(path):
    try:
        d = pd.read_csv(path, header=None, usecols=[1, 2, 3])
    except Exception:
        return None
    lon, lat = d[2].to_numpy(float), d[3].to_numpy(float)
    ok = (lon > 120) & (lon < 123) & (lat > 30) & (lat < 32.5); lon, lat = lon[ok], lat[ok]
    if len(lon) < 50: return None
    g = 20
    def idx(x):
        lo, hi = np.min(x), np.max(x)
        return np.clip(((x - lo) / (hi - lo + 1e-12) * g).astype(int), 0, g - 1)
    ix, iy = idx(lon), idx(lat); cell = ix * g + iy; coarse = (ix // 4) * 5 + (iy // 4)
    # run-length encode; a run of >=3 fixes in one cell is a dwell -> trip boundary
    change = np.r_[True, cell[1:] != cell[:-1]]; starts = np.where(change)[0]
    runs = np.diff(np.r_[starts, len(cell)])
    trips, cur, origin = [], [], None
    for s, r in zip(starts, runs):
        if not cur: origin = coarse[s]
        cur.append(cell[s])
        if r >= 3:
            if len(cur) >= 6: trips.append((np.array(cur), int(origin)))
            cur = [cell[s]]; origin = coarse[s]
    if len(cur) >= 6: trips.append((np.array(cur), int(origin)))
    return analyse('tx' + os.path.basename(path), trips, 3, zlib.crc32(os.path.basename(path).encode()))

if __name__ == '__main__':
    gl = sorted(glob.glob(os.path.join(gr.CACHE, '*.npz')))
    tx = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), '*')))
    with Pool(2) as p:
        G = [r for r in p.imap_unordered(geolife_user, gl) if r]
        T = [r for r in p.imap_unordered(taxi_vehicle, tx) if r]
    G = pd.DataFrame(G); G['dataset'] = 'GeoLife'; T = pd.DataFrame(T); T['dataset'] = 'Taxi'
    G.to_csv(P.out(P.CONTROLS_RESULTS / 'origin_memory_geolife.csv'), index=False)
    T.to_csv(P.out(P.CONTROLS_RESULTS / 'origin_memory_taxi.csv'), index=False)  # per-vehicle; the paper uses the pooled taxi_origin.py
    R = pd.concat([G, T])
    for name, s in R.groupby('dataset'):
        print(name, 'units', len(s), 'median steps', s.n_steps.median())
        for c, p in [('cmi3_excess', 'p3'), ('origin_excess', 'p_origin'), ('origin_beyond3_excess', 'p_origin3')]:
            print(f'  {c:24s} median {s[c].median():.4f}  mean {s[c].mean():.4f}  rel {np.median(s[c] / s.H_given2):.3f}  sig {(s[p] < .05).mean():.2f}')
