"""Taxi origin memory, pooled over the staged vehicles on a global grid (vehicles share coordinates).
Trips: split at dwells >= 5 min in one cell or gaps > 15 min. Origin = trip start on a 10x coarser grid.
Steps >= K0 cells after trip start. Stability: 20 random half-samples of vehicles (no replacement; duplicating vehicles would bias plug-in CMI)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, numpy as np, pandas as pd, sys
import origin_memory as om
G = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
N_VEH = int(os.environ.get('N_VEH', 100))   # the paper pools 100 vehicles; taxi_pick.json lists 300 candidates in random order
rows = []; used = []
import json
PICK = [os.path.join(str(P.TAXI_RAW), x) for x in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taxi_pick.json')))]
for vid, f in enumerate([p for p in PICK if os.path.exists(p)]):
    d = pd.read_csv(f, header=None, usecols=[1, 2, 3]); d[1] = pd.to_datetime(d[1]).astype('int64') // 10**9
    d = d.sort_values(1); t, lon, lat = d[1].to_numpy(), d[2].to_numpy(), d[3].to_numpy()
    ok = (lon > 120.8) & (lon < 122.2) & (lat > 30.6) & (lat < 31.9); t, lon, lat = t[ok], lon[ok], lat[ok]
    if len(t) < 30: continue
    cell = (np.floor(lat / G) * 100000 + np.floor(lon / G)).astype(np.int64)
    coarse = (np.floor(lat / (10 * G)) * 100000 + np.floor(lon / (10 * G))).astype(np.int64)
    trips, cur, org = [], [], None; i = 0; n = len(t)
    while i < n:
        j = i
        while j + 1 < n and cell[j + 1] == cell[i] and t[j + 1] - t[j] <= 900: j += 1
        if not cur: org = coarse[i]
        cur.append(cell[i])
        brk = (t[j] - t[i] >= 300) or (j + 1 < n and t[j + 1] - t[j] > 900)
        if brk:
            if len(cur) >= 6: trips.append((np.array(cur), int(org)))
            cur = [cell[i]] if t[j] - t[i] >= 300 else []; org = coarse[i]
        i = j + 1
    if len(cur) >= 6: trips.append((np.array(cur), int(org)))
    X = om.steps_from_trips(trips, 3)
    if X is not None: rows.append(np.c_[X, np.full(len(X), vid)]); used.append(os.path.basename(f))
    if len(rows) >= N_VEH: break
X = np.vstack(rows); print('grid', G, 'steps', len(X), 'vehicles', len(rows))
def run(X, seed):
    rng = np.random.default_rng(seed)
    a, b, c, y, o = [np.unique(X[:, i], return_inverse=True)[1].astype(np.int64) for i in range(5)]
    c2 = om.enc_pairs(b, c); c3 = om.enc_pairs(a, b, c)
    i3, p3, h2 = om.cmi(y, a, c2, rng); io, po, _ = om.cmi(y, o, c2, rng); io3, po3, _ = om.cmi(y, o, c3, rng)
    # second order for reference
    i2, p2, h1 = om.cmi(y, b, c, rng)
    return dict(H1=h1, cmi2=i2, p2=p2, H2=h2, cmi3=i3, p3=p3, origin=io, p_origin=po, origin3=io3, p_origin3=po3)
res = run(X, 0); print({k: round(v, 4) for k, v in res.items()})
veh = np.unique(X[:, 5]); B = []
rng = np.random.default_rng(1)
for bi in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
    pick = rng.choice(veh, len(veh) // 2, replace=False); Xb = np.vstack([X[X[:, 5] == v] for v in pick])
    B.append(run(Xb, bi + 10))
B = pd.DataFrame(B)
out = pd.DataFrame([res]); out['grid'] = G; out['n_steps'] = len(X); out['n_veh'] = len(rows)
for k in ['cmi2', 'cmi3', 'origin', 'origin3']:
    out[k + '_lo'] = B[k].quantile(.025); out[k + '_hi'] = B[k].quantile(.975)
print(out.T)
out.to_csv(P.out(P.CONTROLS_RESULTS / f'taxi_origin_{G}.csv'), index=False)
open(P.out(P.CONTROLS_RESULTS / f'taxi_origin_{G}_vehicles.txt'), 'w').write('\n'.join(used) + '\n')  # which files were pooled
