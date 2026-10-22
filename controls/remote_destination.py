"""P3-2  Remote cells: does their information persist to long lags, and is it the destination?

P1-2 found that, given the last three cells L3 = (S_{t-2}, S_{t-1}, S_t), the cell 5, 10 or 20 steps back adds about
0.0024 bits in GeoLife without decay over that range: a trip-level hidden variable rather than a decaying memory.
Here, per GeoLife user (cleaned cache, trips = train + test part, 100 m cells as in origin_vs_direction.py):
  I_m      = I(S_{t+1}; S_{t-m} | L3)
  I_m|D    = I(S_{t+1}; S_{t-m} | L3, D)            D = destination (last fix of the trip) on a 1 km grid
  I_m|Dsh  = the same with D shuffled within the L3 strata (same number of strata, no information): power control
  I_D      = I(S_{t+1}; D | L3)                     the destination's own information
on two step sets: 'short' = steps at least 21 cells into a trip, m = 5, 10, 20 (as P1-2);
'long' = steps at least 101 cells into a trip, m = 5, 10, 20, 50, 100 (same steps for every m, so the m-profile is
not confounded with trip length). Every quantity is bias-corrected with the stratified permutation null (the tested
variable permuted within the conditioning strata), 50 permutations, seed crc32('remote|<user>'). Bits.
Criterion (to-do list P3-2): a plateau that drops when D is given, more than in the shuffled control -> the trip-level
component is mainly the destination; a plateau that D does not explain -> another trip-level variable; decay to
zero with m -> memory of finite length.
Each user's result is written atomically to controls/work/remote_destination/<user>.json and skipped on a restart.

    python controls/remote_destination.py   -> results/remote_destination_geolife.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, json, os, time, zlib
import numpy as np, pandas as pd
import geolife_robust as gr
import origin_memory as om
import origin_vs_direction as ov

NPERM = 50
SETS = {'short': (21, (5, 10, 20)), 'long': (101, (5, 10, 20, 50, 100))}
WORK = P.CONTROLS_WORK / 'remote_destination'


def trips_of(path):
    d = np.load(path, allow_pickle=True); out = []
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        lat = np.concatenate([np.asarray(tla, float), np.asarray(sla, float)]); lon = np.concatenate([np.asarray(tlo, float), np.asarray(slo, float)])
        m = np.isfinite(lat) & np.isfinite(lon); lat, lon = lat[m], lon[m]
        if len(lat) < 3: continue
        cells = gr.to_states(lat, lon, 0.001)
        dest = int(np.floor(lat[-1] / 0.01) * 100000 + np.floor(lon[-1] / 0.01))
        out.append((cells, dest))
    return out


def unit(trips, k0, lags, seed):
    rows = [tuple(c[t - m] for m in lags) + (c[t - 2], c[t - 1], c[t], c[t + 1], dest)
            for c, dest in trips for t in range(k0, len(c) - 1)]
    if len(rows) < 300:
        return None
    X = np.array(rows, dtype=np.int64)
    cols = [np.unique(X[:, i], return_inverse=True)[1].astype(np.int64) for i in range(X.shape[1])]
    rem, (a, b, s, y, D) = cols[:len(lags)], cols[len(lags):]
    L3 = om.enc_pairs(a, b, s); rng = np.random.default_rng(seed)
    Dsh = gr.strat_shuffle(D, L3, rng, np.argsort(L3, kind='stable'))
    r = dict(n_steps=len(X))
    r['I_D'], r['p_I_D'] = ov.cmi(y, D, L3, rng, NPERM)
    for m, x in zip(lags, rem):
        r[f'I_{m}'], r[f'p_I_{m}'] = ov.cmi(y, x, L3, rng, NPERM)
        r[f'I_{m}|D'], r[f'p_I_{m}|D'] = ov.cmi(y, x, om.enc_pairs(L3, D), rng, NPERM)
        r[f'I_{m}|Dsh'], r[f'p_I_{m}|Dsh'] = ov.cmi(y, x, om.enc_pairs(L3, Dsh), rng, NPERM)
    return r


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz'))); t0 = time.time()
    for i, f in enumerate(files):
        user = os.path.basename(f)[:3]; out = WORK / f'{user}.json'
        if out.exists():
            continue
        tr = trips_of(f); res = dict(user=user)
        for name, (k0, lags) in SETS.items():
            r = unit(tr, k0, lags, zlib.crc32(f'remote|{name}|{user}'.encode()))
            if r: res.update({f'{name}:{k}': v for k, v in r.items()})
        WORK.mkdir(parents=True, exist_ok=True); tmp = str(out) + '.tmp'; json.dump(res, open(tmp, 'w')); os.replace(tmp, out)
        print(f'{i + 1}/{len(files)} user {user} [{time.time() - t0:.0f}s]', flush=True)
    R = pd.DataFrame([json.load(open(WORK / f'{os.path.basename(f)[:3]}.json')) for f in files])
    R.to_csv(P.out(P.CONTROLS_RESULTS / 'remote_destination_geolife.csv'), index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 100)
    cols = [c for c in R.columns if ':I_' in c]
    print(pd.DataFrame({'users': R[cols].notna().sum(), 'median': R[cols].median(),
                        'share_p<.05': [(R[c.replace(':I_', ':p_I_')] < .05).sum() / R[c].notna().sum() for c in cols]}).round(5).to_string())
