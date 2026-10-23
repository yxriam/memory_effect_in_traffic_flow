"""P1-2  Is the origin information 'purposive', or does the origin only stand in for the direction of arrival
(or for the vehicle)?

On steps at least 21 cells into a trip (so that S_{t-20} exists), with L3 = (S_{t-2}, S_{t-1}, S_t):
  I_m      = I(S_{t+1}; S_{t-m} | L3),            m = 5, 10, 20   (remote direction)
  I_O|3    = I(S_{t+1}; O | L3)                                    (the paper's origin quantity, on these steps)
  I_O|3,10 = I(S_{t+1}; O | L3, S_{t-10})                           (origin beyond the direction 10 cells back)
taxi only:
  I_O|3,veh,per    = I(S_{t+1}; O | L3, vehicle, period)            (origin permuted within vehicle and time of day)
  I_O|3,10,veh,per = I(S_{t+1}; O | L3, S_{t-10}, vehicle, period)
Conditioning on more variables splits the data and costs power by itself, so each added control is repeated with
that variable shuffled within the L3 strata (same number of strata, no information): I_O|3,10shuf, and for taxi
I_O|3,veh,per shuf. Every quantity is bias-corrected with a stratified permutation null (the tested variable permuted within the
conditioning strata), 200 permutations for the pooled taxi sample and 100 per GeoLife user. GeoLife: 100 m cells, origin on a 1 km grid, per user (as origin_memory.py,
which has no timestamps, so the period control is taxi-only). Taxi: the pooled 100 vehicles of taxi_origin.py,
200 m cells, origin on a 2 km grid, trips split at dwells >= 5 min or gaps > 15 min; period = start hour of the trip
in 6 h blocks. Bits.

    python controls/origin_vs_direction.py   -> results/origin_direction_geolife.csv, origin_direction_taxi.csv
    python controls/origin_vs_direction.py taxi   -> only the taxi part
Fix 2026-10-22: the taxi time stamps were in units of 1000 s under pandas 3 (see taxi_origin.py), so every vehicle was one
trip of the whole day; the conversion now gives seconds. The GeoLife part has no time stamps and is not affected.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
import geolife_robust as gr
import origin_memory as om

NPERM_TAXI, NPERM_GEOLIFE = 200, 100
K0 = 21
G_TAXI = 0.002


def cmi(y, x, cond, rng, n_perm):
    K = int(y.max()) + 1; Kx = int(x.max()) + 1
    hc = gr.cond_H(cond, y, K); hf = gr.cond_H(gr.pair_id(cond, x, Kx), y, K)
    real = hc - hf
    order = np.argsort(cond, kind='stable'); null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = hc - gr.cond_H(gr.pair_id(cond, gr.strat_shuffle(x, cond, rng, order), Kx), y, K)
    return real - null.mean(), float((1 + np.sum(null >= real)) / (n_perm + 1))


def steps(trips):
    """rows: S_{t-20}, S_{t-10}, S_{t-5}, S_{t-2}, S_{t-1}, S_t, S_{t+1}, origin, extra..."""
    rows = []
    for tr in trips:
        cells, origin = tr[0], tr[1]; extra = tr[2:]
        for t in range(K0, len(cells) - 1):
            rows.append((cells[t - 20], cells[t - 10], cells[t - 5], cells[t - 2], cells[t - 1], cells[t], cells[t + 1], origin) + tuple(extra))
    return np.array(rows, dtype=np.int64) if rows else None


def quantities(X, seed, taxi=False):
    rng = np.random.default_rng(seed); n_perm = NPERM_TAXI if taxi else NPERM_GEOLIFE
    c = [np.unique(X[:, i], return_inverse=True)[1].astype(np.int64) for i in range(X.shape[1])]
    m20, m10, m5, a, b, s, y, o = c[:8]
    L3 = om.enc_pairs(a, b, s)
    r = {}
    for name, x in (('I_5', m5), ('I_10', m10), ('I_20', m20)):
        r[name], r['p_' + name] = cmi(y, x, L3, rng, n_perm)
    r['I_O|3'], r['p_I_O|3'] = cmi(y, o, L3, rng, n_perm)
    r['I_O|3,10'], r['p_I_O|3,10'] = cmi(y, o, om.enc_pairs(L3, m10), rng, n_perm)
    # power control: S_{t-10} shuffled within L3 strata has the same number of strata but no information
    oL3 = np.argsort(L3, kind='stable')
    m10s = gr.strat_shuffle(m10, L3, rng, oL3)
    r['I_O|3,10shuf'], r['p_I_O|3,10shuf'] = cmi(y, o, om.enc_pairs(L3, m10s), rng, n_perm)
    if taxi:
        veh, per = c[8], c[9]
        r['I_O|3,veh,per'], r['p_I_O|3,veh,per'] = cmi(y, o, om.enc_pairs(L3, veh, per), rng, n_perm)
        r['I_O|3,10,veh,per'], r['p_I_O|3,10,veh,per'] = cmi(y, o, om.enc_pairs(L3, m10, veh, per), rng, n_perm)
        vp = gr.strat_shuffle(om.enc_pairs(veh, per), L3, rng, oL3)
        r['I_O|3,veh,per shuf'], r['p_I_O|3,veh,per shuf'] = cmi(y, o, om.enc_pairs(L3, vp), rng, n_perm)
    r['n_steps'] = len(X)
    return r


def geolife_user(path):
    d = np.load(path, allow_pickle=True); trips = []
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        lat = np.concatenate([np.asarray(tla, float), np.asarray(sla, float)]); lon = np.concatenate([np.asarray(tlo, float), np.asarray(slo, float)])
        m = np.isfinite(lat) & np.isfinite(lon); lat, lon = lat[m], lon[m]
        if len(lat) < 3: continue
        cells = gr.to_states(lat, lon, 0.001)
        if len(cells) < K0 + 2: continue
        trips.append((cells, int(np.floor(lat[0] / 0.01) * 100000 + np.floor(lon[0] / 0.01))))
    X = steps(trips)
    if X is None or len(X) < 300: return None
    u = os.path.basename(path)[:3]
    return dict(user=u, **quantities(X, int(u)))


def taxi_trips(f, vid):
    d = pd.read_csv(f, header=None, usecols=[1, 2, 3]); d[1] = pd.to_datetime(d[1])
    d = d.sort_values(1); hour = d[1].dt.hour.to_numpy()
    t = (d[1] - pd.Timestamp('1970-01-01')).dt.total_seconds().to_numpy().astype(np.int64)  # seconds whatever the datetime unit (10-22 fix; see taxi_origin.py)
    lon, lat = d[2].to_numpy(), d[3].to_numpy()
    ok = (lon > 120.8) & (lon < 122.2) & (lat > 30.6) & (lat < 31.9); t, lon, lat, hour = t[ok], lon[ok], lat[ok], hour[ok]
    if len(t) < 30: return []
    G = G_TAXI
    cell = (np.floor(lat / G) * 100000 + np.floor(lon / G)).astype(np.int64)
    coarse = (np.floor(lat / (10 * G)) * 100000 + np.floor(lon / (10 * G))).astype(np.int64)
    trips, cur, org, per = [], [], None, None; i = 0; n = len(t)
    while i < n:                                     # identical to taxi_origin.py, plus the start hour of each trip
        j = i
        while j + 1 < n and cell[j + 1] == cell[i] and t[j + 1] - t[j] <= 900: j += 1
        if not cur: org = coarse[i]; per = hour[i] // 6
        cur.append(cell[i])
        brk = (t[j] - t[i] >= 300) or (j + 1 < n and t[j + 1] - t[j] > 900)
        if brk:
            if len(cur) >= 6: trips.append((np.array(cur), int(org), vid, int(per)))
            cur = [cell[i]] if t[j] - t[i] >= 300 else []; org = coarse[i]; per = hour[i] // 6
        i = j + 1
    if len(cur) >= 6: trips.append((np.array(cur), int(org), vid, int(per)))
    return trips


if __name__ == '__main__':
    parts = sys.argv[1:] or ['geolife', 'taxi']          # e.g. 'python controls/origin_vs_direction.py taxi' reruns the taxi part only
    pd.set_option('display.width', 200)
    if 'geolife' in parts:
        gl = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz')))
        with Pool(2) as pool:
            G = pd.DataFrame([r for r in pool.imap_unordered(geolife_user, gl) if r]).sort_values('user')
        G.to_csv(P.out(P.CONTROLS_RESULTS / 'origin_direction_geolife.csv'), index=False)
        cols = [c for c in G.columns if c.startswith('I_')]   # p-values are in 'p_' + name
        print('GeoLife users', len(G)); print(pd.DataFrame({'median': G[cols].median(), 'share_p<.05': [(G['p_' + c] < .05).mean() for c in cols]}).round(4))
    if 'taxi' not in parts:
        raise SystemExit
    pick = json.load(open(P.CONTROLS / 'taxi_pick.json'))
    trips = []; used = 0
    for f in pick:
        tr = taxi_trips(os.path.join(str(P.TAXI_RAW), f), used)
        if tr: trips += tr; used += 1                # = taxi_origin.py: the first 100 usable files of taxi_pick.json
        if used >= 100: break
    X = steps(trips)
    r = quantities(X, 1, taxi=True); r['vehicles'] = used
    T = pd.DataFrame([r]); T.to_csv(P.out(P.CONTROLS_RESULTS / 'origin_direction_taxi.csv'), index=False)
    print(T.T.round(5))
