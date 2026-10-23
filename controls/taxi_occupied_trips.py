"""Taxi origin and destination tests on real passenger trips (2026-10-22).

The 7th field of the Shanghai taxi records is the occupancy state (0 vacant, 1 occupied; 2 and 3 on vehicles that do not
move all day; see taxi_status_column.py and the data description of the release). The trip-level test of the paper had
delimited trips by dwells of at least 5 min in one cell or gaps longer than 15 min (taxi_origin.py,
origin_vs_direction.py), because the field had not been identified. Here trips are the occupied runs themselves.

Trips. Per vehicle file: records inside lon 120.8-122.2, lat 30.6-31.9 (as taxi_origin.py), sorted by time, duplicate
times dropped. A run is a maximal sequence of records with the same state and no gap longer than 300 s. A run counts as a
complete trip only if the record before it and the record after it exist, have another state and are at most 300 s
away (so that pick-up and drop-off are observed). Cells: global 0.002 deg grid (about 200 m), run-length compressed,
trips with at least 6 cells. Origin O = cell of the first record, destination D = cell of the last record, both on the
ten times coarser grid (about 2 km). Period = start hour // 6. Vehicle = index of the file in the sorted list.

Samples
  dwell_pick100  the dwell-delimited trips of taxi_origin.py on its 100 vehicles (first 100 usable files of
                 taxi_pick.json); D = coarse cell of the last cell of the trip. Reproduces the paper's numbers.
  occ_pick100    occupied trips of the same 100 vehicles
  occ_all        occupied trips of all vehicles
  vac_all        vacant runs of all vehicles (contrast: cruising without a passenger; O = previous drop-off,
                 D = next pick-up)

Quantities (bits, bias-corrected with the stratified permutation null: the tested variable permuted within the strata of
the conditioning variables; L2 = (S_{t-1}, S_t), L3 = (S_{t-2}, S_{t-1}, S_t)).
On steps at least 3 cells after the trip start (as taxi_origin.py):
  cmi2 = I(S+; S_{t-1} | S_t)     cmi3 = I(S+; S_{t-2} | L2)     IO = I(S+; O | L2)     IO3 = I(S+; O | L3)
  ID3 = I(S+; D | L3)                                     the destination's own information
  IO3_D = I(S+; O | L3, D),  cmi3_D = I(S+; S_{t-2} | L2, D)   memory beyond the destination
  IO3_Dsh, cmi3_Dsh: the same with D shuffled within the strata of L3 (L2): same number of strata, no information
  IO3_VP = I(S+; O | L3, vehicle, period), IO3_VPsh: (vehicle, period) shuffled within L3 (as origin_vs_direction.py)
On steps at least 11 cells after the trip start (remote cells, as remote_destination.py):
  I5, I10 = I(S+; S_{t-m} | L3);  I5_D, I10_D given D as well;  I5_Dsh, I10_Dsh with D shuffled within L3;  ID3_r = ID3
On symmetric steps (current cell at least 3 cells after the start, next cell at least 3 cells before the last cell;
added after the 10-22 review): IO3_sym, ID3_sym (unit <sample>_sym.json, seed crc32('occupied|<sample>|sym')).
Stability: 20 half-samples of vehicles (without replacement, as taxi_origin.py) for cmi2, cmi3, IO, IO3, ID3, IO3_D,
cmi3_D; central 95 % range. Permutations: 50 for the full samples, 20 for the half-samples.
Seeds: zlib.crc32('occupied|<sample>|main') and crc32('occupied|<sample>|boot|<b>') (vehicle draw and permutations).

Resumable: the trips of each vehicle file are cached in controls/work/taxi_occupied/vehicles/<file>.pkl, the full-sample
result of each sample in work/taxi_occupied/<sample>_main.json and each half-sample in <sample>_boot<b>.json (all written
atomically, skipped on a restart). Re-running the command continues where it stopped; the result does not depend on
where it was interrupted.

    python controls/taxi_occupied_trips.py          -> results/taxi_occupied_trips.csv (one row per sample)
    python controls/taxi_occupied_trips.py counts   -> results/taxi_occupied_trip_counts.csv (runs per criterion)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, json, os, platform, sys, time, zlib
import numpy as np, pandas as pd
import geolife_robust as gr
import origin_memory as om
import origin_vs_direction as ov

G = 0.002
GAP = 300
MIN_CELLS = 6
NPERM_MAIN, NPERM_BOOT, NBOOT = 50, 20, 20
BBOX = (120.8, 122.2, 30.6, 31.9)
WORK = P.CONTROLS_WORK / 'taxi_occupied'
BOOT_KEYS = ('cmi2', 'cmi3', 'IO', 'IO3', 'ID3', 'IO3_D', 'cmi3_D')


def atomic_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = str(path) + '.tmp'; pd.to_pickle(obj, tmp); os.replace(tmp, path)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = str(path) + '.tmp'
    json.dump(obj, open(tmp, 'w')); os.replace(tmp, path)


def coarse_of(cell):
    """Coarse (10 G) cell of a fine cell key floor(lat/G)*100000 + floor(lon/G) (all coordinates positive)."""
    return (cell // 100000 // 10) * 100000 + (cell % 100000) // 10


def status_trips(f):
    """Complete runs of state 0 and 1 of one vehicle file: list of (state, cells, O, D, period)."""
    out = WORK / 'vehicles' / (os.path.basename(f) + '.pkl')
    if out.exists():
        return pd.read_pickle(out)
    d = pd.read_csv(f, header=None, names=['id', 't', 'lon', 'lat', 'v', 'h', 's'], skipinitialspace=True)
    d['t'] = pd.to_datetime(d.t, errors='coerce'); d = d.dropna(subset=['t', 'lon', 'lat', 's'])
    d = d[(d.lon > BBOX[0]) & (d.lon < BBOX[1]) & (d.lat > BBOX[2]) & (d.lat < BBOX[3])].sort_values('t').drop_duplicates('t')
    trips = []
    if len(d) >= MIN_CELLS + 2:
        ts = (d.t - pd.Timestamp('2007-02-20')).dt.total_seconds().to_numpy(); hour = d.t.dt.hour.to_numpy()
        s = d.s.to_numpy().astype(int); lat, lon = d.lat.to_numpy(), d.lon.to_numpy()
        cell = (np.floor(lat / G) * 100000 + np.floor(lon / G)).astype(np.int64)
        dt = np.diff(ts)
        cut = np.r_[0, np.where((np.diff(s) != 0) | (dt > GAP))[0] + 1, len(s)]
        for a, b in zip(cut[:-1], cut[1:]):
            if s[a] not in (0, 1) or a == 0 or b == len(s):
                continue
            if not (s[a - 1] != s[a] and dt[a - 1] <= GAP and s[b] != s[a] and dt[b - 1] <= GAP):
                continue                                                  # pick-up or drop-off not observed
            c = cell[a:b]; c = c[np.r_[True, c[1:] != c[:-1]]]
            if len(c) >= MIN_CELLS:
                trips.append((int(s[a]), c, int(coarse_of(cell[a])), int(coarse_of(cell[b - 1])), int(hour[a] // 6)))
    atomic_pickle(out, trips)
    return trips


def steps(trips, k0, lags=()):
    """Rows (S_{t-m} for m in lags..., S_{t-2}, S_{t-1}, S_t, S_{t+1}, O, D, vehicle, period) for t >= k0."""
    rows = []
    for cells, o, dd, veh, per in trips:
        for t in range(k0, len(cells) - 1):
            rows.append(tuple(cells[t - m] for m in lags) + (cells[t - 2], cells[t - 1], cells[t], cells[t + 1], o, dd, veh, per))
    return np.array(rows, dtype=np.int64) if rows else None


def symmetric(trips, seed, n_perm):
    """Origin and destination on the same, symmetric steps: the current cell at least 3 cells after the trip start and
    the next cell at least 3 cells before the last cell (added after the 10-22 review: on the steps of quantities(), which
    are trimmed only at the start, the last cells lie next to the destination)."""
    rng = np.random.default_rng(seed); rows = []
    for cells, o, dd, veh, per in trips:
        for t in range(3, len(cells) - 4):
            rows.append((cells[t - 2], cells[t - 1], cells[t], cells[t + 1], o, dd))
    X = np.array(rows, dtype=np.int64); a, b, s, y, o, dd = enc(X); L3 = om.enc_pairs(a, b, s); r = dict(n_steps_sym=len(X))
    for name, x in (('IO3_sym', o), ('ID3_sym', dd)):
        r[name], r['p_' + name], r['plugin_' + name] = cmi(y, x, L3, rng, n_perm)
    return r


def cmi(y, x, cond, rng, n_perm):
    """Plug-in I(Y; X | cond) minus the mean of its stratified permutation null; p-value; plug-in value. Bits."""
    K = int(y.max()) + 1; Kx = int(x.max()) + 1
    hc = gr.cond_H(cond, y, K); real = hc - gr.cond_H(gr.pair_id(cond, x, Kx), y, K)
    order = np.argsort(cond, kind='stable'); null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = hc - gr.cond_H(gr.pair_id(cond, gr.strat_shuffle(x, cond, rng, order), Kx), y, K)
    return float(real - null.mean()), float((1 + np.sum(null >= real)) / (n_perm + 1)), float(real)


def enc(X):
    return [np.unique(X[:, i], return_inverse=True)[1].astype(np.int64) for i in range(X.shape[1])]


def quantities(trips, seed, n_perm, full=True):
    rng = np.random.default_rng(seed); r = {}
    X = steps(trips, 3)
    a, b, s, y, o, dd, veh, per = enc(X)
    L2 = om.enc_pairs(b, s); L3 = om.enc_pairs(a, b, s)
    oL2, oL3 = np.argsort(L2, kind='stable'), np.argsort(L3, kind='stable')
    K = int(y.max()) + 1
    r.update(n_steps=len(X), n_trips=len(trips), n_vehicles=len({t[3] for t in trips}), H1=float(gr.cond_H(s, y, K)),
             H2=float(gr.cond_H(L2, y, K)), H3=float(gr.cond_H(L3, y, K)))

    def put(name, x, cond):
        r[name], r['p_' + name], r['plugin_' + name] = cmi(y, x, cond, rng, n_perm)

    put('cmi2', b, s); put('cmi3', a, L2); put('IO', o, L2); put('IO3', o, L3); put('ID3', dd, L3)
    put('IO3_D', o, om.enc_pairs(L3, dd)); put('cmi3_D', a, om.enc_pairs(L2, dd))
    if not full:
        return r
    put('IO3_Dsh', o, om.enc_pairs(L3, gr.strat_shuffle(dd, L3, rng, oL3)))
    put('cmi3_Dsh', a, om.enc_pairs(L2, gr.strat_shuffle(dd, L2, rng, oL2)))
    vp = om.enc_pairs(veh, per)
    put('IO3_VP', o, om.enc_pairs(L3, vp)); put('IO3_VPsh', o, om.enc_pairs(L3, gr.strat_shuffle(vp, L3, rng, oL3)))
    Xr = steps(trips, 11, lags=(10, 5))
    r['n_steps_remote'] = 0 if Xr is None else len(Xr)
    if Xr is not None and len(Xr) >= 2000:
        m10, m5, a, b, s, y, o, dd, veh, per = enc(Xr)
        L3 = om.enc_pairs(a, b, s); oL3 = np.argsort(L3, kind='stable'); Dsh = gr.strat_shuffle(dd, L3, rng, oL3)
        put('ID3_r', dd, L3)
        for m, x in ((5, m5), (10, m10)):
            put(f'I{m}', x, L3); put(f'I{m}_D', x, om.enc_pairs(L3, dd)); put(f'I{m}_Dsh', x, om.enc_pairs(L3, Dsh))
    return r


def samples():
    files = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), 'Taxi_*')))
    vid = {os.path.basename(f): i for i, f in enumerate(files)}
    t0 = time.time(); per_file = {}
    for i, f in enumerate(files):
        per_file[os.path.basename(f)] = status_trips(f)
        if (i + 1) % 500 == 0:
            print(f'trips of {i + 1}/{len(files)} files [{time.time() - t0:.0f}s]', flush=True)
    pick, dwell = [], []
    for f in json.load(open(P.CONTROLS / 'taxi_pick.json')):                 # = taxi_origin.py: first 100 usable files
        tr = ov.taxi_trips(os.path.join(str(P.TAXI_RAW), f), vid[f])
        if tr:
            pick.append(f); dwell += [(c, o, int(coarse_of(c[-1])), v, per) for c, o, v, per in tr]
        if len(pick) >= 100:
            break
    occ = lambda names, st: [(c, o, dd, vid[n], per) for n in names for (s_, c, o, dd, per) in per_file[n] if s_ == st]
    allf = sorted(per_file)
    return {'dwell_pick100': dwell, 'occ_pick100': occ(pick, 1), 'occ_all': occ(allf, 1), 'vac_all': occ(allf, 0)}, pick


def env():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, machine=platform.machine())


def run_counts():
    """How many runs of each state survive each criterion (descriptive; deterministic, cached per file in
    work/taxi_occupied/counts/<file>.json). -> results/taxi_occupied_trip_counts.csv"""
    rows = []
    for f in sorted(glob.glob(os.path.join(str(P.TAXI_RAW), 'Taxi_*'))):
        out = WORK / 'counts' / (os.path.basename(f) + '.json')
        if not out.exists():
            d = pd.read_csv(f, header=None, names=['id', 't', 'lon', 'lat', 'v', 'h', 's'], skipinitialspace=True)
            d['t'] = pd.to_datetime(d.t, errors='coerce'); d = d.dropna(subset=['t', 'lon', 'lat', 's'])
            d = d[(d.lon > BBOX[0]) & (d.lon < BBOX[1]) & (d.lat > BBOX[2]) & (d.lat < BBOX[3])].sort_values('t').drop_duplicates('t')
            c = {f'{k}_{st}': 0 for st in (0, 1) for k in ('runs', 'complete', 'complete_6cells')}
            c.update(dt_median_s=float('nan'))
            if len(d) >= 2:
                ts = (d.t - pd.Timestamp('2007-02-20')).dt.total_seconds().to_numpy(); s = d.s.to_numpy().astype(int)
                cell = (np.floor(d.lat.to_numpy() / G) * 100000 + np.floor(d.lon.to_numpy() / G)).astype(np.int64); dt = np.diff(ts)
                c['dt_median_s'] = float(np.median(dt))
                cut = np.r_[0, np.where((np.diff(s) != 0) | (dt > GAP))[0] + 1, len(s)]
                for a, b in zip(cut[:-1], cut[1:]):
                    if s[a] not in (0, 1):
                        continue
                    c[f'runs_{s[a]}'] += 1
                    if a == 0 or b == len(s) or not (s[a - 1] != s[a] and dt[a - 1] <= GAP and s[b] != s[a] and dt[b - 1] <= GAP):
                        continue
                    c[f'complete_{s[a]}'] += 1
                    x = cell[a:b]
                    c[f'complete_6cells_{s[a]}'] += int(np.sum(np.r_[True, x[1:] != x[:-1]]) >= MIN_CELLS)
            atomic_json(out, c)
        rows.append(dict(file=os.path.basename(f), **json.load(open(out))))
    C = pd.DataFrame(rows); tot = C.drop(columns=['file', 'dt_median_s']).sum()
    res = pd.DataFrame([dict(state=st, vehicles_with_runs=int((C[f'runs_{st}'] > 0).sum()), runs=int(tot[f'runs_{st}']),
                             complete=int(tot[f'complete_{st}']), complete_6cells=int(tot[f'complete_6cells_{st}']),
                             vehicles_with_trips=int((C[f'complete_6cells_{st}'] > 0).sum()),
                             dt_median_s_over_vehicles=float(C.loc[C[f'runs_{st}'] > 0, 'dt_median_s'].median())) for st in (0, 1)])
    res.to_csv(P.out(P.CONTROLS_RESULTS / 'taxi_occupied_trip_counts.csv'), index=False)
    print(res.to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'counts':
        run_counts(); raise SystemExit
    S, pick = samples()
    open(P.out(P.CONTROLS_RESULTS / 'taxi_occupied_pick100_vehicles.txt'), 'w').write('\n'.join(pick) + '\n')
    rows = []
    for name, trips in S.items():
        f_main = WORK / f'{name}_main.json'
        if not f_main.exists():
            t0 = time.time()
            atomic_json(f_main, dict(sample=name, **quantities(trips, zlib.crc32(f'occupied|{name}|main'.encode()), NPERM_MAIN),
                                     nperm=NPERM_MAIN, grid=G, gap_s=GAP, min_cells=MIN_CELLS, **env()))
            print(f'{name}: full sample done [{time.time() - t0:.0f}s]', flush=True)
        res = json.load(open(f_main))
        f_sym = WORK / f'{name}_sym.json'
        if not f_sym.exists():
            atomic_json(f_sym, symmetric(trips, zlib.crc32(f'occupied|{name}|sym'.encode()), NPERM_MAIN))
        res.update(json.load(open(f_sym)))
        veh = np.array(sorted({t[3] for t in trips})); B = []
        for bi in range(NBOOT):
            f_b = WORK / f'{name}_boot{bi}.json'
            if not f_b.exists():
                seed = zlib.crc32(f'occupied|{name}|boot|{bi}'.encode()); rng = np.random.default_rng(seed)
                keep = set(rng.choice(veh, len(veh) // 2, replace=False).tolist())
                atomic_json(f_b, quantities([t for t in trips if t[3] in keep], seed + 1, NPERM_BOOT, full=False))
            B.append(json.load(open(f_b)))
        B = pd.DataFrame(B)
        for k in BOOT_KEYS:
            res[k + '_lo'] = float(B[k].quantile(.025)); res[k + '_hi'] = float(B[k].quantile(.975))
        print(f'{name}: half-samples done', flush=True)
        rows.append(res)
    R = pd.DataFrame(rows)
    R.to_csv(P.out(P.CONTROLS_RESULTS / 'taxi_occupied_trips.csv'), index=False)
    pd.set_option('display.width', 250)
    show = ['n_trips', 'n_steps', 'cmi2', 'cmi3', 'IO', 'IO3', 'ID3', 'n_steps_sym', 'IO3_sym', 'ID3_sym', 'IO3_D', 'IO3_Dsh', 'cmi3_D', 'cmi3_Dsh', 'IO3_VP', 'IO3_VPsh',
            'n_steps_remote', 'I5', 'I5_D', 'I5_Dsh', 'I10', 'I10_D', 'I10_Dsh']
    print(R.set_index('sample')[[c for c in show if c in R]].T.round(5))
