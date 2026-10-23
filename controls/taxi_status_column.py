"""P3-8  What is the 7th column of the Shanghai taxi records? (descriptive check, input for the user's decision)

Each raw record is: vehicle id, time, lon, lat, speed field, direction field, column 7 (values 0-3). The paper says the
data carry no occupancy flag. If column 7 is the occupancy state, its runs should look like passenger trips: the
'occupied' value lasts minutes to tens of minutes, the vehicle moves during it and travels a few kilometres, and
switches happen when the vehicle is (nearly) stopped. This script reports, per value of column 7:
  share of records, vehicles (and vehicles that carry only this value), number of runs, quartiles of run duration
  and path length (km) over runs of at least 3 records (shorter runs are mostly one-record flicker), share of
  position-derived speed > 5 km/h during the run, and the median position-derived speed around a switch into the value.
Speed is computed from consecutive positions. (An earlier note here called the recorded speed field unreliable; that
check had taken the speed and direction fields the wrong way round. Column 5 is the speed in km/h and column 6 the heading
in units of 2 degrees; both agree with the positions, see heading_state.py, taxi_device. Column 7 is confirmed as the
occupancy state by the data description of the release, 2026-10-22.)
Deterministic, no random numbers; per-file summaries are cached in controls/work/taxi_status/<file>.pkl (atomic),
so an interrupted run continues where it stopped.

    python controls/taxi_status_column.py   -> results/taxi_status_column.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os
import numpy as np, pandas as pd

WORK = P.CONTROLS_WORK / 'taxi_status'


def one_file(f):
    out = WORK / (os.path.basename(f) + '.pkl')
    if out.exists():
        return pd.read_pickle(out)
    d = pd.read_csv(f, header=None, names=['id', 't', 'lon', 'lat', 'v', 'h', 's'], skipinitialspace=True)
    d['t'] = pd.to_datetime(d.t, errors='coerce'); d = d.dropna(subset=['t', 'lon', 'lat', 's'])
    d = d[(d.lon > 120) & (d.lon < 123) & (d.lat > 30) & (d.lat < 32.5)].sort_values('t').drop_duplicates('t')
    rows = []
    if len(d) >= 3:
        ts = (d.t - pd.Timestamp('2007-02-20')).dt.total_seconds().to_numpy()
        x = d.lon.to_numpy() * 111.32 * np.cos(np.radians(31.2)); y = d.lat.to_numpy() * 110.57; s = d.s.to_numpy().astype(int)
        dt = np.diff(ts); dd = np.hypot(np.diff(x), np.diff(y))
        sp = np.where(dt > 0, dd / np.maximum(dt, 1) * 3600, np.nan)                       # km/h on each interval
        cut = np.r_[0, np.where((np.diff(s) != 0) | (dt > 600))[0] + 1, len(s)]             # runs of constant value, split at gaps
        for a, b in zip(cut[:-1], cut[1:]):
            iv = sp[a:b - 1] if b - a > 1 else np.array([])
            switch = a > 0 and s[a] != s[a - 1] and dt[a - 1] <= 600
            win = np.abs(ts - ts[a]) <= 30
            rows.append(dict(file=os.path.basename(f), value=int(s[a]), n=b - a, dur_s=ts[b - 1] - ts[a],
                             disp_km=float(np.hypot(x[b - 1] - x[a], y[b - 1] - y[a])), path_km=float(dd[a:b - 1].sum()),
                             moving=float(np.nanmean(iv > 5)) if len(iv) else np.nan,
                             entered_by_switch=bool(switch), speed_at_switch=float(np.nanmedian(sp[max(a - 1, 0):a + 1])) if switch else np.nan))
    R = pd.DataFrame(rows)
    WORK.mkdir(parents=True, exist_ok=True); tmp = str(out) + '.tmp'; R.to_pickle(tmp); os.replace(tmp, out)
    return R


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), 'Taxi_*')))
    R = pd.concat([one_file(f) for f in files], ignore_index=True)
    q = lambda c, p: (lambda x: float(np.nanquantile(x, p)) if len(x) else np.nan)
    rows = []
    combo = R.groupby('file').value.apply(lambda s: tuple(sorted(set(s))))
    for v, g in R.groupby('value'):
        sw = g[g.entered_by_switch]; g3 = g[g.n >= 3]                               # runs of >= 3 records: no single-record flicker
        rows.append(dict(value=v, share_records=g.n.sum() / R.n.sum(), vehicles=g.file.nunique(),
                         vehicles_only_this_value=int((combo == (v,)).sum()), vehicles_with_0_and_1=int(combo.apply(lambda c: 0 in c and 1 in c and v in c).sum()),
                         runs=len(g), runs_3plus=len(g3),
                         dur_min_q25=q(0, .25)(g3.dur_s) / 60, dur_min_median=q(0, .5)(g3.dur_s) / 60, dur_min_q75=q(0, .75)(g3.dur_s) / 60,
                         path_km_q25=q(0, .25)(g3.path_km), path_km_median=q(0, .5)(g3.path_km), path_km_q75=q(0, .75)(g3.path_km),
                         disp_km_median=q(0, .5)(g3.disp_km), moving_share_median=q(0, .5)(g3.moving.dropna()),
                         runs_entered_by_switch=len(sw), speed_at_switch_median_kmh=q(0, .5)(sw.speed_at_switch.dropna())))
    S = pd.DataFrame(rows)
    out = P.out(P.CONTROLS_RESULTS / 'taxi_status_column.csv'); tmp = str(out) + '.tmp'; S.to_csv(tmp, index=False); os.replace(tmp, out)
    pd.set_option('display.width', 250); print(S.round(3).to_string(index=False))
