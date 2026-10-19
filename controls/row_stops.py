"""Stop spacing vs inertia (decorrelation time) on clean (30 s split) GeoLife speed segments."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import pandas as pd, numpy as np
from scipy.stats import spearmanr
r = pd.read_csv(P.CONTROLS_RESULTS / 'row_persistence.csv'); r = r[(r.gap == 30) & r.tau_c.notna()]
m = r.groupby('mode').agg(n=('tau_c', 'size'), tau=('tau_c', 'median'), phi=('phi', 'median'),
                          speed=('mean_speed', 'median'), stops=('stops_per_km', 'median'), stops_h=('stops_per_h', 'median'))
m = m[m.n >= 20]
rho, p = spearmanr(m.tau, m.stops); print(m.round(3)); print('mode-level spearman tau vs stops/km', rho, p)
rho2, p2 = spearmanr(m.tau, m.speed); print('mode-level spearman tau vs speed', rho2, p2)
rows = [dict(mode='ALL(mode medians)', rho=rho, p=p, n=len(m))]
for mo, s in r[r['mode'].isin(m.index)].groupby('mode'):
    s = s.dropna(subset=['stops_per_km'])
    a = spearmanr(s.tau_c, s.stops_per_km); rows.append(dict(mode=mo, rho=a[0], p=a[1], n=len(s)))
R = pd.DataFrame(rows); print(R.round(4)); R.to_csv(P.out(P.CONTROLS_RESULTS / 'row_stops_spearman.csv'), index=False); m.to_csv(P.out(P.CONTROLS_RESULTS / 'row_mode_medians.csv'))
