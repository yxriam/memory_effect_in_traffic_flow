import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import pandas as pd, numpy as np
D = pd.read_csv(P.CONTROLS_RESULTS / 'row_persistence.csv')
D = D[~D['mode'].isin(['airplane', 'boat'])].copy()
grp = {'subway': 'exclusive', 'train': 'exclusive', 'car': 'shared', 'taxi': 'shared', 'bus': 'shared', 'walk': 'nonmotor', 'bike': 'nonmotor'}
D['grp'] = D['mode'].map(grp)
D['log_tau'] = np.log(D.tau_c.clip(lower=5)); D['z_phi'] = np.arctanh(D.phi.clip(-0.999, 0.999))
D['log_speed'] = np.log(D.mean_speed.clip(lower=0.1)); D['log_stops'] = np.log1p(D.stops_per_km.fillna(0))
D['bus'] = (D['mode'] == 'bus').astype(float)
def design(d, terms):
    X = [np.ones(len(d))]; names = ['const']
    for t in terms:
        if t == 'excl': X.append((d.grp == 'exclusive').astype(float)); names.append('exclusive ROW')
        elif t == 'nonmotor': X.append((d.grp == 'nonmotor').astype(float)); names.append('non-motorized')
        elif t == 'bus': X.append(d.bus.values); names.append('bus (vs car/taxi)')
        else: X.append(d[t].values.astype(float)); names.append(t)
    return np.column_stack(X), names
def fit(d, y, terms, B=1000, seed=0):
    d = d.dropna(subset=[y] + [t for t in terms if t in d.columns])
    X, names = design(d, terms); beta = np.linalg.lstsq(X, d[y].values, rcond=None)[0]
    users = np.sort(d.user.unique()); rng = np.random.default_rng(seed); bs = []   # sorted: independent of the row order of row_persistence.csv
    by = {u: np.where(d.user.values == u)[0] for u in users}
    for _ in range(B):
        idx = np.concatenate([by[u] for u in rng.choice(users, len(users))])
        bs.append(np.linalg.lstsq(X[idx], d[y].values[idx], rcond=None)[0])
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame({'coef': beta, 'ci_lo': lo, 'ci_hi': hi}, index=names).round(3), len(d)
out = []
for gap in (30.0, 1800.0):
    d = D[D.gap == gap]
    specs = [('M1 groups only', ['excl', 'nonmotor']),
             ('M2 + speed', ['excl', 'nonmotor', 'log_speed']),
             ('M3 + speed + stops/km', ['excl', 'nonmotor', 'log_speed', 'log_stops'])]
    if gap == 1800.0: specs.append(('M4 + interpolation', ['excl', 'nonmotor', 'log_speed', 'log_stops', 'interp_frac']))
    for y in ('log_tau', 'z_phi'):
        for name, terms in specs:
            r, n = fit(d, y, terms); r['gap'] = gap; r['y'] = y; r['model'] = name; r['n'] = n; out.append(r)
            print(f'--- gap={gap:.0f}s  y={y}  {name}  (n={n})'); print(r[['coef', 'ci_lo', 'ci_hi']])
    # bus vs car/taxi mediation (motorized shared only)
    s = d[d.grp == 'shared']
    for name, terms in [('bus only', ['bus']), ('bus + speed', ['bus', 'log_speed']), ('bus + speed + stops/km', ['bus', 'log_speed', 'log_stops'])]:
        r, n = fit(s, 'log_tau', terms); r['gap'] = gap; r['y'] = 'log_tau(shared)'; r['model'] = name; r['n'] = n; out.append(r)
        print(f'--- gap={gap:.0f}s  shared road, y=log_tau  {name}  (n={n})'); print(r[['coef', 'ci_lo', 'ci_hi']])
pd.concat(out).to_csv(P.out(P.CONTROLS_RESULTS / 'row_regression.csv'))

# ---- split exclusive into subway / train; base = car+taxi; clean data only
print('\n===== split modes, gap=30 s, base = car+taxi =====')
d = D[D.gap == 30.0].copy()
for m in ['subway', 'train', 'bus', 'bike', 'walk']: d['is_' + m] = (d['mode'] == m).astype(float)
res = []
for y in ('log_tau', 'z_phi'):
    for name, terms in [('modes only', ['is_subway', 'is_train', 'is_bus', 'is_bike', 'is_walk']),
                        ('+ speed', ['is_subway', 'is_train', 'is_bus', 'is_bike', 'is_walk', 'log_speed'])]:
        r, n = fit(d, y, terms); r['y'] = y; r['model'] = name; res.append(r)
        print(f'--- y={y} {name} (n={n})'); print(r[['coef', 'ci_lo', 'ci_hi']])
pd.concat(res).to_csv(P.out(P.CONTROLS_RESULTS / 'row_regression_split.csv'))
# speed-matched comparison: segments with mean speed 6-15 m/s (overlap of subway, car, taxi, bus)
m = d[(d.mean_speed > 6) & (d.mean_speed < 15)]
print('\n===== speed-matched 6-15 m/s =====')
print(m.groupby('mode').agg(n=('tau_c', 'size'), users=('user', 'nunique'), speed=('mean_speed', 'median'), tau=('tau_c', 'median'), phi=('phi', 'median')).round(3))
