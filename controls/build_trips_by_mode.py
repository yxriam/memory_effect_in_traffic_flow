"""Rebuild trips_by_mode_data from the official GeoLife 1.3 release.
For each user with labels.txt and each labelled interval (start, end, mode), all GPS fixes of that user
whose timestamp falls inside the interval are written, in time order, to <user>_<mode>_combined.plt."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as _P
import sys, os, glob, numpy as np, pandas as pd
# defaults: read the GeoLife release, write to controls/work/ so an existing trips_by_mode_data is never overwritten
DATA = sys.argv[1] if len(sys.argv) > 1 else str(_P.GEOLIFE_RAW)
OUT = sys.argv[2] if len(sys.argv) > 2 else str(_P.CONTROLS_WORK / 'trips_by_mode_data')
HDR = 'Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n'
stats = []
for ud in sorted(glob.glob(os.path.join(DATA, '*'))):
    lab = os.path.join(ud, 'labels.txt')
    if not os.path.exists(lab): continue
    user = os.path.basename(ud)
    L = pd.read_csv(lab, sep='\t'); L.columns = ['start', 'end', 'mode']
    L['start'] = pd.to_datetime(L.start); L['end'] = pd.to_datetime(L.end)
    rows = []
    for f in sorted(glob.glob(os.path.join(ud, 'Trajectory', '*.plt'))):
        with open(f, errors='ignore') as h: lines = h.read().splitlines()[6:]
        rows += [l for l in lines if l.count(',') >= 6]
    P = pd.DataFrame([r.split(',') for r in rows]).iloc[:, :7]
    P['t'] = pd.to_datetime(P[5] + ' ' + P[6], errors='coerce'); P = P.dropna(subset=['t']).sort_values('t').drop_duplicates()
    t = P.t.to_numpy()
    for mode, g in L.groupby('mode'):
        parts = []
        for s, e in zip(g.start, g.end):
            i0, i1 = np.searchsorted(t, np.datetime64(s), 'left'), np.searchsorted(t, np.datetime64(e), 'right')
            if i1 > i0: parts.append(P.iloc[i0:i1])
        if not parts: continue
        Q = pd.concat(parts)
        os.makedirs(os.path.join(OUT, mode, 'plt'), exist_ok=True)
        with open(os.path.join(OUT, mode, 'plt', f'{user}_{mode}_combined.plt'), 'w') as h:
            h.write(HDR); h.write('\n'.join(','.join(map(str, r)) for r in Q[[0, 1, 2, 3, 4, 5, 6]].itertuples(index=False))); h.write('\n')
        stats.append(dict(user=user, mode=mode, n_points=len(Q), n_intervals=len(g)))
S = pd.DataFrame(stats); S.to_csv(os.path.join(OUT, '_build_summary.csv'), index=False)
print(S.groupby('mode').agg(users=('user', 'size'), points=('n_points', 'sum')))
