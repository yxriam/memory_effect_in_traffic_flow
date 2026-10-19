"""Per-vehicle 0.2 s series from the original highD release (`highd_full` in common/paths.py), for P0-2 and P0-3.

For every vehicle the speed and acceleration along the driving direction are computed in five ways:
  bd      backward difference of positions, as in highd_augment.py: v = central difference at 25 Hz,
          a = (v_t - v_{t-0.2 s}) / 0.2 s
  ds      the dataset's own xVelocity and xAcceleration
  sg15/sg21/sg25   Savitzky-Golay fit (order 2) of the 25 Hz position over 15 / 21 / 25 frames (0.6 / 0.84 / 1.0 s),
          first and second derivative
plus the preceding-vehicle variables of the dataset: gap = dhw (m), dv = leader speed - own speed (m/s),
leader id, lane id and the driving-direction sign dir (-1 towards -x; signed v = dir * v, as in highd_augment.py).
Rows are kept every 5th frame (0.2 s). Vehicles with fewer than 50 frames are skipped.
The table is cached in controls/work/highd_full_states.pkl.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, time
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from multiprocessing import Pool

DT_RAW, DS = 0.04, 5
DT = DT_RAW * DS
SG = {'sg15': 15, 'sg21': 21, 'sg25': 25}
CACHE = P.CONTROLS_WORK / 'highd_full_states.pkl'
COLS = ['frame', 'id', 'x', 'xVelocity', 'xAcceleration', 'dhw', 'precedingXVelocity', 'precedingId', 'laneId']


def process(path):
    rec = int(os.path.basename(path)[:2])
    meta = pd.read_csv(path.replace('_tracks.csv', '_tracksMeta.csv'), usecols=['id', 'drivingDirection'])
    sgn = dict(zip(meta.id, np.where(meta.drivingDirection == 1, -1.0, 1.0)))
    df = pd.read_csv(path, usecols=COLS).sort_values(['id', 'frame'])
    out = []
    for vid, g in df.groupby('id', sort=False):
        n = len(g)
        if n < 50:
            continue
        s = sgn[vid]
        fr = g.frame.to_numpy(); x = g.x.to_numpy() * s
        d = {}
        v = np.full(n, np.nan); v[1:-1] = (x[2:] - x[:-2]) / (2 * DT_RAW)
        a = np.full(n, np.nan); a[6:] = (v[6:] - v[1:-5]) / DT
        d['v_bd'], d['a_bd'] = v, a
        d['v_ds'] = g.xVelocity.to_numpy() * s; d['a_ds'] = g.xAcceleration.to_numpy() * s
        for k, w in SG.items():
            if n > w:
                d['v_' + k] = savgol_filter(x, w, 2, deriv=1, delta=DT_RAW)
                d['a_' + k] = savgol_filter(x, w, 2, deriv=2, delta=DT_RAW)
                h = w // 2                                   # edge values are extrapolated: drop them
                for c in ('v_' + k, 'a_' + k):
                    d[c][:h] = np.nan; d[c][-h:] = np.nan
            else:
                d['v_' + k] = np.full(n, np.nan); d['a_' + k] = np.full(n, np.nan)
        lead = g.precedingId.to_numpy()
        d['gap'] = np.where(lead > 0, g.dhw.to_numpy(), np.nan)
        d['dv'] = np.where(lead > 0, (g.precedingXVelocity.to_numpy() - g.xVelocity.to_numpy()) * s, np.nan)
        keep = fr % DS == 0
        t = pd.DataFrame({k: np.asarray(val)[keep] for k, val in d.items()})
        t['frame'] = fr[keep]; t['lead'] = lead[keep]; t['lane'] = g.laneId.to_numpy()[keep]
        t['uid'] = rec * 100000 + vid; t['rec'] = rec; t['dir'] = s   # dir = -1 for vehicles driving towards -x
        out.append(t)
    return pd.concat(out, ignore_index=True)


def add_dir(S):
    """Driving-direction sign per vehicle from *_tracksMeta.csv (-1: towards -x)."""
    m = []
    for f in sorted(glob.glob(os.path.join(str(P.HIGHD_FULL), '*_tracksMeta.csv'))):
        d = pd.read_csv(f, usecols=['id', 'drivingDirection'])
        m.append(pd.DataFrame({'uid': int(os.path.basename(f)[:2]) * 100000 + d.id, 'dir': np.where(d.drivingDirection == 1, -1.0, 1.0)}))
    return S.merge(pd.concat(m), on='uid', how='left')


def load():
    if CACHE.exists():
        S = pd.read_pickle(CACHE)
        if 'dir' not in S:                       # caches written before the 'dir' column existed
            S = add_dir(S); S.to_pickle(CACHE)
        return S
    files = sorted(glob.glob(os.path.join(str(P.HIGHD_FULL), '*_tracks.csv')))
    t0 = time.time()
    with Pool(2) as pool:
        parts = pool.map(process, files)
    S = pd.concat(parts, ignore_index=True).sort_values(['uid', 'frame']).reset_index(drop=True)
    for c in S.columns:
        if S[c].dtype == np.float64:
            S[c] = S[c].astype(np.float32)
    S.to_pickle(P.out(CACHE))
    print(f'highD full: {len(files)} recordings, {S.uid.nunique()} vehicles, {len(S)} rows [{time.time() - t0:.0f}s]', flush=True)
    return S


if __name__ == '__main__':
    S = load()
    print(S.describe().T.round(3))
