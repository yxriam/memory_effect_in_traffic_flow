"""Right-of-way test: does exclusive right-of-way raise short-range persistence of speed, after controlling for
GPS sampling (interpolation), speed level and user? Segments: 5 s resampling, split at gaps > GAP s."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, importlib.util, glob, os
import numpy as np, pandas as pd
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location('m4', str(P.GEOLIFE_PIPELINE / '04_mode_hurst_memory.py')); m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
DT = 5.0
MODES = ['subway', 'train', 'car', 'taxi', 'bus', 'bike', 'walk', 'airplane', 'boat']

def acf(x, maxlag):
    x = np.asarray(x, float) - np.mean(x); n = len(x); v = np.dot(x, x) / n
    return np.array([np.dot(x[:n - k], x[k:]) / n / v for k in range(maxlag + 1)])

def run(args):
    mode, f, gap = args
    rng = np.random.default_rng(abs(hash((f, gap))) % 2**32)
    raw = m4.read_plt(m4.Path(f)); out = []
    user = os.path.basename(f)[:3]
    for i, seg in enumerate(m4.split_by_gap(raw, gap, 30)):
        rs = m4.resample_segment(seg, DT)
        if rs is None: continue
        vx, vy, sp = m4.velocity(rs, DT)
        if len(sp) < 202: continue
        # native sampling: fraction of resampled instants inside a native gap > 2*DT
        ts = (seg['datetime'] - seg['datetime'].iloc[0]).dt.total_seconds().to_numpy()
        ts = np.unique(ts); g = np.diff(ts)
        idx = np.clip(np.searchsorted(ts, rs['t'].to_numpy(), side='right') - 1, 0, len(g) - 1)
        interp = float(np.mean(g[idx] > 2 * DT)) if len(g) else 1.0
        a = acf(sp, 120)
        below = np.where(a < np.exp(-1))[0]
        tau = float(below[0] * DT) if len(below) else np.nan     # censored at 600 s
        stopped = sp < 0.5
        starts = np.sum(stopped[2:] & stopped[1:-1] & ~stopped[:-2])   # entering a stop of >= 2 samples
        dist_km = sp.sum() * DT / 1000
        dur_h = len(sp) * DT / 3600
        out.append(dict(mode=mode, user=user, gap=gap, seg=i, n=len(sp), interp_frac=interp, native_gap_med=float(np.median(g)) if len(g) else np.nan,
                        mean_speed=float(sp.mean()), cv_speed=float(sp.std() / max(sp.mean(), 1e-6)),
                        phi=float(a[1]), tau_c=tau, tau_censored=int(len(below) == 0),
                        stop_frac=float(stopped.mean()), stops_per_km=float(starts / dist_km) if dist_km > 0.05 else np.nan,
                        stops_per_h=float(starts / dur_h), H=m4.hurst_rs(sp)))
    return out

if __name__ == '__main__':
    jobs = []
    for mode in MODES:
        for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), mode, 'plt', '*.plt'))):
            for gap in (1800.0, 30.0):
                jobs.append((mode, f, gap))
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with Pool(2) as p:
        rows = [r for res in p.imap_unordered(run, jobs) for r in res]
    df = pd.DataFrame(rows); df.to_csv(P.out(P.CONTROLS_RESULTS / 'row_persistence.csv'), index=False)
    pd.set_option('display.width', 250)
    print(df.groupby(['gap', 'mode'])[['n', 'interp_frac', 'native_gap_med', 'mean_speed', 'phi', 'tau_c', 'stop_frac', 'stops_per_km', 'H']].median().round(3))
    print(df.groupby(['gap', 'mode']).size())
