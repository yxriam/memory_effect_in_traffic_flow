"""R/S Hurst controls for GeoLife speed series (walk, bus, car).
For each 5 s-resampled segment (same pipeline as 04_mode_hurst_memory.py):
  H_speed      R/S on the speed series (reproduces the paper value)
  H_shuffled   R/S on a random permutation of the same values (no temporal order)
  H_ar1        R/S on an AR(1) (Markov, short-memory) surrogate with the same length, mean, variance, lag-1 ACF
  H_increment  R/S on the speed increments
  dfa_alpha    DFA-1 exponent on the speed series
Variant 'gap30': segments split at gaps > 30 s instead of 1800 s, so no long interpolated stretches.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, importlib.util, glob, os, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location('m4', str(P.GEOLIFE_PIPELINE / '04_mode_hurst_memory.py')); m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
DT = 5.0

def dfa(x):
    x = np.asarray(x, float); y = np.cumsum(x - x.mean()); n = len(y)
    scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n // 4), 10)).astype(int))
    F, S = [], []
    for s in scales:
        k = n // s
        if k < 2: continue
        seg = y[:k * s].reshape(k, s); t = np.arange(s)
        res = []
        for row in seg:
            p = np.polyfit(t, row, 1); res.append(np.mean((row - np.polyval(p, t)) ** 2))
        F.append(np.sqrt(np.mean(res))); S.append(s)
    if len(F) < 3: return np.nan
    return float(np.polyfit(np.log(S), np.log(F), 1)[0])

def ar1(x, rng):
    x = np.asarray(x, float); mu, sd = x.mean(), x.std()
    z = (x - mu) / sd if sd > 0 else x * 0
    phi = float(np.clip(np.corrcoef(z[:-1], z[1:])[0, 1], -0.999, 0.999))
    e = rng.normal(0, np.sqrt(1 - phi ** 2), len(x)); s = np.empty(len(x)); s[0] = rng.normal()
    for i in range(1, len(x)): s[i] = phi * s[i - 1] + e[i]
    return mu + sd * s, phi

def run(args):
    mode, f, gap = args
    # seed from the file name and gap: Python's hash() changes between runs and machines, so it is not reproducible
    rng = np.random.default_rng(zlib.crc32(f'{os.path.basename(f)}|{gap:g}'.encode()))
    raw = m4.read_plt(m4.Path(f)); out = []
    for i, seg in enumerate(m4.split_by_gap(raw, gap, 30)):
        rs = m4.resample_segment(seg, DT)
        if rs is None: continue
        vx, vy, sp = m4.velocity(rs, DT)
        if len(sp) < 202: continue      # same length filter as the main analysis (max_lag 200)
        sur, phi = ar1(sp, rng)
        out.append(dict(mode=mode, gap=gap, file=os.path.basename(f), seg=i, n=len(sp), phi=phi,
                        H_speed=m4.hurst_rs(sp), H_shuffled=m4.hurst_rs(rng.permutation(sp)),
                        H_ar1=m4.hurst_rs(sur), H_increment=m4.hurst_rs(np.diff(sp)), dfa_alpha=dfa(sp),
                        frac_zero_speed=float(np.mean(sp < 0.05))))
    return out

if __name__ == '__main__':
    jobs = []
    for mode in ['walk', 'bus', 'car']:
        for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), mode, 'plt', '*.plt'))):
            for gap in (1800.0, 30.0):
                jobs.append((mode, f, gap))
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with Pool(2) as p:
        rows = [r for res in p.imap_unordered(run, jobs) for r in res]
    df = pd.DataFrame(rows); df.to_csv(P.out(P.CONTROLS_RESULTS / 'hurst_control.csv'), index=False)
    print(df.groupby(['gap', 'mode'])[['n', 'phi', 'H_speed', 'H_shuffled', 'H_ar1', 'H_increment', 'dfa_alpha', 'frac_zero_speed']].median().round(3))
    print(df.groupby(['gap', 'mode']).size())
