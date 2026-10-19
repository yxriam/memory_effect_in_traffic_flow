"""DFA control for GeoLife speed series (walk, bus, car) — completes the DFA item of the to-do list.

Same segments as hurst_control.py (5 s resampling, segments >= 202 steps, gaps of 1800 s and 30 s).
For every segment it computes the DFA-1 exponent on
  dfa_speed     the speed series (reproduces dfa_alpha in hurst_control.csv)
  dfa_ar1       an AR(1) surrogate with the same length, mean, variance and lag-1 ACF (Markov, short memory)
  dfa_shuffled  a random permutation of the speeds (no temporal order)
and, as a check that the segments are the same as before, the R/S exponents H_speed and H_ar1.

If DFA, like R/S, is pushed up by short-range correlation, dfa_ar1 will be close to dfa_speed even though
the surrogate has no long memory. Random seeds are derived from the file name, so a re-run gives the same numbers.

    python controls/dfa_control.py            # walk, bus, car
    python controls/dfa_control.py --all      # all seven modes used in the paper
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, importlib.util, glob, os, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats

spec = importlib.util.spec_from_file_location('m4', str(P.GEOLIFE_PIPELINE / '04_mode_hurst_memory.py'))
m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
DT = 5.0
MODES = ['walk', 'bus', 'car'] + (['bike', 'subway', 'taxi', 'train'] if '--all' in sys.argv else [])


def dfa(x):
    """DFA-1, identical to hurst_control.dfa."""
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
    """AR(1) surrogate, identical to hurst_control.ar1."""
    x = np.asarray(x, float); mu, sd = x.mean(), x.std()
    z = (x - mu) / sd if sd > 0 else x * 0
    phi = float(np.clip(np.corrcoef(z[:-1], z[1:])[0, 1], -0.999, 0.999))
    e = rng.normal(0, np.sqrt(1 - phi ** 2), len(x)); s = np.empty(len(x)); s[0] = rng.normal()
    for i in range(1, len(x)): s[i] = phi * s[i - 1] + e[i]
    return mu + sd * s, phi


def run(args):
    mode, f, gap = args
    rng = np.random.default_rng(zlib.crc32(f'{os.path.basename(f)}|{gap}'.encode()))
    raw = m4.read_plt(m4.Path(f)); out = []
    for i, seg in enumerate(m4.split_by_gap(raw, gap, 30)):
        rs = m4.resample_segment(seg, DT)
        if rs is None: continue
        vx, vy, sp = m4.velocity(rs, DT)
        if len(sp) < 202: continue      # same length filter as the main analysis (max_lag 200)
        sur, phi = ar1(sp, rng)
        out.append(dict(mode=mode, gap=gap, file=os.path.basename(f), seg=i, n=len(sp), phi=phi,
                        H_speed=m4.hurst_rs(sp), H_ar1=m4.hurst_rs(sur),
                        dfa_speed=dfa(sp), dfa_ar1=dfa(sur), dfa_shuffled=dfa(rng.permutation(sp))))
    return out


if __name__ == '__main__':
    jobs = []
    for mode in MODES:
        for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), mode, 'plt', '*.plt'))):
            for gap in (1800.0, 30.0):
                jobs.append((mode, f, gap))
    if not jobs:
        raise SystemExit('no GeoLife trips_by_mode files found under %s (see local_paths.json)' % P.GEOLIFE_TRIPS_BY_MODE)
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with Pool(max(1, min(4, os.cpu_count() or 1))) as p:
        rows = [r for res in p.imap_unordered(run, jobs) for r in res]
    df = pd.DataFrame(rows).sort_values(['gap', 'mode', 'file', 'seg'])
    df.to_csv(P.out(P.CONTROLS_RESULTS / 'dfa_control.csv'), index=False)

    cols = ['n', 'phi', 'H_speed', 'H_ar1', 'dfa_speed', 'dfa_ar1', 'dfa_shuffled']
    summary = df.groupby(['gap', 'mode'])[cols].median().round(3)
    summary['segments'] = df.groupby(['gap', 'mode']).size()
    d = df.dropna(subset=['dfa_speed', 'dfa_ar1'])
    g = d.groupby(['gap', 'mode'])
    summary['dfa_diff_median'] = g.apply(lambda x: np.median(x.dfa_speed - x.dfa_ar1)).round(3)
    summary['share_speed_above_ar1'] = g.apply(lambda x: np.mean(x.dfa_speed > x.dfa_ar1)).round(3)
    summary['wilcoxon_p'] = g.apply(lambda x: stats.wilcoxon(x.dfa_speed, x.dfa_ar1).pvalue)
    summary.to_csv(P.out(P.CONTROLS_RESULTS / 'dfa_control_summary.csv'))
    pd.set_option('display.width', 200)
    print(summary)
