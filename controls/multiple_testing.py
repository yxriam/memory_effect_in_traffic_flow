"""P3-3  How many units really carry second-order memory? Multiple testing for the per-unit permutation tests.

The paper reports the share of units with p < 0.05 (e.g. 66 % of GeoLife users at 10 m). With many units some of these
are false positives, and because the stratified permutation null is conservative (null_calibration.py: false-positive
rate 1.1 % / 3.0 % instead of 5 %) the share also misses units with weak memory. Two summaries that do not depend on
the 0.05 cut are added, both from the p-values already in the result tables:
  pi1     Storey's estimate of the share of units with memory, 1 - #{p > lambda} / (m (1 - lambda)), lambda = 0.5.
          A conservative null makes null p-values pile up near 1, so pi1 is itself conservative.
  BH      share of units rejected by Benjamini-Hochberg at a false-discovery rate of 5 %.
and 95 % bootstrap intervals (2 000 resamples of units, seed 20261022) for the median excess CMI, the share with
p < 0.05 and pi1. Smallest attainable p: 1/51 (GeoLife, 50 permutations), 1/31 (modes, 30), 1/101 (taxi, 100).

Input tables (committed): controls/results/geolife_robust_per_user.csv, controls/results/memory_by_mode_path.csv,
taxi/memory_full_per_file.csv. Runs in seconds; the result table is written atomically (.tmp, then rename).

    python controls/multiple_testing.py   -> results/multiple_testing.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import os
import numpy as np, pandas as pd

LAM, Q, NBOOT, SEED = 0.5, 0.05, 2000, 20261022


def pi1(p, lam=LAM):
    p = np.asarray(p, float)
    return 1.0 - min(1.0, np.mean(p > lam) / (1.0 - lam))


def bh_share(p, q=Q):
    p = np.sort(np.asarray(p, float)); m = len(p)
    ok = p <= q * np.arange(1, m + 1) / m
    return (np.max(np.where(ok)[0]) + 1) / m if ok.any() else 0.0


def summarise(label, excess, p, rng):
    excess, p = np.asarray(excess, float), np.asarray(p, float)
    m = len(p); idx = rng.integers(0, m, size=(NBOOT, m))
    bm = np.median(excess[idx], axis=1); bs = (p[idx] < 0.05).mean(axis=1); bp = np.array([pi1(p[i]) for i in idx])
    ci = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
    return dict(unit_set=label, units=m, median_excess=float(np.median(excess)), median_excess_lo=ci(bm)[0], median_excess_hi=ci(bm)[1],
                share_p05=float((p < 0.05).mean()), share_p05_lo=ci(bs)[0], share_p05_hi=ci(bs)[1],
                pi1=pi1(p), pi1_lo=ci(bp)[0], pi1_hi=ci(bp)[1], bh_share=bh_share(p), min_p=float(p.min()))


if __name__ == '__main__':
    rng = np.random.default_rng(SEED); rows = []
    g = pd.read_csv(P.CONTROLS_RESULTS / 'geolife_robust_per_user.csv')
    for grid, lab in [(0.0001, '10 m'), (0.001, '100 m')]:
        s = g[np.isclose(g.grid, grid) & (g.dejitter.astype(int) == 0)]
        rows.append(summarise(f'GeoLife users, {lab}, second order (bits)', s.cmi2_excess, s.cmi2_p, rng))
        rows.append(summarise(f'GeoLife users, {lab}, third order (bits)', s.cmi3_excess, s.cmi3_p, rng))
    t = pd.read_csv(P.TAXI_RESULTS / 'memory_full_per_file.csv')
    t = t[(t.grid == 20) & t.compressed.astype(bool) & (t.comp_len >= 50)]
    rows.append(summarise('taxi trajectories, 20x20 compressed, second order (nats)', t.cmi2_excess, t.cmi2_p, rng))
    rows.append(summarise('taxi trajectories, 20x20 compressed, third order (nats)', t.cmi3_excess, t.cmi3_p, rng))
    m = pd.read_csv(P.CONTROLS_RESULTS / 'memory_by_mode_path.csv'); m = m[m.dejitter == 0]
    for mode, s in m.groupby('mode'):
        rows.append(summarise(f'GeoLife user x mode, {mode}, 100 m (bits)', s.cmi_excess, s.p, rng))
    R = pd.DataFrame(rows)
    out = P.out(P.CONTROLS_RESULTS / 'multiple_testing.csv'); tmp = str(out) + '.tmp'
    R.to_csv(tmp, index=False); os.replace(tmp, out)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(R.round(4).to_string(index=False))
