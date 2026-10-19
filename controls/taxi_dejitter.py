"""P1-5  Taxi: does GPS jitter at cell borders (A -> B -> A after run-length compression) create the second-order memory?

Same per-trajectory analysis as taxi/taxi_memory_analysis.ipynb (main result: per-trajectory 20 x 20 grid,
run-length compressed whole-day sequence, >= 50 compressed states), run twice: as published and after removing
A -> B -> A patterns iteratively (geolife_robust.dejitter, as used for GeoLife). Per trajectory:
  CMI2 = I(S_{t+1}; S_{t-1} | S_t) with the stratified permutation null (100 permutations, seed from the file name),
  held-out Delta LL(2 vs 1) with the interpolated back-off chain at the tuned beta* = 3 (taxi/memory_backoff_tuned_summary.csv,
  20 x 20 per trajectory), 80/20 split in time.
All in nats. The memory-network part of this check is in prediction/taxi_network_controls.py.

    python controls/taxi_dejitter.py   -> results/taxi_dejitter.csv, taxi_dejitter_summary.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.ROOT / 'taxi'))
import glob, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import taxi_memory_lib as tl
import geolife_robust as gr

GRID, NPERM, BETA = 20, 100, 3.0


def run(path):
    lat, lon = tl.read_lat_lon(path)
    st = tl.map_to_states(lat, lon, GRID)
    if len(st) < 30: return []
    comp = tl.compress_repeats(st)
    out = []
    for variant, s in (('published', comp), ('dejittered', gr.dejitter(np.asarray(comp, dtype=np.int64)).tolist())):
        if len(s) < 50: continue
        c2 = tl.cmi_perm(s, 2, NPERM, tl.stable_seed(path))
        cut = int(len(s) * 0.8)
        l1, l2, _ = tl._ll_orders(s[cut:], s[:cut], GRID * GRID, BETA)
        out.append(dict(file=os.path.basename(path), variant=variant, comp_len=len(s), cmi2_real=c2['real'] if c2 else np.nan,
                        cmi2_excess=c2['excess'] if c2 else np.nan, cmi2_p=c2['p_value'] if c2 else np.nan, dll_2v1=l2 - l1,
                        aba_share=float(np.mean(np.asarray(comp[2:]) == np.asarray(comp[:-2]))) if len(comp) > 2 else np.nan))
    return out


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), '*')))
    with Pool(2) as pool:
        R = pd.DataFrame([r for res in pool.imap(run, files, chunksize=20) for r in res])
    R.to_csv(P.out(P.CONTROLS_RESULTS / 'taxi_dejitter.csv'), index=False)
    both = R.pivot(index='file', columns='variant', values=['cmi2_excess', 'cmi2_p', 'dll_2v1']).dropna()
    rows = []
    for v in ('published', 'dejittered'):
        d = R[R.variant == v]
        rows.append(dict(variant=v, trajectories=len(d), comp_len_median=d.comp_len.median(), cmi2_excess_median=d.cmi2_excess.median(),
                         share_significant=(d.cmi2_p < .05).mean(), dll_2v1_median=d.dll_2v1.median(), dll_2v1_mean=d.dll_2v1.mean(),
                         dll_2v1_positive=(d.dll_2v1 > 0).mean(), aba_share_median=d.aba_share.median()))
    S = pd.DataFrame(rows)
    w = stats.wilcoxon(both[('cmi2_excess', 'published')], both[('cmi2_excess', 'dejittered')])
    S['paired_trajectories'] = len(both); S['wilcoxon_p_cmi2'] = w.pvalue
    S.to_csv(P.out(P.CONTROLS_RESULTS / 'taxi_dejitter_summary.csv'), index=False)
    pd.set_option('display.width', 200); print(S.round(4).to_string(index=False))
