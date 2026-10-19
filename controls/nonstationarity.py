"""P1-1 (GeoLife part)  Does the excess path CMI come from switching between first-order chains (time of day, weekday)?

A mixture of first-order chains is not first-order: if a user follows different chains on weekdays and weekends, or
in the morning and the evening, the pooled sequence shows second-order structure. Test: the per-unit path CMI is
computed twice, I(S_{t+1}; S_{t-1} | S_t) with the null stratified by S_t (as in the paper), and
I(S_{t+1}; S_{t-1} | S_t, type) with the null stratified by (S_t, type), where type = weekday / weekend x
4 periods (0-6, 6-12, 12-18, 18-24 h Beijing time, from the start of the segment). Splitting the data into strata
costs power by itself (the excess CMI is under-recovered in small samples, see null_calibration.py), so the same
computation is repeated with the type labels shuffled across the unit's segments; only the difference between the
real and the shuffled labels is attributed to switching between chains.
Data: GeoLife mode-labelled trips (the only GeoLife data here with timestamps), segments split at gaps > 30 s,
100 m cells, repeats collapsed, units with >= 200 trigrams; units = user x mode (as memory_by_mode.py) and
user (all modes pooled). 50 permutations, bits.
The taxi part (memory network against per-vehicle first-order surrogates) is in prediction/taxi_network_controls.py.

    python controls/nonstationarity.py   -> results/nonstationarity_geolife.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
import geolife_robust as gr
import memory_by_mode as mm

NPERM = 50


def segments(args):
    mode, f = args
    raw = mm.m4.read_plt(mm.m4.Path(f)); user = os.path.basename(f)[:3]
    out = []
    for seg in mm.m4.split_by_gap(raw, 30.0, 30):
        st = gr.to_states(seg.lat.to_numpy(), seg.lon.to_numpy(), mm.GRID)
        if len(st) < 3: continue
        t0 = seg.datetime.iloc[0] + pd.Timedelta(hours=8)                 # GeoLife times are GMT
        typ = int(t0.dayofweek >= 5) * 4 + t0.hour // 6
        out.append((mode, user, st, typ))
    return out


def unit(seqs, types, seed):
    a, b, c = gr.ngrams(seqs, 3)
    if len(a) < 200: return None
    ty = np.concatenate([np.full(max(len(s) - 2, 0), t) for s, t in zip(seqs, types)])
    vocab = np.unique(np.concatenate(seqs)); K = len(vocab); enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c = enc(a), enc(b), enc(c)
    rng = np.random.default_rng(seed)
    r0, n0, p0, h1, _ = gr.cmi_with_null(a, b, c, K, NPERM, rng)
    bt = gr.pair_id(b, ty, 8)
    r1, n1, p1, h1t, _ = gr.cmi_with_null(a, bt, c, K, NPERM, rng)
    # control for the loss of power from splitting the data: the same number of strata, but the type labels shuffled
    # across the unit's segments (keeps how many segments have each label, removes the link to time)
    tys = np.asarray(types)[rng.permutation(len(types))]
    typ = np.concatenate([np.full(max(len(s) - 2, 0), t) for s, t in zip(seqs, tys)])
    r2, n2, p2, _, _ = gr.cmi_with_null(a, gr.pair_id(b, typ, 8), c, K, NPERM, rng)
    return dict(n_trigrams=len(a), n_types=int(len(np.unique(ty))), H1=h1, excess=r0 - n0, p=p0, rel_excess=(r0 - n0) / h1 if h1 > 0 else np.nan,
                H1_type=h1t, excess_type=r1 - n1, p_type=p1, excess_shuffled_type=r2 - n2, p_shuffled_type=p2)


if __name__ == '__main__':
    jobs = [(m, f) for m in mm.MODES for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), m, 'plt', '*.plt')))]
    with Pool(2) as pool:
        segs = [s for res in pool.imap(segments, jobs) for s in res]
    rows = []
    D = pd.DataFrame(segs, columns=['mode', 'user', 'st', 'typ'])
    for (mode, user), g in D.groupby(['mode', 'user']):
        r = unit(list(g.st), list(g.typ), zlib.crc32(f'{mode}|{user}'.encode()))
        if r: rows.append(dict(level='user x mode', mode=mode, user=user, **r))
    for user, g in D.groupby('user'):
        r = unit(list(g.st), list(g.typ), zlib.crc32(f'all|{user}'.encode()))
        if r: rows.append(dict(level='user', mode='all', user=user, **r))
    R = pd.DataFrame(rows); R.to_csv(P.out(P.CONTROLS_RESULTS / 'nonstationarity_geolife.csv'), index=False)
    pd.set_option('display.width', 200)
    print(R.groupby(['level', 'mode']).agg(units=('user', 'size'), excess=('excess', 'median'), excess_type=('excess_type', 'median'),
                                          excess_shuffled_type=('excess_shuffled_type', 'median'), sig=('p', lambda x: (x < .05).mean()),
                                          sig_type=('p_type', lambda x: (x < .05).mean()), sig_shuffled=('p_shuffled_type', lambda x: (x < .05).mean())).round(4))
