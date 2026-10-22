"""P0-3  Is the residual speed memory after state augmentation measurement noise?

The augmented state (fine speed + acceleration) leaves 0.007 nats (0.010 bits) per 0.2 s step for v_{t-1}, and the
backward-difference acceleration has a lag-0.2 s autocorrelation of -0.33. Both could come from differentiating
noisy positions. Here the state-augmentation table and the acceleration ACF are recomputed with five definitions of
(v, a) on the same steps (see highd_full.py): bd (as in the paper), ds (dataset values), sg15 / sg21 / sg25
(Savitzky-Golay, 0.6 / 0.84 / 1.0 s windows).

Model and splits as in highd_fine.py / highd_fine3.py: target v_{t+1} in 20 quantile bins, add-0.5 smoothing,
two states: signed velocity (as in the paper's augmentation table) and speed along the driving direction,
3 vehicle splits (seed 100 + split, 80 / 20). Gains in nats per step (bits = nats / ln 2).
The original release covers 23 of the 60 recordings here (01-11, 15-24, 58, 59), so the 'bd' row is the reference
for this subset rather than the paper's number.

    python controls/highd_accel_noise.py   -> results/highd_accel_noise<TAG>.csv, _acf, _gains (TAG from highd_full.py:
                                              '' for the first 23 recordings, '_60rec' for the full release)
Each (state, split) block is cached in controls/work/highd_accel_noise<TAG>/ (atomic writes); a restart skips finished blocks.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import os, pickle
import numpy as np, pandas as pd
import highd_augment as h
import highd_full as hf

DEFS = ['bd', 'ds', 'sg15', 'sg21', 'sg25']
MODELS = {'V1 (20)': ['v0_20'], 'V2 (20,20)': ['v0_20', 'v1_20'], 'V1 (100)': ['v0_100'], 'V2 (100,20)': ['v0_100', 'v1_20'],
          'V2 (100,100)': ['v0_100', 'v1_100'], 'VA (100,a20)': ['v0_100', 'a_20'], 'VA2 (100,a20,v1_20)': ['v0_100', 'a_20', 'v1_20'],
          'V3 (100,20,20)': ['v0_100', 'v1_20', 'v2_20'], 'VA3 (100,a20,v1,v2)': ['v0_100', 'a_20', 'v1_20', 'v2_20']}
RADIX = {'v0_20': 20, 'v0_100': 100, 'v1_20': 20, 'v1_100': 100, 'v2_20': 20, 'a_20': 20}
GAINS = {'coarse: V2(20,20) - V1(20)': ('V2 (20,20)', 'V1 (20)'), 'fine: V2(100,20) - V1(100)': ('V2 (100,20)', 'V1 (100)'),
         'fine, v_{t-1} also fine: V2(100,100) - V1(100)': ('V2 (100,100)', 'V1 (100)'),
         'residual after a: VA2 - VA': ('VA2 (100,a20,v1_20)', 'VA (100,a20)'),
         'third step: V3 - V2(100,20)': ('V3 (100,20,20)', 'V2 (100,20)'), 'third step with a: VA3 - VA2': ('VA3 (100,a20,v1,v2)', 'VA2 (100,a20,v1_20)')}

if __name__ == '__main__':
    S = hf.load()
    uid = S.uid.to_numpy(); fr = S.frame.to_numpy()
    n = len(S); i = np.arange(2, n - 1)
    ok = (uid[i - 2] == uid[i + 1]) & (fr[i + 1] - fr[i - 2] == 3 * hf.DS)
    for d in DEFS:                                    # common steps: every definition finite at t-2 .. t+1
        v = S['v_' + d].to_numpy(); a = S['a_' + d].to_numpy()
        ok &= np.isfinite(v[i - 2]) & np.isfinite(v[i - 1]) & np.isfinite(v[i]) & np.isfinite(v[i + 1]) & np.isfinite(a[i])
    i = i[ok]
    print('steps', len(i), 'vehicles', len(np.unique(uid[i])), flush=True)
    rows = []
    sgn = S['dir'].to_numpy().astype(float)
    WORK = P.CONTROLS_WORK / f'highd_accel_noise{hf.TAG}'; WORK.mkdir(parents=True, exist_ok=True)
    for state in ('signed_v', 'speed'):
      for sp in range(3):
        blk = WORK / f'{state}_s{sp}.pkl'
        if blk.exists():
            rows += pickle.load(open(blk, 'rb')); print(state, 'split', sp, '(cached)', flush=True); continue
        r0 = len(rows)
        rng = np.random.default_rng(100 + sp); u = rng.permutation(np.unique(uid[i])); tr = np.isin(uid[i], u[:int(.8 * len(u))])
        for d in DEFS:
            v = S['v_' + d].to_numpy().astype(float); a = S['a_' + d].to_numpy().astype(float)
            if state == 'signed_v':               # the paper's state: signed velocity along x, both directions binned together
                v, a = v * sgn, a * sgn
            E = {k: h.qedges(v[i[tr]], k) for k in (20, 100)}; EA = h.qedges(a[i[tr]], 20)
            def st(ix):
                return dict(y=h.qbins(v[ix + 1], E[20]), v0_20=h.qbins(v[ix], E[20]), v0_100=h.qbins(v[ix], E[100]),
                            v1_20=h.qbins(v[ix - 1], E[20]), v1_100=h.qbins(v[ix - 1], E[100]), v2_20=h.qbins(v[ix - 2], E[20]),
                            a_20=h.qbins(a[ix], EA))
            T, V = st(i[tr]), st(i[~tr])
            for name, cols in MODELS.items():
                ltr, lva, nc = h.eval_model(T, V, cols, RADIX, 20, alpha=0.5)
                rows.append(dict(state=state, split=sp, definition=d, model=name, train=ltr, val=lva, nctx=nc, n_val=len(V['y'])))
        pickle.dump(rows[r0:], open(str(blk) + '.tmp', 'wb')); os.replace(str(blk) + '.tmp', blk)
        print(state, 'split', sp, flush=True)
    R = pd.DataFrame(rows); R.to_csv(P.out(P.CONTROLS_RESULTS / f'highd_accel_noise{hf.TAG}.csv'), index=False)
    Gs = []
    for state, Rs in R.groupby('state'):
        M = Rs.groupby(['definition', 'model']).val.mean().unstack(0)[DEFS]
        G = pd.DataFrame({k: M.loc[a_] - M.loc[b_] for k, (a_, b_) in GAINS.items()}).T
        G.index.name = 'gain (nats/step)'; G.insert(0, 'state', state); Gs.append(G)
        print(state); print(M.round(4))
    G = pd.concat(Gs); G.to_csv(P.out(P.CONTROLS_RESULTS / f'highd_accel_noise{hf.TAG}_gains.csv'))
    # acceleration autocorrelation, lags 0.2 ... 3 s, pairs within one vehicle and consecutive 0.2 s rows
    acf = []
    same = lambda L: (uid[L:] == uid[:-L]) & (fr[L:] - fr[:-L] == L * hf.DS)
    for d in DEFS:
        a = S['a_' + d].to_numpy().astype(float)
        for L in range(1, 16):
            m = same(L) & np.isfinite(a[L:]) & np.isfinite(a[:-L])
            acf.append(dict(definition=d, lag=L, lag_s=0.2 * L, acf=float(np.corrcoef(a[L:][m], a[:-L][m])[0, 1]), n=int(m.sum())))
    A = pd.DataFrame(acf); A.to_csv(P.out(P.CONTROLS_RESULTS / f'highd_accel_noise{hf.TAG}_acf.csv'), index=False)
    pd.set_option('display.width', 200)
    print(G.round(4).to_string())
    print(A.pivot(index='lag_s', columns='definition', values='acf')[DEFS].round(3))
