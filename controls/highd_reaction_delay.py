"""P0-2  Driver reaction delay in highD: does the leader's state 0.4-2 s ago predict the own future speed change?

'Speed memory vanishes' was tested only with the own v_{t-1} at a 0.2 s horizon. A driver reacts to the leader with a
delay of about 1 s, so the leader's earlier state could matter. Here:
  horizons h = 0.2, 1, 2 s (1, 5, 10 steps of 0.2 s); target y = v_{t+h} - v_t (20 quantile bins of the training part)
  base state  (v, a, gap, dv) from the dataset (xVelocity, xAcceleration, dhw, precedingXVelocity - xVelocity along the
              driving direction), binned 10 / 10 / 5 / 5
  added       dv_{t-k} or gap_{t-k} (5 bins), k = 2..10 steps (0.4-2.0 s), one at a time, then dv and gap jointly at
              each k, and dv at k = 2, 5, 10 together
  samples     same vehicle, same leader (precedingId > 0) and same lane from t - 2 s to t + h; for each h the same
              samples are used for every k
  estimator   held-out log-likelihood gain with interpolated back-off (P(y|base,lag) -> P(y|base) -> P(y)),
              beta chosen on an inner 80/20 vehicle split from {1, 3, 10, 30, 100}; 3 outer vehicle splits
              (seed 200 + split, 80/20). Null band: 20 permutations of the added variable within base-state strata
              (outer split 0, betas of the real fit), 2.5-97.5 %.
  cross-check k-nearest-neighbour CMI (Frenzel-Pompe, k = 10, max-norm, rank-transformed variables) of y and dv_{t-k}
              given the base state on 3 000 random samples, with a local permutation test (Runge 2018: dv_{t-k}
              replaced by that of one of the 5 nearest neighbours in the base state; 30 permutations).
Gains in nats per step and bits. Data: original highD release, 23 recordings (see highd_full.py).
--lead-accel: the leader's own acceleration at t (dataset xAcceleration of the preceding vehicle, 5 bins) is added to
the base state (samples where it is known), and its own gain over the base state is reported as added = 'a_lead'.
If the lagged dv only stands in for the leader's acceleration (the trend of dv), its gain vanishes in this run.

    python controls/highd_reaction_delay.py                -> results/highd_reaction_delay.csv, _null.csv, _knn.csv
    python controls/highd_reaction_delay.py --lead-accel   -> results/highd_reaction_delay_leadacc.csv, _null, _knn
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, time
import numpy as np, pandas as pd
from scipy.special import digamma
from scipy.stats import rankdata
from sklearn.neighbors import KDTree
import highd_augment as h
import highd_full as hf

HORIZONS = (1, 5, 10)
LAGS = tuple(range(2, 11))
KMAX = 10
BETAS = (1.0, 3.0, 10.0, 30.0, 100.0)
NY = 20


def eligible(S, hz):
    """Indices t with one vehicle, one leader (>0), one lane and consecutive 0.2 s rows over [t - KMAX, t + hz]."""
    uid = S.uid.to_numpy(); fr = S.frame.to_numpy(); lead = S['lead'].to_numpy(); lane = S.lane.to_numpy()
    fin = np.isfinite(S.gap.to_numpy()) & np.isfinite(S.dv.to_numpy())
    brk = np.r_[True, (uid[1:] != uid[:-1]) | (lead[1:] != lead[:-1]) | (lane[1:] != lane[:-1]) | (np.diff(fr) != hf.DS)
                | ~fin[1:] | ~fin[:-1]]
    brk |= (lead <= 0)
    run = np.cumsum(brk)
    start = np.r_[0, np.flatnonzero(np.diff(run)) + 1]
    pos = np.arange(len(S)) - start[run - 1]
    length = np.bincount(run)[run]
    t = np.where((pos >= KMAX) & (pos + hz <= length - 1) & (lead > 0))[0]
    return t


def fit_counts(key, y):
    k, n = np.unique(key * NY + y, return_counts=True)
    kc, nc = np.unique(key, return_counts=True)
    return k, n, kc, nc


def look(keys, cnt, q):
    idx = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return np.where(keys[idx] == q, cnt[idx], 0).astype(float)


def probs(train, test, beta_b, beta_f=None):
    """P(y|base) (and P(y|base,lag) if 'lag' in the dicts) for the test rows, interpolated back-off."""
    p0 = (np.bincount(train['y'], minlength=NY) + 0.5) / (len(train['y']) + 0.5 * NY)
    kb, nb, kcb, ncb = fit_counts(train['b'], train['y'])
    pb = (look(kb, nb, test['b'] * NY + test['y']) + beta_b * p0[test['y']]) / (look(kcb, ncb, test['b']) + beta_b)
    if beta_f is None:
        return pb
    kf, nf, kcf, ncf = fit_counts(train['f'], train['y'])
    return (look(kf, nf, test['f'] * NY + test['y']) + beta_f * pb) / (look(kcf, ncf, test['f']) + beta_f)


def heldout_gain(train, test, inner):
    """Tune beta_b, beta_f on the inner split of train, refit on train, return (ll_base, ll_full) per step in nats."""
    itr = {k: v[inner] for k, v in train.items()}; iva = {k: v[~inner] for k, v in train.items()}
    bb = max(BETAS, key=lambda b: np.log(probs(itr, iva, b)).mean())
    bf = max(BETAS, key=lambda b: np.log(probs(itr, iva, bb, b)).mean())
    return np.log(probs(train, test, bb)).mean(), np.log(probs(train, test, bb, bf)).mean(), bb, bf


def strat_perm(x, strata, rng):
    order = np.lexsort((rng.random(len(x)), strata))
    out = np.empty_like(x); idx = np.argsort(strata, kind='stable')
    out[idx] = x[order]
    return out


def knn_cmi(x, y, z, k=10):
    """Frenzel-Pompe CMI I(X;Y|Z) in nats, max-norm, inputs already standardised (n x d arrays)."""
    xyz = np.c_[x, y, z]; xz = np.c_[x, z]; yz = np.c_[y, z]
    eps = KDTree(xyz, metric='chebyshev').query(xyz, k=k + 1)[0][:, -1] - 1e-12
    cnt = lambda A: KDTree(A, metric='chebyshev').query_radius(A, eps, count_only=True) - 1
    return float(digamma(k) - np.mean(digamma(cnt(xz) + 1) + digamma(cnt(yz) + 1) - digamma(cnt(z) + 1)))


def local_perm_test(x, y, z, n_perm, rng, k=10, k_perm=5):
    obs = knn_cmi(x, y, z, k)
    nb = KDTree(z, metric='chebyshev').query(z, k=k_perm + 1)[1][:, 1:]
    null = []
    for _ in range(n_perm):
        pick = nb[np.arange(len(x)), rng.integers(0, k_perm, len(x))]
        null.append(knn_cmi(x[pick], y, z, k))
    null = np.array(null)
    return obs, float(null.mean()), float(np.quantile(null, .975)), float((1 + np.sum(null >= obs)) / (n_perm + 1))


if __name__ == '__main__':
    t0 = time.time()
    S = hf.load()
    LEAD = '--lead-accel' in sys.argv; SUF = '_leadacc' if LEAD else ''
    v = S.v_ds.to_numpy().astype(float); a = S.a_ds.to_numpy().astype(float)
    if LEAD:   # the leader's dataset acceleration (along the driving direction) at the same frame
        own = pd.DataFrame({'luid': S.uid.to_numpy(), 'frame': S.frame.to_numpy(), 'a_lead': a})
        q = pd.DataFrame({'luid': S.rec.to_numpy().astype(np.int64) * 100000 + S['lead'].to_numpy().astype(np.int64), 'frame': S.frame.to_numpy()})
        alead = np.array(q.merge(own, how='left', on=['luid', 'frame']).a_lead, dtype=float)
        alead[S['lead'].to_numpy() <= 0] = np.nan
    gap = S.gap.to_numpy().astype(float); dv = S.dv.to_numpy().astype(float); uid = S.uid.to_numpy()
    rows, nulls, knn = [], [], []
    for hz in HORIZONS:
        t = eligible(S, hz)
        if LEAD: t = t[np.isfinite(alead[t])]
        y_cont = v[t + hz] - v[t]
        print(f'h = {0.2 * hz:.1f} s: {len(t)} samples, {len(np.unique(uid[t]))} vehicles [{time.time() - t0:.0f}s]', flush=True)
        for sp in range(3):
            rng = np.random.default_rng(200 + sp)
            u = rng.permutation(np.unique(uid[t])); trv = set(u[:int(.8 * len(u))].tolist())
            tr = np.array([x in trv for x in uid[t]])
            E = dict(y=h.qedges(y_cont[tr], NY), v=h.qedges(v[t][tr], 10), a=h.qedges(a[t][tr], 10),
                     g=h.qedges(gap[t][tr], 5), d=h.qedges(dv[t][tr], 5))
            yb = h.qbins(y_cont, E['y'])
            base = ((h.qbins(v[t], E['v']) * 10 + h.qbins(a[t], E['a'])) * 5 + h.qbins(gap[t], E['g'])) * 5 + h.qbins(dv[t], E['d'])
            ui = np.unique(uid[t][tr]); inner_ids = set(np.random.default_rng(300 + sp).permutation(ui)[:int(.8 * len(ui))].tolist())
            inner = np.array([x in inner_ids for x in uid[t][tr]])
            if LEAD:
                base0 = base; base = base0 * 5 + h.qbins(alead[t], h.qedges(alead[t][tr], 5))
                lb, lf, bb, bf = heldout_gain(dict(y=yb[tr], b=base0[tr], f=base[tr]), dict(y=yb[~tr], b=base0[~tr], f=base[~tr]), inner)
                rows.append(dict(horizon_s=0.2 * hz, split=sp, added='a_lead', lag_s=0.0, ll_base=lb, ll_full=lf, gain_nats=lf - lb,
                                 gain_bits=(lf - lb) / np.log(2), beta_base=bb, beta_full=bf, n_test=int((~tr).sum())))
            feats = {}
            for k in LAGS:
                feats[('dv', k)] = h.qbins(dv[t - k], E['d'])
                feats[('gap', k)] = h.qbins(gap[t - k], E['g'])
                feats[('dv+gap', k)] = feats[('dv', k)] * 5 + feats[('gap', k)]
            feats[('dv@2,5,10', 0)] = (feats[('dv', 2)] * 5 + feats[('dv', 5)]) * 5 + feats[('dv', 10)]
            for (name, k), lag in feats.items():
                R = 125 if name.startswith('dv@') else (25 if '+' in name else 5)
                full = base * R + lag
                trd = dict(y=yb[tr], b=base[tr], f=full[tr]); ted = dict(y=yb[~tr], b=base[~tr], f=full[~tr])
                lb, lf, bb, bf = heldout_gain(trd, ted, inner)
                rows.append(dict(horizon_s=0.2 * hz, split=sp, added=name, lag_s=0.2 * k, ll_base=lb, ll_full=lf,
                                 gain_nats=lf - lb, gain_bits=(lf - lb) / np.log(2), beta_base=bb, beta_full=bf, n_test=int((~tr).sum())))
                if sp == 0 and name in ('dv', 'gap'):
                    prng = np.random.default_rng(1000 + hz * 100 + k)
                    for r in range(20):
                        lp = strat_perm(lag, base, prng)
                        full = base * R + lp
                        trn = dict(y=yb[tr], b=base[tr], f=full[tr]); ten = dict(y=yb[~tr], b=base[~tr], f=full[~tr])
                        g0 = np.log(probs(trn, ten, bb, bf)).mean() - np.log(probs(trn, ten, bb)).mean()   # betas of the real fit
                        nulls.append(dict(horizon_s=0.2 * hz, added=name, lag_s=0.2 * k, perm=r, gain_nats=g0))
            print(f'  split {sp} done [{time.time() - t0:.0f}s]', flush=True)
        # kNN CMI cross-check on 3 000 samples
        rng = np.random.default_rng(500 + hz)
        sel = rng.choice(len(t), min(3000, len(t)), replace=False)
        rk = lambda w: (rankdata(w) / len(w) - 0.5) / 0.2887 + rng.normal(0, 1e-6, len(w))
        Z = np.c_[rk(v[t][sel]), rk(a[t][sel]), rk(gap[t][sel]), rk(dv[t][sel])]
        if LEAD: Z = np.c_[Z, rk(alead[t][sel])]
        Y = rk(y_cont[sel])
        for name, k in [('dv', 2), ('dv', 5), ('dv', 10), ('gap', 5)]:
            X = rk((dv if name == 'dv' else gap)[t[sel] - k])
            obs, nm, q975, p = local_perm_test(X, Y, Z, 30, np.random.default_rng(600 + hz * 10 + k))
            knn.append(dict(horizon_s=0.2 * hz, added=name, lag_s=0.2 * k, cmi_nats=obs, null_mean=nm, null_q975=q975,
                            excess_nats=obs - nm, excess_bits=(obs - nm) / np.log(2), p=p, n=len(sel)))
        print(f'  kNN done [{time.time() - t0:.0f}s]', flush=True)
        pd.DataFrame(rows).to_csv(P.out(P.CONTROLS_RESULTS / f'highd_reaction_delay{SUF}.csv'), index=False)
        pd.DataFrame(nulls).to_csv(P.out(P.CONTROLS_RESULTS / f'highd_reaction_delay{SUF}_null.csv'), index=False)
        pd.DataFrame(knn).to_csv(P.out(P.CONTROLS_RESULTS / f'highd_reaction_delay{SUF}_knn.csv'), index=False)
    R = pd.DataFrame(rows); N = pd.DataFrame(nulls); K = pd.DataFrame(knn)
    G = R.groupby(['horizon_s', 'added', 'lag_s']).gain_bits.agg(['mean', 'std']).reset_index()
    B = N.groupby(['horizon_s', 'added', 'lag_s']).gain_nats.quantile([.025, .975]).unstack().div(np.log(2)).reset_index()
    B.columns = ['horizon_s', 'added', 'lag_s', 'null_lo_bits', 'null_hi_bits']
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300)
    print(G.merge(B, how='left').round(5).to_string(index=False)); print(K.round(5).to_string(index=False))
    print('done', round(time.time() - t0))
