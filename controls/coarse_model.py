"""Minimal physical model for apparent speed memory in highD.
Empirical: held-out gain of v_{t-1} over v_t, G(k) = LL[V(v_t^k, v_{t-1}^k)] - LL[V(v_t^k)], target v_{t+1} on 20 bins,
for k current/history bins. Same for two surrogate data sets with identical vehicles, lengths and mean speeds:
  AR1  : x_{t+1} = rho x_t + s_u e           (Markov in v; any gain is pure coarse-graining artefact)
  AR2  : x_{t+1} = p1 x_t + p2 x_{t-1} + s_u e (underdamped Langevin = Markov in (v,a); genuine 2nd order)
x = v - vehicle mean, dt = 0.2 s, pooled coefficients, vehicle-specific noise scale."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, highd_augment as h
S = pd.read_pickle(P.CONTROLS_WORK / 'highd_states.pkl')
uid = S.uid.to_numpy(); fr = S.frame.to_numpy(); v = S.v.to_numpy()
# contiguous segments
brk = np.r_[True, (uid[1:] != uid[:-1]) | (np.diff(fr) != 5)]
seg = np.cumsum(brk) - 1
mu = pd.Series(v).groupby(seg).transform('mean').to_numpy(); x = v - mu
i = np.arange(2, len(v)); ok = (seg[i] == seg[i - 2]); i = i[ok]
X2 = np.c_[x[i - 1], x[i - 2]]; y = x[i]
p = np.linalg.lstsq(X2, y, rcond=None)[0]; r2 = y - X2 @ p
rho = np.linalg.lstsq(x[i - 1][:, None], y, rcond=None)[0][0]; r1 = y - rho * x[i - 1]
s2 = pd.Series(r2).groupby(seg[i]).std().reindex(range(seg.max() + 1)).fillna(np.std(r2)).to_numpy()
s1 = pd.Series(r1).groupby(seg[i]).std().reindex(range(seg.max() + 1)).fillna(np.std(r1)).to_numpy()
print('AR2', p, 'AR1', rho, 'resid sd', r2.std(), r1.std())
# implied continuous parameters of v'' = -gamma v' - omega^2 x + noise (AR2 roots)
roots = np.roots([1, -p[0], -p[1]]); print('AR2 roots', roots, 'timescales (s)', -0.2 / np.log(np.abs(roots)))

def simulate(kind, rng):
    out = np.empty_like(x); starts = np.where(brk)[0]; ends = np.r_[starts[1:], len(x)]
    e = rng.standard_normal(len(x))
    for sg, (a, b) in enumerate(zip(starts, ends)):
        n = b - a; burn = 60; z = np.zeros(n + burn); ee = np.r_[rng.standard_normal(burn), e[a:b]]
        if kind == 'AR1':
            for t in range(1, n + burn): z[t] = rho * z[t - 1] + s1[sg] * ee[t]
        else:
            for t in range(2, n + burn): z[t] = p[0] * z[t - 1] + p[1] * z[t - 2] + s2[sg] * ee[t]
        out[a:b] = z[burn:]
    return out + mu

KS = [5, 10, 20, 40, 100, 200, 400]
def gains(vv, label, rng):
    j = np.arange(1, len(vv) - 1); j = j[(seg[j - 1] == seg[j + 1])]
    u = rng.permutation(np.unique(uid)); tr = np.isin(uid[j], u[:int(.8 * len(u))])
    vt = vv[j[tr]]; Ey = h.qedges(vt, 20); rows = []
    for k in KS:
        E = h.qedges(vt, k)
        def st(ix): return dict(y=h.qbins(vv[ix + 1], Ey), v0=h.qbins(vv[ix], E), v1=h.qbins(vv[ix - 1], E))
        T, V = st(j[tr]), st(j[~tr]); rad = dict(v0=k, v1=k)
        _, l1, _ = h.eval_model(T, V, ['v0'], rad, 20); _, l2, _ = h.eval_model(T, V, ['v0', 'v1'], rad, 20)
        rows.append(dict(data=label, k=k, ll1=l1, ll2=l2, gain=l2 - l1)); print(rows[-1], flush=True)
    return rows
rng = np.random.default_rng(5); R = []
R += gains(v, 'highD', np.random.default_rng(1))
for kind in ('AR1', 'AR2'):
    R += gains(simulate(kind, rng), kind, np.random.default_rng(1))
R = pd.DataFrame(R); R.to_csv(P.out(P.CONTROLS_RESULTS / 'coarse_model.csv'), index=False)
import json; json.dump(dict(ar2=p.tolist(), ar1=float(rho), roots=[complex(r).real for r in roots]), open(P.out(P.CONTROLS_RESULTS / 'coarse_model_params.json'), 'w'))
print(R.pivot(index='k', columns='data', values='gain').round(4))
