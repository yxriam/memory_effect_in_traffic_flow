"""P0-5  Positive control for the history ablation: do the LSTM and the Transformer pick up a planted long-range dependence?

In the paper the LSTM that sees the last 4 symbols (2.84 bits/step) is as good as or better than the one with a
128-step context (2.90), and we conclude that the usable memory stops at 3-4 steps. That conclusion needs the
128-step models to be able to use a long dependence when there is one. Test: synthetic taxi-like sequences in which
such a dependence is planted with a known size.

Data (python prediction/positive_control.py make):
  - the pieces of data/taxi_traj20.npz (same first cell, length, unit and train/val/test split); 20 x 20 per-trajectory grid.
  - generator: second-order chain fitted on all pieces of taxi_traj20 (interpolated back-off to the first-order chain,
    beta = 3), successors restricted to the observed successors of the current cell, so the sequences stay
    run-length compressed.
  - planted dependence: each cell c carries a fixed random bit z(c) (seed 2026); the successors of each cell i are
    split once into two halves A_i / B_i of nearly equal first-order mass. When the target x_u is drawn (u - 1 >= L),
    the mass m_A of A_i under P2(. | x_{u-2}, x_{u-1}) is raised (z(x_{u-1-L}) = 1) or lowered (z = 0) by
    delta = gamma * min(m_A, 1 - m_A), proportionally inside each half. Averaged over z the step distribution is P2,
    so the short-range statistics are unchanged; the information planted in the step is
    I_u = h(m_A) - [h(m_A + delta) + h(m_A - delta)] / 2 bits (h: binary entropy). A model sees x_{u-1-L} only if its
    history reaches L + 1 symbols back (k = 4 never does for L >= 8; the 128-step models always do).
  - gamma is set (bisection on the contexts of the real test pieces) so that the mean of I_u over all scored test
    targets is 0.1, 0.2 or 0.3 bits/step. Datasets: L = 8, 16, 32 at 0.2 bits/step and L = 16 at 0.1 and 0.3 bits/step
    -> data/pc_L<L>_I<10 x bits>.npz + .json (generator cross-entropies on the test part, md5 of the tokens).
  - in the .json: H_full = generator's cross-entropy knowing z; H_short = with z averaged 50/50 (planted = H_short - H_full);
    H_ctx2 / H_ctx4 = with z predicted from the last 2 / 4 symbols (estimated on the training part). The expected
    k=4 vs k=128 gap is min(H_short, H_ctx2, H_ctx4) - H_full: what a 4-step model misses even if it predicts z from
    its context as well as these estimates do.
CPU check (python prediction/positive_control.py ngram): Kneser-Ney orders 2-4 (blind to the planted dependence)
and a skip model P(x_u | x_{u-2}, x_{u-1}, x_{u-1-L}) (interpolated back-off to KN order 3, beta tuned on val), on the
synthetic sets and on the real taxi_traj20 -> results/ngram/positive_control_ngram.csv.
GPU part: prediction/scripts/run_positive_control.bat (neural_hist.py k = 4 and neural_ch5.py ctx = 128, LSTM and
Transformer, seeds 0-2 -> results/neural/positive_control/), then
  python prediction/positive_control.py summary   -> results/ngram/positive_control_summary.csv, figures/output/fig_positive_control.*
Bits per step throughout.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.PREDICTION))
import hashlib, json, sys, time
import numpy as np, pandas as pd

BETA = 3.0
Z_SEED = 2026
SETS = [(16, 0.2), (8, 0.2), (32, 0.2), (16, 0.1), (16, 0.3)]           # (L, planted bits/step), in run order
OUT_NEURAL = P.PRED_NEURAL / 'positive_control'


def name_of(L, I):
    return f'pc_L{L}_I{int(round(10 * I))}'


def hb(p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_taxi():
    d = np.load(P.PRED_DATA / 'taxi_traj20.npz')
    return d['tokens'].astype(np.int64), d['offsets'].astype(np.int64), d['part'], d['unit'], int(d['vocab_size'])


class Chain:
    def __init__(self, tok, off, V):
        self.V = V
        P_ = [tok[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        a = np.concatenate([p[:-1] for p in P_ if len(p) >= 2]); b = np.concatenate([p[1:] for p in P_ if len(p) >= 2])
        C1 = np.zeros((V, V)); np.add.at(C1, (a, b), 1)
        self.succ = [np.flatnonzero(C1[i]) for i in range(V)]
        self.p1 = [C1[i, s] / C1[i, s].sum() if len(s) else None for i, s in enumerate(self.succ)]
        uni = np.bincount(tok, minlength=V).astype(float)
        self.fallback = uni / uni.sum()                                   # for cells never followed by anything
        x = np.concatenate([p[:-2] for p in P_ if len(p) >= 3]); y = np.concatenate([p[1:-1] for p in P_ if len(p) >= 3])
        z = np.concatenate([p[2:] for p in P_ if len(p) >= 3])
        u, cnt = np.unique((x * V + y) * V + z, return_counts=True)
        ctx, nxt = u // V, u % V
        cu, st = np.unique(ctx, return_index=True); en = np.r_[st[1:], len(ctx)]
        self.c2 = {int(c): (nxt[s:e], cnt[s:e].astype(float)) for c, s, e in zip(cu, st, en)}
        self.inA = []                                                     # halves of nearly equal first-order mass
        for i in range(V):
            s = self.succ[i]; m = np.zeros(len(s), bool)
            if len(s):
                mA = mB = 0.0
                for j in np.argsort(-self.p1[i], kind='stable'):
                    if mA <= mB: m[j] = True; mA += self.p1[i][j]
                    else: mB += self.p1[i][j]
            self.inA.append(m)
        self.memo = {}

    def p2(self, a, b):
        """(successors, P2(. | a, b)); a = -1 when there is no second-order context."""
        key = a * self.V + b
        r = self.memo.get(key)
        if r is not None: return r
        s = self.succ[b]
        if not len(s):
            r = (np.arange(self.V), self.fallback)
        elif a < 0 or key not in self.c2:
            r = (s, self.p1[b])
        else:
            nx, cn = self.c2[key]; q = BETA * self.p1[b]; q[np.searchsorted(s, nx)] += cn
            r = (s, q / (cn.sum() + BETA))
        self.memo[key] = r
        return r

    def mass_A(self, a, b):
        s, p = self.p2(a, b)
        if not len(self.succ[b]): return 0.0
        return float(p[self.inA[b]].sum())


def tilt(p, mask, zbit, gamma):
    mA = float(p[mask].sum())
    d = gamma * min(mA, 1 - mA)
    if d <= 0: return p, p, p, 0.0
    q1 = p.copy(); q1[mask] *= (mA + d) / mA; q1[~mask] *= (1 - mA - d) / (1 - mA)
    q0 = p.copy(); q0[mask] *= (mA - d) / mA; q0[~mask] *= (1 - mA + d) / (1 - mA)
    return (q1 if zbit else q0), q1, q0, float(hb(mA) - 0.5 * (hb(mA + d) + hb(mA - d)))


def calibrate(ch, tok, off, part, L, target):
    """gamma such that the mean planted information over all scored targets of the real test pieces equals target."""
    mA, n_targets = [], 0
    for i in np.where(part == 2)[0]:
        s = tok[off[i]:off[i + 1]]; n_targets += max(len(s) - 1, 0)
        for u in range(L + 1, len(s)):
            mA.append(ch.mass_A(int(s[u - 2]), int(s[u - 1])))
    mA = np.array(mA); ok = (mA > 0) & (mA < 1); mA = mA[ok]
    f = lambda g: float((hb(mA) - 0.5 * (hb(mA + g * np.minimum(mA, 1 - mA)) + hb(mA - g * np.minimum(mA, 1 - mA)))).sum() / n_targets)
    if f(1.0) < target: return 1.0, f(1.0)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2, f((lo + hi) / 2)


def generate(ch, tok, off, L, gamma, seed):
    rng = np.random.default_rng(seed); zb = np.random.default_rng(Z_SEED).integers(0, 2, ch.V)
    out = np.empty_like(tok); rec = []                                    # per target: piece, u, p(z=1), p(z=0), p_true, I
    for i in range(len(off) - 1):
        a0, a1 = off[i], off[i + 1]; n = a1 - a0
        x = np.empty(n, np.int64); x[0] = tok[a0]
        for u in range(1, n):
            s, p = ch.p2(int(x[u - 2]) if u >= 2 else -1, int(x[u - 1]))
            I, q1, q0 = 0.0, p, p
            if u - 1 >= L and len(ch.succ[x[u - 1]]):
                p, q1, q0, I = tilt(p, ch.inA[x[u - 1]], zb[x[u - 1 - L]], gamma)
            j = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum(), side='right')), len(s) - 1)
            x[u] = s[j]
            rec.append((i, u, q1[j], q0[j], p[j], I, zb[x[u - 1 - L]] if u - 1 >= L else -1))
        out[a0:a1] = x
    R = pd.DataFrame(rec, columns=['piece', 'u', 'p_z1', 'p_z0', 'p_true', 'I', 'z'])
    return out, R


def z_given_context(out, off, part, R, k):
    """P(z = 1 | last k symbols), estimated on the training part (Beta(1,1) prior, back-off from k to k-1 ... to 0
    with weight 3); evaluated for every target of R."""
    ctx = np.zeros((len(R), k), np.int64) - 1
    starts = off[R.piece.to_numpy()]; u = R.u.to_numpy()
    for j in range(1, k + 1):
        ok = u - j >= 0; ctx[ok, k - j] = out[starts[ok] + u[ok] - j]
    tr = (part[R.piece.to_numpy()] == 0) & (R.z.to_numpy() >= 0); zz = R.z.to_numpy().astype(float)
    pr = np.full(len(R), (zz[tr].sum() + 1) / (tr.sum() + 2))
    V1 = int(out.max()) + 2                                               # -1 (before the piece start) -> 0
    for j in range(1, k + 1):
        key = np.zeros(len(R), np.int64)
        for m in range(k - j, k): key = key * V1 + (ctx[:, m] + 1)
        uk, inv = np.unique(key, return_inverse=True)
        n1 = np.bincount(inv[tr], weights=zz[tr], minlength=len(uk)); n = np.bincount(inv[tr], minlength=len(uk))
        pr = (n1[inv] + 3 * pr) / (n[inv] + 3)
    return pr


def make():
    tok, off, part, unit, V = load_taxi()
    ch = Chain(tok, off, V)
    for L, I in SETS:
        t0 = time.time(); nm = name_of(L, I)
        gamma, I_cal = calibrate(ch, tok, off, part, L, I)
        out, R = generate(ch, tok, off, L, gamma, 1000 * L + int(round(10 * I)))
        np.savez_compressed(P.PRED_DATA / f'{nm}.npz', tokens=out.astype(np.int32), offsets=off, unit=unit, part=part, vocab_size=V,
                            keys=np.arange(V))
        meta = dict(dataset=nm, source='taxi_traj20 pieces; second-order chain + planted dependence on the cell L+1 steps back',
                    L=L, target_bits=I, gamma=gamma, planted_bits_calibration=I_cal, beta=BETA, z_seed=Z_SEED,
                    seed=1000 * L + int(round(10 * I)), md5_tokens=hashlib.md5(out.astype(np.int32).tobytes()).hexdigest())
        te = part[R.piece.to_numpy()] == 2
        p_short = 0.5 * (R.p_z1 + R.p_z0)
        for k in (2, 4):
            pz = z_given_context(out, off, part, R, k)
            meta[f'H_ctx{k}'] = float(-np.log2(pz * R.p_z1 + (1 - pz) * R.p_z0)[te].mean())
        meta.update(n_test=int(te.sum()), H_full=float(-np.log2(R.p_true[te]).mean()), H_short=float(-np.log2(p_short[te]).mean()),
                    planted_bits=float(R.I[te].mean()), share_tilted_steps=float((R.I[te] > 0).mean()))
        meta['expected_gap_k4_vs_k128'] = min(meta['H_short'], meta['H_ctx2'], meta['H_ctx4']) - meta['H_full']
        json.dump(meta, open(P.PRED_DATA / f'{nm}.json', 'w'), indent=1)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()}, f'[{time.time() - t0:.0f}s]', flush=True)


def ngram():
    import lm
    rows = []
    for nm, L in [(name_of(L, I), L) for L, I in SETS] + [('taxi_traj20', L) for L in (8, 16, 32)]:
        d = np.load(P.PRED_DATA / f'{nm}.npz'); tok, off, part = d['tokens'].astype(np.int64), d['offsets'], d['part']; V = int(d['vocab_size'])
        seqs = {p: [tok[off[i]:off[i + 1]].tolist() for i in np.where(part == p)[0] if off[i + 1] - off[i] >= 1] for p in (0, 1, 2)}
        full = lm.NGram(4, V, 'kn').fit(seqs[0])
        r = dict(dataset=nm, L=L)
        if nm != 'taxi_traj20' or L == 8:
            for N in (2, 3, 4):
                r[f'kn{N}'] = lm.evaluate(lm.view(full, N, 'kn'), seqs[2], topk=False)['bits']
        kn3 = lm.view(full, 3, 'kn')
        cnt = {}                                                        # (x_{u-1}, x_{u-1-L}) -> Counter of x_u
        for s in seqs[0]:
            for u in range(L + 1, len(s)):
                cnt.setdefault((s[u - 1], s[u - 1 - L]), {}).setdefault(s[u], 0); cnt[(s[u - 1], s[u - 1 - L])][s[u]] += 1
        tot = {k: sum(v.values()) for k, v in cnt.items()}

        def score(seqs_, beta):
            ll = ll0 = 0.0; n = 0
            for s in seqs_:
                for u in range(1, len(s)):
                    p0 = kn3.prob(tuple(s[max(0, u - 2):u]), s[u]); p = p0
                    if u >= L + 1:
                        k = (s[u - 1], s[u - 1 - L]); c = cnt.get(k)
                        if c is not None: p = (c.get(s[u], 0) + beta * p0) / (tot[k] + beta)
                    ll += np.log2(p); ll0 += np.log2(p0); n += 1
            return -ll / n, -ll0 / n
        val = {b: score(seqs[1], b)[0] for b in (1, 3, 10, 30, 100, 300)}
        b = min(val, key=val.get); h, h0 = score(seqs[2], b)
        r.update(skip_beta=b, kn3_test=h0, skip_test=h, skip_gain=h0 - h)
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(P.out(P.PRED_NGRAM / 'positive_control_ngram.csv'), index=False)


def summary():
    rows = []
    for f in sorted(OUT_NEURAL.glob('*.json')):
        d = json.load(open(f))
        hist = d.get('hist')
        rows.append(dict(dataset=d['dataset'], model=d.get('model', 'lstm'), context=hist if hist is not None else d['args']['ctx'],
                         script='neural_hist' if hist is not None else 'neural_ch5', seed=d.get('seed', d['args']['seed']),
                         test_bits=d['test']['bits'], val_bits=d['val_bits'], epochs=len(d['history'])))
    R = pd.DataFrame(rows)
    meta = {nm: json.load(open(P.PRED_DATA / f'{nm}.json')) for nm in R.dataset.unique()}
    S = []
    for (ds, model), g in R.groupby(['dataset', 'model']):
        k4 = g[g.context == 4].set_index('seed').test_bits; k128 = g[g.context == 128].set_index('seed').test_bits
        both = k4.index.intersection(k128.index); gap = (k4[both] - k128[both])
        m = meta[ds]
        S.append(dict(dataset=ds, L=m['L'], planted_bits=m['planted_bits'], expected_gap=m['expected_gap_k4_vs_k128'], model=model,
                      H_full=m['H_full'], H_ctx4=m['H_ctx4'], k4_bits=k4.mean(), k128_bits=k128.mean(), seeds=len(both),
                      gap_mean=gap.mean(), gap_sd=gap.std(ddof=1) if len(both) > 1 else np.nan, gap_min=gap.min(),
                      recovery=gap.mean() / m['expected_gap_k4_vs_k128']))
    S = pd.DataFrame(S).sort_values(['model', 'L', 'planted_bits'])
    S.to_csv(P.out(P.PRED_NGRAM / 'positive_control_summary.csv'), index=False)
    R.sort_values(['dataset', 'model', 'context', 'seed']).to_csv(P.out(P.PRED_NGRAM / 'positive_control_runs.csv'), index=False)
    pd.set_option('display.width', 200); print(S.round(4).to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    at02 = np.isclose(S.planted_bits.round(1), 0.2)
    for model, mk in (('lstm', 'o'), ('transformer', 's')):
        a = S[(S.model == model) & at02].sort_values('L')
        ax[0].errorbar(a.L, a.gap_mean, a.gap_sd, marker=mk, label=model, capsize=3)
        b = S[(S.model == model) & (S.L == 16)].sort_values('planted_bits')
        ax[1].errorbar(b.expected_gap, b.gap_mean, b.gap_sd, marker=mk, label=model, capsize=3)
    a = S[at02].drop_duplicates('dataset').sort_values('L'); ax[0].plot(a.L, a.expected_gap, 'k--', label='available (generator)')
    lim = [0, max(0.35, S.expected_gap.max() * 1.1)]; ax[1].plot(lim, lim, 'k--', lw=1)
    ax[0].set_xscale('log', base=2); ax[0].set_xticks([8, 16, 32]); ax[0].set_xticklabels(['8', '16', '32']); ax[0].set_xlabel('lag L of the planted dependence'); ax[0].set_ylabel('bits(k=4) - bits(128)  [bits/step]')
    ax[0].axhline(0, color='grey', lw=.5); ax[0].legend(fontsize=8); ax[0].set_title('~0.2 bits/step planted')
    ax[1].set_xlabel('available gain (generator) [bits/step]'); ax[1].set_ylabel('recovered gain [bits/step]'); ax[1].set_title('L = 16')
    fig.tight_layout(); od = P.PREDICTION / 'figures' / 'output'; od.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(od / f'fig_positive_control.{ext}', dpi=200)


if __name__ == '__main__':
    {'make': make, 'ngram': ngram, 'summary': summary}[sys.argv[1]]()
