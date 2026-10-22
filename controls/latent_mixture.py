"""P3-7  Is GeoLife path memory a trip-level hidden variable? A mixture of first-order chains on held-out data.

If each trip follows one of K first-order chains (a hidden trip type: destination, purpose, route family) the pooled
sequence looks second order even though, given the hidden variable, it is first order. The paper's held-out test
(geolife_robust.heldout) compares first- and second-order back-off models; here a third model is added:
  M_K   mixture of K first-order chains with one hidden class per trip, fitted by EM on the training parts.
        Component k: P_k(y | s) = (n_k(s, y) + beta P1(y | s)) / (n_k(s) + beta), soft counts n_k from the
        responsibilities, P1 = the pooled first-order back-off model (so K = 1 is a re-smoothed P1).
        Test prediction is sequential: the class posterior starts from the trip's own training part and is updated
        after every test step, so the model only uses the past.
  M2_K  the same classes with second-order soft counts on top: P2_k(y | r, s) = (n_k(r, s, y) + beta P_k(y | s)) / (n_k(r, s) + beta).
Gains in bits per test step, all on the paper's test steps (4-grams inside the test parts, as in geolife_robust.heldout):
  G2 = LL2 - LL1 (paper; counts from the training 4-grams), G2own = the same with all training bigrams / trigrams
  counted (the counting used by the mixture), GM_K = LL(M_K) - LL(M_1), R_K = LL(M2_K) - LL(M_K) (second order left
  given the class), GM2_K = LL(M2_K) - LL2own (mixture of second-order chains against the pooled second order).
  Control (added after a two-user pilot showed that splitting the counts over classes lowers R_K by itself):
  the fitted class sizes are shuffled over trips (seed crc32('mixture-random|...')) and GMrand_K, Rrand_K are
  computed the same way. Only R_K below Rrand_K is attributed to the hidden class.
Users: GeoLife cleaned cache, trips = (train part, test part) as in the paper; 100 m cells by default, no de-jittering.
K = 1, 2, 4, 8 (K <= training trips / 5); EM: 40 iterations, 3 random starts (seed crc32('mixture|<grid>|<user>|K|start')),
the start with the highest training log-likelihood is kept; beta = 3 and 10.

Criterion (written before the full run): if the best GM_K reaches half of G2own and R_K falls below half of G2own,
and below Rrand_K, a large part of the second-order path memory is a trip-level hidden variable (first order given
the trip type); if GM_K stays below half of G2own and R_K is no lower than Rrand_K, the second-order structure is
local (consistent with P3-1's kinematic reading) and not a trip-level class.
Each user's result is written atomically to controls/work/latent_mixture/<grid>/<user>.json and skipped on a restart.

    python controls/latent_mixture.py [0.001|0.0001]   -> results/latent_mixture_geolife_<100m|10m>.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, json, os, sys, time, zlib
import numpy as np, pandas as pd
import geolife_robust as gr

BETAS, KS, NIT, NSTART = (3.0, 10.0), (1, 2, 4, 8), 40, 3


def trips(path, g):
    """List of (train cells, test cells) per trip, run-length compressed, as in geolife_robust.load_user."""
    d = np.load(path, allow_pickle=True); out = []
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        tla, tlo, sla, slo = [np.asarray(x, float) for x in (tla, tlo, sla, slo)]
        m1 = np.isfinite(tla) & np.isfinite(tlo); m2 = np.isfinite(sla) & np.isfinite(slo)
        out.append((gr.to_states(tla[m1], tlo[m1], g), gr.to_states(sla[m2], slo[m2], g)))
    return out


def fit_user(T, seed_key):
    tr_seqs = [a for a, _ in T if len(a) >= 3]; te_seqs = [b for _, b in T if len(b) >= 3]
    base = gr.heldout(tr_seqs, te_seqs)                                          # the paper's LL1, LL2 (same test steps)
    if base is None:
        return None
    vocab = np.unique(np.concatenate([np.concatenate([a, b]) for a, b in T if len(a) + len(b)]))
    K = len(vocab); enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    # training steps: bigrams (s, y) and trigrams (r, s, y) of every training part, with the trip index
    R, S, Y, J, has_r = [], [], [], [], []
    for j, (a, _) in enumerate(T):
        if len(a) < 2:
            continue
        a = enc(a); n = len(a) - 1
        R.append(np.r_[-1, a[:-2]]); S.append(a[:-1]); Y.append(a[1:]); J.append(np.full(n, j)); has_r.append(np.r_[False, np.ones(n - 1, bool)])
    R, S, Y, J, has_r = map(np.concatenate, (R, S, Y, J, has_r))
    # test steps: every bigram of every test part (for the posterior); scored = positions >= 3 (4-grams, as the paper)
    tR, tS, tY, tJ, score = [], [], [], [], []
    for j, (_, b) in enumerate(T):
        if len(b) < 2:
            continue
        b = enc(b); n = len(b) - 1
        tR.append(np.r_[-1, b[:-2]]); tS.append(b[:-1]); tY.append(b[1:]); tJ.append(np.full(n, j)); score.append(np.arange(1, n + 1) >= 3)
    tR, tS, tY, tJ, score = map(np.concatenate, (tR, tS, tY, tJ, score))
    ntrip = len(T)
    # pooled first-order back-off P1 (add-1 unigram base over the joint vocabulary, as in geolife_robust.heldout)
    uni = np.bincount(np.concatenate([enc(a) for a, _ in T if len(a) >= 3]) if tr_seqs else Y, minlength=K).astype(float)
    P0 = lambda y: (uni[y] + 1.0) / (uni.sum() + K)
    pair_all = np.unique(np.r_[S * K + Y, tS * K + tY], return_inverse=True)[1]; pid, tpid = pair_all[:len(S)], pair_all[len(S):]
    npair = int(pair_all.max()) + 1
    trip_all = np.unique(np.r_[np.where(has_r, R * K + S, -1), np.where(tR >= 0, tR * K + tS, -1)], return_inverse=True)[1]
    c2, tc2 = trip_all[:len(S)], trip_all[len(S):]
    tri_all = np.unique(np.r_[c2 * K + Y, tc2 * K + tY], return_inverse=True)[1]; t3, tt3 = tri_all[:len(S)], tri_all[len(S):]
    nc2, nt3 = int(trip_all.max()) + 1, int(tri_all.max()) + 1
    n_tr = np.bincount(J, minlength=ntrip) > 0
    out = dict(n_train_steps=int(len(S)), n_test_steps=int(score.sum()), n_trips=int(n_tr.sum()), n_states=K)
    for beta in BETAS:
        b = f'_b{beta:g}'
        npool = np.bincount(pid, minlength=npair).astype(float); cpool = np.bincount(S, minlength=K).astype(float)
        P1 = (npool[pid] + beta * P0(Y)) / (cpool[S] + beta); tP1 = (npool[tpid] + beta * P0(tY)) / (cpool[tS] + beta)
        out['LL1' + b] = float(base[f'll1_b{beta:g}']); out['LL2' + b] = float(base[f'll2_b{beta:g}']); out['G2' + b] = out['LL2' + b] - out['LL1' + b]
        # pooled first and second order with the counting used here (all bigrams / trigrams of the training parts)
        n3p = np.bincount(t3[has_r], minlength=nt3).astype(float); c3p = np.bincount(c2[has_r], minlength=nc2).astype(float)
        tP2 = np.where(tR >= 0, (n3p[tt3] + beta * tP1) / (c3p[tc2] + beta), tP1)
        out['LL1own' + b] = float(np.log2(tP1[score]).mean()); out['LL2own' + b] = float(np.log2(tP2[score]).mean())
        out['G2own' + b] = out['LL2own' + b] - out['LL1own' + b]

        def evaluate(r_cnt, r_pri, Kc):
            """Test log-likelihoods (bits/step) of M_K and M2_K for given class weights (counts) and prior per trip."""
            w = r_cnt[J]
            nk = np.stack([np.bincount(pid, weights=w[:, k], minlength=npair) for k in range(Kc)], 1)
            ck = np.stack([np.bincount(S, weights=w[:, k], minlength=K) for k in range(Kc)], 1)
            n3 = np.stack([np.bincount(t3[has_r], weights=w[has_r, k], minlength=nt3) for k in range(Kc)], 1)
            c3 = np.stack([np.bincount(c2[has_r], weights=w[has_r, k], minlength=nc2) for k in range(Kc)], 1)
            tPk = (nk[tpid] + beta * tP1[:, None]) / (ck[tS] + beta)
            tP2k = np.where((tR >= 0)[:, None], (n3[tt3] + beta * tPk) / (c3[tc2] + beta), tPk)
            logpost = np.log(np.maximum(r_pri, 1e-300)); lm1, lm2 = np.zeros(len(tS)), np.zeros(len(tS))
            for j in np.unique(tJ):
                idx = np.where(tJ == j)[0]
                cum = np.vstack([np.zeros(Kc), np.cumsum(np.log(tPk[idx]), 0)[:-1]]) + logpost[j]   # posterior before each step
                q = np.exp(cum - cum.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
                lm1[idx] = np.log2((q * tPk[idx]).sum(1)); lm2[idx] = np.log2((q * tP2k[idx]).sum(1))
            return float(lm1[score].mean()), float(lm2[score].mean())

        for Kc in KS:
            if Kc > 1 and n_tr.sum() < 5 * Kc:
                continue
            best = None
            for st in range(NSTART if Kc > 1 else 1):
                rng = np.random.default_rng(zlib.crc32(f'mixture|{seed_key}|{Kc}|{st}'.encode()))
                r = rng.dirichlet(np.ones(Kc), size=ntrip)
                for it in range(NIT):
                    pi = (r[n_tr].sum(0) + 1.0) / (n_tr.sum() + Kc)
                    w = r[J]                                                        # steps x K
                    nk = np.stack([np.bincount(pid, weights=w[:, k], minlength=npair) for k in range(Kc)], 1)
                    ck = np.stack([np.bincount(S, weights=w[:, k], minlength=K) for k in range(Kc)], 1)
                    Pk = (nk[pid] + beta * P1[:, None]) / (ck[S] + beta)
                    ll = np.stack([np.bincount(J, weights=np.log(Pk[:, k]), minlength=ntrip) for k in range(Kc)], 1) + np.log(pi)
                    m = ll.max(1, keepdims=True); r = np.exp(ll - m); tot = r.sum(1, keepdims=True); r /= tot
                    train_ll = float((np.log(tot[:, 0]) + m[:, 0])[n_tr].sum())
                if best is None or train_ll > best[0]:
                    best = (train_ll, r.copy(), nk, ck, pi, ll)
            train_ll, r, nk, ck, pi, ll = best
            r_tr = np.where(n_tr[:, None], r, pi[None, :])                          # trips without a training part: prior
            out[f'LLM_K{Kc}{b}'], out[f'LLM2_K{Kc}{b}'] = evaluate(r_tr, r_tr, Kc)
            out[f'train_ll_K{Kc}{b}'] = train_ll
            out[f'class_entropy_K{Kc}{b}'] = float(-(r[n_tr] * np.log2(np.maximum(r[n_tr], 1e-300))).sum(1).mean())
            if Kc > 1:                                                              # control: random trip classes, same sizes
                rng = np.random.default_rng(zlib.crc32(f'mixture-random|{seed_key}|{Kc}'.encode()))
                hard = np.argmax(r_tr, 1); lab = hard[rng.permutation(ntrip)]       # the fitted class sizes, shuffled over trips
                oh = np.eye(Kc)[lab]
                out[f'LLMrand_K{Kc}{b}'], out[f'LLM2rand_K{Kc}{b}'] = evaluate(oh, oh, Kc)
        for Kc in KS:
            if f'LLM_K{Kc}{b}' in out:
                out[f'GM_K{Kc}{b}'] = out[f'LLM_K{Kc}{b}'] - out[f'LLM_K1{b}']
                out[f'R_K{Kc}{b}'] = out[f'LLM2_K{Kc}{b}'] - out[f'LLM_K{Kc}{b}']
                out[f'GM2_K{Kc}{b}'] = out[f'LLM2_K{Kc}{b}'] - out['LL2own' + b]     # mixture of second order vs pooled second order
            if f'LLMrand_K{Kc}{b}' in out:
                out[f'GMrand_K{Kc}{b}'] = out[f'LLMrand_K{Kc}{b}'] - out[f'LLM_K1{b}']
                out[f'Rrand_K{Kc}{b}'] = out[f'LLM2rand_K{Kc}{b}'] - out[f'LLMrand_K{Kc}{b}']
    return out


if __name__ == '__main__':
    g = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    lab = {0.0001: '10m', 0.001: '100m'}[g]; wd = P.CONTROLS_WORK / 'latent_mixture' / lab
    files = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz'))); t0 = time.time()
    for i, f in enumerate(files):
        user = os.path.basename(f)[:3]; out = wd / f'{user}.json'
        if out.exists():
            continue
        res = dict(user=user, **(fit_user(trips(f, g), f'{g:g}|{user}') or {}))
        wd.mkdir(parents=True, exist_ok=True); tmp = str(out) + '.tmp'; json.dump(res, open(tmp, 'w')); os.replace(tmp, out)
        print(f'{lab} {i + 1}/{len(files)} user {user} [{time.time() - t0:.0f}s]', flush=True)
    R = pd.DataFrame([json.load(open(wd / f'{os.path.basename(f)[:3]}.json')) for f in files])
    R.to_csv(P.out(P.CONTROLS_RESULTS / f'latent_mixture_geolife_{lab}.csv'), index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
    cols = [c for c in R.columns if c.startswith(('G2', 'GM_', 'R_', 'class_entropy'))]
    print(pd.DataFrame({'users': R[cols].notna().sum(), 'median': R[cols].median(), 'mean': R[cols].mean()}).round(5).to_string())
