"""
GeoLife memory-flow robustness analysis (supplement to 02_memory_flow_metrics.py).

For every user and every (grid size, de-jitter) configuration:
  * plug-in second-order CMI  I(S_{t+1}; S_{t-1} | S_t)  = H1 - H2   (reproduces the original metric)
  * stratified permutation null: S_{t-1} shuffled among positions sharing the same S_t
    (keeps P(S_{t+1}|S_t) exactly, destroys only origin dependence) -> bias-corrected excess CMI, p-value
  * third-order CMI  I(S_{t+1}; S_{t-2} | S_{t-1}, S_t)  with the analogous null
  * held-out cross-entropy: models fitted on the training part of each trip, evaluated on the
    held-out part, interpolated back-off P0 -> P1 -> P2 -> P3 with strength beta
De-jitter: after repeat collapse, A,B,A ping-pong patterns are removed iteratively (B treated as
GPS jitter). This also removes genuine one-cell U-turns, so it is a conservative test.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, os, glob, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

CACHE = str(P.GEOLIFE_CACHE)
GRIDS = [0.0001, 0.0005, 0.001, 0.002]
N_PERM = int(os.environ.get('N_PERM', 30))
BETAS = [1.0, 3.0, 10.0]
MIN_TRIGRAMS = 50


# ---------------------------------------------------------------- states
def to_states(lat, lon, g):
    la = np.floor(lat / g).astype(np.int64)
    lo = np.floor(lon / g).astype(np.int64)
    key = la * 10_000_000 + lo
    if len(key) == 0:
        return key
    keep = np.ones(len(key), bool)
    keep[1:] = key[1:] != key[:-1]          # collapse consecutive repeats
    return key[keep]


def dejitter(seq):
    out = []
    for s in seq.tolist():
        if out and out[-1] == s:
            continue
        if len(out) >= 2 and out[-2] == s:   # A,B,A  -> A
            out.pop()
            continue
        out.append(s)
    return np.asarray(out, dtype=np.int64)


def load_user(path, g, dj):
    d = np.load(path, allow_pickle=True)
    full, train, test = [], [], []
    for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
        tla, tlo = np.asarray(tla, float), np.asarray(tlo, float)
        sla, slo = np.asarray(sla, float), np.asarray(slo, float)
        m1 = np.isfinite(tla) & np.isfinite(tlo); m2 = np.isfinite(sla) & np.isfinite(slo)
        tla, tlo, sla, slo = tla[m1], tlo[m1], sla[m2], slo[m2]
        f = to_states(np.concatenate([tla, sla]), np.concatenate([tlo, slo]), g)
        a = to_states(tla, tlo, g); b = to_states(sla, slo, g)
        if dj:
            f, a, b = dejitter(f), dejitter(a), dejitter(b)
        if len(f) >= 3: full.append(f)
        if len(a) >= 3: train.append(a)
        if len(b) >= 3: test.append(b)
    return full, train, test


def ngrams(seqs, n):
    cols = [[] for _ in range(n)]
    for s in seqs:
        if len(s) < n:
            continue
        L = len(s) - n + 1
        for j in range(n):
            cols[j].append(s[j:j + L])
    if not cols[0]:
        return [np.zeros(0, np.int64)] * n
    return [np.concatenate(c) for c in cols]


# ---------------------------------------------------------------- entropies
def _H(counts):
    N = counts.sum()
    p = counts / N
    return float(-(p * np.log2(p)).sum())


def cond_H(ctx, y, K):
    """H(Y | ctx) in bits, ctx = int64 ids in [0, M)."""
    joint = ctx * K + y
    _, cj = np.unique(joint, return_counts=True)
    _, cc = np.unique(ctx, return_counts=True)
    return _H(cj) - _H(cc)


def pair_id(a, b, K):
    return np.unique(a * K + b, return_inverse=True)[1].astype(np.int64)


def strat_shuffle(x, strata, rng, order):
    r = rng.random(len(x))
    o2 = np.lexsort((r[order], strata[order]))
    out = np.empty_like(x)
    out[order] = x[order][o2]
    return out


def cmi_with_null(hist, cond, y, K, n_perm, rng):
    """I(Y; hist | cond); cond already an id array. Returns real, null mean, p."""
    h_cond = cond_H(cond, y, K)
    h_full = cond_H(pair_id(cond, hist, K), y, K)
    real = h_cond - h_full
    order = np.argsort(cond, kind='stable')
    null = np.empty(n_perm)
    for i in range(n_perm):
        hp = strat_shuffle(hist, cond, rng, order)
        null[i] = h_cond - cond_H(pair_id(cond, hp, K), y, K)
    p = (1 + np.sum(null >= real)) / (n_perm + 1)
    return real, float(null.mean()), float(p), h_cond, h_full


# ---------------------------------------------------------------- held-out back-off
def lookup(keys_sorted, counts, q):
    idx = np.searchsorted(keys_sorted, q)
    idx = np.clip(idx, 0, len(keys_sorted) - 1)
    hit = keys_sorted[idx] == q
    return np.where(hit, counts[idx], 0).astype(float)


def heldout(train, test):
    tr = ngrams(train, 4); te = ngrams(test, 4)
    if len(te[0]) < 20 or len(tr[0]) < 50:
        return None
    allst = np.concatenate([np.concatenate(train), np.concatenate(test)])
    vocab, inv = np.unique(allst, return_inverse=True)
    K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c, d = [enc(x) for x in tr]
    A, B, C, D = [enc(x) for x in te]
    # unigram (add-1 over vocab)
    tr_states = enc(np.concatenate(train))
    uni = np.bincount(tr_states, minlength=K).astype(float)
    P0 = (uni[D] + 1.0) / (uni.sum() + K)
    # counts
    def table(*cols):
        key = cols[0].copy()
        for cc in cols[1:]:
            key = key * K + cc
        k, n = np.unique(key, return_counts=True)
        return k, n
    def keyof(*cols):
        key = cols[0].copy()
        for cc in cols[1:]:
            key = key * K + cc
        return key
    # contexts -- K^4 may overflow when K>~5e4; factorize pairs first
    pid_tr = np.unique(np.concatenate([b * K + c, B * K + C]), return_inverse=True)[1]
    pbc_tr, pbc_te = pid_tr[:len(b)].astype(np.int64), pid_tr[len(b):].astype(np.int64)
    M = int(pid_tr.max()) + 1
    k1, n1 = table(c, d);           kc1, nc1 = table(c)
    k2, n2 = table(pbc_tr, d);      kc2, nc2 = table(pbc_tr)
    c3 = np.unique(np.concatenate([a * M + pbc_tr, A * M + pbc_te]), return_inverse=True)[1].astype(np.int64)
    ctx3_tr, ctx3_te = c3[:len(a)], c3[len(a):]
    k3, n3 = np.unique(ctx3_tr * K + d, return_counts=True); kc3, nc3 = np.unique(ctx3_tr, return_counts=True)
    out = {}
    for beta in BETAS:
        P1 = (lookup(k1, n1, C * K + D) + beta * P0) / (lookup(kc1, nc1, C) + beta)
        P2 = (lookup(k2, n2, pbc_te * K + D) + beta * P1) / (lookup(kc2, nc2, pbc_te) + beta)
        P3 = (lookup(k3, n3, ctx3_te * K + D) + beta * P2) / (lookup(kc3, nc3, ctx3_te) + beta)
        l1, l2, l3 = np.log2(P1).mean(), np.log2(P2).mean(), np.log2(P3).mean()
        out[f'll1_b{beta:g}'] = l1; out[f'll2_b{beta:g}'] = l2; out[f'll3_b{beta:g}'] = l3
    out['n_test'] = len(D)
    return out


# ---------------------------------------------------------------- per user
def analyze(args):
    path, g, dj = args
    user = os.path.basename(path)[:3]
    rng = np.random.default_rng(int(user) * 1000 + int(round(g * 1e5)) + 7 * dj)
    full, train, test = load_user(path, g, dj)
    a, b, c = ngrams(full, 3)
    if len(a) < MIN_TRIGRAMS:
        return None
    allst = np.concatenate(full)
    vocab = np.unique(allst)
    K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c = enc(a), enc(b), enc(c)
    row = dict(user=user, grid=g, dejitter=int(dj), n_trigrams=len(a), n_states=K,
               two_step_return=float(np.mean(a == c)))
    real, nullm, p, h1, h2 = cmi_with_null(a, b, c, K, N_PERM, rng)
    row.update(H1=h1, H2=h2, cmi2_plugin=real, cmi2_null=nullm, cmi2_excess=real - nullm, cmi2_p=p,
               n_hist2=int(len(np.unique(a * K + b))))
    q = ngrams(full, 4)
    if len(q[0]) >= MIN_TRIGRAMS:
        A, B, C, D = [enc(x) for x in q]
        bc = pair_id(B, C, K)
        r3, n3, p3, _, _ = cmi_with_null(A, bc, D, K, N_PERM, rng)
        row.update(cmi3_plugin=r3, cmi3_null=n3, cmi3_excess=r3 - n3, cmi3_p=p3)
    ho = heldout(train, test)
    if ho:
        row.update(ho)
    return row


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(CACHE, '*.npz')))
    out_csv = sys.argv[1] if len(sys.argv) > 1 else P.out(P.CONTROLS_RESULTS / 'geolife_robust_per_user.csv')
    grids = [float(x) for x in os.environ.get('GRIDS', ','.join(map(str, GRIDS))).split(',')]
    rows = []
    t0 = time.time()
    for g in grids:
        for dj in (False, True):
            args = [(f, g, dj) for f in files]
            # biggest users first for load balance
            args.sort(key=lambda x: -os.path.getsize(x[0]))
            with Pool(2) as pool:
                res = [r for r in pool.imap_unordered(analyze, args) if r]
            rows += res
            df = pd.DataFrame(res)
            print(f'grid={g} dejitter={dj} users={len(df)} '
                  f'H1={df.H1.median():.4f} H2={df.H2.median():.4f} plug={df.cmi2_plugin.median():.4f} '
                  f'null={df.cmi2_null.median():.4f} excess={df.cmi2_excess.median():.4f} '
                  f'sig={(df.cmi2_p < .05).mean():.3f} cmi3x={df.cmi3_excess.median():.4f} '
                  f'sig3={(df.cmi3_p < .05).mean():.3f} [{time.time()-t0:.0f}s]', flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('done', time.time() - t0)
