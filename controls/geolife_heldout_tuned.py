"""P3-9  GeoLife held-out gain of the second order with the back-off strength tuned for each order.

The paper's GeoLife held-out comparison (geolife_robust.heldout; Fig. 3d, Table 7) uses one beta for every level of the
interpolated back-off chain P_k = (n(c, s) + beta P_{k-1}) / (n(c) + beta). With a fixed beta, every added level also
lowers the effective smoothing of the lower-order estimate (a pilot in latent_mixture.py showed that a first-order
model smoothed twice towards itself beats the once-smoothed one by up to 0.18 bits per step at 100 m, beta = 10),
so part of the 'second-order gain' may be a smoothing effect. The paper's sign of this: the gain grows with beta
(+0.065 -> +0.118 bits at 100 m, +0.000 -> +0.017 at 10 m). Here each order gets its own strengths, chosen on an inner
split of the training data (as the paper already does for the taxi data):
  inner split: the last 20 % of every training part (at least 3 cells) is the validation part;
  order 1: beta1 from the grid; order 2: (beta1, beta2) jointly; order 3: beta3 given the best (beta1, beta2);
  grid 0.1, 0.3, 1, 3, 10, 30, 100; the chosen strengths are refitted on the whole training part and scored on the
  test 4-grams, i.e. the same steps as geolife_robust.heldout.
Reported per user (bits per test step): LL1, LL2, LL3 and the gains G2 = LL2 - LL1, G3 = LL3 - LL2 for the tuned
strengths and, as a check against the paper, for beta = 3 and 10 at every level.
Deterministic (no random numbers). Each user is written atomically to controls/work/heldout_tuned/<grid>/<user>.json.

    python controls/geolife_heldout_tuned.py   -> results/geolife_heldout_tuned.csv (10 m and 100 m)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, itertools, json, os, time
import numpy as np, pandas as pd
import geolife_robust as gr

GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
WORK = P.CONTROLS_WORK / 'heldout_tuned'


def counts(train, test):
    """Count lookups for the test 4-grams (as geolife_robust.heldout). Returns a dict of arrays or None."""
    tr = gr.ngrams(train, 4); te = gr.ngrams(test, 4)
    if len(te[0]) < 20 or len(tr[0]) < 50:
        return None
    vocab = np.unique(np.concatenate([np.concatenate(train), np.concatenate(test)])); K = len(vocab)
    enc = lambda x: np.searchsorted(vocab, x).astype(np.int64)
    a, b, c, d = [enc(x) for x in tr]; A, B, C, D = [enc(x) for x in te]
    uni = np.bincount(enc(np.concatenate(train)), minlength=K).astype(float)
    pid = np.unique(np.concatenate([b * K + c, B * K + C]), return_inverse=True)[1].astype(np.int64)
    p_tr, p_te = pid[:len(b)], pid[len(b):]; M = int(pid.max()) + 1
    c3 = np.unique(np.concatenate([a * M + p_tr, A * M + p_te]), return_inverse=True)[1].astype(np.int64)
    x_tr, x_te = c3[:len(a)], c3[len(a):]
    tab = lambda key: np.unique(key, return_counts=True)
    L = gr.lookup
    k1, n1 = tab(c * K + d); kc1, nc1 = tab(c)
    k2, n2 = tab(p_tr * K + d); kc2, nc2 = tab(p_tr)
    k3, n3 = tab(x_tr * K + d); kc3, nc3 = tab(x_tr)
    return dict(P0=(uni[D] + 1.0) / (uni.sum() + K),
                n1=L(k1, n1, C * K + D), c1=L(kc1, nc1, C), n2=L(k2, n2, p_te * K + D), c2=L(kc2, nc2, p_te),
                n3=L(k3, n3, x_te * K + D), c3=L(kc3, nc3, x_te))


def lls(T, b1, b2=None, b3=None):
    P1 = (T['n1'] + b1 * T['P0']) / (T['c1'] + b1)
    out = [float(np.log2(P1).mean())]
    if b2 is not None:
        P2 = (T['n2'] + b2 * P1) / (T['c2'] + b2); out.append(float(np.log2(P2).mean()))
        if b3 is not None:
            P3 = (T['n3'] + b3 * P2) / (T['c3'] + b3); out.append(float(np.log2(P3).mean()))
    return out


def user(path, g):
    full, train, test = gr.load_user(path, g, False)
    T = counts(train, test)
    if T is None:
        return {}
    itr, iva = [], []
    for s in train:
        cut = int(round(len(s) * 0.8))
        if cut >= 3 and len(s) - cut >= 3:
            itr.append(s[:cut]); iva.append(s[cut:])
        else:
            itr.append(s)
    V = counts(itr, iva) if iva else None
    out = dict(n_test=int(len(T['P0'])), n_val=int(len(V['P0'])) if V else 0)
    for beta in (3.0, 10.0):
        l1, l2, l3 = lls(T, beta, beta, beta)
        out.update({f'LL1_b{beta:g}': l1, f'G2_b{beta:g}': l2 - l1, f'G3_b{beta:g}': l3 - l2})
    if V is None:
        return out
    b1 = max(GRID, key=lambda x: lls(V, x)[0])
    b12 = max(itertools.product(GRID, GRID), key=lambda x: lls(V, x[0], x[1])[1])
    b3 = max(GRID, key=lambda x: lls(V, b12[0], b12[1], x)[2])
    l1 = lls(T, b1)[0]; _, l2, l3 = lls(T, b12[0], b12[1], b3)
    out.update(beta1_order1=b1, beta1_order2=b12[0], beta2_order2=b12[1], beta3_order3=b3,
               LL1_tuned=l1, LL2_tuned=l2, LL3_tuned=l3, G2_tuned=l2 - l1, G3_tuned=l3 - l2)
    return out


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz'))); rows = []; t0 = time.time()
    for g, lab in ((0.0001, '10m'), (0.001, '100m')):
        for i, f in enumerate(files):
            u = os.path.basename(f)[:3]; out = WORK / lab / f'{u}.json'
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True); tmp = str(out) + '.tmp'
                json.dump(dict(user=u, grid=lab, **user(f, g)), open(tmp, 'w')); os.replace(tmp, out)
                print(f'{lab} {i + 1}/{len(files)} user {u} [{time.time() - t0:.0f}s]', flush=True)
            rows.append(json.load(open(out)))
    R = pd.DataFrame(rows)
    out = P.out(P.CONTROLS_RESULTS / 'geolife_heldout_tuned.csv'); tmp = str(out) + '.tmp'; R.to_csv(tmp, index=False); os.replace(tmp, out)
    pd.set_option('display.width', 220)
    cols = ['G2_b3', 'G2_b10', 'G2_tuned', 'G3_b3', 'G3_b10', 'G3_tuned', 'beta1_order1', 'beta1_order2', 'beta2_order2']
    S = R.groupby('grid')[cols].median().T
    S.loc['share_G2_tuned>0'] = R.groupby('grid').G2_tuned.apply(lambda x: (x.dropna() > 0).mean())
    S.loc['users'] = R.groupby('grid').G2_tuned.count()
    print(S.round(4).to_string())
