"""Memory in airline O-D itineraries (BTS DB1B), measured with the same controls as the trajectory datasets.

For every itinerary (one line of data/sequences_split/<YYYYMM>/<YYYYMM>_<n>.txt, airports in travel order)
each flight leg is one step. On the pooled steps of a quarter the script reports

  I2      I(S_{t+1}; S_{t-1} | S_t)                       plug-in, stratified-permutation null, excess, p
  I3      I(S_{t+1}; S_{t-2} | S_{t-1}, S_t)              same, on steps with >= 2 previous airports
  IO|3    I(S_{t+1}; S_0 | S_{t-2}, S_{t-1}, S_t)         origin beyond the last three airports (t >= 3)
  held-out cross-entropy of interpolated back-off chains of order 1-4 (beta tuned on a validation split),
  of a "return-only" model that uses the previous airport only through P(S_{t+1} = S_{t-1}), and of a
  first-order chain on (trip origin, current airport); ce_curve: orders 1-4 and (origin, current) on the same steps.

States: airport (IATA code) or city (airports sharing a city in airport_dict.json; consecutive repeats merged).
Results are cached per (quarter, state, quantity) in --cache, so an interrupted run resumes where it stopped.

Data location: common/paths.py key `airports_seq` (local_paths.json or MEMFX_AIRPORTS_SEQ).
Seeds are fixed per (quarter, state, quantity) with zlib.crc32, so an interrupted and resumed run gives the same numbers.
The cache is written to a .tmp file and renamed; rerunning the same command resumes.

    cd airports
    python airport_memory.py                        # 202409 and 202412, airport and city states, 20 permutations
"""
import argparse, gc, glob, json, os, platform, re, sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import AIRPORTS_SEQ  # noqa: E402
import numpy as np
import pandas as pd

LOG2 = np.log(2.0)


def load(seq_dir, quarter, state, city_of):
    """Itineraries grouped by length: {n: (array m x n of state indices, itinerary ids)}."""
    seqs = []
    for f in sorted(glob.glob(os.path.join(seq_dir, quarter, f"{quarter}_*.txt"))):
        with open(f) as fh:
            for line in fh:
                s = line.split()
                if state == "city":
                    s = [city_of.get(a, a) for a in s]
                    s = [x for i, x in enumerate(s) if i == 0 or x != s[i - 1]]
                if len(s) >= 2:
                    seqs.append(s)
    vocab = {a: i for i, a in enumerate(sorted({a for s in seqs for a in s}))}
    by_len, ids = {}, {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append([vocab[a] for a in s])
        ids.setdefault(len(s), []).append(i)
    groups = {n: (np.array(v, dtype=np.int64), np.array(ids[n], dtype=np.int64)) for n, v in by_len.items()}
    return groups, len(vocab), len(seqs)


def windows(groups, k, min_t):
    """Rows (s_{t-k+1}, ..., s_t, s_{t+1}, origin, itinerary id) for steps t >= min_t."""
    parts = []
    for n, (arr, ids) in groups.items():
        for t in range(max(k - 1, min_t), n - 1):
            parts.append(np.column_stack([arr[:, t - k + 1:t + 2], arr[:, 0], ids]))
    return np.concatenate(parts) if parts else np.zeros((0, k + 3), dtype=np.int64)


def entropy_of_keys(k):
    _, c = np.unique(k, return_counts=True)
    n = c.sum()
    return float((np.log(n) - (c * np.log(c)).sum() / n) / LOG2)


def cond_entropy(y, x_keys, V):
    """Plug-in H(Y | X) in bits = H(X, Y) - H(X)."""
    return entropy_of_keys(x_keys * (V + 1) + y) - entropy_of_keys(x_keys)


def key(cols, V):
    k = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        k = k * (V + 1) + c
    return k


def cmi_with_null(y, cond_cols, extra, V, perms, rng):
    """I(Y; extra | cond) plug-in, stratified-permutation null (extra shuffled within cond strata)."""
    kc = key(cond_cols, V)
    h_c = cond_entropy(y, kc, V)
    obs = h_c - cond_entropy(y, key(cond_cols + [extra], V), V)
    order = np.argsort(kc, kind="stable")
    null = []
    for _ in range(perms):
        perm = np.lexsort((rng.random(len(kc)), kc))
        e = np.empty_like(extra)
        e[order] = extra[perm]
        null.append(h_c - cond_entropy(y, key(cond_cols + [e], V), V))
    null = np.array(null)
    return dict(H_cond=h_c, plugin=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                excess=float(obs - null.mean()), p=float((1 + (null >= obs).sum()) / (perms + 1)))


# ---------------- interpolated back-off chains -------------------------------------------------
def split_ids(n_itin, seed=0):
    h = np.random.default_rng(seed).integers(0, 10, n_itin)          # fixed seed: reproducible split
    return np.where(h < 7, 0, np.where(h < 8, 1, 2))          # 0 train, 1 validation, 2 test


def counts(rows, ctx_len):
    """Counts of (context of the last ctx_len airports, next) from window rows (k = 4 columns of history)."""
    hist = rows[:, 4 - ctx_len:4] if ctx_len else np.zeros((len(rows), 0), dtype=np.int64)
    return hist, rows[:, 4]


class Backoff:
    def __init__(self, train4, V):
        self.V = V
        y = train4[:, 4]
        self.n0 = np.bincount(y, minlength=V).astype(float)
        self.tables = {}
        for L in range(1, 5):
            h = train4[:, 4 - L:4]
            kc = key([h[:, j] for j in range(L)], V)
            df = pd.DataFrame({"c": kc, "y": y})
            self.tables[L] = (df.groupby(["c", "y"]).size(), df.groupby("c").size())

    def prob(self, rows, order, betas):
        """P(next | last `order` airports) for every row, recursively interpolated down to add-one unigram."""
        y = rows[:, 4]
        p = (self.n0[y] + 1.0) / (self.n0.sum() + self.V)
        for L in range(1, order + 1):
            h = rows[:, 4 - L:4]
            kc = key([h[:, j] for j in range(L)], self.V)
            cy, c = self.tables[L]
            ncy = cy.reindex(pd.MultiIndex.from_arrays([kc, y])).fillna(0).to_numpy()
            nc = c.reindex(kc).fillna(0).to_numpy()
            b = betas[L - 1]
            p = (ncy + b * p) / (nc + b)
        return p


def ce(p):
    return float(-np.mean(np.log(p)) / LOG2)


def tune(model, rows, order, betas_fixed, grid):
    best = None
    for b in grid:
        bs = list(betas_fixed) + [b]
        v = ce(model.prob(rows, order, bs))
        if best is None or v < best[0]:
            best = (v, b)
    return best[1]


def return_model_ce(train, test, V, beta1, kappa=10.0):
    """Previous airport used only through 'return to it': P = lam_b 1[c=a] + (1-lam_b) Q(c|b, c != a)."""
    a, b, y = train[:, 2], train[:, 3], train[:, 4]
    ret = (y == a)
    nb = np.bincount(b, minlength=V).astype(float)
    rb = np.bincount(b[ret], minlength=V).astype(float)
    lam_bar = ret.mean()
    lam = (rb + kappa * lam_bar) / (nb + kappa)
    nr = ~ret
    n0 = np.bincount(y[nr], minlength=V).astype(float)
    p0 = (n0 + 1.0) / (n0.sum() + V)
    df = pd.DataFrame({"b": b[nr], "y": y[nr]})
    cby = df.groupby(["b", "y"]).size()
    cb = np.bincount(b[nr], minlength=V).astype(float)

    def q(bb, yy):
        ncy = cby.reindex(pd.MultiIndex.from_arrays([bb, yy])).fillna(0).to_numpy()
        return (ncy + beta1 * p0[yy]) / (cb[bb] + beta1)

    ta, tb, ty = test[:, 2], test[:, 3], test[:, 4]
    p_ret = lam[tb]
    q_y = q(tb, ty)
    q_a = q(tb, ta)
    p = np.where(ty == ta, p_ret, (1 - p_ret) * q_y / np.clip(1 - q_a, 1e-12, None))
    return ce(np.clip(p, 1e-12, None))


# ---------------- main ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-dir", default=str(AIRPORTS_SEQ))
    ap.add_argument("--quarters", nargs="+", default=["202409", "202412"])
    ap.add_argument("--states", nargs="+", default=["airport", "city"])
    ap.add_argument("--dict", default=str(Path(__file__).resolve().parent / "airport_dict.json"))
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "airport_memory.csv"))
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parent / "results" / "airport_memory_cache.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    def save():
        tmp = args.cache + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cache, fh, indent=1)
        os.replace(tmp, args.cache)

    def seeded(quarter, state, what):
        return np.random.default_rng(zlib.crc32(f"{quarter}|{state}|{what}".encode()))

    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, platform=platform.platform())

    city_of = json.load(open(args.dict, encoding="utf-8"))
    grid = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    for quarter in args.quarters:
        for state in args.states:
            tag = f"{quarter}|{state}"
            res = cache.get(tag, {})
            if res.get("done") and "ce_curve" in res:
                print("cached", tag); continue
            t0 = time.time()
            groups, V, n_itin = load(args.seq_dir, quarter, state, city_of)
            lens = np.concatenate([np.full(len(g[1]), n) for n, g in groups.items()])
            rt = np.concatenate([g[0][:, 0] == g[0][:, -1] for g in groups.values()])
            res.update(params=dict(perms=args.perms, beta_grid=grid, split="70/10/20 by itinerary, numpy seed 0",
                                   seeds="zlib.crc32(quarter|state|quantity)", seq_dir=args.seq_dir), env=env)
            res.update(quarter=quarter, state=state, itineraries=int(n_itin), states=int(V),
                       mean_len=float(lens.mean()), share_len3plus=float((lens >= 3).mean()),
                       share_round_trip=float(rt.mean()))
            if "I2" not in res:
                w2 = windows(groups, 2, 1)                   # columns: a, b, y, origin, id
                res["steps_I2"] = int(len(w2))
                res["I2"] = cmi_with_null(w2[:, 2], [w2[:, 1]], w2[:, 0], V, args.perms, seeded(quarter, state, "I2"))
                a, b, y = w2[:, 0], w2[:, 1], w2[:, 2]
                res["return_rate"] = float((y == a).mean())
                # return rate implied by a first-order chain: sum_{a,b} p(a,b) P1(a|b)
                df = pd.DataFrame({"b": b, "y": y}); p1 = df.groupby(["b", "y"]).size() / df.groupby("b").size()
                res["return_rate_first_order"] = float(p1.reindex(pd.MultiIndex.from_arrays([b, a])).fillna(0).mean())
                cache[tag] = res; save()
                del w2, a, b, y, df, p1; gc.collect()
            if "I3" not in res:
                w3 = windows(groups, 3, 2)                   # z, a, b, y, origin, id
                res["steps_I3"] = int(len(w3))
                res["I3"] = cmi_with_null(w3[:, 3], [w3[:, 1], w3[:, 2]], w3[:, 0], V, args.perms, seeded(quarter, state, "I3"))
                cache[tag] = res; save()
                del w3; gc.collect()
            if "IO3" not in res:
                w4 = windows(groups, 3, 3)                   # steps t >= 3: origin is outside the last three airports
                res["steps_IO3"] = int(len(w4))
                res["IO3"] = cmi_with_null(w4[:, 3], [w4[:, 0], w4[:, 1], w4[:, 2]], w4[:, 4], V, args.perms, seeded(quarter, state, "IO3"))
                cache[tag] = res; save()
                del w4; gc.collect()
            if "heldout" not in res:
                # rows with 4 columns of history (padded with V = 'start') + next
                parts = []
                for n, (arr, ids) in groups.items():
                    pad = np.column_stack([np.full((len(arr), 4), V, dtype=np.int64), arr])
                    for t in range(1, n):
                        parts.append(np.column_stack([pad[:, t:t + 5], ids, np.full(len(arr), t), arr[:, 0]]))
                rows = np.concatenate(parts)                # h-4 h-3 h-2 h-1 | y | id | t (index of y) | origin
                sp = split_ids(n_itin)[rows[:, 5]]
                tr, va, te = rows[sp == 0], rows[sp == 1], rows[sp == 2]
                m = Backoff(tr[:, :5], V + 1)
                betas = []
                for L in range(1, 5):
                    betas.append(tune(m, va[va[:, 6] >= L][:, :5], L, betas, grid))
                ho = dict(betas=betas)
                for L in range(1, 5):
                    sub = te[te[:, 6] >= L][:, :5]          # steps with at least L previous airports
                    ho[f"steps_ge{L}"] = int(len(sub))
                    ho[f"CE{L-1}_on_ge{L}"] = ce(m.prob(sub, L - 1, betas)) if L > 1 else ce(m.prob(sub, 0, betas))
                    ho[f"CE{L}_on_ge{L}"] = ce(m.prob(sub, L, betas))
                    ho[f"gain{L}"] = ho[f"CE{L-1}_on_ge{L}"] - ho[f"CE{L}_on_ge{L}"]
                sub = te[te[:, 6] >= 2]
                trw = tr[tr[:, 6] >= 2]
                ho["CE_return_only"] = return_model_ce(trw[:, :5], sub[:, :5], V + 1, betas[0])
                ho["gain_return_only"] = ho["CE1_on_ge2"] - ho["CE_return_only"]
                ho["share_of_gain2_from_return"] = ho["gain_return_only"] / ho["gain2"]
                # first-order chain on the augmented state (trip origin, current airport), backed off to first order
                def origin_prob(train, rows_, beta):
                    ko_tr = train[:, 7] * (V + 2) + train[:, 3]
                    df = pd.DataFrame({"c": ko_tr, "y": train[:, 4]})
                    cy, c = df.groupby(["c", "y"]).size(), df.groupby("c").size()
                    p1 = m.prob(rows_[:, :5], 1, betas)
                    ko = rows_[:, 7] * (V + 2) + rows_[:, 3]
                    ncy = cy.reindex(pd.MultiIndex.from_arrays([ko, rows_[:, 4]])).fillna(0).to_numpy()
                    nc = c.reindex(ko).fillna(0).to_numpy()
                    return (ncy + beta * p1) / (nc + beta)
                vb = va[va[:, 6] >= 2]
                b_o = min(grid, key=lambda b: ce(origin_prob(tr, vb, b)))
                ho["beta_origin"] = b_o
                for L in (2, 3, 4):
                    sub = te[te[:, 6] >= L]
                    ho[f"CE_origin_on_ge{L}"] = ce(origin_prob(tr, sub, b_o))
                    # share of the gain from first order to order L recovered by (origin, current) alone
                    ce1 = ce(m.prob(sub[:, :5], 1, betas)); ceL = ce(m.prob(sub[:, :5], L, betas))
                    ho[f"CE1_on_ge{L}_b"] = ce1
                    ho[f"CEorder{L}_on_ge{L}"] = ceL
                    ho[f"share_recovered_by_origin_ge{L}"] = (ce1 - ho[f"CE_origin_on_ge{L}"]) / (ce1 - ceL)
                res["heldout"] = ho
                cache[tag] = res; save()
                del rows, parts, tr, va, te, m, sub, trw, vb; gc.collect()
            if "ce_curve" not in res:
                # cross-entropy of orders 1-4 and of (origin, current) on the same test steps (t >= 2 and t >= 4)
                parts = []
                for n, (arr, ids) in groups.items():
                    pad = np.column_stack([np.full((len(arr), 4), V, dtype=np.int64), arr])
                    for t in range(1, n):
                        parts.append(np.column_stack([pad[:, t:t + 5], ids, np.full(len(arr), t), arr[:, 0]]))
                rows = np.concatenate(parts); del parts
                sp = split_ids(n_itin)[rows[:, 5]]
                tr, te = rows[sp == 0], rows[sp == 2]
                m = Backoff(tr[:, :5], V + 1)
                betas = res["heldout"]["betas"]
                b_o = res["heldout"]["beta_origin"]
                ko_tr = tr[:, 7] * (V + 2) + tr[:, 3]
                dfo = pd.DataFrame({"c": ko_tr, "y": tr[:, 4]})
                cyo, co = dfo.groupby(["c", "y"]).size(), dfo.groupby("c").size()
                curve = {}
                for lo in (2, 4):
                    sub = te[te[:, 6] >= lo]
                    d = {f"order{L}": ce(m.prob(sub[:, :5], L, betas)) for L in range(1, 5)}
                    p1 = m.prob(sub[:, :5], 1, betas)
                    ko = sub[:, 7] * (V + 2) + sub[:, 3]
                    ncy = cyo.reindex(pd.MultiIndex.from_arrays([ko, sub[:, 4]])).fillna(0).to_numpy()
                    nc = co.reindex(ko).fillna(0).to_numpy()
                    d["origin_current"] = ce((ncy + b_o * p1) / (nc + b_o))
                    d["steps"] = int(len(sub))
                    curve[f"ge{lo}"] = d
                res["ce_curve"] = curve
                cache[tag] = res; save()
                del rows, tr, te, m, dfo, sub; gc.collect()
            del groups, lens, rt; gc.collect()
            res["done"] = True
            res["seconds"] = round(time.time() - t0, 1)
            cache[tag] = res; save()
            print(tag, json.dumps({k: v for k, v in res.items() if k in ("I2", "I3", "IO3", "heldout")}, indent=0)[:1500])

    # flat table
    out = []
    for tag, r in cache.items():
        if not r.get("done"):
            continue
        row = {k: r[k] for k in ("quarter", "state", "itineraries", "states", "mean_len", "share_len3plus",
                                 "share_round_trip", "steps_I2", "return_rate", "return_rate_first_order",
                                 "steps_I3", "steps_IO3")}
        for q in ("I2", "I3", "IO3"):
            for k, v in r[q].items():
                row[f"{q}_{k}"] = v
        for k, v in r["heldout"].items():
            row[f"ho_{k}"] = v if not isinstance(v, list) else "/".join(str(x) for x in v)
        for sub, d in r.get("ce_curve", {}).items():
            for k, v in d.items():
                row[f"curve_{sub}_{k}"] = v
        out.append(row)
    pd.DataFrame(out).to_csv(args.out + ".tmp", index=False)
    os.replace(args.out + ".tmp", args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
