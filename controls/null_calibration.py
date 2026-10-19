"""P0-1  Calibration of the stratified permutation null (negative and positive controls).

The excess CMI and the share of significant units rest on the stratified permutation null. Overlapping triples of
one chain are treated as exchangeable there, so the null mean may be right while its spread (and the p-values) is not.
This script runs the unchanged estimators on sequences whose true second-order CMI is known.

Negative control: for each unit, a first-order Markov chain fitted to the unit's own compressed sequences
  (same start state and length per trip; a trip ends early if it reaches a state without successors).
  True I2 = 0, so the share of p < 0.05 is the false-positive rate.
Positive control: P(c | a, b) = lam * P2s(c | a, b) + (1 - lam) * P1(c | b), lam in {0.25, 0.5, 1}, where P2s is the
  unit's second-order chain smoothed towards P1 by interpolated back-off (beta = 3). The true I2 of the generator is
  computed exactly on the pair distribution the generated sample visits; recovery = excess CMI / true I2.
  The length dependence uses the first 25 / 50 / 100 % of each unit's trips as templates.

GeoLife: cleaned cache, 10 m grid (0.0001 deg), no de-jitter, CMI in bits, 50 permutations (= geolife_robust.py).
Taxi: one compressed whole-day sequence per vehicle on its own 20 x 20 grid, CMI in nats, 100 permutations
  (= taxi/taxi_memory_lib.cmi_perm).
Seeds: units are drawn with seed 0; every replicate has its own fixed seed (crc32 of dataset|unit|kind|rep), so a
run with fewer replicates gives exactly the first replicates of a longer run.
Replicates: taxi 100 negative replicates per unit and 25 positive units x 10; GeoLife 30 negative replicates per unit
and 10 positive units x 5. One GeoLife replicate (10 m grid, up to 2.7 x 10^5 trigrams, 50 permutations) costs about
100 taxi replicates, so the planned 100 per unit would take ~15 CPU hours; 50 x 30 = 1 500 runs still give the
false-positive rate to +-0.6 % (one s.e.). Override with NC_REP_NEG_GEOLIFE, NC_POS_UNITS_GEOLIFE, NC_REP_POS_GEOLIFE.
Finished units are cached in work/null_calibration/, so an interrupted run resumes.

    python controls/null_calibration.py          -> results/null_calibration_{negative,positive}.csv, _summary.json
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.ROOT / 'taxi'))
import glob, json, os, time, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
import geolife_robust as gr
import taxi_memory_lib as tl

N_UNITS = int(os.environ.get('NC_UNITS', 50))     # units per dataset (negative control)
N_POS_UNITS = {'taxi': int(os.environ.get('NC_POS_UNITS_TAXI', 25)), 'geolife': int(os.environ.get('NC_POS_UNITS_GEOLIFE', 10))}
N_REP_NEG = {'taxi': int(os.environ.get('NC_REP_NEG_TAXI', 100)), 'geolife': int(os.environ.get('NC_REP_NEG_GEOLIFE', 30))}
N_REP_POS = {'taxi': int(os.environ.get('NC_REP_POS_TAXI', 10)), 'geolife': int(os.environ.get('NC_REP_POS_GEOLIFE', 5))}
CACHE = P.CONTROLS_WORK / 'null_calibration'
LAMBDAS = (0.25, 0.5, 1.0)
FRACS = (0.25, 0.5, 1.0)
BETA = 3.0
GRID_GL = 0.0001
GRID_TX = 20
NPERM_GL, NPERM_TX = 50, 100


# ---------------------------------------------------------------- fitted chains
class Chain:
    """First-order and second-order successor tables of a set of sequences (states 0..K-1)."""

    def __init__(self, seqs, K):
        self.K = K
        b = np.concatenate([s[:-1] for s in seqs if len(s) >= 2]); c = np.concatenate([s[1:] for s in seqs if len(s) >= 2])
        key, cnt = np.unique(b * K + c, return_counts=True)
        self.p1_b, self.p1_c, self.p1_n = key // K, key % K, cnt.astype(float)
        self.p1_start = np.searchsorted(self.p1_b, np.arange(K)); self.p1_end = np.searchsorted(self.p1_b, np.arange(K), 'right')
        tot = np.bincount(self.p1_b, weights=self.p1_n, minlength=K)
        self.p1_p = self.p1_n / tot[self.p1_b]
        self.p1_g = self.p1_b + _seg_cumsum(self.p1_p, self.p1_start, self.p1_end)   # increasing key for vectorised sampling
        tri = [(s[:-2], s[1:-1], s[2:]) for s in seqs if len(s) >= 3]
        a = np.concatenate([t[0] for t in tri]); bb = np.concatenate([t[1] for t in tri]); cc = np.concatenate([t[2] for t in tri])
        ctx = a * K + bb
        ck, inv = np.unique(ctx, return_inverse=True)
        self.ctx_key = ck
        kk, cnt2 = np.unique(inv.astype(np.int64) * K + cc, return_counts=True)
        self.p2_ctx, self.p2_c, self.p2_n = kk // K, kk % K, cnt2.astype(float)
        self.p2_start = np.searchsorted(self.p2_ctx, np.arange(len(ck))); self.p2_end = np.searchsorted(self.p2_ctx, np.arange(len(ck)), 'right')
        self.ctx_tot = np.bincount(self.p2_ctx, weights=self.p2_n, minlength=len(ck))
        self.p2_p = self.p2_n / self.ctx_tot[self.p2_ctx]
        self.p2_g = self.p2_ctx + _seg_cumsum(self.p2_p, self.p2_start, self.p2_end)

    def step1(self, cur, rng):
        """Sample successors of states cur from P1; returns next states and a mask of chains that could move."""
        can = self.p1_end[cur] > self.p1_start[cur]
        nxt = cur.copy()
        if can.any():
            q = cur[can] + rng.random(can.sum()) * (1 - 1e-12)
            nxt[can] = self.p1_c[np.searchsorted(self.p1_g, q, side='right')]
        return nxt, can

    def ctx_index(self, prev, cur):
        q = prev * self.K + cur
        j = np.clip(np.searchsorted(self.ctx_key, q), 0, len(self.ctx_key) - 1)
        return np.where(self.ctx_key[j] == q, j, -1)

    def step2(self, prev, cur, lam, rng):
        nxt, can = self.step1(cur, rng)
        j = self.ctx_index(prev, cur)
        w = np.where(j >= 0, self.ctx_tot[np.maximum(j, 0)] / (self.ctx_tot[np.maximum(j, 0)] + BETA), 0.0)
        use2 = can & (j >= 0) & (rng.random(len(cur)) < lam * w)
        if use2.any():
            q = j[use2] + rng.random(use2.sum()) * (1 - 1e-12)
            nxt[use2] = self.p2_c[np.searchsorted(self.p2_g, q, side='right')]
        return nxt, can

    def simulate(self, templates, rng, lam=None):
        """templates: list of sequences (start states and lengths are copied). lam=None -> first-order chain."""
        n = len(templates); L = np.array([len(t) for t in templates])
        out = [[int(t[0])] for t in templates]
        if lam is not None:
            for i, t in enumerate(templates):
                if len(t) >= 2: out[i].append(int(t[1]))
        alive = np.ones(n, bool)
        while True:
            idx = np.where(alive & (np.array([len(o) for o in out]) < L))[0]
            if len(idx) == 0: break
            cur = np.array([out[i][-1] for i in idx])
            if lam is None:
                nxt, can = self.step1(cur, rng)
            else:
                prev = np.array([out[i][-2] for i in idx])
                nxt, can = self.step2(prev, cur, lam, rng)
            for i, x, c in zip(idx, nxt, can):
                if c: out[i].append(int(x))
                else: alive[i] = False
        return [np.array(o, dtype=np.int64) for o in out]

    def true_cmi(self, seqs, lam, log=np.log2):
        """Exact I(S_{t+1}; S_{t-1} | S_t) = H(C|B) - H(C|A,B) of the mixture generator, on the (a, b) pairs that seqs
        visit with a successor; H(C|B) uses the generator's P(c|b) = sum_a q(a|b) P(c|a,b)."""
        tri = [s for s in seqs if len(s) >= 3]
        if not tri: return np.nan
        a = np.concatenate([s[:-2] for s in tri]); b = np.concatenate([s[1:-1] for s in tri])
        pk, pc = np.unique(a * self.K + b, return_counts=True)
        pa, pb = pk // self.K, pk % self.K
        st, en = self.p1_start[pb], self.p1_end[pb]
        m = en > st; pa, pb, pc, st, en = pa[m], pb[m], pc[m], st[m], en[m]
        q = pc / pc.sum()
        deg = en - st
        off = np.r_[0, np.cumsum(deg)[:-1]]
        rowid = np.repeat(np.arange(len(pa)), deg)
        gidx = st[rowid] + (np.arange(deg.sum()) - off[rowid])          # global index into the P1 table
        j = self.ctx_index(pa, pb)
        w = np.where(j >= 0, self.ctx_tot[np.maximum(j, 0)] / (self.ctx_tot[np.maximum(j, 0)] + BETA), 0.0)
        Pm = (1 - lam * w)[rowid] * self.p1_p[gidx]
        r2 = np.where(j >= 0)[0]
        if len(r2):
            d2 = self.p2_end[j[r2]] - self.p2_start[j[r2]]
            o2 = np.r_[0, np.cumsum(d2)[:-1]]
            rr = np.repeat(r2, d2)
            e_idx = self.p2_start[j[rr]] + (np.arange(d2.sum()) - np.repeat(o2, d2))
            cg = np.searchsorted(self.p1_b * self.K + self.p1_c, pb[rr] * self.K + self.p2_c[e_idx])
            np.add.at(Pm, off[rr] + (cg - st[rr]), lam * w[rr] * self.p2_p[e_idx])
        with np.errstate(divide='ignore', invalid='ignore'):
            plogp = np.where(Pm > 0, Pm * log(Pm), 0.0)
        H_ab = -np.bincount(rowid, weights=plogp, minlength=len(pa))
        qb = np.bincount(pb, weights=q, minlength=self.K)
        Pbar = np.zeros(len(self.p1_c))
        np.add.at(Pbar, gidx, (q / qb[pb])[rowid] * Pm)
        with np.errstate(divide='ignore', invalid='ignore'):
            plb = np.where(Pbar > 0, Pbar * log(Pbar), 0.0)
        H_b = -np.bincount(self.p1_b, weights=plb, minlength=self.K)
        return float((qb * H_b).sum() - (q * H_ab).sum())


def _seg_cumsum(p, start, end):
    """Cumulative sums restarted at each segment [start, end); the last entry of every segment is set to 1."""
    c = np.cumsum(p)
    seg = np.repeat(np.arange(len(start)), end - start)
    base = np.r_[0.0, c][start]
    out = c - base[seg]
    last = end[end > start] - 1
    out[last] = 1.0
    return out


# ---------------------------------------------------------------- estimators (unchanged from the paper)
def cmi_geolife(seqs, K, seed):
    a, b, c = gr.ngrams(seqs, 3)
    if len(a) < gr.MIN_TRIGRAMS: return None
    real, nullm, p, h1, h2 = gr.cmi_with_null(a, b, c, K, NPERM_GL, np.random.default_rng(seed))
    return dict(plugin=real, null=nullm, excess=real - nullm, p=p, n_steps=len(a))


def cmi_taxi(seq, seed):
    r = tl.cmi_perm(list(seq), 2, NPERM_TX, seed)
    if r is None: return None
    return dict(plugin=r['real'], null=r['real'] - r['excess'], excess=r['excess'], p=r['p_value'], n_steps=len(seq) - 2)


# ---------------------------------------------------------------- units
def geolife_unit(path):
    full, _, _ = gr.load_user(path, GRID_GL, False)
    vocab = np.unique(np.concatenate(full)); K = len(vocab)
    return [np.searchsorted(vocab, s).astype(np.int64) for s in full], K


def taxi_unit(path):
    lat, lon = tl.read_lat_lon(path)
    st = tl.compress_repeats(tl.map_to_states(lat, lon, GRID_TX))
    return np.asarray(st, dtype=np.int64)


def job(args):
    kind, ds, path, n_rep = args
    uid = os.path.basename(path)
    cache = CACHE / f'{ds}_{kind}_{uid}_{n_rep}.json'
    if cache.exists():
        return kind, json.load(open(cache))
    rows = json.loads(json.dumps(run_unit(kind, ds, path, n_rep, uid), default=_plain))   # same types as a cached unit
    json.dump(rows, open(P.out(cache), 'w'))
    return kind, rows


def _plain(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return float(x)


def run_unit(kind, ds, path, n_rep, uid):
    rows = []
    if ds == 'geolife':
        seqs, K = geolife_unit(path)
    else:
        seqs = [taxi_unit(path)]; K = int(seqs[0].max()) + 1
    ch = Chain(seqs, K)
    log = np.log2 if ds == 'geolife' else np.log
    est = (lambda sim, sd: cmi_geolife(sim, K, sd)) if ds == 'geolife' else (lambda sim, sd: cmi_taxi(sim[0], sd))
    true_cache = {}
    for rep in range(n_rep):
        seed = zlib.crc32(f'{ds}|{uid}|{kind}|{rep}'.encode())
        rng = np.random.default_rng(seed)
        if kind == 'neg':
            r = est(ch.simulate(seqs, rng), seed + 1)
            if r: rows.append(dict(dataset=ds, unit=uid, rep=rep, **r))
            continue
        for f in FRACS:
            if ds == 'geolife':
                tmpl = seqs[:max(1, int(np.ceil(f * len(seqs))))]
            else:
                tmpl = [seqs[0][:max(3, int(np.ceil(f * len(seqs[0]))))]]
            for lam in LAMBDAS:
                sim = ch.simulate(tmpl, rng, lam)
                if (f, lam) not in true_cache:
                    true_cache[(f, lam)] = ch.true_cmi(sim, lam, log)
                r = est(sim, seed + 1)
                if r: rows.append(dict(dataset=ds, unit=uid, rep=rep, frac=f, lam=lam, true=true_cache[(f, lam)], **r))
    return rows


def pick_units():
    rng = np.random.default_rng(0)
    gl = sorted(glob.glob(os.path.join(str(P.GEOLIFE_CACHE), '*.npz')))
    ok = []
    for p in gl:
        a, _, _ = gr.ngrams(gr.load_user(p, GRID_GL, False)[0], 3)
        if len(a) >= gr.MIN_TRIGRAMS: ok.append(p)
    gl_pick = sorted(rng.choice(ok, N_UNITS, replace=False).tolist())
    tx = sorted(glob.glob(os.path.join(str(P.TAXI_RAW), '*')))
    tx_ok = [p for p in rng.permutation(tx) if len(taxi_unit(p)) >= 50]
    tx_pick = sorted(tx_ok[:N_UNITS])
    return gl_pick, tx_pick


if __name__ == '__main__':
    t0 = time.time()
    gl, tx = pick_units()
    print('units', len(gl), len(tx), f'[{time.time() - t0:.0f}s]', flush=True)
    jobs = [('neg', 'taxi', p, N_REP_NEG['taxi']) for p in tx] + [('pos', 'taxi', p, N_REP_POS['taxi']) for p in tx[:N_POS_UNITS['taxi']]]
    jobs += [('neg', 'geolife', p, N_REP_NEG['geolife']) for p in gl] + [('pos', 'geolife', p, N_REP_POS['geolife']) for p in gl[:N_POS_UNITS['geolife']]]
    neg, pos = [], []
    with Pool(2) as pool:
        for k, (kind, rows) in enumerate(pool.imap_unordered(job, jobs)):
            (neg if kind == 'neg' else pos).extend(rows)
            if k % 10 == 0:
                print(k, len(jobs), f'[{time.time() - t0:.0f}s]', flush=True)
                pd.DataFrame(neg).to_csv(P.out(P.CONTROLS_RESULTS / 'null_calibration_negative.csv'), index=False)
                pd.DataFrame(pos).to_csv(P.out(P.CONTROLS_RESULTS / 'null_calibration_positive.csv'), index=False)
    N = pd.DataFrame(neg).sort_values(['dataset', 'unit', 'rep']); Q = pd.DataFrame(pos).sort_values(['dataset', 'unit', 'rep', 'frac', 'lam'])
    N.to_csv(P.out(P.CONTROLS_RESULTS / 'null_calibration_negative.csv'), index=False)
    Q.to_csv(P.out(P.CONTROLS_RESULTS / 'null_calibration_positive.csv'), index=False)
    summ = {}
    for ds, g in N.groupby('dataset'):
        unit_fpr = g.groupby('unit').p.apply(lambda x: (x < .05).mean())
        summ[ds] = dict(units=int(g.unit.nunique()), runs=len(g), fpr=float((g.p < .05).mean()),
                        fpr_unit_q10_q90=[float(unit_fpr.quantile(.1)), float(unit_fpr.quantile(.9))],
                        excess_median=float(g.excess.median()), excess_mean=float(g.excess.mean()),
                        unit_median_excess_median=float(g.groupby('unit').excess.median().median()),
                        unit_share_significant_median=float(g.groupby('rep').p.apply(lambda x: (x < .05).mean()).median()))
    pos_s = []
    for (ds, f, lam), g in Q.groupby(['dataset', 'frac', 'lam']):
        pos_s.append(dict(dataset=ds, frac=f, lam=lam, runs=len(g), steps_median=float(g.n_steps.median()),
                          true_median=float(g['true'].median()), excess_median=float(g.excess.median()),
                          recovery_median=float((g.excess / g['true']).median()), detection=float((g.p < .05).mean())))
    summ['positive'] = pos_s
    summ['settings'] = dict(units=N_UNITS, rep_neg=N_REP_NEG, pos_units=N_POS_UNITS, rep_pos=N_REP_POS, perm=dict(geolife=NPERM_GL, taxi=NPERM_TX))
    summ['seconds'] = time.time() - t0
    json.dump(summ, open(P.out(P.CONTROLS_RESULTS / 'null_calibration_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
