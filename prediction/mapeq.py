"""Two-level map equation for first-order and second-order (memory) networks, with a simple Infomap-style
optimiser (greedy local moving + aggregation, repeated with several random orders; keeps the best).
Flows are empirical link flows (relative transition frequencies), no teleportation.
Memory networks: nodes are state nodes (i->j) with physical node j; the index codebook of a module encodes
physical nodes, i.e. visit rates of state nodes of the same physical node inside one module are summed
(Rosvall et al. 2014, Nat. Commun. 5:4630; Edler et al. 2017, PRX 7:041025)."""
import numpy as np
from collections import defaultdict

def plogp(x):
    x = np.asarray(x, float); out = np.zeros_like(x); m = x > 0; out[m] = x[m] * np.log2(x[m]); return out

class Net:
    def __init__(self, src, dst, w, phys):
        """src,dst: int arrays of node ids 0..n-1; w: link flow (sums to 1); phys: physical id of each node."""
        self.n = len(phys); self.phys = np.asarray(phys)
        f = defaultdict(float)
        for a, b, x in zip(src, dst, w): f[(a, b)] += x
        self.out = [dict() for _ in range(self.n)]; self.inn = [dict() for _ in range(self.n)]
        for (a, b), x in f.items():
            self.out[a][b] = self.out[a].get(b, 0) + x; self.inn[b][a] = self.inn[b].get(a, 0) + x
        self.p = np.array([sum(self.inn[i].values()) for i in range(self.n)])
        self.p = self.p / self.p.sum()
        self.outflow = np.array([sum(v for k, v in self.out[i].items() if k != i) for i in range(self.n)])

    def codelength(self, mod):
        """L(M) for a module assignment of the (state) nodes."""
        mod = np.asarray(mod); K = mod.max() + 1
        q = np.zeros(K)
        for a in range(self.n):
            for b, x in self.out[a].items():
                if mod[a] != mod[b]: q[mod[a]] += x
        pm = np.bincount(mod, weights=self.p, minlength=K)
        # physical-node visit rates inside modules
        key = mod.astype(np.int64) * (self.phys.max() + 1) + self.phys
        _, inv = np.unique(key, return_inverse=True); pp = np.bincount(inv, weights=self.p)
        qt = q.sum()
        return float(plogp(qt) - 2 * plogp(q).sum() - plogp(pp).sum() + plogp(q + pm).sum())

    def one_module(self):
        return self.codelength(np.zeros(self.n, int))

def optimise(net, trials=5, seed=0, max_levels=10):
    """Greedy local moving on the full objective (computed incrementally per module), then aggregation."""
    rng = np.random.default_rng(seed); best = (net.one_module(), np.zeros(net.n, int))
    for tr in range(trials):
        mod = np.arange(net.n)
        for level in range(max_levels):
            moved = _local_moving(net, mod, rng)
            if not moved: break
        mod = np.unique(mod, return_inverse=True)[1]
        L = net.codelength(mod)
        if L < best[0]: best = (L, mod.copy())
    return best

def _local_moving(net, mod, rng):
    """One sweep sequence of single-node moves with exact delta-L (module-level state kept incrementally)."""
    n = net.n; phys = net.phys
    K = n
    q = np.zeros(K); pm = np.zeros(K)
    np.add.at(pm, mod, net.p)
    for a in range(n):
        for b, x in net.out[a].items():
            if mod[a] != mod[b]: q[mod[a]] += x
    pp = defaultdict(float)                       # (module, physical) -> rate
    for a in range(n): pp[(mod[a], phys[a])] += net.p[a]
    qt = q.sum()
    def term_mod(qm, pmm): return -2 * _pl(qm) + _pl(qm + pmm)
    any_move = False
    for sweep in range(20):
        moves = 0
        for a in rng.permutation(n):
            ma = mod[a]
            # flows between a and each neighbouring module
            outm = defaultdict(float); inm = defaultdict(float)
            for b, x in net.out[a].items():
                if b != a: outm[mod[b]] += x
            for b, x in net.inn[a].items():
                if b != a: inm[mod[b]] += x
            cands = set(outm) | set(inm); cands.discard(ma)
            if not cands: continue
            oa = net.outflow[a]; ia = sum(v for k, v in net.inn[a].items() if k != a); pa = net.p[a]; ph = phys[a]
            # remove a from ma
            q_old_a = q[ma]
            q_new_a = q[ma] - (oa - outm.get(ma, 0)) + inm.get(ma, 0)
            pp_a = pp[(ma, ph)]
            best_d, best_m = 0.0, None
            base_rm = term_mod(q_new_a, pm[ma] - pa) - term_mod(q_old_a, pm[ma]) \
                      - (_pl(pp_a - pa) - _pl(pp_a))
            for m2 in cands:
                q_old_b = q[m2]
                q_new_b = q[m2] + (oa - outm.get(m2, 0)) - inm.get(m2, 0)
                pp_b = pp.get((m2, ph), 0.0)
                dq = (q_new_a - q_old_a) + (q_new_b - q_old_b)
                d = base_rm + term_mod(q_new_b, pm[m2] + pa) - term_mod(q_old_b, pm[m2]) \
                    - (_pl(pp_b + pa) - _pl(pp_b)) + _pl(qt + dq) - _pl(qt)
                if d < best_d - 1e-12: best_d, best_m = d, m2
            if best_m is not None:
                m2 = best_m
                dqa = -(oa - outm.get(ma, 0)) + inm.get(ma, 0); dqb = (oa - outm.get(m2, 0)) - inm.get(m2, 0)
                q[ma] += dqa; q[m2] += dqb; qt += dqa + dqb
                pm[ma] -= pa; pm[m2] += pa
                pp[(ma, ph)] -= pa; pp[(m2, ph)] = pp.get((m2, ph), 0.0) + pa
                mod[a] = m2; moves += 1; any_move = True
        if moves == 0: break
    return any_move

def _pl(x): return x * np.log2(x) if x > 1e-300 else 0.0

def optimise_ml(net, trials=3, seed=0):
    """Multi-level: local moving of units (nodes, then modules as super-units) until no move at any level."""
    rng = np.random.default_rng(seed); best = (net.one_module(), np.zeros(net.n, int))
    for tr in range(trials):
        # units
        member = np.arange(net.n)                              # node -> unit
        uout = [dict((b, x) for b, x in net.out[a].items() if b != a) for a in range(net.n)]
        up = net.p.copy(); uphys = [{net.phys[a]: net.p[a]} for a in range(net.n)]
        while True:
            umod, moved = _move_units(uout, up, uphys, rng)
            if not moved: break
            # aggregate
            newid = np.unique(umod, return_inverse=True)[1]; K = newid.max() + 1
            nout = [defaultdict(float) for _ in range(K)]; npv = np.zeros(K); nph = [defaultdict(float) for _ in range(K)]
            for u in range(len(up)):
                a = newid[u]; npv[a] += up[u]
                for ph, x in uphys[u].items(): nph[a][ph] += x
                for v, x in uout[u].items():
                    b = newid[v]
                    if a != b: nout[a][b] += x
            member = newid[member]; uout = [dict(d) for d in nout]; up = npv; uphys = [dict(d) for d in nph]
        mod = np.unique(member, return_inverse=True)[1]; L = net.codelength(mod)
        if L < best[0] - 1e-12: best = (L, mod.copy())
    return best

def _move_units(uout, up, uphys, rng):
    n = len(up); uin = [dict() for _ in range(n)]
    for a in range(n):
        for b, x in uout[a].items(): uin[b][a] = x
    mod = np.arange(n); q = np.array([sum(d.values()) for d in uout]); pm = up.copy()
    pp = [dict(d) for d in uphys]                              # module -> {phys: rate}
    qt = q.sum(); moved_any = False
    for sweep in range(30):
        moves = 0
        for a in rng.permutation(n):
            ma = mod[a]; outm = defaultdict(float); inm = defaultdict(float)
            for b, x in uout[a].items(): outm[mod[b]] += x
            for b, x in uin[a].items(): inm[mod[b]] += x
            cands = (set(outm) | set(inm)) - {ma}
            if not cands: continue
            oa = sum(uout[a].values()); pa = up[a]; Da = uphys[a]
            qa_new = q[ma] - (oa - outm.get(ma, 0)) + inm.get(ma, 0)
            rm = -2 * _pl(qa_new) + _pl(qa_new + pm[ma] - pa) + 2 * _pl(q[ma]) - _pl(q[ma] + pm[ma])
            rm -= sum(_pl(pp[ma].get(ph, 0) - x) - _pl(pp[ma].get(ph, 0)) for ph, x in Da.items())
            best_d, best_m = 0.0, None
            for m2 in cands:
                qb_new = q[m2] + (oa - outm.get(m2, 0)) - inm.get(m2, 0)
                add = -2 * _pl(qb_new) + _pl(qb_new + pm[m2] + pa) + 2 * _pl(q[m2]) - _pl(q[m2] + pm[m2])
                add -= sum(_pl(pp[m2].get(ph, 0) + x) - _pl(pp[m2].get(ph, 0)) for ph, x in Da.items())
                dqt = (qa_new - q[ma]) + (qb_new - q[m2])
                d = rm + add + _pl(qt + dqt) - _pl(qt)
                if d < best_d - 1e-12: best_d, best_m = d, m2
            if best_m is not None:
                m2 = best_m; qb_new = q[m2] + (oa - outm.get(m2, 0)) - inm.get(m2, 0)
                qt += (qa_new - q[ma]) + (qb_new - q[m2]); q[ma] = qa_new; q[m2] = qb_new
                pm[ma] -= pa; pm[m2] += pa
                for ph, x in Da.items():
                    pp[ma][ph] = pp[ma].get(ph, 0) - x; pp[m2][ph] = pp[m2].get(ph, 0) + x
                mod[a] = m2; moves += 1; moved_any = True
        if moves == 0: break
    return mod, moved_any
