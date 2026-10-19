"""n-gram / variable-order Markov predictors for symbol sequences (chapter 5).
Models (all conditional on up to N-1 previous symbols inside the same sequence piece):
  addalpha(k)   : order-k chain, add-alpha smoothing over the full vocabulary, no back-off (the naive baseline)
  jm(N, beta)   : interpolated back-off P_k = (c(h,w) + beta P_{k-1}) / (c(h) + beta)   (as in chapter 4)
  wb(N)         : interpolated Witten-Bell = PPM-C style blending, a variable-order Markov model
  kn(N)         : interpolated modified Kneser-Ney (Chen & Goodman 1999), discounts D1, D2, D3+ per order
Base distribution for jm/wb: add-one unigram over V. Evaluation: log2-loss on every position t>=1 of each
test piece (history truncated at the piece start), top-1 / top-5 accuracy."""
import numpy as np
from collections import defaultdict, Counter

class NGram:
    def __init__(self, N, V, kind='kn', beta=1.0, alpha=0.5):
        self.N, self.V, self.kind, self.beta, self.alpha = N, V, kind, beta, alpha

    def fit(self, seqs):
        N = self.N
        self.c = [defaultdict(Counter) for _ in range(N)]          # c[k][h][w], k = len(h)
        for s in seqs:
            s = list(s)
            for t in range(1, len(s)) if self.kind == 'addalpha' else range(len(s)):
                for k in range(0, min(t, N - 1) + 1):
                    if self.kind == 'addalpha' and k != N - 1: continue
                    self.c[k][tuple(s[t - k:t])][s[t]] += 1
        self.uni = self.c[0][()] if self.kind != 'addalpha' else None
        self.freq = self.c[0][()] if len(self.c[0]) else Counter()
        self.ntok = sum(self.uni.values()) if self.uni is not None else 0
        if self.kind == 'kn': self._prep_kn()
        self.tot = [{h: sum(d.values()) for h, d in ck.items()} for ck in self.c]
        self.typ = [{h: len(d) for h, d in ck.items()} for ck in self.c]
        return self

    def _prep_kn(self):
        N = self.N
        # continuation counts for k < N-1: cc[k][h][w] = #distinct u with c[k+1][(u,)+h][w] > 0
        self.cc = [defaultdict(Counter) for _ in range(N)]
        for k in range(1, N):
            for h, d in self.c[k].items():
                for w in d: self.cc[k - 1][h[1:]][w] += 1
        self.D = []; self.stats = []
        for k in range(N):
            tab = self.c[k] if k == N - 1 else self.cc[k]
            n = Counter()
            for d in tab.values():
                for v in d.values():
                    if v <= 4: n[v] += 1
            Y = n[1] / (n[1] + 2 * n[2]) if n[1] + 2 * n[2] > 0 else 0.5
            D = [0.0] + [max(0.05, min(j - (j + 1) * Y * n[j + 1] / n[j], j)) if n[j] > 0 else 0.5 for j in (1, 2, 3)]
            self.D.append(D)
            st = {}
            for h, d in tab.items():
                vals = list(d.values()); tot = sum(vals)
                n1 = sum(1 for v in vals if v == 1); n2 = sum(1 for v in vals if v == 2); n3 = len(vals) - n1 - n2
                st[h] = (tot, (D[1] * n1 + D[2] * n2 + D[3] * n3) / tot)
            self.stats.append(st)
        self.tabs = [self.c[k] if k == N - 1 else self.cc[k] for k in range(N)]

    def prob(self, hist, w):
        """P(w | hist) with hist a tuple (most recent last), truncated to N-1."""
        N = self.N; hist = tuple(hist[-(N - 1):]) if N > 1 else ()
        if self.kind == 'addalpha':
            h = hist[-(N - 1):] if N > 1 else ()
            if len(h) < N - 1: return 1.0 / self.V
            d = self.c[N - 1].get(h); return ((d[w] if d else 0) + self.alpha) / ((self.tot[N - 1].get(h, 0)) + self.alpha * self.V)
        if self.kind == 'kn':
            p = 1.0 / self.V
            for k in range(0, len(hist) + 1):
                h = hist[len(hist) - k:] if k else ()
                st = self.stats[k].get(h)
                if st is None: continue
                tot, gamma = st; v = self.tabs[k][h].get(w, 0)
                D = self.D[k][min(v, 3)]
                p = max(v - D, 0.0) / tot + gamma * p
            return p
        p = (self.uni.get(w, 0) + 1.0) / (self.ntok + self.V)
        for k in range(1, len(hist) + 1):
            h = hist[len(hist) - k:]
            d = self.c[k].get(h)
            if d is None: continue
            tot = self.tot[k][h]; lam = self.beta if self.kind == 'jm' else self.typ[k][h]
            p = (d.get(w, 0) + lam * p) / (tot + lam)
        return p

    def candidates(self, hist):
        hist = tuple(hist[-(self.N - 1):]) if self.N > 1 else ()
        cand = set()
        for k in range(1, len(hist) + 1):
            d = self.c[k].get(hist[len(hist) - k:]) if self.kind != 'kn' or k == self.N - 1 else self.c[k].get(hist[len(hist) - k:])
            if d: cand.update(d.keys())
        cand.update(w for w, _ in self.freq.most_common(5))
        if self.kind == 'kn': cand.update(w for w, _ in self.cc[0][()].most_common(5)) if self.N > 1 else None
        return cand

def evaluate(model, seqs, topk=True, max_ctx=None):
    ll, n, a1, a5 = 0.0, 0, 0, 0
    for s in seqs:
        s = list(s)
        for t in range(1, len(s)):
            h = tuple(s[max(0, t - model.N + 1):t]); w = s[t]
            p = model.prob(h, w); ll += np.log2(p); n += 1
            if topk:
                cand = model.candidates(h)
                fr = model.freq
                sc = sorted(((model.prob(h, c), fr.get(c, 0), -c, c) for c in cand), reverse=True)
                top = [x[3] for x in sc[:5]]; a1 += top[0] == w; a5 += w in top
    r = dict(bits=-ll / max(n, 1), n=n)
    if topk: r.update(acc1=a1 / max(n, 1), acc5=a5 / max(n, 1))
    return r

def view(full, N, kind, beta=1.0, alpha=0.5):
    """A model of order N sharing the counts of a model fitted with a larger N."""
    m = NGram(N, full.V, kind, beta, alpha)
    m.c = full.c[:N]; m.tot = full.tot[:N]; m.typ = full.typ[:N]; m.uni = full.uni; m.ntok = full.ntok; m.freq = full.freq
    if kind == 'addalpha':
        m.uni = None
    if kind == 'kn': m._prep_kn()
    return m
