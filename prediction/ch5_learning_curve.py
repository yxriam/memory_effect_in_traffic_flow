"""Learning curve: test bits/step of KN models of order 1..4 when trained on a random fraction of training pieces."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, numpy as np, pandas as pd
import lm, run_ch5 as r
rows = []
for name in sys.argv[1:]:
    seqs, part, unit, V = r.load(name)
    tr = [s for s, p in zip(seqs, part) if p == 0]; te = [s for s, p in zip(seqs, part) if p == 2]
    rng = np.random.default_rng(0)
    for frac in (0.03, 0.1, 0.3, 1.0):
        idx = rng.permutation(len(tr))[:max(1, int(frac * len(tr)))]
        full = lm.NGram(5, V, 'wb').fit([tr[i] for i in idx]); ntr = sum(len(tr[i]) for i in idx)
        for N in (2, 3, 4, 5):
            e = lm.evaluate(lm.view(full, N, 'kn'), te, topk=False)
            rows.append(dict(dataset=name, frac=frac, train_tokens=ntr, order=N - 1, test_bits=e['bits'])); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P.out(P.PRED_NGRAM / f'ch5_learning_curve_{"_".join(sys.argv[1:])}.csv'), index=False)
