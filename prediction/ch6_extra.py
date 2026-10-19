"""Decompose the memory-network codelength gain: L_first(opt) -> L_mem(lifted first-order partition) -> L_mem(opt).
'lifted' assigns every state node the module of its physical node in the optimised first-order partition, so
L_mem(lifted) - L_mem(opt) is the gain from letting a physical node belong to several modules (overlap)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, numpy as np, pandas as pd
import ch6_networks as c, mapeq
rows = []
for w in sys.argv[1:]:
    seqs = c.taxi_sequences() if w == 'taxi' else c.npz_sequences('geolife_bj500m') if w == 'geolife' else c.npz_sequences('highd_v20', parts=(0, 1, 2, 3))
    S, first, mem, states, V = c.build(seqs)
    for lab, (f, m) in [('data', (first, mem)), ('markov_surrogate', c.build(c.surrogate(S, V, np.random.default_rng(1)))[1:3])]:
        Lf, mf = mapeq.optimise_ml(f, trials=4, seed=7)
        lifted = mf[m.phys]; Ll = m.codelength(lifted)
        Lm, mm = mapeq.optimise_ml(m, trials=4, seed=7)
        rows.append(dict(dataset=w, network=lab, first_L1=f.one_module(), first_L=Lf, first_modules=int(mf.max() + 1), mem_L1=m.one_module(),
                         mem_L_lifted=Ll, mem_L=Lm, mem_modules=int(mm.max() + 1), gain_entropy=Lf - Ll, gain_overlap=Ll - Lm))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P.out(P.PRED_NGRAM / f'ch6_decomp_{"_".join(sys.argv[1:])}.csv'), index=False)
