import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, numpy as np
import ch6_networks as c
for w in ['taxi','geolife','highd']:
    seqs = c.taxi_sequences() if w=='taxi' else c.npz_sequences('geolife_bj500m') if w=='geolife' else c.npz_sequences('highd_v20', parts=(0,1,2,3))
    S, first, mem, states, V = c.build(seqs)
    sur = c.surrogate(S, V, np.random.default_rng(1))
    S2, f2, m2, st2, V2 = c.build(sur)
    c.export_infomap(w + '_surrogate', f2, m2, st2, V2); print(w, m2.n, flush=True)
