import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import numpy as np, pandas as pd
S = pd.read_pickle(P.CONTROLS_WORK / 'highd_states.pkl')
uid = S.uid.to_numpy(); fr = S.frame.to_numpy(); v = S.v.to_numpy()
a = np.r_[np.nan, np.diff(v)] / 0.2; ok = np.r_[False, (uid[1:] == uid[:-1]) & (np.diff(fr) == 5)]; a[~ok] = np.nan
rows = []
for L in range(1, 16):
    m = np.isfinite(a[L:]) & np.isfinite(a[:-L]) & (uid[L:] == uid[:-L])
    rows.append(dict(lag=L, lag_s=0.2 * L, acf=np.corrcoef(a[L:][m], a[:-L][m])[0, 1]))
pd.DataFrame(rows).to_csv(P.out(P.CONTROLS_RESULTS / 'highd_accel_acf.csv'), index=False); print(pd.DataFrame(rows).round(3))
