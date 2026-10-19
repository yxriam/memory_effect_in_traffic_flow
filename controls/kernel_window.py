"""Power-law exponent of |K(t)| for five fitting windows (lags in 5 s steps), from mode_kernel_curves.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import sys, numpy as np, pandas as pd
from scipy import stats
W = {'10-400': (2, 80), '10-100': (2, 20), '10-200': (2, 40), '20-400': (4, 80), '5-50': (1, 10)}
def alpha(y, lo, hi):
    y = np.asarray(y, float); idx = np.arange(len(y)); t = idx * 5.0
    m = (idx >= lo) & (idx <= hi) & np.isfinite(y) & (np.abs(y) > 0)
    return float(-stats.linregress(np.log(t[m]), np.log(np.abs(y[m])))[0]) if m.sum() >= 4 else np.nan
SRC = sys.argv[1] if len(sys.argv) > 1 else P.GEOLIFE_OUTPUTS / '04_mode_memory' / 'mode_kernel_curves.csv'
DST = sys.argv[2] if len(sys.argv) > 2 else P.out(P.CONTROLS_RESULTS / 'geolife_kernel_window_sensitivity.csv')
C = pd.read_csv(SRC); out = []
for mo, s in C.groupby('mode'):
    s = s.sort_values('lag'); out.append(dict(mode=mo, **{k: round(alpha(s.kernel, *v), 2) for k, v in W.items()}))
R = pd.DataFrame(out); print(R.to_string(index=False)); R.to_csv(DST, index=False)
