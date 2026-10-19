import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import pandas as pd, numpy as np, matplotlib.pyplot as plt
C = pd.read_csv(P.GEOLIFE_OUTPUTS / '04_mode_memory' / 'mode_kernel_curves.csv'); A = pd.read_csv(P.CONTROLS_RESULTS / 'geolife_kernel_window_sensitivity.csv').set_index('mode')
fig, ax = plt.subplots(figsize=(10, 5.2))
for m in ['walk', 'bus', 'bike', 'car', 'subway', 'taxi', 'train', 'airplane']:
    s = C[(C['mode'] == m) & (C.lag > 0)].sort_values('lag')
    ax.plot(s.time_s, np.abs(s.kernel), lw=1.6, alpha=0.9, label=f"{m} (alpha={A.loc[m, '10-400']:.2f})")
ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3)
ax.set_xlabel('Lag time, seconds'); ax.set_ylabel('|Memory kernel|'); ax.set_title('GeoLife memory-kernel decay by transportation mode')
ax.legend(ncol=2, fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(P.out(P.CONTROLS_FIGURES / 'geolife_kernel.png'), dpi=160)
