import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as _P
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json
from common.fig_style import *
nature_style()
R = pd.read_csv(_P.CONTROLS_RESULTS / 'coarse_model.csv'); A = pd.read_csv(_P.CONTROLS_RESULTS / 'highd_accel_acf.csv'); P = json.load(open(_P.CONTROLS_RESULTS / 'coarse_model_params.json'))
fig, (p1, p2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.32))
sty = {'highD': (NPG[3], 'o', '-', 'highD data'), 'AR2': (NPG[0], 's', '--', 'AR(2), underdamped'),
       'AR1': (NPG[2], '^', ':', 'AR(1), Markov in $v$')}
for d, (c, m, ls, lab) in sty.items():
    s = R[R.data == d].sort_values('k'); p1.plot(s.k, s.gain, marker=m, ls=ls, color=c, ms=3.5, label=lab)
p1.axhline(0, color='grey', lw=0.6); p1.set_xscale('log'); p1.set_xticks([5, 10, 20, 40, 100, 200, 400]); p1.set_xticklabels([5, 10, 20, 40, 100, 200, 400])
p1.set_xlabel('Number of speed bins $k$ for $v_t$ and $v_{t-1}$'); p1.set_ylabel('Held-out gain of $v_{t-1}$ (nats/step)')
p1.axvspan(15, 27, color='#eeeeee', zorder=0); p1.text(20, -0.013, 'usual\nresolution', ha='center', va='bottom', fontsize=5.5, color='#666')
p1.legend(loc='lower left', fontsize=5.5, bbox_to_anchor=(0.0, 0.12)); p1.set_ylim(-0.015, 0.085); panel_label(p1, 'a')
p2.bar(A.lag_s, A.acf, width=0.14, color=[NPG[0] if x < 0 else NPG[3] for x in A.acf])
p2.axhline(0, color='grey', lw=0.6); p2.set_xlabel('Lag (s)'); p2.set_ylabel('Autocorrelation of $a_t$')
p2.text(0.97, 0.12, f'lag 0.2 s: {A.acf.iloc[0]:.2f}\n(pure position noise: $-0.5$)', transform=p2.transAxes, ha='right', fontsize=6)
panel_label(p2, 'b')
fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_coarse_model.pdf')); fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_coarse_model.png'), dpi=150)
