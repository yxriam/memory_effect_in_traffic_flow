import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style()
H = pd.read_csv(P.CONTROLS_RESULTS / 'hurst_control.csv')
modes = ['walk', 'bus', 'car']
series = [('H_speed', 'speed', NPG[3], 1800.0), ('H_ar1', 'AR(1) surrogate', NPG[0], 1800.0), ('H_shuffled', 'shuffled', NPG[5], 1800.0),
          ('H_increment', 'increments', NPG[1], 1800.0), ('H_speed', 'speed, gaps > 30 s split', NPG[2], 30.0)]
fig, ax = plt.subplots(figsize=(7.2, 2.8))
w = 0.15
for k, (col, lab, colr, gap) in enumerate(series):
    for i, m in enumerate(modes):
        v = H[(H['mode'] == m) & (H.gap == gap)][col].dropna()
        pos = i + (k - 2) * w
        ax.boxplot([v], positions=[pos], widths=w * 0.85, showfliers=False, patch_artist=True, whis=(5, 95),
                   medianprops=dict(color='white', lw=1.1), boxprops=dict(facecolor=colr, edgecolor=colr, lw=0.6),
                   whiskerprops=dict(color=colr, lw=0.6), capprops=dict(color=colr, lw=0.6))
    ax.plot([], [], color=colr, lw=5, label=lab)
n = H[H.gap == 1800.0].groupby('mode').size()
ax.set_xticks(range(3)); ax.set_xticklabels([f'{m} ($n$={n[m]})' for m in modes])
ax.axhline(0.5, ls='--', color='grey', lw=0.7); ax.set_ylabel('R/S Hurst exponent')
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=6)
plt.tight_layout(); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_hurst_control.pdf')); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_hurst_control.png'), dpi=200)
