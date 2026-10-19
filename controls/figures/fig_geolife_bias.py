import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style()
D = pd.read_csv(P.CONTROLS_RESULTS / 'geolife_robust_per_user.csv')
D['m'] = D.grid.map({0.0001: 10, 0.0005: 50, 0.001: 100, 0.002: 200})
for b in (1, 3, 10):
    D[f'd21_b{b}'] = D[f'll2_b{b}'] - D[f'll1_b{b}']; D[f'd32_b{b}'] = D[f'll3_b{b}'] - D[f'll2_b{b}']
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6)); (a, b), (c, d) = axs
s = D[(D.m == 10) & (D.dejitter == 0)]
a.scatter(s.cmi2_null, s.cmi2_plugin, s=7, color=NPG[3], alpha=0.7, lw=0)
mx = s.cmi2_plugin.max() * 1.05; a.plot([0, mx], [0, mx], ls='--', color='grey', lw=0.7)
a.set_xlabel('Permutation-null mean CMI (bits)'); a.set_ylabel('Plug-in CMI $\\hat I_2$ (bits)')
a.text(0.04, 0.96, f'10 m grid, {len(s)} users\nmedian plug-in {s.cmi2_plugin.median():.3f}\nmedian null {s.cmi2_null.median():.3f}\nmedian excess {s.cmi2_excess.median():.3f}',
       transform=a.transAxes, va='top', fontsize=6)
panel_label(a, 'a')
ms = [10, 50, 100, 200]; x = np.arange(4); w = 0.2
for k, (dj, al) in enumerate([(0, 1.0), (1, 0.45)]):
    g = D[D.dejitter == dj].groupby('m')
    b.bar(x + (k - 0.5) * 2 * w - w / 2, [g.cmi2_plugin.median()[m] for m in ms], w, color=NPG[5], alpha=al, label='plug-in' if dj == 0 else None)
    b.bar(x + (k - 0.5) * 2 * w + w / 2, [g.cmi2_excess.median()[m] for m in ms], w, color=NPG[0], alpha=al, label='bias-corrected' if dj == 0 else None)
b.set_xticks(x); b.set_xticklabels([f'{m} m' for m in ms]); b.set_xlabel('Grid cell size')
b.set_ylabel('Median second-order CMI (bits)'); b.legend(loc='upper left')
b.set_ylim(0, 0.78); b.text(0.02, 0.86, 'solid: collapsed paths\nlight: + de-jitter (A$\\to$B$\\to$A removed)', transform=b.transAxes, ha='left', va='top', fontsize=5.5, color='#444')
panel_label(b, 'b')
for dj, ls, mk in [(0, '-', 'o'), (1, ':', 's')]:
    g = D[D.dejitter == dj].groupby('m')
    c.plot(x, [(g.cmi2_p.apply(lambda p: (p < .05).mean()))[m] for m in ms], ls=ls, marker=mk, ms=3.5, color=NPG[3], label=f'2nd order{" (de-jitter)" if dj else ""}')
    c.plot(x, [(g.cmi3_p.apply(lambda p: (p < .05).mean()))[m] for m in ms], ls=ls, marker=mk, ms=3.5, color=NPG[1], label=f'3rd order{" (de-jitter)" if dj else ""}')
c.axhline(0.05, color=NPG[0], ls='--', lw=0.7); c.set_ylim(0, 1.05)
c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
c.set_xticks(x); c.set_xticklabels([f'{m} m' for m in ms]); c.set_xlabel('Grid cell size'); c.set_ylabel('Users with $p<0.05$')
c.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07), fontsize=5.5); panel_label(c, 'c')
s0 = D[D.dejitter == 0].groupby('m')
for col, lab, colr in [('d21', '2nd $-$ 1st', NPG[3]), ('d32', '3rd $-$ 2nd', NPG[1])]:
    med = {bb: [s0[f'{col}_b{bb}'].median()[m] for m in ms] for bb in (1, 3, 10)}
    lo = np.minimum(med[1], med[10]); hi = np.maximum(med[1], med[10])
    d.fill_between(x, lo, hi, color=colr, alpha=0.18, lw=0)
    d.plot(x, med[3], marker='o', ms=3.5, color=colr, label=lab + ' ($\\beta=3$)')
pos = [(D[(D.dejitter == 0) & (D.m == m)]['d21_b3'] > 0).mean() for m in ms]
for xi, m in zip(x, ms):
    d.text(xi, s0['d21_b3'].median()[m] + 0.02, f'{pos[xi]*100:.0f}%', ha='center', fontsize=5.5, color=NPG[3])
d.axhline(0, color='k', lw=0.5); d.set_xticks(x); d.set_xticklabels([f'{m} m' for m in ms]); d.set_xlabel('Grid cell size')
d.set_ylabel('Median held-out gain (bits/step)'); d.legend(loc='upper left')
d.text(0.98, 0.03, 'numbers: users with positive 2nd$-$1st gain\nband: $\\beta\\in[1,10]$', transform=d.transAxes, ha='right', va='bottom', fontsize=5.5, color='#444')
panel_label(d, 'd')
plt.tight_layout()
fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_bias.pdf')); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_bias.png'), dpi=200)
