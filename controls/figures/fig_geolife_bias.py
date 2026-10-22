"""Fig. 1 of the paper: apparent versus genuine memory in GeoLife (redrawn 2026-10-22 at the text width, text at
9-9.5 pt; numbers moved to the caption).

  (a) per-user plug-in CMI against the mean of its stratified permutation null, 10 m grid
  (b) median plug-in and bias-corrected CMI per grid, with and without de-jittering
  (c) share of users with p < 0.05 for the second- and third-order excess CMI
  (d) held-out gain of order 2 over 1 and 3 over 2 with the back-off strengths tuned for each order on an inner split
      (geolife_heldout_tuned.py, P3-9); the band spans the medians with one beta for all levels (beta = 3 and 10)
Inputs: controls/results/geolife_robust_per_user.csv, geolife_heldout_tuned.csv.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common.fig_style import NPG, nature_style

nature_style()
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'lines.linewidth': 1.4})
RED, BLUE, SALMON = NPG[0], NPG[3], NPG[4]
W = 6.3


def label(ax, s):
    ax.text(-0.22, 1.1, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


D = pd.read_csv(P.CONTROLS_RESULTS / 'geolife_robust_per_user.csv')
D['m'] = D.grid.map({0.0001: 10, 0.0005: 50, 0.001: 100, 0.002: 200})
T = pd.read_csv(P.CONTROLS_RESULTS / 'geolife_heldout_tuned.csv'); T['m'] = T.grid.str.rstrip('m').astype(int)
ms = [10, 50, 100, 200]; x = np.arange(4)
fig, axs = plt.subplots(2, 2, figsize=(W, 5.0), gridspec_kw=dict(wspace=0.42, hspace=0.55)); (a, b), (c, d) = axs

s = D[(D.m == 10) & (D.dejitter == 0)]
a.scatter(s.cmi2_null, s.cmi2_plugin, s=9, color=BLUE, alpha=0.7, lw=0)
mx = s.cmi2_plugin.max() * 1.05; a.plot([0, mx], [0, mx], ls='--', color='grey', lw=0.8)
a.set_xlabel('Null mean CMI (bits)'); a.set_ylabel('Plug-in CMI (bits)'); label(a, 'a')

w = 0.2
for k, (dj, al) in enumerate([(0, 1.0), (1, 0.5)]):
    g = D[D.dejitter == dj].groupby('m')
    b.bar(x + (k - 0.5) * 2 * w - w / 2, [g.cmi2_plugin.median()[m] for m in ms], w, color=SALMON, alpha=al)
    b.bar(x + (k - 0.5) * 2 * w + w / 2, [g.cmi2_excess.median()[m] for m in ms], w, color=BLUE, alpha=al)
b.set_xticks(x); b.set_xticklabels([f'{m} m' for m in ms]); b.set_xlabel('Grid cell size')
b.set_ylabel('Median CMI (bits)'); b.set_ylim(0, 0.75); label(b, 'b')
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SALMON, label='plug-in'), plt.Rectangle((0, 0), 1, 1, color=BLUE, label='bias-corrected')],
         loc='upper left', handlelength=1.2)

for dj, ls, mk in [(0, '-', 'o'), (1, ':', 's')]:
    g = D[D.dejitter == dj].groupby('m')
    c.plot(x, [100 * g.cmi2_p.apply(lambda p: (p < .05).mean())[m] for m in ms], ls=ls, marker=mk, ms=4, color=BLUE)
    c.plot(x, [100 * g.cmi3_p.apply(lambda p: (p < .05).mean())[m] for m in ms], ls=ls, marker=mk, ms=4, color=RED)
c.axhline(5, color='grey', ls='--', lw=0.8); c.set_ylim(0, 100)
c.set_xticks(x); c.set_xticklabels([f'{m} m' for m in ms]); c.set_xlabel('Grid cell size'); c.set_ylabel('Users with $p<0.05$ (%)')
label(c, 'c')

D0 = D[D.dejitter == 0]
for (hi, lo), tcol, colr in [(('ll2', 'll1'), 'G2', BLUE), (('ll3', 'll2'), 'G3', RED)]:
    fixed = {bb: [D0.assign(v=D0[f'{hi}_b{bb}'] - D0[f'{lo}_b{bb}']).groupby('m').v.median()[m] for m in ms] for bb in (3, 10)}
    d.fill_between(x, np.minimum(fixed[3], fixed[10]), np.maximum(fixed[3], fixed[10]), color=colr, alpha=0.18, lw=0)
    tuned = [T[T.m == m][f'{tcol}_tuned'].median() for m in ms]
    d.plot(x, tuned, marker='o', ms=4, color=colr)
    if tcol == 'G2':
        for xi, m in zip(x, ms):
            pos = (T[T.m == m].G2_tuned.dropna() > 0).mean()
            d.text(xi, tuned[xi] + 0.012, f'{100 * pos:.0f}%', ha='center', va='bottom', fontsize=8, color=colr)
d.axhline(0, color='k', lw=0.6); d.set_xticks(x); d.set_xticklabels([f'{m} m' for m in ms]); d.set_xlabel('Grid cell size')
d.set_ylabel('Held-out gain (bits/step)'); d.set_ylim(-0.01, 0.16); label(d, 'd')

h = [Line2D([], [], color=BLUE, marker='o', ms=4, label='2nd order'), Line2D([], [], color=RED, marker='o', ms=4, label='3rd order'),
     Line2D([], [], color='k', ls=':', marker='s', ms=4, label='de-jittered (c)'), plt.Rectangle((0, 0), 1, 1, color='0.75', label='one $\\beta$ for all orders (d)')]
fig.subplots_adjust(bottom=0.15, left=0.1, right=0.98, top=0.95)
fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_bias.pdf')); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_geolife_bias.png'), dpi=200)
print('panel a medians: plug-in %.4f null %.4f excess %.4f (n=%d)' % (s.cmi2_plugin.median(), s.cmi2_null.median(), s.cmi2_excess.median(), len(s)))
