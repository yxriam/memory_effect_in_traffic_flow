import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from common.fig_style import *
T = str(P.TAXI_RESULTS) + os.sep
C2, C3 = NPG[3], NPG[1]
def box(ax, data, pos, color, width=0.32, alpha=1.0):
    ax.boxplot(data, positions=pos, widths=width, showfliers=False, patch_artist=True, whis=(5, 95),
               medianprops=dict(color='white', lw=1.2), boxprops=dict(facecolor=color, edgecolor=color, lw=0.6, alpha=alpha),
               whiskerprops=dict(color=color, lw=0.6, alpha=alpha), capprops=dict(color=color, lw=0.6, alpha=alpha))
both = pd.read_csv(T + 'memory_full_summary.csv').set_index(['grid_mode', 'grid', 'compressed']).sort_index()
tun = pd.read_csv(T + 'memory_backoff_tuned_summary.csv')
sweeps = {'per_traj': pd.read_csv(T + 'memory_backoff_sweep.csv'), 'global': pd.read_csv(T + 'memory_backoff_sweep_global.csv')}
grids = [10, 20, 30]
nature_style()
fig2, (e, f, h) = plt.subplots(1, 3, figsize=(7.2, 2.7))
MODES = [('per_traj', 'Per-trajectory grid', 'o', '-'), ('global', 'Global grid', 's', '--')]
xs = np.arange(3); w = 0.36
for k, (mode, lab, mk, ls) in enumerate(MODES):
    Sm = both.loc[mode]
    e.bar(xs + (k - 0.5) * w, [Sm.loc[(g, True), 'stay_frac_median'] for g in grids], w, color=NPG[3], alpha=1 if mode == 'per_traj' else 0.4, label=f'{lab}: stay')
e2 = e.twinx(); e2.spines['top'].set_visible(False)
for k, (mode, lab, mk, ls) in enumerate(MODES):
    Sm = both.loc[mode]
    e2.plot(xs, [Sm.loc[(g, True), 'len_median'] for g in grids], marker=mk, ls=ls, ms=3.5, color=NPG[0], label=f'{lab}: length')
e.set_ylim(0, 1.55); e.set_yticks([0, .25, .5, .75, 1]); e.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
e2.set_ylim(0, 650)
e.set_ylabel('Stay fraction\n(removed by compression)'); e2.set_ylabel('Compressed length (steps)', color=NPG[0]); e2.tick_params(axis='y', colors=NPG[0])
e.set_xticks(xs); e.set_xticklabels(grids); e.set_xlabel('Grid resolution $g$ ($g\\times g$)')
h1, l1 = e.get_legend_handles_labels(); h2, l2 = e2.get_legend_handles_labels()
e.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=5, handlelength=1.4, labelspacing=0.3); panel_label(e, 'a', x=-0.34)
for k, (mode, lab, mk, ls) in enumerate(MODES):
    Sm = both.loc[mode]
    f.plot(xs, [Sm.loc[(g, True), 'cmi2_sig05'] for g in grids], marker=mk, ls=ls, ms=3.5, color=C2, label=f'{lab}, 2nd order')
    f.plot(xs, [Sm.loc[(g, True), 'cmi3_sig05'] for g in grids], marker=mk, ls=ls, ms=3.5, color=C3, label=f'{lab}, 3rd order')
    for xi, g in zip(xs, grids):
        f.annotate(f"{Sm.loc[(g, True), 'cmi2_median']:.3f}", (xi, Sm.loc[(g, True), 'cmi2_sig05']), textcoords='offset points',
                   xytext=(0, 5) if mode == 'per_traj' else (7, -2), ha='center' if mode == 'per_traj' else 'left', fontsize=5, color=C2)
f.axhline(0.05, color=NPG[0], ls='--', lw=0.7)
f.set_ylim(0, 1.25); f.set_yticks([0, .2, .4, .6, .8]); f.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
f.set_xlim(-0.4, 2.4); f.set_xticks(xs); f.set_xticklabels(grids); f.set_xlabel('Grid resolution $g$ (compressed)')
f.set_ylabel('Fraction with $p<0.05$'); f.legend(loc='upper left', fontsize=5, labelspacing=0.3, ncol=1); panel_label(f, 'b', x=-0.3)
for k, (mode, lab, mk, ls) in enumerate(MODES):
    sw_ = sweeps[mode]; tu_ = tun[tun.grid_mode == mode].set_index('grid')
    for i, g in enumerate(grids):
        s = sw_[(sw_.grid == g) & sw_.compressed]
        b1, b2 = tu_.loc[g, 'beta1_star'], tu_.loc[g, 'beta2_star']
        d21 = s[f'll2_b{b2:g}'] - s[f'll1_b{b1:g}']
        box(h, [d21], [i + (k - 0.5) * 0.36], C2, width=0.3, alpha=1 if mode == 'per_traj' else 0.4)
        h.text(i + (k - 0.5) * 0.36, -0.13 - 0.025 * k, f"{(d21 > 0).mean()*100:.0f}%", ha='center', fontsize=5, color=C2, alpha=1 if mode == 'per_traj' else 0.6)
h.axhline(0, color='0.3', lw=0.5); h.set_ylim(-0.17, 0.27)
h.set_xticks(xs); h.set_xticklabels(grids); h.set_xlabel('Grid resolution $g$ (compressed)')
h.set_ylabel(r'$\Delta$LL 2nd $-$ 1st, tuned $\beta^*$ (nat/step)')
h.plot([], [], color=C2, lw=4, label='Per-trajectory grid'); h.plot([], [], color=C2, lw=4, alpha=0.4, label='Global grid')
h.legend(loc='upper left', fontsize=5, labelspacing=0.3); panel_label(h, 'c', x=-0.3)
fig2.tight_layout(w_pad=1.6)
for ext in ('png', 'pdf'): fig2.savefig(P.out(P.CONTROLS_FIGURES / f'fig_taxi_gridmode.{ext}'), dpi=200 if ext == 'png' else None)
