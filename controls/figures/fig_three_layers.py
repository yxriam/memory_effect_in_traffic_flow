import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as _P
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style()
MODES = ['walk', 'bike', 'bus', 'subway', 'taxi', 'car', 'train']
ROW = {'walk': 'non-motorised', 'bike': 'non-motorised', 'bus': 'shared road', 'taxi': 'shared road', 'car': 'shared road', 'subway': 'exclusive track', 'train': 'exclusive track'}
RC = {'non-motorised': NPG[2], 'shared road': NPG[3], 'exclusive track': NPG[0]}
sp = pd.read_csv(_P.CONTROLS_RESULTS / 'memory_by_mode_speed.csv').groupby('mode')[['gain_coarse', 'gain_fine', 'gain_after_a']].agg(['mean', 'std'])
r = pd.read_csv(_P.CONTROLS_RESULTS / 'row_persistence.csv'); r = r[(r.gap == 30) & r['mode'].isin(MODES)]
pa = pd.read_csv(_P.CONTROLS_RESULTS / 'memory_by_mode_path.csv'); pa = pa[pa.dejitter == 0]
om = pd.read_csv(_P.CONTROLS_RESULTS / 'origin_memory_geolife.csv').query('dataset=="GeoLife"')
t2 = pd.read_csv(_P.CONTROLS_RESULTS / 'taxi_origin_0.002.csv').iloc[0]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.45, wspace=0.32)); (p1, p2), (p3, p4) = axs
# a speed memory by mode
x = np.arange(len(MODES)); w = 0.26
for j, (c, lab, col) in enumerate([('gain_coarse', 'coarse $v$ [10]', NPG[4]), ('gain_fine', 'fine $v$ [50]', NPG[0]), ('gain_after_a', 'fine $v$ [50] + $a$ [10]', NPG[3])]):
    p1.bar(x + (j - 1) * w, sp.loc[MODES, (c, 'mean')], w, yerr=sp.loc[MODES, (c, 'std')], color=col, label=lab, error_kw=dict(lw=0.5, capsize=1))
p1.axhline(0, color='grey', lw=0.6); p1.set_xticks(x); p1.set_xticklabels(MODES)
p1.set_ylabel('Held-out gain of $v_{t-1}$ (nats/step)'); p1.legend(loc='upper right'); p1.set_title('Layer 1: speed memory', loc='left')
panel_label(p1, 'a')
# b inertia vs stop frequency
for m in MODES:
    s = r[r['mode'] == m]; st = s.stops_per_km.fillna(0) + 0.02
    q = st.quantile([.25, .5, .75]).to_numpy(); t = s.tau_c.quantile([.25, .5, .75]).to_numpy()
    p2.errorbar(q[1], t[1], xerr=[[q[1] - q[0]], [q[2] - q[1]]], yerr=[[t[1] - t[0]], [t[2] - t[1]]], fmt='o', color=RC[ROW[m]], ms=4, lw=0.6, capsize=1)
    off = {'car': (-14, 5), 'taxi': (4, 5)}.get(m, (4, 3))
    p2.annotate(m, (q[1], t[1]), textcoords='offset points', xytext=off, fontsize=6)
p2.set_xscale('log'); p2.set_yscale('log'); p2.set_xlabel('Stops per km (+0.02)'); p2.set_ylabel('Speed decorrelation time $\\tau_c$ (s)')
for k, c in RC.items(): p2.plot([], [], 'o', color=c, label=k, ms=4)
p2.legend(loc='lower left'); p2.set_title('Layer 2: inertia', loc='left')
p2.text(0.97, 0.95, 'Spearman (mode medians)\n$\\rho=-0.98$, $p<10^{-4}$', transform=p2.transAxes, ha='right', va='top', fontsize=6)
panel_label(p2, 'b')
# c phase diagram
for m in MODES:
    s = r[r['mode'] == m]; P = pa[pa['mode'] == m].rel_excess
    t = s.tau_c.median(); y = P.median(); yl, yh = P.quantile(.25), P.quantile(.75)
    tl, th = s.tau_c.quantile(.25), s.tau_c.quantile(.75)
    mk = 'o' if m != 'train' else 'D'
    p3.errorbar(t, y, xerr=[[t - tl], [th - t]], yerr=[[y - yl], [yh - y]], fmt=mk, color=RC[ROW[m]], ms=4.5, lw=0.6, capsize=1, mfc='white' if m == 'train' else RC[ROW[m]])
    off = {'bus': (-16, 5), 'bike': (4, -9)}.get(m, (4, 4))
    p3.annotate(m, (t, y), textcoords='offset points', xytext=off, fontsize=6)
p3.set_xscale('log'); p3.set_xlabel('Inertia: speed decorrelation time $\\tau_c$ (s)'); p3.set_ylabel('Path memory: relative excess CMI')
p3.text(0.98, 0.97, 'open marker: mostly untestable\n(one predecessor per cell)', transform=p3.transAxes, ha='right', va='top', fontsize=5.5, color='#555')
p3.set_title('Layer 3 vs layer 2', loc='left'); panel_label(p3, 'c')
# d origin memory
labs = ['third-order\n$I(S_{t+1};S_{t-2}|S_{t-1},S_t)$', 'trip origin\n$I(S_{t+1};O|S_{t-1},S_t)$', 'origin beyond\nthird order']
g = [om.cmi3_excess.median(), om.origin_excess.median(), om.origin_beyond3_excess.median()]
tx = [t2.cmi3, t2.origin, t2.origin3]
xx = np.arange(3); p4.bar(xx - 0.18, g, 0.36, color=NPG[1], label='GeoLife (per user, median; 100 m)')
p4.bar(xx + 0.18, tx, 0.36, color=NPG[8], label='Taxi (100 vehicles pooled; 200 m)')
for i in range(3):
    p4.text(i - 0.18, g[i] + 0.0003, f'{g[i]:.4f}', ha='center', fontsize=5.5); p4.text(i + 0.18, tx[i] + 0.0003, f'{tx[i]:.4f}', ha='center', fontsize=5.5)
p4.set_xticks(xx); p4.set_xticklabels(labs, fontsize=5.8); p4.set_ylabel('Bias-corrected CMI (bits/step)')
p4.legend(loc='upper right'); p4.set_title('Layer 3: purposive component', loc='left'); p4.set_ylim(0, 0.019); panel_label(p4, 'd')
fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_three_layers.pdf')); fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_three_layers.png'), dpi=150)
