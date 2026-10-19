import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from common.fig_style import *
nature_style()
A = pd.read_csv(P.CONTROLS_RESULTS / 'augment_markov.csv'); F = pd.read_csv(P.CONTROLS_RESULTS / 'fine_resolution.csv')
a = A[A.state=='signed_v'].groupby('model').val_ll.agg(['mean','std'])
f = F.groupby('model').val.agg(['mean','std'])
V = np.load(P.CONTROLS_RESULTS / 'vacf_kernels.npz'); S = json.load(open(P.CONTROLS_RESULTS / 'vacf_summary.json'))
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4)); (p1, p2), (p3, p4) = axs
# a: validation LL by state representation
items = [('V1 (20)', f, '$v_t$ [20]'), ('V2 (20,20)', f, '$v_t,v_{t-1}$ [20,20]'), ('V3', a, '$v_t,v_{t-1},v_{t-2}$ [20]'),
         ('V1 (40)', f, '$v_t$ [40]'), ('V1 (100)', f, '$v_t$ [100]'), ('V1 (400)', f, '$v_t$ [400]'),
         ('V2 (100,20)', f, '$v_t,v_{t-1}$ [100,20]'), ('V2 (100,100)', f, '$v_t,v_{t-1}$ [100,100]'),
         ('VA (100,a20)', f, '$v_t,a_t$ [100,20]'), ('VA2 (100,a20,v1_20)', f, '$v_t,a_t,v_{t-1}$'),
         ('VA1', a, '$v_t,a_t$ [20,7]'), ('VAI1', a, '$v_t,a_t,\\Delta v,g$ [20,7,5,5]')]
cols = []
for k, src, lab in items:
    hist = ('v_{t-1}' in lab) or ('v_{t-2}' in lab)
    cols.append(NPG[0] if hist else NPG[3])
y = np.arange(len(items))[::-1]
vals = [src.loc[k, 'mean'] for k, src, _ in items]
p1.barh(y, [v + 0.40 for v in vals], left=-0.40, color=cols, height=0.65)
for yi, v in zip(y, vals): p1.text(v + 0.003, yi, f'{v:.3f}', va='center', fontsize=5.5)
p1.set_yticks(y); p1.set_yticklabels([l for _, _, l in items], fontsize=6)
p1.set_xlim(-0.40, -0.19); p1.set_xticks([-0.40,-0.35,-0.30,-0.25,-0.20]); p1.set_xlabel('Validation log-likelihood per step (nats)')
p1.axvline(f.loc['V2 (20,20)','mean'], ls=':', lw=0.7, color='grey')
p1.text(0.98, 0.01, 'blue: current state only\nred: state + history\n[ ] = number of quantile bins', transform=p1.transAxes, ha='right', va='bottom', fontsize=5.5, color='#444')
panel_label(p1, 'a', x=-0.52)
# b: history gain under different state representations
g = [('coarse $v$ [20]', f.loc['V2 (20,20)','mean'] - f.loc['V1 (20)','mean']),
     ('fine $v$ [100]', f.loc['V2 (100,20)','mean'] - f.loc['V1 (100)','mean']),
     ('fine $v$ [100]\n+ fine history', f.loc['V2 (100,100)','mean'] - f.loc['V1 (100)','mean']),
     ('fine $v$ + $a_t$', f.loc['VA2 (100,a20,v1_20)','mean'] - f.loc['VA (100,a20)','mean'])]
p2.bar(range(len(g)), [x for _, x in g], color=[NPG[0], NPG[4], NPG[4], NPG[2]], width=0.6)
for i, (_, x) in enumerate(g): p2.text(i, x + 0.002, f'+{x:.3f}', ha='center', fontsize=6)
p2.set_xticks(range(len(g))); p2.set_xticklabels([l for l, _ in g], fontsize=6)
p2.set_ylabel('Gain from adding $v_{t-1}$ (nats/step)'); p2.set_ylim(0, 0.09)
p2.set_title('Apparent memory shrinks as the state is refined', fontsize=7)
panel_label(p2, 'b')
# c: VACF all vs long tracks
for key, lab, c in [('C_all', 'all tracks ($n$=%d)' % S['n_all_tracks'], NPG[3]), ('C_long', 'tracks $\\geq$24 s ($n$=%d)' % S['n_long_tracks'], NPG[0])]:
    C = V[key]; t = np.arange(len(C)) * 0.2
    p3.plot(t, C, marker='o', ms=2, color=c, label=lab)
    tau = S['dt0.2_all' if key == 'C_all' else 'dt0.2_long']['tau_C']
    p3.plot(t, np.exp(-t / tau), ls='--', lw=0.8, color=c, alpha=0.7)
p3.axhline(0, color='k', lw=0.5); p3.set_xlabel('Lag $t$ (s)'); p3.set_ylabel('Normalized VACF')
p3.legend(loc='lower left'); p3.text(0.98, 0.95, 'dashed: exponential (OU)\nwith the same $\\tau_C$', transform=p3.transAxes, ha='right', va='bottom', fontsize=5.5, color='#444')
panel_label(p3, 'c')
# d: memory fraction vs integration limit
for key, lab, c, dt in [('dt0.2_all', 'all tracks, $\\Delta t$=0.2 s', NPG[3], 0.2), ('dt0.2_long', 'long tracks, $\\Delta t$=0.2 s', NPG[0], 0.2), ('dt0.04_long', 'long tracks, $\\Delta t$=0.04 s', NPG[2], 0.04)]:
    K = V['K_all' if key == 'dt0.2_all' else ('K_long' if key == 'dt0.2_long' else 'K_raw')]
    Ts = np.arange(1, len(K)) * dt; gi = K[0] * dt; gm = np.cumsum(K[1:]) * dt
    p4.plot(Ts, gm / (gi + gm), color=c, label=lab + f'  ($K_1/K_0$={S[key]["K1_over_K0"]:+.2f})')
p4.axhline(0, color='k', lw=0.5); p4.set_ylim(-2, 1); p4.set_xlabel('Integration limit $T$ (s)')
p4.set_ylabel('$\\gamma_{mem}(T)/\\gamma_{total}(T)$'); p4.legend(loc='lower right')
panel_label(p4, 'd')
plt.tight_layout(); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_highd_augment.pdf')); fig.savefig(P.out(P.CONTROLS_FIGURES / 'fig_highd_augment.png'), dpi=200)
print({l: round(x, 4) for l, x in g})
