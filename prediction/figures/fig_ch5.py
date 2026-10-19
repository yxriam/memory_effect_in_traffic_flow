import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import os
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style(); R = str(P.PRED_NGRAM) + os.sep
DS = [('taxi_traj20', 'Taxi (per-trajectory 20×20)'), ('geolife_bj500m', 'GeoLife (Beijing, 500 m)'), ('highd_v20', 'highD (speed, 20 states)'), ('highd_v100', 'highD (speed, 100 states)')]
ST = {'addalpha': (NPG[4], 's', 'add-α (no back-off)'), 'jm': (NPG[1], '^', 'interpolated back-off'), 'wb': (NPG[2], 'D', 'Witten–Bell / PPM-C (VOM)'), 'kn': (NPG[0], 'o', 'modified Kneser–Ney')}
def hi_lim(g):
    lo = g[g.kind != 'unigram'].test_bits.min(); hi = g[(g.kind != 'unigram') & (g.kind != 'addalpha')].test_bits.max(); return hi + 0.6 * (hi - lo) + 0.01
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8), gridspec_kw=dict(hspace=0.5, wspace=0.35)); axs = axs.ravel()
for ax, (d, lab), L in zip(axs, DS, 'abcd'):
    g = pd.read_csv(R + f'ch5_pooled_grid_{d}.csv'); g['order'] = g.N - 1
    u = g[g.kind == 'unigram'].test_bits.iloc[0]
    for k, (c, m, l) in ST.items():
        s = g[g.kind == k]
        if k == 'jm': s = s.loc[s.groupby('N').val_bits.idxmin()]
        s = s.sort_values('order'); ax.plot(s.order, s.test_bits, marker=m, color=c, ms=3, label=l)
    ax.set_title(lab, loc='left', fontsize=7); ax.set_xlabel('Markov order k')
    off = g[(g.kind=='addalpha')&(g.test_bits>hi_lim(g))]
    if len(off): ax.text(0.98, 0.85, 'add-α off scale: ' + ', '.join(f'{v:.1f}' for v in off.sort_values('N').test_bits), transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color=NPG[4]); ax.set_ylabel('test bits / step')
    lo = g[g.kind != 'unigram'].test_bits.min(); hi = g[(g.kind != 'unigram') & (g.kind != 'addalpha')].test_bits.max()
    ax.set_ylim(lo - 0.15 * (hi - lo) - 0.01, hi + 0.6 * (hi - lo) + 0.01); ax.text(0.98, 0.95, f'order 0: {u:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=6)
    panel_label(ax, L, x=-0.25)
axs[0].legend(fontsize=5.5, loc='upper left', bbox_to_anchor=(0, 0.88))
lc = pd.read_csv(R + 'ch5_learning_curve_taxi_traj20_geolife_bj500m_highd_v20.csv')
for ax, (d, lab), L in zip(axs[4:], DS[:2], 'ef'):
    s = lc[lc.dataset == d]
    for o, c in zip((1, 2, 3, 4), [NPG[3], NPG[0], NPG[2], NPG[4]]):
        t = s[s.order == o]; ax.plot(t.train_tokens, t.test_bits, marker='o', ms=3, color=c, label=f'order {o}')
    ax.set_xscale('log'); ax.set_xlabel('training symbols'); ax.set_ylabel('test bits / step (KN)'); ax.set_title(lab.split(' (')[0] + ': learning curve (KN)', loc='left', fontsize=7)
    from matplotlib.ticker import LogLocator, NullFormatter; ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=5.5); panel_label(ax, L, x=-0.25)
fig.savefig(P.out(P.PRED_FIGURES / 'fig_ch5_ngram.pdf')); fig.savefig(P.out(P.PRED_FIGURES / 'fig_ch5_ngram.png'), dpi=150)
