import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import os
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style(); R = str(P.PRED_NGRAM) + os.sep
dec = pd.read_csv(R + 'ch6_decomp_highd_geolife_taxi.csv')
base = pd.concat([pd.read_csv(R + f'ch6_{w}.csv') for w in ['taxi', 'geolife', 'highd']])
DS = [('taxi', 'Taxi\n(1 km grid)'), ('geolife', 'GeoLife\n(500 m)'), ('highd', 'highD\n(speed states)')]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.45))
x = np.arange(3); w = 0.36
# a: entropy rates
for j, (lab, col, sel) in enumerate([('data', NPG[0], lambda s: s.dataset == s.dataset.iloc[0]), ('first-order surrogate', NPG[3], None)]):
    pass
H = []
for k, _ in DS:
    d = base[base.dataset == k].iloc[0]; s = base[base.dataset.str.startswith(k + '_markov')]
    H.append((d.H1, d.H2, s.H1.mean(), s.H2.mean()))
H = np.array(H); ax = axs[0]
ax.bar(x - w / 2, 1 - H[:, 1] / H[:, 0], w, color=NPG[0], label='data'); ax.bar(x + w / 2, 1 - H[:, 3] / H[:, 2], w, color=NPG[3], label='Markov surrogate')
ax.set_xticks(x); ax.set_xticklabels([l for _, l in DS], fontsize=6); ax.set_ylabel('1 − H₂/H₁ (plug-in)'); ax.legend(fontsize=5.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0)); ax.set_ylim(0, 0.6); ax.set_title('Entropy-rate reduction', loc='left'); panel_label(ax, 'a', x=-0.3)
# b: codelength first vs memory (optimised), data vs surrogate
ax = axs[1]
for j, (net, col) in enumerate([('data', NPG[0]), ('markov_surrogate', NPG[3])]):
    s = dec[dec.network == net].set_index('dataset').loc[[k for k, _ in DS]]
    ax.bar(x + (j - 0.5) * w, s.first_L - s.mem_L, w, color=col, label='data' if net == 'data' else 'Markov surrogate')
    for xi, v in zip(x + (j - 0.5) * w, s.first_L - s.mem_L): ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in DS], fontsize=6); ax.set_ylabel('L(first-order) − L(memory)  [bits]'); ax.set_title('Codelength gain', loc='left'); panel_label(ax, 'b', x=-0.3)
# c: overlap
ax = axs[2]
for j, (sel, col) in enumerate([(lambda k: base[base.dataset == k], NPG[0]), (lambda k: base[base.dataset.str.startswith(k + '_markov')], NPG[3])]):
    v = [sel(k).overlap_frac_flow.mean() for k, _ in DS]
    ax.bar(x + (j - 0.5) * w, v, w, color=col)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in DS], fontsize=6); ax.set_ylabel('flow share of physical nodes\nin ≥2 modules'); ax.set_title('Overlapping modules', loc='left'); panel_label(ax, 'c', x=-0.3)
fig.savefig(P.out(P.PRED_FIGURES / 'fig_ch6_memory_network.pdf')); fig.savefig(P.out(P.PRED_FIGURES / 'fig_ch6_memory_network.png'), dpi=150)
