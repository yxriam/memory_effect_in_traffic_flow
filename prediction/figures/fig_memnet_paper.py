"""Paper figure: memory networks with the official Infomap 2.8.0 (two-level, directed, rawdir)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as P
import os
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.fig_style import *
nature_style(); R = str(P.PRED_NGRAM) + os.sep
off = pd.read_csv(P.PRED_INFOMAP / 'infomap_summary.csv').set_index('network')
base = pd.concat([pd.read_csv(R + f'ch6_{w}.csv') for w in ['taxi', 'geolife', 'highd']])
DS = [('taxi', 'Taxi\n1 km'), ('geolife', 'GeoLife\n500 m'), ('highd', 'highD\n20 bins')]
x = np.arange(3); w = 0.36
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.5))
H = []
for k, _ in DS:
    d = base[base.dataset == k].iloc[0]; s = base[base.dataset.str.startswith(k + '_markov')]
    H.append((1 - d.H2 / d.H1, (1 - s.H2 / s.H1).mean()))
H = np.array(H); ax = axs[0]
ax.bar(x - w/2, H[:, 0], w, color=NPG[0], label='data'); ax.bar(x + w/2, H[:, 1], w, color=NPG[3], label='first-order surrogate')
ax.set_ylabel(r'$1-H_2/H_1$ (plug-in)'); ax.set_ylim(0, 0.68); ax.legend(fontsize=5.5, loc='upper center', ncol=2, frameon=False)
ax.set_title('Entropy-rate reduction', loc='left'); panel_label(ax, 'a', x=-0.32)
ax = axs[1]
for j, (suf, col) in enumerate([('', NPG[0]), ('_surrogate', NPG[3])]):
    v = [off.loc[f'{k}{suf}_first', 'codelength'] - off.loc[f'{k}{suf}_memory', 'codelength'] for k, _ in DS]
    ax.bar(x + (j - .5) * w, v, w, color=col)
    for xi, vi in zip(x + (j - .5) * w, v): ax.text(xi, vi + 0.02, f'{vi:.2f}', ha='center', fontsize=5.5)
ax.set_ylabel(r'$L_{\rm first}-L_{\rm memory}$ (bits)'); ax.set_ylim(0, 1.25); ax.set_title('Codelength saving', loc='left'); panel_label(ax, 'b', x=-0.32)
ax = axs[2]
for j, (suf, col) in enumerate([('', NPG[0]), ('_surrogate', NPG[3])]):
    v = [off.loc[f'{k}{suf}_memory', 'physical_in_multiple_modules'] / off.loc[f'{k}{suf}_memory', 'physical_nodes'] for k, _ in DS]
    ax.bar(x + (j - .5) * w, v, w, color=col)
ax.set_ylabel('share of physical nodes\nin two or more modules'); ax.set_ylim(0, 1); ax.set_title('Overlapping modules', loc='left'); panel_label(ax, 'c', x=-0.32)
for ax in axs: ax.set_xticks(x); ax.set_xticklabels([l for _, l in DS], fontsize=6)
fig.savefig(P.out(P.PRED_FIGURES / 'fig_memnet_paper.pdf')); fig.savefig(P.out(P.PRED_FIGURES / 'fig_memnet_paper.png'), dpi=150)
print(H)
