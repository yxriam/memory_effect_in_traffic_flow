"""Fig. 9 of the paper (2026-10-22): second-order path memory is mostly the heading, and what is left further back.

  (a) Held-out gains with tuned back-off (P3-1), as a share of the second-order gain G2 of the same units:
      the heading alone (GA) and the previous cell once the heading is in the state (R). GeoLife: sums over
      users weighted by test steps; taxi: all vehicles pooled, heading = direction recorded by the device at the entry
      fix (2026-10-22; before: bearing from the previous fix, which mostly lies in the previous cell).
  (b) Excess CMI I(S+; S- | S, heading) and the same with the heading shuffled within cells (power control), as a
      share of the excess I(S+; S- | S); sums over GeoLife users, pooled taxi sample.
  (c) Remote cells (P3-2): median excess I(S+; S(t-m) | last 3 cells) on steps at least 101 cells into a trip, alone,
      given the destination, and given a shuffled destination.
Inputs: controls/results/heading_state_geolife_{10m,100m}.csv, heading_state_taxi_200m_device.csv,
remote_destination_geolife.csv. Sized for the text width (16 cm), text at 9-9.5 pt; numbers are in the caption.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as _P
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common.fig_style import NPG, nature_style

nature_style()
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'lines.linewidth': 1.4})
R = _P.CONTROLS_RESULTS
W = 6.3
RED, BLUE, LAV = NPG[0], NPG[3], NPG[5]


def label(ax, s):
    ax.text(-0.3, 1.17, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


def shares():
    rows = []
    for lab, f, A in (('GeoLife\n10 m', 'heading_state_geolife_10m.csv', 'heading'), ('GeoLife\n100 m', 'heading_state_geolife_100m.csv', 'heading'),
                      ('taxi\n200 m', 'heading_state_taxi_200m_device.csv', 'heading+side')):
        d = pd.read_csv(R / f)
        if 'visits' in d:                                   # taxi: direction recorded by the device, all visits
            d = d[(d.heading == 'device') & (d.visits == 'all')]
        h = d[d.G2_tuned.notna()]; w = h.n_test
        g2 = np.average(h.G2_tuned, weights=w)
        e = d[['E0', 'Eh', 'Eh_shuffled']].dropna().sum()
        rows.append(dict(unit=lab, label=A, G2=g2, GA=np.average(h[f'GA_{A}_tuned'], weights=w) / g2,
                         R=np.average(h[f'R_{A}_tuned'], weights=w) / g2, Eh=e.Eh / e.E0, Esh=e.Eh_shuffled / e.E0))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    S = shares(); print(S.round(4).to_string(index=False))
    D = pd.read_csv(R / 'remote_destination_geolife.csv')
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.85), gridspec_kw=dict(wspace=0.62, width_ratios=[1.2, 1.2, 1]))
    x = np.arange(len(S)); w = 0.38
    def bars(a, left, right, cl, cr, top):
        for xs, vals, c in ((x - w / 2, left, cl), (x + w / 2, right, cr)):
            a.bar(xs, 100 * vals, w, color=c)
            for xi, v in zip(xs, vals):
                a.text(xi, 100 * v + top * 0.015, f'{100 * v:.0f}' if v >= 0.095 else f'{100 * v:.1f}', ha='center', va='bottom', fontsize=7.5)
        a.set_xticks(x); a.set_xticklabels(['10 m', '100 m', '200 m']); a.set_ylim(0, top)
        tr = matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes)
        a.text(0.5, -0.2, 'GeoLife', transform=tr, ha='center', va='top'); a.text(2, -0.2, 'taxi', transform=tr, ha='center', va='top')
        a.plot([-0.35, 1.35], [-0.18, -0.18], transform=tr, color='k', lw=0.6, clip_on=False)
    bars(ax[0], S.GA, S.R, BLUE, RED, 140); ax[0].axhline(100, color='grey', lw=0.8, ls=':')
    ax[0].set_ylabel('Share of $G_2$ (%)'); label(ax[0], 'a')
    bars(ax[1], S.Eh, S.Esh, RED, LAV, 100); ax[1].set_ylabel('Excess CMI left (%)'); label(ax[1], 'b')
    ms = [5, 10, 20, 50, 100]
    for suf, c, ls, mk, lab in (('', BLUE, '-', 'o', '$S_{t-m}$'), ('|D', RED, '-', 's', '+ destination'), ('|Dsh', LAV, '--', '^', '+ shuffled dest.')):
        ax[2].plot(ms, [1000 * D[f'long:I_{m}{suf}'].median() for m in ms], color=c, ls=ls, marker=mk, ms=4, label=lab)
    ax[2].set_xscale('log'); ax[2].set_xticks(ms); ax[2].set_xticklabels([str(m) for m in ms]); ax[2].minorticks_off()
    ax[2].axhline(0, color='grey', lw=0.8); ax[2].set_xlabel('Cells back, $m$'); ax[2].set_ylabel('Excess CMI (mbit/step)')
    ax[2].set_ylim(-0.1, 2.3); label(ax[2], 'c')
    hb = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label='heading alone'), plt.Rectangle((0, 0), 1, 1, color=RED, label='previous cell, heading in state'),
          plt.Rectangle((0, 0), 1, 1, color=LAV, label='heading shuffled within cells')]
    hl, ll = ax[2].get_legend_handles_labels()
    h = [hb[0], hl[0], hb[1], hl[1], hb[2], hl[2]]
    fig.subplots_adjust(bottom=0.36, left=0.09, right=0.99)
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.4)
    fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_path_heading.pdf')); fig.savefig(_P.out(_P.CONTROLS_FIGURES / 'fig_path_heading.png'), dpi=200)
