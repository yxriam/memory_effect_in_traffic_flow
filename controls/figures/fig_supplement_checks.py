"""Figures for the checks of the to-do list (2026-10-19), sized for the paper's appendix: 16 cm wide, text at
9-10 pt so that it reads at about the size of the body text; legends are short and the details are in the captions.
One PDF/PNG per check; a figure whose input table is missing is skipped. All panels are drawn from the result tables.

  fig_s_null_calibration   P0-1  negative-control p-values; positive-control recovery and detection vs amount of data
  fig_s_reaction_delay     P0-2  gain of the leader's lagged dv / gap above the permutation null, per horizon
  fig_s_accel_noise        P0-3  acceleration autocorrelation and augmentation gains for five derivative definitions
  fig_s_stops              P0-4  tau_c on whole vs stop-free windows; mixed-model coefficients
  fig_s_hurst_power        P1-3  detection rate of R/S and DFA for fGn / ARFIMA / AR(1) at the real segment lengths
  fig_s_positive_control   P0-5  gain of 128-step over 4-step networks on sequences with a planted long-range dependence
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as _P
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common.fig_style import NPG, nature_style

nature_style()
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'lines.linewidth': 1.4})
R = _P.CONTROLS_RESULTS
W = 6.3                                                        # inches = 16 cm, the text width of the paper
MODES = ['walk', 'bike', 'bus', 'subway', 'car', 'taxi', 'train']
RED, BLUE, TEAL, CYAN, SALMON, LAV = NPG[0], NPG[3], NPG[2], NPG[1], NPG[4], NPG[5]


def label(ax, s):
    ax.text(-0.2, 1.08, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


def save(fig, name):
    fig.savefig(_P.out(_P.CONTROLS_FIGURES / f'{name}.pdf')); fig.savefig(_P.out(_P.CONTROLS_FIGURES / f'{name}.png'), dpi=200)
    plt.close(fig)


def null_calibration():
    if not (R / 'null_calibration_summary.json').exists(): return
    N = pd.read_csv(R / 'null_calibration_negative.csv'); Q = pd.read_csv(R / 'null_calibration_positive.csv')
    Q = Q.assign(rec=Q.excess / Q['true'])
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw=dict(wspace=0.6))
    for ds, c, nm in (('geolife', BLUE, 'GeoLife'), ('taxi', RED, 'taxi')):
        p = N[N.dataset == ds].p
        ax[0].hist(p, bins=np.linspace(0, 1, 11), density=True, histtype='step', color=c, lw=1.4, label=nm)
    ax[0].axhline(1, color='grey', lw=0.8, ls=':'); ax[0].set_xlabel('p-value'); ax[0].set_ylabel('Density')
    ax[0].legend(loc='upper left', handlelength=1.2); label(ax[0], 'a')
    lam_c = {0.25: CYAN, 0.5: TEAL, 1.0: RED}
    for ds, ls in (('geolife', '-'), ('taxi', '--')):
        for lam, c in lam_c.items():
            g = Q[(Q.dataset == ds) & (Q.lam == lam)].groupby('frac').agg(rec=('rec', 'median'), det=('p', lambda x: (x < .05).mean()))
            ax[1].plot(100 * g.index, g.rec, ls=ls, marker='o', ms=3.5, color=c)
            ax[2].plot(100 * g.index, g.det, ls=ls, marker='o', ms=3.5, color=c)
    ax[1].axhline(1, color='grey', lw=0.8, ls=':'); ax[1].set_ylim(-0.6, 1.2)
    ax[1].set_xlabel('Data used (%)'); ax[1].set_ylabel('Excess / true $I_2$'); label(ax[1], 'b')
    ax[2].set_xlabel('Data used (%)'); ax[2].set_ylabel('Detection rate'); ax[2].set_ylim(0, 1.02); label(ax[2], 'c')
    for a in ax[1:]: a.set_xticks([25, 50, 100])
    h = [Line2D([], [], color=c, marker='o', ms=3.5, label=f'λ = {l:g}') for l, c in lam_c.items()]
    h += [Line2D([], [], color='k', ls='-', label='GeoLife'), Line2D([], [], color='k', ls='--', label='taxi')]
    fig.subplots_adjust(bottom=0.3); fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
    save(fig, 'fig_s_null_calibration')


def reaction_delay():
    """Gain of each lagged leader variable minus the mean of its permutation null (same betas); band = null 2.5-97.5 %."""
    if not (R / 'highd_reaction_delay.csv').exists(): return
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw=dict(wspace=0.4))
    for suf, ls, alpha in (('', '-', 0.18), ('_leadacc', '--', 0.0)):
        if not (R / f'highd_reaction_delay{suf}.csv').exists(): continue
        G = pd.read_csv(R / f'highd_reaction_delay{suf}.csv'); Nl = pd.read_csv(R / f'highd_reaction_delay{suf}_null.csv')
        for i, hz in enumerate(sorted(G.horizon_s.unique())):
            for var, c in (('dv', RED), ('gap', BLUE)):
                g = G[(G.horizon_s == hz) & (G.added == var)].groupby('lag_s').gain_nats.agg(['mean', 'std'])
                nl = Nl[(Nl.horizon_s == hz) & (Nl.added == var)].groupby('lag_s').gain_nats
                m = nl.mean(); q = nl.quantile([.025, .975]).unstack(); k = 1000 / np.log(2)
                ax[i].errorbar(g.index, (g['mean'] - m) * k, g['std'] * k, color=c, ls=ls, marker='o', ms=3, capsize=1.5, lw=1.2)
                if alpha: ax[i].fill_between(q.index, (q[0.025] - m) * k, (q[0.975] - m) * k, color=c, alpha=alpha, lw=0)
            ax[i].axhline(0, color='grey', lw=0.8); ax[i].set_title(f'horizon {hz:g} s'); ax[i].set_xlabel('Lag (s)')
            ax[i].set_xticks([0.4, 1.0, 1.6, 2.0])
    ax[0].set_ylabel('Gain above null (mbit)')
    for i, l in enumerate('abc'): label(ax[i], l)
    h = [Line2D([], [], color=RED, marker='o', ms=3, label='$\\Delta v$ lag'), Line2D([], [], color=BLUE, marker='o', ms=3, label='gap lag'),
         Line2D([], [], color='k', ls='-', label='state $(v,a,g,\\Delta v)$'), Line2D([], [], color='k', ls='--', label='+ leader acceleration')]
    fig.subplots_adjust(bottom=0.3); fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
    save(fig, 'fig_s_reaction_delay')


def accel_noise():
    if not (R / 'highd_accel_noise_gains.csv').exists(): return
    A = pd.read_csv(R / 'highd_accel_noise_acf.csv'); G = pd.read_csv(R / 'highd_accel_noise_gains.csv')
    defs = {'bd': ('backward diff.', RED), 'ds': ('dataset', BLUE), 'sg15': ('S–G 0.6 s', SALMON), 'sg21': ('S–G 0.84 s', TEAL), 'sg25': ('S–G 1.0 s', LAV)}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.25]))
    for d, (lab, c) in defs.items():
        a = A[(A.definition == d) & (A.lag_s <= 2.0)]; ax[0].plot(a.lag_s, a.acf, marker='o', ms=2.5, color=c, label=lab)
    ax[0].axhline(0, color='grey', lw=0.8); ax[0].set_xlabel('Lag (s)'); ax[0].set_ylabel('Autocorrelation of $a_t$'); label(ax[0], 'a')
    rows = ['coarse: V2(20,20) - V1(20)', 'fine: V2(100,20) - V1(100)', 'residual after a: VA2 - VA', 'third step with a: VA3 - VA2']
    g = G[G.state == 'signed_v'].set_index(G.columns[0]).loc[rows]
    x = np.arange(len(rows)); w = 0.16
    for j, (d, (lab, c)) in enumerate(defs.items()):
        ax[1].bar(x + (j - 2) * w, 1000 * g[d], w, color=c, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels(['coarse', 'fine', 'fine + $a$', '3rd step'])
    ax[1].axhline(0, color='grey', lw=0.8); ax[1].set_ylabel('Held-out gain (mnat/step)'); label(ax[1], 'b')
    ax[1].legend(loc='upper right', handlelength=1.2, borderaxespad=0.2)
    save(fig, 'fig_s_accel_noise')


def stops():
    if not (R / 'stops_tautology_summary.json').exists(): return
    S = json.load(open(R / 'stops_tautology_summary.json')); L = pd.read_csv(R / 'stops_lmm.csv')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(wspace=0.55))
    x = np.arange(len(MODES))
    for j, (kind, c, lab) in enumerate((('whole', BLUE, 'whole windows'), ('stop_free', RED, 'stop-free'))):
        m = pd.DataFrame(S[kind]['by_mode']).set_index('mode').reindex(MODES)
        ax[0].bar(x + (j - 0.5) * 0.38, m.tau_median, 0.38, color=c, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(MODES, rotation=35, ha='right'); ax[0].set_ylabel('Median $\\tau_c$ (s)')
    ax[0].legend(loc='upper left', handlelength=1.2); label(ax[0], 'a')
    full = L[L.model == 'speed + stops + length + mode'].set_index('term').drop('const')
    names = {'log_speed': 'log speed', 'log1p_stops_per_km': 'log(1+stops/km)', 'log_length': 'log length'}
    y = np.arange(len(full))[::-1]
    ax[1].errorbar(full.coef, y, xerr=1.96 * full.se, fmt='o', color=BLUE, ms=4, capsize=2)
    ax[1].axvline(0, color='grey', lw=0.8); ax[1].set_yticks(y); ax[1].set_yticklabels([names.get(t, t.replace('mode_', '')) for t in full.index])
    ax[1].set_xlabel('Coefficient on log $\\tau_c$'); label(ax[1], 'b')
    save(fig, 'fig_s_stops')


def hurst_power():
    if not (R / 'hurst_power_summary.csv').exists(): return
    S = pd.read_csv(R / 'hurst_power_summary.csv')
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw=dict(wspace=0.12), sharey=True)
    for i, mode in enumerate(['walk', 'bus', 'car']):
        for fam, ls in (('fGn', '-'), ('ARFIMA', '--')):
            g = S[(S.family == fam) & (S['mode'] == mode)].sort_values('H_true')
            ax[i].plot(g.H_true, g.det_rs, ls=ls, marker='o', ms=3.5, color=RED)
            ax[i].plot(g.H_true, g.det_dfa, ls=ls, marker='s', ms=3.5, color=BLUE)
        a = S[(S.family == 'AR1') & (S['mode'] == mode)]; d = S[(S.family == 'data') & (S['mode'] == mode)]
        ax[i].scatter([0.5], a.det_rs, color=RED, marker='x', s=22, zorder=3); ax[i].scatter([0.5], a.det_dfa, color=BLUE, marker='x', s=22, zorder=3)
        if len(d):
            ax[i].scatter([0.9], d.det_rs, color=RED, marker='D', s=20, zorder=3); ax[i].scatter([0.9], d.det_dfa, color=BLUE, marker='D', s=20, zorder=3)
        ax[i].set_title(mode); ax[i].set_xlabel('Hurst exponent $H$'); ax[i].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax[i].set_xticklabels(['0.5', '0.6', '0.7', '0.8', 'data'])
        ax[i].axhline(0.05, color='grey', lw=0.8, ls=':'); ax[i].axvline(0.85, color='0.8', lw=0.8); ax[i].set_xlim(0.45, 0.95); label(ax[i], 'abc'[i])
    ax[0].set_ylabel('Detection rate'); ax[0].set_ylim(0, 1.02)
    h = [Line2D([], [], color=RED, marker='o', ms=3.5, label='R/S'), Line2D([], [], color=BLUE, marker='s', ms=3.5, label='DFA'),
         Line2D([], [], color='k', ls='-', label='fGn'), Line2D([], [], color='k', ls='--', label='ARFIMA')]
    fig.subplots_adjust(bottom=0.3); fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
    save(fig, 'fig_s_hurst_power')


def positive_control():
    """P0-5: prediction/results/ngram/positive_control_runs.csv (made by prediction/positive_control.py summary)."""
    f = _P.PRED_NGRAM / 'positive_control_runs.csv'
    if not f.exists(): return
    Rn = pd.read_csv(f)
    sets = [('pc_L8_I2', '$L$ = 8\n0.2'), ('pc_L16_I1', '16\n0.1'), ('pc_L16_I2', '16\n0.2'), ('pc_L16_I3', '16\n0.3'), ('pc_L32_I2', '32\n0.2')]
    meta = {d: json.load(open(_P.PRED_DATA / f'{d}.json')) for d, _ in sets}
    piv = Rn.pivot_table(index=['dataset', 'model', 'seed'], columns='context', values='test_bits')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(wspace=0.38))
    x = np.arange(len(sets)); mc = {'lstm': BLUE, 'transformer': RED}; off = {'lstm': -0.13, 'transformer': 0.13}
    o4 = {'lstm': -0.3, 'transformer': 0.1}; o128 = {'lstm': -0.1, 'transformer': 0.3}   # (b): 4 and 128 steps side by side
    for i, (d, _) in enumerate(sets):
        ax[0].plot([i - 0.3, i + 0.3], [meta[d]['expected_gap_k4_vs_k128']] * 2, color='k', lw=1.6)
        for m in ('lstm', 'transformer'):
            g = piv.loc[(d, m)]; gain = g[4] - g[128]
            ax[0].scatter(np.full(len(gain), i + off[m]), gain, s=12, color=mc[m], alpha=0.55, lw=0)
            ax[0].scatter([i + off[m]], [gain.mean()], s=28, color=mc[m], marker='o', zorder=3)
            ax[1].scatter([i + o4[m]], [g[4].mean()], s=26, color=mc[m], marker='o', zorder=3)
            ax[1].scatter([i + o128[m]], [g[128].mean()], s=26, facecolor='white', edgecolor=mc[m], marker='s', lw=1.2, zorder=3)
        ax[1].plot([i - 0.4, i + 0.4], [meta[d]['H_full']] * 2, color='k', lw=1.6)
        ax[1].plot([i - 0.4, i + 0.4], [meta[d]['H_ctx4']] * 2, color='0.55', lw=1.6, ls='--')
    ax[0].axhline(0, color='grey', lw=0.8)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([t for _, t in sets]); a.set_xlim(-0.6, len(sets) - 0.4)
        a.set_xlabel('Lag $L$ and planted bits/step')
    ax[0].set_ylabel('Gain, 128 vs 4 steps (bits)'); ax[0].set_ylim(-0.3, 0.38)
    ax[1].set_ylabel('Cross-entropy (bits/step)')
    for a, l in zip(ax, 'ab'): a.text(-0.27, 1.08, l, transform=a.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
    h = [Line2D([], [], color=BLUE, marker='o', ls='', ms=5, label='LSTM'), Line2D([], [], color=RED, marker='o', ls='', ms=5, label='Transformer'),
         Line2D([], [], color='k', marker='o', ls='', ms=5, label='4 steps'),
         Line2D([], [], color='k', marker='s', ls='', ms=5, markerfacecolor='white', label='128 steps'),
         Line2D([], [], color='k', lw=1.6, label='available (a) / generator (b)'), Line2D([], [], color='0.55', lw=1.6, ls='--', label='generator, 4 steps')]
    fig.subplots_adjust(bottom=0.4, left=0.11, right=0.98); fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4)
    save(fig, 'fig_s_positive_control')


if __name__ == '__main__':
    for f in (null_calibration, reaction_delay, accel_noise, stops, hurst_power, positive_control):
        f()
