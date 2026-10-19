"""Supplementary figures for the checks of the to-do list (2026-10-19). One PDF/PNG per check; a figure whose
input table is missing is skipped.

  fig_s_null_calibration   P0-1  negative-control p-values, positive-control recovery and detection vs length
  fig_s_reaction_delay     P0-2  held-out gain of the leader's lagged dv / gap vs lag, per horizon, with null band
  fig_s_accel_noise        P0-3  acceleration autocorrelation and the augmentation gains for five derivative definitions
  fig_s_stops              P0-4  tau_c on whole vs stop-free windows; mixed-model coefficients
  fig_s_hurst_power        P1-3  detection rate of R/S and DFA for fGn / ARFIMA / AR(1) at the real segment lengths
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repository root
from common import paths as _P
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.fig_style import *
nature_style()
R = _P.CONTROLS_RESULTS
MODES = ['walk', 'bike', 'bus', 'subway', 'car', 'taxi', 'train']


def save(fig, name):
    fig.savefig(_P.out(_P.CONTROLS_FIGURES / f'{name}.pdf')); fig.savefig(_P.out(_P.CONTROLS_FIGURES / f'{name}.png'), dpi=150)
    plt.close(fig)


def null_calibration():
    if not (R / 'null_calibration_summary.json').exists(): return
    N = pd.read_csv(R / 'null_calibration_negative.csv'); Q = pd.read_csv(R / 'null_calibration_positive.csv')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.38))
    for i, (ds, c) in enumerate((('geolife', NPG[3]), ('taxi', NPG[0]))):
        p = N[N.dataset == ds].p
        ax[0].hist(p, bins=np.linspace(0, 1, 21), density=True, histtype='step', color=c, lw=1.1,
                   label=f'{"GeoLife" if ds == "geolife" else "taxi"}: FPR {(p < .05).mean():.3f}')
    ax[0].axhline(1, color='grey', lw=0.6, ls=':'); ax[0].set_xlabel('p-value (first-order surrogates)'); ax[0].set_ylabel('Density')
    ax[0].legend(loc='upper right'); panel_label(ax[0], 'a')
    Q = Q.assign(rec=Q.excess / Q['true'])
    for ds, ls in (('geolife', '-'), ('taxi', '--')):
        for lam, c in zip((0.25, 0.5, 1.0), (NPG[1], NPG[2], NPG[3])):
            g = Q[(Q.dataset == ds) & (Q.lam == lam)].groupby('frac').agg(rec=('rec', 'median'), det=('p', lambda x: (x < .05).mean()), n=('n_steps', 'median')).reset_index()
            lab = f'{"GeoLife" if ds == "geolife" else "taxi"}, λ = {lam}'
            ax[1].plot(g.frac, g.rec, ls=ls, marker='o', ms=2.5, color=c, label=lab)
            ax[2].plot(g.frac, g.det, ls=ls, marker='o', ms=2.5, color=c, label=lab)
    ax[1].axhline(1, color='grey', lw=0.6, ls=':'); ax[1].axhspan(0.8, 1.2, color='#eeeeee', zorder=0)
    ax[1].set_ylim(-0.6, 1.3); ax[1].text(0.04, 0.03, 'taxi, λ = 0.25: down to −2.1\n(true $I_2$ ≈ 0.002 nats)', transform=ax[1].transAxes, ha='left', fontsize=5)
    ax[1].set_xlabel('Share of the unit\'s data used'); ax[1].set_ylabel('Recovery: excess / true $I_2$ (median)'); panel_label(ax[1], 'b')
    ax[2].set_xlabel('Share of the unit\'s data used'); ax[2].set_ylabel('Detection rate (p < 0.05)'); ax[2].set_ylim(0, 1.02)
    h, l = ax[2].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=6, fontsize=5.5, bbox_to_anchor=(0.5, -0.1)); panel_label(ax[2], 'c')
    save(fig, 'fig_s_null_calibration')


def reaction_delay():
    """Gain of each lagged leader variable minus the mean of its permutation null (same betas); band = null 2.5-97.5 %."""
    if not (R / 'highd_reaction_delay.csv').exists(): return
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.32))
    for suf, ls, tag, alpha in (('', '-', 'base state', 0.15), ('_leadacc', '--', '+ leader acc.', 0.08)):
        if not (R / f'highd_reaction_delay{suf}.csv').exists(): continue
        G = pd.read_csv(R / f'highd_reaction_delay{suf}.csv'); Nl = pd.read_csv(R / f'highd_reaction_delay{suf}_null.csv')
        for i, hz in enumerate(sorted(G.horizon_s.unique())):
            for var, c in (('dv', NPG[0]), ('gap', NPG[3])):
                g = G[(G.horizon_s == hz) & (G.added == var)].groupby('lag_s').gain_nats.agg(['mean', 'std'])
                nl = Nl[(Nl.horizon_s == hz) & (Nl.added == var)].groupby('lag_s').gain_nats
                m = nl.mean(); q = nl.quantile([.025, .975]).unstack()
                k = 1 / np.log(2)
                ax[i].errorbar(g.index, (g['mean'] - m) * k, g['std'] * k, color=c, ls=ls, marker='o', ms=2.2, capsize=1.5, label=f'{var} lag, {tag}')
                ax[i].fill_between(q.index, (q[0.025] - m) * k, (q[0.975] - m) * k, color=c, alpha=alpha, lw=0)
            ax[i].axhline(0, color='grey', lw=0.6); ax[i].set_title(f'horizon {hz:g} s'); ax[i].set_xlabel('Lag of the leader variable (s)')
    ax[0].set_ylabel('Gain above permutation null (bits/sample)'); ax[1].legend(fontsize=5, loc='upper left')
    for i, l in enumerate('abc'): panel_label(ax[i], l)
    save(fig, 'fig_s_reaction_delay')


def accel_noise():
    if not (R / 'highd_accel_noise_gains.csv').exists(): return
    A = pd.read_csv(R / 'highd_accel_noise_acf.csv'); G = pd.read_csv(R / 'highd_accel_noise_gains.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.3]))
    lab = {'bd': 'backward difference (paper)', 'ds': 'dataset v, a', 'sg15': 'Savitzky–Golay 0.6 s', 'sg21': 'S–G 0.84 s', 'sg25': 'S–G 1.0 s'}
    for d, c in zip(lab, (NPG[0], NPG[3], NPG[1], NPG[2], NPG[5])):
        a = A[(A.definition == d) & (A.lag_s <= 2.0)]; ax[0].plot(a.lag_s, a.acf, marker='o', ms=2, color=c, label=lab[d])
    ax[0].axhline(0, color='grey', lw=0.6); ax[0].set_xlabel('Lag (s)'); ax[0].set_ylabel('Autocorrelation of $a_t$')
    panel_label(ax[0], 'a')
    rows = {'coarse: V2(20,20) - V1(20)': '$v_{t-1}$, 20 bins', 'fine: V2(100,20) - V1(100)': '$v_{t-1}$ | fine $v_t$',
            'residual after a: VA2 - VA': '$v_{t-1}$ | ($v_t$, $a_t$)', 'third step with a: VA3 - VA2': '$v_{t-2}$ | ($v_t$, $a_t$, $v_{t-1}$)'}
    g = G[G.state == 'signed_v'].set_index(G.columns[0]).loc[list(rows)]
    x = np.arange(len(rows)); w = 0.16
    for j, (d, c) in enumerate(zip(lab, (NPG[0], NPG[3], NPG[1], NPG[2], NPG[5]))):
        ax[1].bar(x + (j - 2) * w, g[d], w, color=c, label=lab[d])
    ax[1].set_xticks(x); ax[1].set_xticklabels(list(rows.values()), fontsize=5.5); ax[1].axhline(0, color='grey', lw=0.6)
    ax[1].set_ylabel('Held-out gain (nats/step)'); ax[1].legend(fontsize=5.5, loc='upper right'); panel_label(ax[1], 'b')
    save(fig, 'fig_s_accel_noise')


def stops():
    if not (R / 'stops_tautology_summary.json').exists(): return
    S = json.load(open(R / 'stops_tautology_summary.json')); L = pd.read_csv(R / 'stops_lmm.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.35))
    x = np.arange(len(MODES))
    for j, (kind, c, lab) in enumerate((('whole', NPG[3], 'whole windows'), ('stop_free', NPG[0], 'stop-free windows'))):
        m = pd.DataFrame(S[kind]['by_mode']).set_index('mode').reindex(MODES)
        ax[0].bar(x + (j - 0.5) * 0.38, m.tau_median, 0.38, color=c, label=f'{lab} (η² = {S[kind]["eta2_log_tau"]:.2f})')
    ax[0].set_xticks(x); ax[0].set_xticklabels(MODES, rotation=30); ax[0].set_ylabel('Median $\\tau_c$ in 300 s windows (s)')
    ax[0].legend(fontsize=5.5); panel_label(ax[0], 'a')
    full = L[L.model == 'speed + stops + length + mode'].set_index('term').drop('const')
    y = np.arange(len(full))[::-1]
    ax[1].errorbar(full.coef, y, xerr=1.96 * full.se, fmt='o', color=NPG[3], ms=3, capsize=2)
    ax[1].axvline(0, color='grey', lw=0.6); ax[1].set_yticks(y); ax[1].set_yticklabels([t.replace('mode_', '') for t in full.index], fontsize=6)
    ax[1].set_xlabel('Coefficient on log $\\tau_c$ (±1.96 se; car = reference)'); panel_label(ax[1], 'b')
    save(fig, 'fig_s_stops')


def hurst_power():
    if not (R / 'hurst_power_summary.csv').exists(): return
    S = pd.read_csv(R / 'hurst_power_summary.csv')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.3), sharey=True)
    for i, mode in enumerate(['walk', 'bus', 'car']):
        for fam, ls in (('fGn', '-'), ('ARFIMA', '--')):
            g = S[(S.family == fam) & (S['mode'] == mode)].sort_values('H_true')
            ax[i].plot(g.H_true, g.det_rs, ls=ls, marker='o', ms=2.5, color=NPG[0], label=f'R/S, {fam}')
            ax[i].plot(g.H_true, g.det_dfa, ls=ls, marker='s', ms=2.5, color=NPG[3], label=f'DFA, {fam}')
        a = S[(S.family == 'AR1') & (S['mode'] == mode)]; d = S[(S.family == 'data') & (S['mode'] == mode)]
        ax[i].scatter([0.5], a.det_rs, color=NPG[0], marker='x', s=12); ax[i].scatter([0.5], a.det_dfa, color=NPG[3], marker='x', s=12)
        if len(d):
            ax[i].axhline(float(d.det_rs.iloc[0]), color=NPG[0], lw=0.6, ls=':'); ax[i].axhline(float(d.det_dfa.iloc[0]), color=NPG[3], lw=0.6, ls=':')
        n = int(S[(S.family == 'fGn') & (S['mode'] == mode)].n_median.iloc[0])
        ax[i].set_title(f'{mode} (median length {n} × 5 s)'); ax[i].set_xlabel('Hurst exponent of the series')
        ax[i].axhline(0.05, color='grey', lw=0.5); panel_label(ax[i], 'abc'[i])
    ax[0].plot([], [], ':', color='grey', lw=0.8, label='real segments'); ax[0].scatter([], [], marker='x', color='grey', s=12, label='AR(1) series (false positives)')
    ax[0].set_ylabel('Detected vs 19 AR(1) surrogates'); ax[0].legend(fontsize=5); ax[0].set_ylim(0, 1.02)
    save(fig, 'fig_s_hurst_power')


if __name__ == '__main__':
    for f in (null_calibration, reaction_delay, accel_noise, stops, hurst_power):
        f()
