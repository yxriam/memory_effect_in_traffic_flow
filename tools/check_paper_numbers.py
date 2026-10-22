"""Check every number of the paper that comes from controls/results and prediction/results.

    python tools/check_paper_numbers.py            # prints a table, one row per number
    python tools/check_paper_numbers.py --csv out.csv

Each row: the number as printed in the paper, the value re-computed from the committed result files, and
'match' if the value rounds to the printed number. Run it after re-running any analysis and paste the summary
line into the experiment log (EXPERIMENT_LOG.md). The expected values already include the corrections found on
2026-10-19 (GeoLife LSTM 2.36, highD-100 k=2 LSTM 1.44, train interval 1.12; see experiment_log/).
Group '5 checks' covers the supplementary checks added to the paper on 2026-10-20 (v3); its expected values include
the four rounding fixes of that day (bike 20 %, Table 5 savings 1.109 and 0.363, bus DFA 4 %), and since 2026-10-22
the positive control of the history ablation (P0-5).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument('--csv', help='also write the table to this CSV file'); a = ap.parse_args()
LN2 = np.log(2)

ROWS = []
def item(group, what, paper, value, digits, unit=''):
    """paper: number as printed; value: re-computed; match if value rounds to the printed number."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        ROWS.append(dict(group=group, item=what, paper=paper, recomputed='n/a', status='not available')); return
    shown = f'{value:.{digits}f}'
    ok = np.isclose(round(value, digits), float(paper), atol=10 ** -(digits + 3))
    ROWS.append(dict(group=group, item=what, paper=f'{paper}{unit}', recomputed=f'{shown}{unit}  ({value:.6g})', status='match' if ok else 'MISMATCH'))

def rd(name):
    p = paths.CONTROLS_RESULTS / name.replace('highd_augment/', '')
    return pd.read_csv(p) if p.exists() else None

g = rd('geolife_robust_per_user.csv')
if g is not None:
    for grid, dj, lab in [(0.0001, 0, '10 m'), (0.001, 0, '100 m')]:
        s = g[(np.isclose(g.grid, grid)) & (g.dejitter.astype(int) == dj)]
        if lab == '10 m':
            item('1 GeoLife null', 'median H(S_t+1|S_t), 10 m', '0.5000', s.H1.median(), 4, ' bits')
            item('1 GeoLife null', 'median H(S_t+1|S_t-1,S_t), 10 m', '0.1243', s.H2.median(), 4, ' bits')
            item('1 GeoLife null', 'median plug-in CMI, 10 m', '0.3464', s.cmi2_plugin.median(), 4, ' bits')
            item('1 GeoLife null', 'median permutation-null CMI, 10 m', '0.325', s.cmi2_null.median(), 3, ' bits')
            item('1 GeoLife null', 'median bias-corrected CMI, 10 m', '0.012', s.cmi2_excess.median(), 3, ' bits')
            item('1 GeoLife null', 'share of users with p<0.05, 10 m', '66', 100 * (s.cmi2_p < .05).mean(), 0, ' %')
        else:
            item('1 GeoLife null', 'median bias-corrected CMI, 100 m', '0.109', s.cmi2_excess.median(), 3, ' bits')
            item('1 GeoLife null', 'share of users with p<0.05, 100 m', '82', 100 * (s.cmi2_p < .05).mean(), 0, ' %')

am, fr, f3, cm = rd('highd_augment/augment_markov.csv'), rd('highd_augment/fine_resolution.csv'), rd('highd_augment/fine_third_order.csv'), rd('coarse_model.csv')
if am is not None:
    p = am[am.state == 'signed_v'].pivot_table(index='split', columns='model', values='val_ll').mean()
    gain = p['V2'] - p['V1']
    item('2 highD', 'coarse 2nd-order gain V2-V1 (nats)', '0.076', gain, 3, ' nats')
    item('2 highD', 'same in bits (report used 0.110: wrong rounding)', '0.109', gain / LN2, 3, ' bits')
if fr is not None:
    p = fr.pivot_table(index='split', columns='model', values='val').mean()
    item('2 highD', 'V1 with 100 bins, no history', '-0.257', p['V1 (100)'], 3, ' nats')
    item('2 highD', 'gain of v_t-1 at fine state (100,20)', '0.025', p['V2 (100,20)'] - p['V1 (100)'], 3, ' nats')
    item('2 highD', 'gain of v_t-1 at fine state (100,100)', '0.033', p['V2 (100,100)'] - p['V1 (100)'], 3, ' nats')
    r = p['VA2 (100,a20,v1_20)'] - p['VA (100,a20)']
    item('2 highD', 'residual gain after adding a_t', '0.007', r, 3, ' nats')
    item('2 highD', 'same in bits', '0.010', r / LN2, 3, ' bits')
if f3 is not None:
    p = f3.pivot_table(index='split', columns='model', values='val').mean()
    item('2 highD', 'third step at fine state', '0.007', p['V3 (100,20,20)'] - p['V2 (100,20)'], 3, ' nats')
    item('2 highD', 'third step once a_t in state', '0.002', p['VA3 (100,a20,v1,v2)'] - p['VA2 (100,a20,v1_20)'], 3, ' nats')
if cm is not None:
    c = cm[cm.k == 20].set_index('data').gain
    item('2 highD', 'data gain at k=20 (one split)', '0.072', c['highD'], 3, ' nats')
    item('2 highD', 'AR(1) surrogate gain at k=20', '0.034', c['AR1'], 3, ' nats')
    item('2 highD', 'AR(1) surrogate gain at k=20, bits', '0.049', c['AR1'] / LN2, 3, ' bits')
    item('2 highD', 'AR(2) surrogate gain at k=20', '0.059', c['AR2'], 3, ' nats')
    item('2 highD', 'AR(2) / data', '82', 100 * c['AR2'] / c['highD'], 0, ' %')
pj = paths.CONTROLS_RESULTS / 'coarse_model_params.json'
if pj.exists():
    j = json.load(open(pj))
    item('2 highD', 'AR(2) phi1', '0.710', j['ar2'][0], 3); item('2 highD', 'AR(2) phi2', '0.274', j['ar2'][1], 3)
    item('2 highD', 'AR(1) rho', '0.977', j['ar1'], 3)
    item('2 highD', 'AR(2) roots', '0.987', j['roots'][0], 3); item('2 highD', 'AR(2) second root', '-0.277', j['roots'][1], 3)
ac = rd('highd_accel_acf.csv')
if ac is not None:
    item('2 highD', 'acceleration ACF at lag 0.2 s', '-0.33', float(ac.loc[ac['lag'] == 1, 'acf'].iloc[0]), 2)

mm, sp, rp = rd('row_mode_medians.csv'), rd('row_stops_spearman.csv'), rd('row_persistence.csv')
if mm is not None:
    t = mm.set_index('mode')
    for m, v in [('walk', '10'), ('bike', '20'), ('bus', '20'), ('subway', '40'), ('car', '75'), ('taxi', '75'), ('train', '175')]:
        item('3 layers', f'median tau_c, {m}', v, float(t.loc[m, 'tau']), 0, ' s')
    item('3 layers', 'lag-1 ACF of mode medians, min', '0.49', t.phi.min(), 2); item('3 layers', 'lag-1 ACF of mode medians, max', '0.98', t.phi.max(), 2)
    item('3 layers', 'Spearman tau_c vs speed (mode medians)', '0.93', spearmanr(t.tau, t.speed)[0], 2)
if sp is not None:
    s = sp.set_index('mode')
    item('3 layers', 'Spearman tau_c vs stops/km (mode medians)', '-0.98', s.loc['ALL(mode medians)', 'rho'], 2)
    item('3 layers', 'p of that correlation (x1e-5)', '8', s.loc['ALL(mode medians)', 'p'] * 1e5, 0)
    for m, v in [('bus', '-0.47'), ('subway', '-0.60'), ('taxi', '-0.23')]:
        item('3 layers', f'within-mode Spearman, {m}', v, s.loc[m, 'rho'], 2)
    item('3 layers', 'p, taxi within-mode', '0.02', s.loc['taxi', 'p'], 2)
if rp is not None:
    q = rp[rp.gap == 1800]
    item('3 layers', 'train: median share of interpolated samples (gap 30 min)', '60', 100 * q[q['mode'] == 'train'].interp_frac.median(), 0, ' %')
    item('3 layers', 'subway: median share of interpolated samples (gap 30 min)', '33', 100 * q[q['mode'] == 'subway'].interp_frac.median(), 0, ' %')
rr, rs = rd('row_regression.csv'), rd('row_regression_split.csv')
if rr is not None:
    r = rr[(rr.iloc[:, 0] == 'exclusive ROW') & (rr.gap == 1800) & (rr.model == 'M4 + interpolation') & (rr.y == 'log_tau')].iloc[0]
    item('3 layers', 'right-of-way coef. on log tau_c, + interpolation', '-0.08', r.coef, 2)
    ROWS.append(dict(group='3 layers', item='  ...its 95% CI contains 0 (not significant)', paper='yes', recomputed=f'{r.ci_lo:.3f}..{r.ci_hi:.3f}',
                  status='match' if r.ci_lo < 0 < r.ci_hi else 'MISMATCH'))
if rs is not None:
    for nm, (f, lo, hi) in {'is_subway': ('0.55', '0.39', '0.72'), 'is_train': ('1.64', '1.12', '1.93'), 'is_bus': ('0.48', '0.36', '0.60')}.items():
        r = rs[(rs.iloc[:, 0] == nm) & (rs.y == 'log_tau') & (rs.model == '+ speed')].iloc[0]
        lab = nm[3:]
        note = ' (paper prints 1.11; 1.12 with the order-independent bootstrap)' if nm == 'is_train' else ''
        item('3 layers', f'{lab} vs car/taxi at same speed: factor', f, np.exp(r.coef), 2)
        item('3 layers', f'{lab}: 95% user-bootstrap lower{note}', lo, np.exp(r.ci_lo), 2)
        item('3 layers', f'{lab}: 95% user-bootstrap upper', hi, np.exp(r.ci_hi), 2)
pm = rd('memory_by_mode_path.csv')
if pm is not None:
    med = pm.groupby(['dejitter', 'mode']).rel_excess.median()
    for m, raw, dj in [('bus', '29', '35'), ('bike', '28', '43'), ('subway', '20', '24'), ('taxi', '17', None), ('car', '8', '9'), ('walk', '9', '26')]:
        item('3 layers', f'path memory (rel. excess CMI), {m}, raw', raw, 100 * med[(0, m)], 0, ' %')
        if dj: item('3 layers', f'path memory, {m}, de-jittered', dj, 100 * med[(1, m)], 0, ' %')
    sig = pm[pm.dejitter == 0].groupby('mode').p.apply(lambda x: (x < .05).mean())
    item('3 layers', 'share significant, min over bus/bike/subway/taxi/car/walk', '67', 100 * sig.drop('train').min(), 0, ' %')
    item('3 layers', 'share significant, max over these modes', '94', 100 * sig.drop('train').max(), 0, ' %')
    tr = pm[(pm.dejitter == 0) & (pm['mode'] == 'train')]
    ROWS.append(dict(group='3 layers', item='train: users significant', paper='3 of 12', recomputed=f'{int((tr.p < .05).sum())} of {len(tr)}',
                  status='match' if (int((tr.p < .05).sum()), len(tr)) == (3, 12) else 'MISMATCH'))
og = rd('origin_memory_geolife.csv')
if og is not None:
    ROWS.append(dict(group='3 layers', item='origin memory: GeoLife users', paper='157', recomputed=str(len(og)), status='match' if len(og) == 157 else 'MISMATCH'))
    item('3 layers', 'GeoLife origin CMI I_O (median)', '0.0028', og.origin_excess.median(), 4, ' bits')
    item('3 layers', 'GeoLife third-order residual I_3 (median)', '0.0072', og.cmi3_excess.median(), 4, ' bits')
    item('3 layers', 'GeoLife I_O|3 (median)', '0.0013', og.origin_beyond3_excess.median(), 4, ' bits')
    item('3 layers', 'GeoLife share sig. I_O', '43', 100 * (og.p_origin < .05).mean(), 0, ' %')
    item('3 layers', 'GeoLife share sig. I_O|3', '34', 100 * (og.p_origin3 < .05).mean(), 0, ' %')
for G, vals in [('0.002', ('0.0148', '0.0058', '0.0024')), ('0.005', ('0.039', '0.015', '0.004'))]:
    t = rd(f'taxi_origin_{G}.csv')
    if t is None: continue
    t = t.iloc[0]; d = len(vals[0].split('.')[1])
    if G == '0.002':
        ROWS.append(dict(group='3 layers', item='taxi pooled sample (200 m)', paper='100 vehicles, 86413 steps',
                      recomputed=f'{int(t.n_veh)} vehicles, {int(t.n_steps)} steps', status='match' if (int(t.n_veh), int(t.n_steps)) == (100, 86413) else 'MISMATCH'))
        item('3 layers', 'taxi permutation p (all three)', '0.0323', max(t.p3, t.p_origin, t.p_origin3), 4)
    for k, v in zip(['cmi3', 'origin', 'origin3'], vals):
        item('3 layers', f'taxi {k} ({G} deg grid)', v, t[k], d, ' bits')
    if G == '0.002':
        ROWS.append(dict(group='3 layers', item='taxi half-sample 95% ranges exclude 0', paper='yes',
                      recomputed=', '.join(f'{k}: {t[k + "_lo"]:.4f}..{t[k + "_hi"]:.4f}' for k in ['cmi3', 'origin', 'origin3']),
                      status='match' if all(t[k + '_lo'] > 0 for k in ['cmi3', 'origin', 'origin3']) else 'MISMATCH'))

# ---------------------------------------------------------------- group 4: neural models and memory networks
def js(name):
    p = paths.PRED_NEURAL / name
    return json.load(open(p)) if p.exists() else None
for ds, lab, lstm, trf in [('taxi_traj20', 'taxi', '2.90', '3.25'), ('geolife_bj500m', 'GeoLife', '2.36', '2.64'),
                           ('highd_v20', 'highD 20', '0.36', '0.37'), ('highd_v100', 'highD 100', '1.29', '1.29')]:
    for mdl, v in [('lstm', lstm), ('transformer', trf)]:
        j = js(f'neural_{ds}_{mdl}.json')
        note = ' (paper prints 2.37: should be 2.36)' if (ds, mdl) == ('geolife_bj500m', 'lstm') else ''
        item('4 neural', f'{lab} {mdl}, 128 steps{note}', v, j['test']['bits'] if j else None, 2, ' bits')
seeds = [js(f'neural_taxi_traj20_lstm{s}.json') for s in ('', '_s1', '_s2')]
if all(seeds):
    x = np.array([j['test']['bits'] for j in seeds])
    item('4 neural', 'taxi LSTM, mean of 3 seeds', '2.91', x.mean(), 2, ' bits'); item('4 neural', 'taxi LSTM, sd of 3 seeds', '0.01', x.std(ddof=1), 2, ' bits')
    item('4 neural', 'taxi LSTM top-1 accuracy', '0.35', seeds[0]['test']['acc1'], 2)
for ds, lab, ks in [('taxi_traj20', 'taxi', {1: '4.46', 2: '2.97', 3: '2.87', 4: '2.84', 8: '2.85'}),
                    ('highd_v20', 'highD 20', {1: '0.56', 2: '0.44', 4: '0.39', 8: '0.38', 32: '0.38'}),
                    ('highd_v100', 'highD 100', {1: '1.64', 2: '1.44', 32: '1.34'}),
                    ('geolife_bj500m', 'GeoLife', {1: '3.04', 2: '2.37', 4: '2.26', 8: '2.29'})]:
    for k, v in ks.items():
        j = js(f'hist/hist_{ds}_k{k}_s0.json')
        note = ' (paper prints 1.45: should be 1.44)' if (ds, k) == ('highd_v100', 2) else ''
        item('4 neural', f'{lab} LSTM restricted to k={k}{note}', v, j['test']['bits'] if j else None, 2, ' bits')
j = js('hist/hist_taxi_traj20_k2_s1.json')
item('4 neural', 'taxi k=2, second seed', '2.972', j['test']['bits'] if j else None, 3, ' bits')
p = paths.PRED_INFOMAP / 'infomap_summary.csv'
if p.exists():
    im = pd.read_csv(p).set_index('network')
    L = lambda n: im.loc[n, 'codelength']
    sh = lambda n: im.loc[n, 'physical_in_multiple_modules'] / im.loc[n, 'physical_nodes']
    item('4 network', 'taxi: first-order codelength', '6.66', L('taxi_first'), 2, ' bits'); item('4 network', 'taxi: memory codelength', '5.55', L('taxi_memory'), 2, ' bits')
    item('4 network', 'taxi: saving', '1.11', L('taxi_first') - L('taxi_memory'), 2, ' bits')
    item('4 network', 'taxi surrogate: saving', '0.37', L('taxi_surrogate_first') - L('taxi_surrogate_memory'), 2, ' bits')
    item('4 network', 'taxi: modules first-order', '172', im.loc['taxi_first', 'modules'], 0); item('4 network', 'taxi: modules memory', '4249', im.loc['taxi_memory', 'modules'], 0)
    item('4 network', 'taxi: share of cells in >=2 modules', '70', 100 * sh('taxi_memory'), 0, ' %'); item('4 network', 'taxi surrogate: same share', '40', 100 * sh('taxi_surrogate_memory'), 0, ' %')
    item('4 network', 'highD: saving', '0.15', L('highd_first') - L('highd_memory'), 2, ' bits'); item('4 network', 'highD surrogate: saving', '0.00', L('highd_surrogate_first') - L('highd_surrogate_memory'), 2, ' bits')
    item('4 network', 'highD: speed states split', '18', im.loc['highd_memory', 'physical_in_multiple_modules'], 0)
    item('4 network', 'GeoLife: saving', '0.25', L('geolife_first') - L('geolife_memory'), 2, ' bits'); item('4 network', 'GeoLife surrogate: saving', '0.47', L('geolife_surrogate_first') - L('geolife_surrogate_memory'), 2, ' bits')
    item('4 network', 'GeoLife: share in >=2 modules', '35', 100 * sh('geolife_memory'), 0, ' %'); item('4 network', 'GeoLife surrogate: same share', '18', 100 * sh('geolife_surrogate_memory'), 0, ' %')
for w, dv, sv in [('taxi', '44', '13'), ('geolife', '45', '5'), ('highd', '19', '0')]:
    p = paths.PRED_NGRAM / f'ch6_{w}.csv'
    if not p.exists(): continue
    c = pd.read_csv(p); red = 1 - c.H2 / c.H1
    item('4 network', f'{w}: plug-in 1-H2/H1, data', dv, 100 * red.iloc[0], 0, ' %')
    item('4 network', f'{w}: plug-in 1-H2/H1, surrogates (mean of 2)', sv, 100 * red.iloc[1:].mean(), 0, ' %')
    if w == 'geolife':
        item('4 network', 'GeoLife: links per state node', '1.7', c.mem_links.iloc[0] / c.state_nodes.iloc[0], 1)


h = rd('hurst_control.csv')
if h is not None:
    m = h.groupby(['gap', 'mode'])[['H_speed', 'H_shuffled', 'H_ar1']].median()
    g = m.loc[1800.0]
    item('extra Hurst', 'observed R/S, min over walk/bus/car (gap 30 min)', '0.87', g.H_speed.min(), 2)
    item('extra Hurst', 'observed R/S, max', '0.96', g.H_speed.max(), 2)
    item('extra Hurst', 'AR(1) surrogate R/S, min', '0.83', g.H_ar1.min(), 2)
    item('extra Hurst', 'AR(1) surrogate R/S, max', '0.98', g.H_ar1.max(), 2)
    item('extra Hurst', 'shuffled R/S (mean of the three medians)', '0.59', g.H_shuffled.mean(), 2)
    w = h[(h.gap == 1800.0) & (h['mode'] == 'walk')]
    item('extra Hurst', 'walk: median of observed minus surrogate', '0.05', (w.H_speed - w.H_ar1).median(), 2)
    g30 = m.loc[30.0]
    item('extra Hurst', 'observed R/S with gap 30 s, min', '0.77', g30.H_speed.min(), 2)
    item('extra Hurst', 'observed R/S with gap 30 s, max', '0.93', g30.H_speed.max(), 2)
p = paths.PRED_NGRAM / 'highd_recrossing.json'
if p.exists():
    r = json.load(open(p))
    item('4 network', 'highD: return to previous bin after a bin change', '43', 100 * r['p_return'], 0, ' %')
    item('4 network', 'highD: return predicted by first-order chain', '6', 100 * r['p_return_first_order_chain'], 0, ' %')
    ROWS.append(dict(group='4 network', item='highD: move on to a third bin', paper='< 1 %', recomputed=f"{100 * r['p_third_bin']:.2f} %",
                     status='match' if r['p_third_bin'] < 0.01 else 'MISMATCH'))

# ---- 5: supplementary checks added to the paper on 2026-10-20 (v3; Appendix B, Sections 4.1, 4.4, 5.4, 6.4, 7.2-7.4, 8.3)
G5 = '5 checks'
def rj(name):
    p = paths.CONTROLS_RESULTS / name
    return json.load(open(p)) if p.exists() else None

s = rj('null_calibration_summary.json')                                        # P0-1
if s is not None:
    item(G5, 'null calibration: GeoLife false-positive rate', '1.1', 100 * s['geolife']['fpr'], 1, ' %')
    item(G5, 'null calibration: GeoLife negative runs', '1483', s['geolife']['runs'], 0)
    item(G5, 'null calibration: GeoLife median excess (negative)', '-0.0008', s['geolife']['excess_median'], 4, ' bits')
    item(G5, 'null calibration: taxi false-positive rate', '3.0', 100 * s['taxi']['fpr'], 1, ' %')
    item(G5, 'null calibration: taxi negative runs', '4991', s['taxi']['runs'], 0)
    P = pd.DataFrame(s['positive'])
    gl, tx = P[P.dataset == 'geolife'], P[P.dataset == 'taxi']
    item(G5, 'recovery, GeoLife all data: min over lambda', '70', 100 * gl[gl.frac == 1].recovery_median.min(), 0, ' %')
    item(G5, 'recovery, GeoLife all data: max over lambda', '79', 100 * gl[gl.frac == 1].recovery_median.max(), 0, ' %')
    item(G5, 'recovery, GeoLife quarter of data: min', '45', 100 * gl[gl.frac == .25].recovery_median.min(), 0, ' %')
    item(G5, 'recovery, GeoLife quarter of data: max', '53', 100 * gl[gl.frac == .25].recovery_median.max(), 0, ' %')
    item(G5, 'taxi true I2 (all data): min over lambda', '0.002', tx[tx.frac == 1].true_median.min(), 3, ' nats')
    item(G5, 'taxi true I2 (all data): max over lambda', '0.037', tx[tx.frac == 1].true_median.max(), 3, ' nats')
    item(G5, 'taxi recovery, lambda 0.25, quarter of data', '-2.1', tx[(tx.frac == .25) & (tx.lam == .25)].recovery_median.iloc[0], 1)
    g0 = rd('geolife_robust_per_user.csv')
    if g0 is not None:
        e10 = g0[np.isclose(g0.grid, 0.0001) & (g0.dejitter.astype(int) == 0)]
        item(G5, 'GeoLife excess / 0.7 (genuine memory at recovery 0.7)', '0.017', e10.cmi2_excess.median() / 0.7, 3, ' bits')
        item(G5, 'same as share of the median plug-in CMI', '5', 100 * e10.cmi2_excess.median() / 0.7 / e10.cmi2_plugin.median(), 0, ' %')

gn = rd('highd_accel_noise_gains.csv'); ac = rd('highd_accel_noise_acf.csv')   # P0-3
if gn is not None:
    gn = gn[gn.state == 'signed_v'].set_index(gn.columns[0])
    co, re = gn.loc['coarse: V2(20,20) - V1(20)'], gn.loc['residual after a: VA2 - VA']
    item(G5, 'highD 23 rec.: coarse gain, backward differences', '0.078', co.bd, 3, ' nats')
    item(G5, 'highD 23 rec.: residual after a_t, backward differences', '0.0062', re.bd, 4, ' nats')
    for col, v in [('ds', '-0.0027'), ('sg15', '0.0013'), ('sg21', '-0.0015'), ('sg25', '-0.0025')]:
        item(G5, f'highD residual after a_t, {col}', v, re[col], 4, ' nats')
    item(G5, 'coarse gain with smoothed speeds: min (dataset v)', '0.0002', co[['ds', 'sg15', 'sg21', 'sg25']].min(), 4, ' nats')
    item(G5, 'coarse gain with smoothed speeds: max (S-G 0.6 s)', '0.0083', co[['ds', 'sg15', 'sg21', 'sg25']].max(), 4, ' nats')
if ac is not None:
    a1 = ac[ac.lag == 1].set_index('definition').acf
    for col, v in [('bd', '-0.32'), ('sg15', '0.14'), ('sg21', '0.58'), ('sg25', '0.75')]:
        item(G5, f'highD lag-1 autocorrelation of a, {col}', v, a1[col], 2)
# (the 1.45 million steps and 24,804 vehicles are printed by highd_accel_noise.py, not stored in a result file)

def delay_excess(suf, hz, var='dv', lag=2.0):                                  # P0-2: mean gain over splits minus null mean, bits
    G = rd(f'highd_reaction_delay{suf}.csv'); N = rd(f'highd_reaction_delay{suf}_null.csv')
    if G is None or N is None: return None
    gg = G[(G.horizon_s == hz) & (G.added == var) & np.isclose(G.lag_s, lag)].gain_nats.mean()
    return (gg - N[(N.horizon_s == hz) & (N.added == var) & np.isclose(N.lag_s, lag)].gain_nats.mean()) / LN2
for hz, v in [(0.2, '0.003'), (1.0, '0.019'), (2.0, '0.018')]:
    item(G5, f'leader dv 2 s back above null, horizon {hz:g} s', v, delay_excess('', hz), 3, ' bits')
for hz, v in [(1.0, '0.005'), (2.0, '0.003')]:
    item(G5, f'same with leader acceleration in the state, horizon {hz:g} s', v, delay_excess('_leadacc', hz), 3, ' bits')
la = rd('highd_reaction_delay_leadacc.csv')
if la is not None:
    for hz, v in [(1.0, '0.006'), (2.0, '0.018')]:
        item(G5, f'gain of the leader acceleration itself, horizon {hz:g} s', v, la[(la.horizon_s == hz) & (la.added == 'a_lead')].gain_bits.mean(), 3, ' bits')
kn = rd('highd_reaction_delay_leadacc_knn.csv')
if kn is not None:
    ROWS.append(dict(group=G5, item='kNN CMI with leader acceleration: significant cells', paper='none', recomputed=f'{(kn.p < .05).sum()} of {len(kn)}',
                     status='match' if (kn.p < .05).sum() == 0 else 'MISMATCH'))

s = rj('stops_tautology_summary.json')                                          # P0-4
if s is not None:
    sf = {r['mode']: r['tau_median'] for r in s['stop_free']['by_mode']}
    item(G5, 'stop-free 300 s windows: median tau_c walk', '5', sf['walk'], 0, ' s')
    item(G5, 'stop-free: median tau_c train', '40', sf['train'], 0, ' s')
    item(G5, 'stop-free: min of the other five modes', '10', min(sf[m] for m in ['bike', 'bus', 'subway', 'car', 'taxi']), 0, ' s')
    item(G5, 'stop-free: max of the other five modes', '25', max(sf[m] for m in ['bike', 'bus', 'subway', 'car', 'taxi']), 0, ' s')
    item(G5, 'eta^2 of log tau_c, stop-free windows', '47', 100 * s['stop_free']['eta2_log_tau'], 0, ' %')
    item(G5, 'eta^2 of log tau_c, all windows', '37', 100 * s['whole']['eta2_log_tau'], 0, ' %')
    ab = {r['mode']: r['ratio_median'] for r in s['ar1_bias']['by_mode']}
    item(G5, 'AR(1) bias ratio, walk', '1.00', ab['walk'], 2); item(G5, 'AR(1) bias ratio, train', '0.58', ab['train'], 2)
    item(G5, 'cross-mode rho after bias correction', '-0.96', s['ar1_bias']['spearman_corrected'][0], 2)
lm = rd('stops_lmm.csv')
if lm is not None:
    t = lm.set_index(['model', 'term'])
    full, nosp = t.loc['speed + stops + length + mode'], t.loc['stops + length + mode']
    item(G5, 'LMM stops coefficient without speed', '-0.14', nosp.loc['log1p_stops_per_km', 'coef'], 2)
    item(G5, 'its s.e.', '0.03', nosp.loc['log1p_stops_per_km', 'se'], 2)
    item(G5, 'LMM stops coefficient with speed', '0.27', full.loc['log1p_stops_per_km', 'coef'], 2)
    item(G5, 'its s.e.', '0.04', full.loc['log1p_stops_per_km', 'se'], 2)
    for m, v in [('walk', '0.51'), ('bike', '0.64'), ('bus', '0.55'), ('subway', '0.67')]:
        item(G5, f'LMM factor relative to car, {m}', v, full.loc[f'mode_{m}', 'factor'], 2)
    item(G5, 'LMM segments', '2599', full.n.iloc[0], 0); item(G5, 'LMM users', '34', full.users.iloc[0], 0)

ns = rd('nonstationarity_geolife.csv')                                          # P1-1 GeoLife
if ns is not None:
    u = ns[ns.level == 'user x mode']
    item(G5, 'user x mode: median excess, original strata', '0.074', u.excess.median(), 3, ' bits')
    item(G5, 'user x mode: median excess, strata x trip type', '0.011', u.excess_type.median(), 3, ' bits')
    item(G5, 'user x mode: median excess, shuffled trip type', '0.015', u.excess_shuffled_type.median(), 3, ' bits')
    item(G5, 'share of the excess due to trip-type switching (sums)', '27', 100 * (u.excess_shuffled_type.sum() - u.excess_type.sum()) / u.excess.sum(), 0, ' %')

re_ = rd('route_equal_sample.csv')                                              # P1-6
if re_ is not None:
    rel = re_.groupby(['mode', 'draw']).rel_excess.median().groupby('mode').mean()
    for m, v in [('subway', '22'), ('bike', '20'), ('bus', '10'), ('walk', '8'), ('taxi', '6'), ('car', '4'), ('train', '0')]:
        item(G5, f'equal samples: relative excess, {m}', v, 100 * rel[m], 0, ' %')
rc, rw = rd('route_cells.csv'), rd('route_branching_wls.csv')
if rc is not None: item(G5, 'cells visited >= 20 times', '4249', len(rc), 0)
if rw is not None:
    b = rw[(rw.model == 'branching + mode') & (rw.term == 'log2_branching')].iloc[0]
    item(G5, 'log2 branching coefficient', '0.07', b.coef, 2); item(G5, 'its 95% interval, low', '-0.06', b.lo, 2); item(G5, 'its 95% interval, high', '0.15', b.hi, 2)

og, ot = rd('origin_direction_geolife.csv'), rd('origin_direction_taxi.csv')    # P1-2
if og is not None:
    from scipy.stats import wilcoxon
    item(G5, 'GeoLife users in the trip-level test', '153', len(og), 0)
    im_ = [og[f'I_{m}'].median() for m in (5, 10, 20)]                       # printed as "about 0.0024 bits" for m = 5, 10, 20
    ROWS.append(dict(group=G5, item='GeoLife median I_5, I_10, I_20 ("about 0.0024")', paper='0.0024 bits', recomputed=' / '.join(f'{x:.4f}' for x in im_),
                     status='match' if all(abs(x - 0.0024) <= 0.00015 for x in im_) else 'MISMATCH'))
    item(G5, 'GeoLife median origin CMI given 3 cells and S_t-10', '0.0000', og['I_O|3,10'].median(), 4, ' bits')
    item(G5, 'GeoLife paired Wilcoxon, real vs shuffled S_t-10', '0.29', wilcoxon(og['I_O|3,10'], og['I_O|3,10shuf']).pvalue, 2)
if ot is not None:
    o = ot.iloc[0]
    for m, v in [(5, '0.0013'), (10, '0.0004'), (20, '0.0003')]:
        item(G5, f'taxi I_{m}', v, o[f'I_{m}'], 4, ' bits')
    item(G5, 'taxi origin CMI given 3 cells and S_t-10', '0.00003', o['I_O|3,10'], 5, ' bits'); item(G5, 'its p', '0.13', o['p_I_O|3,10'], 2)
    item(G5, 'taxi origin CMI given 3 cells, vehicle, period: p', '1', o['p_I_O|3,veh,per'], 0)
    item(G5, 'same strata shuffled', '0.0004', o['I_O|3,veh,per shuf'], 4, ' bits'); item(G5, 'its p', '0.005', o['p_I_O|3,veh,per shuf'], 3)

dj, dd = rd('taxi_dejitter_summary.csv'), rd('taxi_dejitter.csv')               # P1-5 (CMI)
if dj is not None and dd is not None:
    pv = dd.pivot_table(index='file', columns='variant', values=['comp_len', 'cmi2_excess', 'cmi2_p', 'dll_2v1'])
    both = pv.dropna()
    item(G5, 'de-jitter: trajectories with >= 50 states in both', '2195', len(both), 0)
    item(G5, 'de-jitter: median share of compressed states removed', '29', 100 * (1 - both.comp_len.dejittered / both.comp_len.published).median(), 0, ' %')
    item(G5, 'de-jitter: median excess I2, original (paired)', '0.0276', both.cmi2_excess.published.median(), 4, ' nats')
    item(G5, 'de-jitter: median excess I2, de-jittered', '0.0251', both.cmi2_excess.dejittered.median(), 4, ' nats')
    item(G5, 'de-jitter: paired Wilcoxon p', '0.14', dj.wilcoxon_p_cmi2.iloc[0], 2)
    item(G5, 'de-jitter: significant share, original (paired)', '56.0', 100 * (both.cmi2_p.published < .05).mean(), 1, ' %')
    item(G5, 'de-jitter: significant share, de-jittered', '56.5', 100 * (both.cmi2_p.dejittered < .05).mean(), 1, ' %')
    item(G5, 'de-jitter: median held-out gain, original (paired)', '0.0088', both.dll_2v1.published.median(), 4, ' nats')
    item(G5, 'de-jitter: median held-out gain, de-jittered', '0.0077', both.dll_2v1.dejittered.median(), 4, ' nats')

nc = paths.PRED_NGRAM / 'taxi_network_controls.csv'                            # P1-1 / P1-5 (memory network)
if nc.exists():
    nc = pd.read_csv(nc); K = nc.groupby('kind')
    sav, red, sh, st = K.saving.mean(), K.entropy_rate_reduction.mean(), K.share_multi_module.mean(), K.steps.mean()
    for k, v in [('data', '1.109'), ('pooled_surrogate', '0.363'), ('per_vehicle_surrogate', '0.971'), ('dejittered', '0.183'),
                 ('dejittered_pooled_surrogate', '0.420'), ('dejittered_per_vehicle_surrogate', '0.330')]:
        item(G5, f'Table 5 saving: {k}', v, sav[k], 3, ' bits')
    for k, v in [('pooled_surrogate', '0.017'), ('per_vehicle_surrogate', '0.003'), ('dejittered_pooled_surrogate', '0.004'), ('dejittered_per_vehicle_surrogate', '0.005')]:
        item(G5, f'Table 5 saving s.d.: {k}', v, K.saving.std(ddof=1)[k], 3, ' bits')
    for k, v in [('data', '44.2'), ('pooled_surrogate', '12.7'), ('per_vehicle_surrogate', '33.3'), ('dejittered', '34.1'),
                 ('dejittered_pooled_surrogate', '8.6'), ('dejittered_per_vehicle_surrogate', '19.4')]:
        item(G5, f'Table 5 1-H2/H1: {k}', v, 100 * red[k], 1, ' %')
    for k, v in [('data', '69.5'), ('pooled_surrogate', '40.0'), ('per_vehicle_surrogate', '68.1'), ('dejittered', '65.6'),
                 ('dejittered_pooled_surrogate', '33.1'), ('dejittered_per_vehicle_surrogate', '60.1')]:
        item(G5, f'Table 5 cells in >= 2 modules: {k}', v, 100 * sh[k], 1, ' %')
    for k, v in [('data', '855788'), ('pooled_surrogate', '852654'), ('per_vehicle_surrogate', '811428'), ('dejittered', '564007'),
                 ('dejittered_pooled_surrogate', '562599'), ('dejittered_per_vehicle_surrogate', '513937')]:
        item(G5, f'Table 5 steps: {k}', v, st[k], 0)
    item(G5, 'per-vehicle surrogate saving / data saving', '88', 100 * sav['per_vehicle_surrogate'] / sav['data'], 0, ' %')
    item(G5, 'data minus per-vehicle surrogate', '0.14', sav['data'] - sav['per_vehicle_surrogate'], 2, ' bits')
    item(G5, 'steps deleted by de-jittering', '34', 100 * (1 - st['dejittered'] / st['data']), 0, ' %')

hp, hr = rd('hurst_power_summary.csv'), rd('hurst_power_real.csv')             # P1-3
if hp is not None:
    f7 = hp[(hp.family == 'fGn') & np.isclose(hp.H_true, 0.7)]
    item(G5, 'fGn H=0.7 detected by R/S: min over modes', '44', 100 * f7.det_rs.min(), 0, ' %')
    item(G5, 'fGn H=0.7 detected by R/S: max', '62', 100 * f7.det_rs.max(), 0, ' %')
    item(G5, 'fGn H=0.7 detected by DFA (all three modes)', '57', 100 * f7.det_dfa.mean(), 0, ' %')
    ROWS.append(dict(group=G5, item='fGn H=0.7 DFA identical in the three modes', paper='57 / 57 / 57', recomputed=' / '.join(f'{100 * x:.0f}' for x in f7.det_dfa),
                     status='match' if np.allclose(f7.det_dfa, 0.57) else 'MISMATCH'))
    af = hp[hp.family == 'ARFIMA']
    item(G5, 'ARFIMA H=0.7 detected (R/S or DFA): max', '13', 100 * af[np.isclose(af.H_true, 0.7)][['det_rs', 'det_dfa']].max().max(), 0, ' %')
    a1_ = hp[hp.family == 'AR1']
    item(G5, 'AR(1) false positives: min', '2', 100 * a1_[['det_rs', 'det_dfa']].min().min(), 0, ' %')
    item(G5, 'AR(1) false positives: max', '8', 100 * a1_[['det_rs', 'det_dfa']].max().max(), 0, ' %')
    lo_, hi_ = hp.n_median.min(), hp.n_median.max()
    ROWS.append(dict(group=G5, item='median segment length per mode within 450-550', paper='450--550', recomputed=f'{lo_:g}--{hi_:g}',
                     status='match' if 450 <= lo_ and hi_ <= 550 else 'MISMATCH'))
if hr is not None:
    d = hr.groupby('mode')[['det_rs', 'det_dfa']].mean()
    for m, c, v in [('walk', 'det_rs', '45'), ('walk', 'det_dfa', '41'), ('bus', 'det_rs', '10'), ('bus', 'det_dfa', '4'), ('car', 'det_rs', '15'), ('car', 'det_dfa', '3')]:
        item(G5, f'real segments detected, {m} {c[4:]}', v, 100 * d.loc[m, c], 0, ' %')

pc = paths.PRED_NGRAM / 'positive_control_summary.csv'                            # P0-5 (added 2026-10-22)
if pc.exists():
    pc = pd.read_csv(pc); lw, tr = pc[pc.model == 'lstm'], pc[pc.model == 'transformer']
    item(G5, 'positive control: planted bits/step, min', '0.10', pc.planted_bits.min(), 2, ' bits')
    item(G5, 'positive control: planted bits/step, max', '0.30', pc.planted_bits.max(), 2, ' bits')
    item(G5, 'positive control: available gain, min', '0.10', pc.expected_gap.min(), 2, ' bits')
    item(G5, 'positive control: available gain, max', '0.30', pc.expected_gap.max(), 2, ' bits')
    item(G5, 'LSTM: 128 steps worse than 4 steps, min', '0.003', -lw.gap_mean.max(), 3, ' bits')
    item(G5, 'LSTM: 128 steps worse than 4 steps, max', '0.011', -lw.gap_mean.min(), 3, ' bits')
    item(G5, 'Transformer: 128 steps worse than 4 steps, min', '0.21', -tr.gap_mean.max(), 2, ' bits')
    item(G5, 'Transformer: 128 steps worse than 4 steps, max', '0.23', -tr.gap_mean.min(), 2, ' bits')
    above = pc.k4_bits - pc.H_ctx4
    item(G5, '4-step models above the generator with 4 steps, min', '0.30', above.min(), 2, ' bits')
    item(G5, '4-step models above the generator with 4 steps, max', '0.32', above.max(), 2, ' bits')
    item(G5, 'positive control: seeds per model and set (all)', '3', pc.seeds.min(), 0)

ta = paths.PRED_NGRAM / 'ch5_hist_ablation_transformer.csv'                       # Transformer history ablation (2026-10-22)
if ta.exists():
    ta = pd.read_csv(ta); ta = ta[ta.dataset == 'taxi_traj20']; tb = ta.groupby('hist').test_bits.mean()
    for k, v in [(1, '4.44'), (2, '3.05'), (3, '2.83'), (4, '2.80'), (8, '2.77'), (16, '2.79'), (32, '2.82')]:
        item(G5, f'taxi Transformer restricted to k={k} (k=4: mean of 3 seeds)', v, tb[k], 2, ' bits')
    s4 = ta[ta['hist'] == 4].test_bits
    item(G5, 'taxi Transformer k=4, lowest seed', '2.78', s4.min(), 2, ' bits'); item(G5, 'taxi Transformer k=4, highest seed', '2.84', s4.max(), 2, ' bits')
    item(G5, 'taxi Transformer k=4, seeds', '3', len(s4), 0)
tr = [js(f) for f in ('neural_taxi_traj20_transformer.json', 'neural_taxi_traj20_transformer_s1.json', 'neural_taxi_traj20_transformer_s2.json')]
if all(tr):
    item(G5, 'taxi Transformer, 128 steps, mean of 3 seeds', '3.22', np.mean([t['test']['bits'] for t in tr]), 2, ' bits')

Q = pd.DataFrame(ROWS)
if a.csv:
    Q.to_csv(a.csv, index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 90); pd.set_option('display.max_rows', 500)
print(Q.to_string(index=False))
print('\nsummary:', Q.status.value_counts().to_dict())
