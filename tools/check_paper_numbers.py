"""Check every number of the paper that comes from controls/results and prediction/results.

    python tools/check_paper_numbers.py            # prints a table, one row per number
    python tools/check_paper_numbers.py --csv out.csv

Each row: the number as printed in the paper, the value re-computed from the committed result files, and
'match' if the value rounds to the printed number. Run it after re-running any analysis and paste the summary
line into the experiment log (EXPERIMENT_LOG.md). The expected values already include the corrections found on
2026-10-19 (GeoLife LSTM 2.36, highD-100 k=2 LSTM 1.44, train interval 1.12; see experiment_log/).
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

Q = pd.DataFrame(ROWS)
if a.csv:
    Q.to_csv(a.csv, index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 90); pd.set_option('display.max_rows', 500)
print(Q.to_string(index=False))
print('\nsummary:', Q.status.value_counts().to_dict())
