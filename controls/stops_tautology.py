"""P0-4  Persistence time and stops: is 'tau_c follows the stop frequency' a tautology?

A stop is a large drop in speed, so it shortens the speed autocorrelation time by construction; across modes there are
only 7 points, and stop frequency and speed are themselves correlated (rho = 0.93). Three checks:

(i)  tau_c on stop-free stretches. Clean segments (split at gaps > 30 s, 5 s resampling, as row_persistence.py);
     stop samples are speed < 0.5 m/s. Non-overlapping windows of W = 60 samples (300 s) are cut (a) from whole
     segments and (b) from stop-free runs only (run ends trimmed by 2 samples, i.e. 10 s, to drop approach and
     departure). tau_c = first lag with ACF < 1/e (lags up to 30 samples; censored at 150 s). Same window length for
     every mode, so the finite-length bias is the same.
(ii) Segment-level mixed model (REML, random intercept per user) on results/row_persistence.csv, gap 30 s:
     log tau_c ~ log mean speed + log(1 + stops/km) + log length + mode (car = reference) + (1 | user).
(iii) Finite-length bias of tau_c: per segment, 5 AR(1) series with the segment's lag-1 autocorrelation and length;
     tau_c is estimated exactly as for the data and compared with the true AR(1) value (first lag with phi^k < 1/e).
     The data tau_c is divided by the median ratio estimated / true of its own surrogates.

    python controls/stops_tautology.py   -> results/stops_windows.csv, stops_tautology_summary.json, stops_lmm.csv, stops_ar1_bias.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import importlib.util, glob, os, json, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import optimize, stats

spec = importlib.util.spec_from_file_location('m4', str(P.GEOLIFE_PIPELINE / '04_mode_hurst_memory.py'))
m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
DT = 5.0
MODES = ['walk', 'bike', 'bus', 'subway', 'car', 'taxi', 'train']
W, MAXLAG, TRIM = 60, 30, 2


def acf(x, maxlag):
    x = np.asarray(x, float) - np.mean(x); n = len(x); v = np.dot(x, x) / n
    if v <= 0: return None
    return np.array([np.dot(x[:n - k], x[k:]) / n / v for k in range(maxlag + 1)])


def tau_of(x, maxlag):
    a = acf(x, maxlag)
    if a is None: return np.nan, 1
    below = np.where(a < np.exp(-1))[0]
    return (float(below[0] * DT), 0) if len(below) else (float((maxlag + 1) * DT), 1)


def runs(mask):
    """Start / end (exclusive) of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def windows(args):
    mode, f = args
    raw = m4.read_plt(m4.Path(f)); user = os.path.basename(f)[:3]
    out = []
    for i, seg in enumerate(m4.split_by_gap(raw, 30.0, 30)):
        rs = m4.resample_segment(seg, DT)
        if rs is None: continue
        _, _, sp = m4.velocity(rs, DT)
        if len(sp) < W: continue
        stopped = sp < 0.5
        for w0 in range(0, len(sp) - W + 1, W):
            x = sp[w0:w0 + W]; st = stopped[w0:w0 + W]
            tau, cen = tau_of(x, MAXLAG)
            n_stop = int(np.sum(st[2:] & st[1:-1] & ~st[:-2]))
            out.append(dict(mode=mode, user=user, seg=i, kind='whole', w0=w0, tau=tau, censored=cen, mean_speed=float(x.mean()),
                            stops=n_stop, km=float(x.sum() * DT / 1000)))
        s0, s1 = runs(~stopped)
        for a, b in zip(s0 + TRIM, s1 - TRIM):
            for w0 in range(a, b - W + 1, W):
                x = sp[w0:w0 + W]
                tau, cen = tau_of(x, MAXLAG)
                out.append(dict(mode=mode, user=user, seg=i, kind='stop_free', w0=int(w0), tau=tau, censored=cen,
                                mean_speed=float(x.mean()), stops=0, km=float(x.sum() * DT / 1000)))
    return out


def reml_lmm(y, X, groups):
    """Random-intercept linear mixed model by REML. Returns beta, se, sigma_u^2, sigma_e^2."""
    g = pd.factorize(groups)[0]; G = g.max() + 1; n, p = X.shape
    ng = np.bincount(g, minlength=G).astype(float)
    Sx = np.zeros((G, p)); np.add.at(Sx, g, X); Sy = np.bincount(g, weights=y, minlength=G)
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y

    def parts(lg):
        gam = np.exp(lg); c = gam / (1 + gam * ng)
        A = XtX - (Sx * c[:, None]).T @ Sx                    # X' H^-1 X
        b = Xty - (Sx * (c * Sy)[:, None]).sum(0)              # X' H^-1 y
        beta = np.linalg.solve(A, b)
        rHr = (yty - (c * Sy ** 2).sum()) - b @ beta           # r' H^-1 r at beta
        return gam, A, beta, rHr

    def negll(lg):
        gam, A, beta, rHr = parts(lg)
        s2 = rHr / (n - p)
        return 0.5 * ((n - p) * np.log(s2) + np.log1p(gam * ng).sum() + np.linalg.slogdet(A)[1])

    lg = optimize.minimize_scalar(negll, bounds=(-12, 6), method='bounded').x
    gam, A, beta, rHr = parts(lg)
    s2 = rHr / (n - p)
    se = np.sqrt(np.diag(np.linalg.inv(A)) * s2)
    return beta, se, gam * s2, s2


if __name__ == '__main__':
    jobs = [(m, f) for m in MODES for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), m, 'plt', '*.plt')))]
    with Pool(2) as pool:
        Wn = pd.DataFrame([r for res in pool.imap(windows, jobs) for r in res])
    Wn.to_csv(P.out(P.CONTROLS_RESULTS / 'stops_windows.csv'), index=False)
    summ = {'window_samples': W, 'max_lag_s': MAXLAG * DT}
    # (i) mode medians on equal-length windows
    summ['windows_without_variance'] = int(Wn.tau.isna().sum())          # constant speed in the window (tau undefined)
    for kind, d in Wn[Wn.tau.notna()].groupby('kind'):
        med = d.groupby('mode').agg(windows=('tau', 'size'), users=('user', 'nunique'), tau_median=('tau', 'median'),
                                   censored=('censored', 'mean'), speed=('mean_speed', 'median'),
                                   stops_per_km=('stops', 'sum'), km=('km', 'sum'))
        med['stops_per_km'] = med.stops_per_km / med.km
        med = med.drop(columns='km').reindex(MODES)
        lt = np.log(d.tau)
        eta2 = float(d.assign(lt=lt).groupby('mode').lt.apply(lambda x: len(x) * (x.mean() - lt.mean()) ** 2).sum() / ((lt - lt.mean()) ** 2).sum())
        kw = stats.kruskal(*[np.log(g.tau) for _, g in d.groupby('mode')])
        summ[kind] = dict(by_mode=med.round(4).reset_index().to_dict('records'), eta2_log_tau=eta2, kruskal_H=float(kw.statistic),
                          kruskal_p=float(kw.pvalue), max_over_min_median=float(med.tau_median.max() / med.tau_median.min()))
        if kind == 'whole':
            r = stats.spearmanr(med.tau_median, med.stops_per_km); summ[kind]['spearman_tau_stops'] = [float(r.statistic), float(r.pvalue)]
        print(kind); print(med.round(3)); print('eta2', round(eta2, 3), 'KW p', kw.pvalue)
    # (ii) mixed model on the segment table
    R = pd.read_csv(P.CONTROLS_RESULTS / 'row_persistence.csv')
    R = R[(R.gap == 30) & R['mode'].isin(MODES) & R.tau_c.notna()].copy()
    y = np.log(R.tau_c.clip(lower=5)).to_numpy()
    cols = {'const': np.ones(len(R)), 'log_speed': np.log(R.mean_speed.clip(lower=0.1)), 'log1p_stops_per_km': np.log1p(R.stops_per_km.fillna(0)),
            'log_length': np.log(R.n)}
    for m in MODES:
        if m != 'car': cols['mode_' + m] = (R['mode'] == m).astype(float)
    X = np.column_stack(list(cols.values()))
    rows = []
    for spec_name, keep in [('speed + stops + length + mode', list(cols)), ('speed + length + mode', [c for c in cols if c != 'log1p_stops_per_km']),
                            ('stops + length + mode', [c for c in cols if c != 'log_speed']), ('mode only', ['const', 'log_length'] + [c for c in cols if c.startswith('mode_')])]:
        idx = [list(cols).index(c) for c in keep]
        b, se, su2, se2 = reml_lmm(y, X[:, idx], R.user.astype(str).to_numpy())
        for c, bb, ss in zip(keep, b, se):
            rows.append(dict(model=spec_name, term=c, coef=bb, se=ss, z=bb / ss, factor=np.exp(bb), sigma_user2=su2, sigma_resid2=se2, n=len(y), users=R.user.nunique()))
    L = pd.DataFrame(rows); L.to_csv(P.out(P.CONTROLS_RESULTS / 'stops_lmm.csv'), index=False)
    print(L.round(3).to_string(index=False))
    # (iii) AR(1) finite-length bias
    B = []
    for _, r in R.iterrows():
        phi = float(np.clip(r.phi, 1e-6, 0.9995)); n = int(r.n)
        true = DT * (np.floor(-1 / np.log(phi)) + 1)
        rng = np.random.default_rng(zlib.crc32(f"{r['mode']}|{r.user}|{r.seg}".encode()))
        est = []
        for _ in range(5):
            e = rng.normal(0, np.sqrt(1 - phi ** 2), n); s = np.empty(n); s[0] = rng.normal()
            for k in range(1, n): s[k] = phi * s[k - 1] + e[k]
            a = acf(s, 120); below = np.where(a < np.exp(-1))[0]
            est.append(below[0] * DT if len(below) else np.nan)
        ratio = np.nanmedian(np.array(est) / true)
        B.append(dict(mode=r['mode'], user=r.user, seg=r.seg, n=n, phi=phi, tau_true_ar1=true, ratio_est_true=ratio,
                      tau_c=r.tau_c, tau_c_corrected=r.tau_c / ratio, stops_per_km=r.stops_per_km))
    B = pd.DataFrame(B); B.to_csv(P.out(P.CONTROLS_RESULTS / 'stops_ar1_bias.csv'), index=False)
    bm = B.groupby('mode').agg(ratio_median=('ratio_est_true', 'median'), tau=('tau_c', 'median'), tau_corrected=('tau_c_corrected', 'median'),
                               stops_per_km=('stops_per_km', 'median')).reindex(MODES)
    r1 = stats.spearmanr(bm.tau, bm.stops_per_km); r2 = stats.spearmanr(bm.tau_corrected, bm.stops_per_km)
    summ['ar1_bias'] = dict(by_mode=bm.round(4).reset_index().to_dict('records'), spearman_raw=[float(r1.statistic), float(r1.pvalue)],
                            spearman_corrected=[float(r2.statistic), float(r2.pvalue)],
                            ratio_vs_length_spearman=float(stats.spearmanr(B.n, B.ratio_est_true, nan_policy='omit').statistic))
    print(bm.round(3)); print('rho raw / corrected', r1.statistic, r2.statistic)
    json.dump(summ, open(P.out(P.CONTROLS_RESULTS / 'stops_tautology_summary.json'), 'w'), indent=1, default=float)
