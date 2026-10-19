"""P1-3  Can R/S and DFA detect genuine long memory at the segment lengths of GeoLife?

'Reproduced by a Markov surrogate' says the diagnostic is not sufficient; it does not say long memory is absent.
Test: series with known long memory, of the same lengths as the real 5 s speed segments (walk, bus, car; segments
split at gaps > 30 min, as in hurst_control.py), analysed with the paper's R/S estimator and DFA-1:
  fGn      fractional Gaussian noise, H = 0.6, 0.7, 0.8 (Davies-Harte)
  ARFIMA   ARFIMA(1, d, 0), d = H - 0.5, AR coefficient = the median lag-1 autocorrelation of the mode's real segments
           (walk / bus / car), i.e. strong short-range persistence plus long memory
  AR(1)    Markov series with the same lag-1 autocorrelation as the ARFIMA series (false-positive check, H = 0.5)
Each series is compared with 19 AR(1) surrogates that match its own length, mean, variance and lag-1
autocorrelation; it counts as 'detected' if its exponent exceeds all 19 (one-sided p = 0.05). 100 series per cell,
lengths drawn from the real segments of that mode. The same test is applied to every real segment.

    python controls/hurst_power.py   -> results/hurst_power_synthetic.csv, hurst_power_real.csv, hurst_power_summary.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import glob, os, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter
from scipy.special import gammaln
import hurst_control as hc

MODES = ['walk', 'bus', 'car']
HS = (0.6, 0.7, 0.8)
N_SER, N_SUR = 100, 19


def fgn(n, H, rng):
    """Davies-Harte fractional Gaussian noise of length n, unit variance."""
    k = np.arange(n + 1)
    r = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    lam = np.fft.rfft(np.r_[r, r[-2:0:-1]]).real
    lam = np.maximum(lam, 0)
    m = 2 * n
    z = rng.normal(size=m // 2 + 1) + 1j * rng.normal(size=m // 2 + 1)
    z[0] = z[0].real * np.sqrt(2); z[-1] = z[-1].real * np.sqrt(2)
    x = np.fft.irfft(np.sqrt(lam / (2 * m)) * z * np.sqrt(m), m)[:n]
    return (x - x.mean()) / x.std()


def arfima(n, d, phi, rng, burn=2000):
    """ARFIMA(1, d, 0): (1 - phi B)(1 - B)^d x = e, via the truncated MA(inf) weights of (1 - B)^-d."""
    L = n + burn
    k = np.arange(L)
    psi = np.exp(gammaln(k + d) - gammaln(k + 1) - gammaln(d))
    e = rng.normal(size=L)
    u = np.fft.irfft(np.fft.rfft(e, 2 * L) * np.fft.rfft(psi, 2 * L), 2 * L)[:L]
    x = lfilter([1.0], [1.0, -phi], u)[burn:]
    return (x - x.mean()) / x.std()


def ar1_like(x, rng):
    s, _ = hc.ar1(x, rng)
    return s


def test_series(x, rng):
    """Exponents of x and of 19 matched AR(1) surrogates; detected if x exceeds all of them."""
    h, a = hc.m4.hurst_rs(x), hc.dfa(x)
    hs, as_ = [], []
    for _ in range(N_SUR):
        s = ar1_like(x, rng); hs.append(hc.m4.hurst_rs(s)); as_.append(hc.dfa(s))
    hs, as_ = np.array(hs), np.array(as_)
    return dict(n=len(x), phi=float(np.corrcoef(x[:-1], x[1:])[0, 1]), H_rs=h, H_rs_sur_max=float(np.nanmax(hs)), H_rs_sur_median=float(np.nanmedian(hs)),
                det_rs=bool(h > np.nanmax(hs)), dfa=a, dfa_sur_max=float(np.nanmax(as_)), dfa_sur_median=float(np.nanmedian(as_)), det_dfa=bool(a > np.nanmax(as_)))


def synth_job(args):
    family, mode, H, i, n, phi = args
    rng = np.random.default_rng(zlib.crc32(f'{family}|{mode}|{H}|{i}'.encode()))
    if family == 'fGn': x = fgn(n, H, rng)
    elif family == 'ARFIMA': x = arfima(n, H - 0.5, phi, rng)
    else: x = lfilter([1.0], [1.0, -phi], rng.normal(size=n + 500))[500:]
    return dict(family=family, mode=mode, H_true=H, i=i, **test_series(x, rng))


def real_job(args):
    mode, f = args
    raw = hc.m4.read_plt(hc.m4.Path(f)); out = []
    rng = np.random.default_rng(zlib.crc32(f'{mode}|{os.path.basename(f)}|real'.encode()))
    for i, seg in enumerate(hc.m4.split_by_gap(raw, 1800.0, 30)):
        rs = hc.m4.resample_segment(seg, hc.DT)
        if rs is None: continue
        _, _, sp = hc.m4.velocity(rs, hc.DT)
        if len(sp) < 202: continue
        out.append(dict(mode=mode, file=os.path.basename(f), seg=i, **test_series(sp, rng)))
    return out


if __name__ == '__main__':
    HC = pd.read_csv(P.CONTROLS_RESULTS / 'hurst_control.csv')
    HC = HC[HC.gap == 1800]
    jobs = []
    for mode in MODES:
        lens = HC[HC['mode'] == mode].n.to_numpy(); phi = float(HC[HC['mode'] == mode].phi.median())
        rng = np.random.default_rng(zlib.crc32(mode.encode()))
        for i in range(N_SER):
            n = int(rng.choice(lens))
            for H in HS:
                jobs.append(('fGn', mode, H, i, n, phi)); jobs.append(('ARFIMA', mode, H, i, n, phi))
            jobs.append(('AR1', mode, 0.5, i, n, phi))
    with Pool(2) as pool:
        Sy = pd.DataFrame(pool.map(synth_job, jobs, chunksize=20))
    Sy.to_csv(P.out(P.CONTROLS_RESULTS / 'hurst_power_synthetic.csv'), index=False)
    rjobs = [(m, f) for m in MODES for f in sorted(glob.glob(os.path.join(str(P.GEOLIFE_TRIPS_BY_MODE), m, 'plt', '*.plt')))]
    with Pool(2) as pool:
        Re = pd.DataFrame([r for res in pool.imap(real_job, rjobs) for r in res])
    Re.to_csv(P.out(P.CONTROLS_RESULTS / 'hurst_power_real.csv'), index=False)
    A = Sy.groupby(['family', 'mode', 'H_true']).agg(series=('n', 'size'), n_median=('n', 'median'), H_rs=('H_rs', 'median'),
                                                       det_rs=('det_rs', 'mean'), dfa=('dfa', 'median'), det_dfa=('det_dfa', 'mean')).reset_index()
    B = Re.groupby('mode').agg(series=('n', 'size'), n_median=('n', 'median'), H_rs=('H_rs', 'median'), det_rs=('det_rs', 'mean'),
                                dfa=('dfa', 'median'), det_dfa=('det_dfa', 'mean')).reset_index()
    B['family'] = 'data'; B['H_true'] = np.nan
    out = pd.concat([A, B], ignore_index=True); out.to_csv(P.out(P.CONTROLS_RESULTS / 'hurst_power_summary.csv'), index=False)
    pd.set_option('display.width', 200); print(out.round(3).to_string(index=False))
