"""Taxi memory-network controls (to-do P1-1 and P1-5) and a re-run of the published taxi networks.

Networks as in ch6_networks.py (all taxi files, global 0.01 deg grid, repeats collapsed; first-order network and
second-order memory network with empirical link flows), partitioned with Infomap 2.8.0 exactly as run_infomap.py
(--two-level --directed --flow-model rawdir --num-trials 20 --seed 123). Variants:
  data                    the published network (should reproduce results/infomap/infomap_summary.csv)
  pooled_surrogate        first-order Markov surrogate from the pooled P(j|i) (= the published surrogate, rng seed 1)
  per_vehicle_surrogate   P1-1: every vehicle's sequence redrawn from that vehicle's own first-order chain
                          (same start and length), so heterogeneity between vehicles is kept and only memory removed
  dejittered              P1-5: A -> B -> A patterns removed from every sequence (geolife_robust.dejitter)
  dejittered_pooled_surrogate   pooled surrogate of the dejittered sequences
  dejittered_per_vehicle_surrogate   per-vehicle surrogate of the dejittered sequences
Surrogates are drawn with 3 seeds each (pooled 1-3, per vehicle 11-13); seed 1 is the published surrogate.
For each: plug-in 1 - H2/H1, codelength saving L_first - L_memory, modules, share of physical cells in >= 2 modules.
Infomap: the official binary if INFOMAP_BIN is set (e.g. a build of github.com/mapequation/infomap v2.8.0), else
the Python package (pip install infomap==2.8.0).

    python prediction/taxi_network_controls.py   -> results/ngram/taxi_network_controls.csv
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.PREDICTION)); _sys.path.insert(0, str(P.CONTROLS))
import json, os, re, subprocess, time
import numpy as np, pandas as pd
import ch6_networks as c6
import geolife_robust as gr

FLAGS = '--two-level --directed --flow-model rawdir --num-trials 20 --seed 123'
WORK = P.PRED_RESULTS / 'work_taxi_network_controls'


def per_vehicle_surrogate(S, rng):
    out = []
    for s in S:
        a, b = s[:-1], s[1:]
        order = np.argsort(a, kind='stable'); a_s, b_s = a[order], b[order]
        x = [s[0]]
        for _ in range(len(s) - 1):
            i = x[-1]; lo, hi = np.searchsorted(a_s, i), np.searchsorted(a_s, i, 'right')
            if hi == lo: break
            x.append(b_s[rng.integers(lo, hi)])
        if len(x) >= 3: out.append(np.array(x))
    return out


def infomap(net):
    """Returns codelength (bits), number of top modules and {state or node id: top module} for one .net file."""
    out = WORK / 'im'; out.mkdir(parents=True, exist_ok=True)
    tag = os.path.basename(net)[:-4]
    binp = os.environ.get('INFOMAP_BIN')
    if binp:
        log = subprocess.run([binp, str(net), str(out)] + FLAGS.split() + ['--tree', '--silent'], capture_output=True, text=True, check=True)
        is_mem = (out / f'{tag}_states.tree').exists()
        tree = out / (f'{tag}_states.tree' if is_mem else f'{tag}.tree')
        head = [l for l in open(tree) if l.startswith('#')]
        L = float(re.search(r'codelength ([0-9.eE+-]+) bits', ''.join(head)).group(1))
        nmod = int(re.search(r'with (\d+) top modules', ''.join(head)).group(1))
        mods = {}
        for l in open(tree):
            if l.startswith('#'): continue
            f = l.split()
            mods[int(f[3] if is_mem else f[-1])] = int(f[0].split(':')[0])
        return L, nmod, mods
    import infomap as imp
    im = imp.Infomap(FLAGS + ' --silent'); im.read_file(str(net)); im.run()
    mods = im.get_modules(states=True) if im.have_memory else im.get_modules()
    return im.codelength, im.num_top_modules, dict(mods)


def analyse(label, seqs):
    t0 = time.time()
    S, first, mem, states, V = c6.build(seqs)
    c6.X = str(WORK / 'net'); os.makedirs(c6.X, exist_ok=True)
    c6.export_infomap(label, first, mem, states, V)
    h1, h2 = c6.entropy_rates(S)
    Lf, mf, _ = infomap(WORK / 'net' / f'{label}_first.net')
    Lm, mm, smod = infomap(WORK / 'net' / f'{label}_memory.net')
    phys = {}
    for sid, m in smod.items():
        phys.setdefault(int(mem.phys[sid - 1]), set()).add(m)
    multi = sum(len(v) > 1 for v in phys.values())
    r = dict(variant=label, sequences=len(S), steps=int(sum(len(s) - 1 for s in S)), physical_nodes=V, state_nodes=mem.n, H1=h1, H2=h2,
             entropy_rate_reduction=1 - h2 / h1, L_first=Lf, L_memory=Lm, saving=Lf - Lm, modules_first=mf, modules_memory=mm,
             share_multi_module=multi / len(phys), seconds=time.time() - t0)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


VARIANTS = [('data', None), ('pooled_surrogate', 1), ('per_vehicle_surrogate', 11), ('dejittered', None),
            ('dejittered_pooled_surrogate', 1), ('pooled_surrogate', 2), ('pooled_surrogate', 3),
            ('per_vehicle_surrogate', 12), ('per_vehicle_surrogate', 13), ('dejittered_per_vehicle_surrogate', 11),
            ('dejittered_per_vehicle_surrogate', 12), ('dejittered_per_vehicle_surrogate', 13),
            ('dejittered_pooled_surrogate', 2), ('dejittered_pooled_surrogate', 3)]


if __name__ == '__main__':
    # every finished variant is kept in WORK/rows/<label>.json, so an interrupted run resumes where it stopped
    (WORK / 'rows').mkdir(parents=True, exist_ok=True)
    seqs = c6.taxi_sequences()
    S, first, mem, states, V = c6.build(seqs)
    dj = [d for d in (gr.dejitter(s) for s in seqs) if len(d) >= 3]
    Sd, _, _, _, Vd = c6.build(dj)
    make = {'data': lambda sd: seqs, 'dejittered': lambda sd: dj,
            'pooled_surrogate': lambda sd: c6.surrogate(S, V, np.random.default_rng(sd)),
            'per_vehicle_surrogate': lambda sd: per_vehicle_surrogate(S, np.random.default_rng(sd)),
            'dejittered_pooled_surrogate': lambda sd: c6.surrogate(Sd, Vd, np.random.default_rng(sd)),
            'dejittered_per_vehicle_surrogate': lambda sd: per_vehicle_surrogate(Sd, np.random.default_rng(sd))}
    rows = []
    for kind, sd in VARIANTS:
        label = kind if sd in (None, 1, 11) else f'{kind}_s{sd}'
        cache = WORK / 'rows' / f'{label}.json'
        if cache.exists():
            r = json.load(open(cache))
        else:
            r = analyse(label, make[kind](sd)); r['seed'] = sd; json.dump(r, open(cache, 'w'))
        r['kind'] = kind; rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(P.out(P.PRED_NGRAM / 'taxi_network_controls.csv'), index=False)
    pd.set_option('display.width', 220); print(R.drop(columns='seconds').round(4).to_string(index=False))
    print(R.groupby('kind')[['saving', 'entropy_rate_reduction', 'share_multi_module', 'modules_memory']].agg(['mean', 'std', 'count']).round(4))
