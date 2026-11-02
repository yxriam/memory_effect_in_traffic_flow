"""T09 highD backward-window features and preflight noise comparison.

python controls/absorption_highd.py noise --recording 01
CPU only. Ctrl+C to interrupt; rerun to resume atomic recording-level units.
The upstream position smoothing remains a limitation of all three estimates.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths as P
import numpy as np
import pandas as pd

VERSION = 't09-highd-backward2-v1'
WORK = P.CONTROLS_WORK / 'absorption' / 'highd'


def atomic_json(path, obj):
    # Same existing helper; imported lazily to leave scripts' old paths untouched.
    from heading_state import atomic_json as write
    write(path, obj)


def environment():
    return dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                platform=platform.platform(), compute='CPU', implementation=VERSION)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def velocities(x, frame):
    """Three velocity estimates with explicit continuity checks at 25 Hz."""
    n = len(x)
    out = {k: np.full(n, np.nan) for k in ('center', 'backward1', 'backward2')}
    good1 = np.diff(frame) == 1
    good2 = good1[:-1] & good1[1:]
    ix = np.flatnonzero(good1) + 1
    out['backward1'][ix] = (x[ix] - x[ix - 1]) / .04
    ix = np.flatnonzero(good2) + 2
    out['backward2'][ix] = (x[ix] - x[ix - 2]) / .08
    ix = np.flatnonzero(good2) + 1
    out['center'][ix] = (x[ix + 1] - x[ix - 1]) / .08
    return out


def acceleration(v, frame):
    out = np.full(len(v), np.nan)
    i = np.arange(5, len(v))
    i = i[frame[i] - frame[i - 5] == 5]
    out[i] = (v[i] - v[i - 5]) / .2
    return out


def scale(x):
    x = np.asarray(x)
    if not len(x):
        return dict(n=0, rms=None, robust_sd=None, median=None)
    median = np.median(x)
    return dict(n=len(x), rms=float(np.sqrt(np.mean(x ** 2))),
                robust_sd=float(1.4826 * np.median(np.abs(x - median))),
                median=float(median))


def noise(recording):
    start = time.monotonic()
    source = P.HIGHD_TRACKS / f'{recording}_tracks.csv'
    signature = dict(version=VERSION, source=fingerprint(source), recording=recording)
    cache = WORK / f'noise_{recording}.json'
    if cache.exists():
        result = json.loads(cache.read_text())
        if result['signature'] != signature:
            raise RuntimeError('Noise input/version changed; archive the unit before rerunning.')
        export_noise(result)
        print(json.dumps(result, indent=2))
        return result
    df = pd.read_csv(source, usecols=['id', 'frame', 'x'])
    df.sort_values(['id', 'frame'], inplace=True)
    metrics = {k: {m: [] for m in ('v', 'd2v', 'a', 'jerk', 'delta_center')}
               for k in ('center', 'backward1', 'backward2')}
    nvehicles = 0
    shift_max = 0.
    for _, g in df.groupby('id', sort=False):
        if len(g) < 50:  # highd_augment inclusion rule
            continue
        x, frame = g.x.to_numpy(float), g.frame.to_numpy(np.int64)
        v = velocities(x, frame)
        a = {k: acceleration(z, frame) for k, z in v.items()}
        valid = (frame % 5 == 0)
        for k in v:
            valid &= np.isfinite(v[k]) & np.isfinite(a[k])
        ix = np.flatnonzero(valid)
        if len(ix) < 3:
            continue
        nvehicles += 1
        triplets = np.flatnonzero((np.diff(frame[ix])[:-1] == 5) &
                                 (np.diff(frame[ix])[1:] == 5)) + 2
        pairs = np.flatnonzero(np.diff(frame[ix]) == 5) + 1
        for k in v:
            vv, aa = v[k][ix], a[k][ix]
            metrics[k]['v'].append(vv)
            metrics[k]['a'].append(aa)
            metrics[k]['d2v'].append((vv[triplets] - 2*vv[triplets-1] + vv[triplets-2]) / .2**2)
            metrics[k]['jerk'].append((aa[pairs] - aa[pairs-1]) / .2)
            metrics[k]['delta_center'].append(vv - v['center'][ix])
        j = np.arange(1, len(x))
        good = np.isfinite(v['backward2'][j]) & np.isfinite(v['center'][j-1])
        shift_max = max(shift_max, float(np.max(np.abs(v['backward2'][j[good]] - v['center'][j[good]-1]), initial=0)))
    rows = []
    for k, measures in metrics.items():
        row = dict(recording=recording, method=k, vehicles=nvehicles)
        for name, blocks in measures.items():
            for stat, value in scale(np.concatenate(blocks)).items():
                row[f'{name}_{stat}'] = value
        rows.append(row)
    result = dict(signature=signature, environment=environment(), rows=rows,
                  backward2_vs_center_shift1_max_abs=shift_max,
                  elapsed_seconds=time.monotonic()-start,
                  limitation='Descriptive noise proxies include true motion. Upstream smoothing may use future information.',
                  sampling='frame % 5 == 0; identical finite samples; tracks >=50 frames; contiguous derivatives',
                  acceleration='(v[t]-v[t-5])/0.2', random_seed='none; deterministic')
    atomic_json(cache, result)
    export_noise(result)
    print(json.dumps(result, indent=2))
    return result


def export_noise(result):
    rec = result['signature']['recording']
    target = P.CONTROLS_RESULTS / f'absorption_highd_noise_{rec}.csv'
    rows = [dict(**r, implementation=VERSION, source_sha256=result['signature']['source']['sha256'],
                 sampling=result['sampling'], acceleration=result['acceleration'],
                 environment=json.dumps(result['environment']), seed='none',
                 upstream_smoothing_limit=result['limitation']) for r in result['rows']]
    tmp = target.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(P.out(tmp), index=False)
    tmp.replace(target)


def prepare_recording(recording):
    """Cache both definitions without running or changing the legacy script."""
    source = P.HIGHD_TRACKS / f'{recording}_tracks.csv'
    full_source = P.HIGHD_FULL / f'{recording}_tracks.csv'
    dest = WORK / f'states_{recording}.npz'
    meta_path = WORK / f'states_{recording}.json'
    signature = dict(version=VERSION, source=fingerprint(source), lane_source=fingerprint(full_source))
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature'] != signature:
            raise RuntimeError('State input/version changed; archive the recording unit first.')
        return dest
    start = time.monotonic()
    raw = pd.read_csv(source, usecols=['id', 'frame', 'x'],
                      dtype={'id': np.int32, 'frame': np.int32, 'x': np.float64})
    lanes = pd.read_csv(full_source, usecols=['id', 'frame', 'laneId'],
                        dtype={'id': np.int32, 'frame': np.int32, 'laneId': np.int16})
    raw = raw.merge(lanes, on=['id', 'frame'], how='left', validate='one_to_one')
    if raw.laneId.isna().any():
        raise ValueError('Missing lane metadata for original frame/vehicle.')
    raw.sort_values(['id', 'frame'], inplace=True)
    blocks = []
    for vid, g in raw.groupby('id', sort=False):
        if len(g) < 50:
            continue
        x, fr = g.x.to_numpy(), g.frame.to_numpy()
        v = velocities(x, fr)
        ac = acceleration(v['center'], fr)
        ab = acceleration(v['backward2'], fr)
        # Legacy S retains every sampled frame with a finite centered v, even if a is NaN.
        ix = np.flatnonzero((fr % 5 == 0) & np.isfinite(v['center']))
        lane = g.laneId.to_numpy(np.int16)
        change = np.r_[False, np.diff(lane) != 0]
        # A lane change at frame j is observed only when j has arrived.
        last_change = np.maximum.accumulate(np.where(change, fr, -1000000))
        # No claim of 'no change' until 2 s of contiguous preceding frames were observed.
        window_start = np.searchsorted(fr, fr - 50)
        complete = (fr - fr[window_start] == 50) & (np.arange(len(fr)) - window_start == 50)
        lane_group = np.where(complete, (fr - last_change <= 50).astype(np.int8), -1)
        blocks.append(pd.DataFrame(dict(uid=np.full(len(ix), int(recording)*100000+int(vid), np.int64),
                                       frame=fr[ix], center=v['center'][ix], backward2=v['backward2'][ix],
                                       a_center=ac[ix], a_backward2=ab[ix], lane_recent=lane_group[ix])))
    data = pd.concat(blocks, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.with_suffix('.npz.tmp').open('wb') as f:
        np.savez_compressed(f, **{c: data[c].to_numpy() for c in data})
    dest.with_suffix('.npz.tmp').replace(dest)
    atomic_json(meta_path, dict(signature=signature, environment=environment(),
                               rows=len(data), vehicles=int(data.uid.nunique()),
                               elapsed_seconds=time.monotonic()-start))
    print(f'prepared {recording}: {len(data)} rows, {time.monotonic()-start:.1f}s', flush=True)
    return dest


def load_states(recordings):
    paths = [prepare_recording(rec) for rec in recordings]
    blocks = []
    for path in paths:
        with np.load(path) as data:
            blocks.append(pd.DataFrame({k: data[k] for k in data.files}))
    return pd.concat(blocks, ignore_index=True)


def legacy_check(recordings):
    """Replay old V1/V2/V3/VA1/VA2 with legacy alpha=0.5, nats units.

    All-recording runs must match the existing table before interpreting differences.
    Single-recording runs are timing pilots, never a reproduction verdict.
    """
    import highd_augment as h
    start = time.monotonic()
    S = load_states(recordings).rename(columns={'center': 'v', 'a_center': 'a'})
    # build_steps expects leader covariates even when unused by these models.
    S['dv'], S['gap'] = np.nan, np.nan
    models = {k: h.MODELS[k] for k in ('V1', 'V2', 'V3', 'VA1', 'VA2')}
    tag = 'all' if len(recordings) == 60 else '-'.join(recordings)
    signature = dict(version=VERSION, recordings=recordings, models=models,
                     units='nats/step', split_seeds=[100,101,102])
    cached = WORK / f'legacy_check_{tag}.json'
    if cached.exists():
        r = json.loads(cached.read_text())
        if r['signature'] != signature:
            raise ValueError('Legacy check signature mismatch')
        if r['verdict']=='FAILED':
            raise RuntimeError('Cached legacy reproduction gate failed; inspect before continuing.')
        return pd.DataFrame(r['rows'])
    rows = []
    for split in range(3):
        checkpoint = WORK / f'legacy_check_{tag}_split{split}.json'
        if checkpoint.exists():
            r = json.loads(checkpoint.read_text())
            if r['signature'] != signature:
                raise ValueError('Legacy split signature mismatch')
            rows.extend(r['rows'])
            continue
        # h.markov_block sp=0 always uses seed100. Keep its exact processing,
        # vary seed through the identical documented permutation explicitly.
        uids = S.uid.unique()
        perm = np.random.default_rng(100+split).permutation(uids)
        ntr = int(.8*len(perm))
        train_ids, test_ids = set(perm[:ntr]), set(perm[ntr:])
        T = S[S.uid.isin(train_ids)]
        edges = dict(v=h.qedges(T.v.to_numpy(),20), a=h.qedges(T.a.to_numpy(),7),
                     dv=np.array([-np.inf,0.,np.inf]), g=np.array([-np.inf,0.,np.inf,np.inf]))
        tr = h.build_steps(S, train_ids, edges)
        te = h.build_steps(S, test_ids, edges)
        radix = dict(v0=20,v1=20,v2=20,a=7,dv=5,g=5)
        current = []
        for name, cols in models.items():
            ltr,lte,nc = h.eval_model(tr,te,cols,radix,20,alpha=.5)
            current.append(dict(split=split,model=name,train_ll=ltr,val_ll=lte,n_ctx=nc,
                                n_train=len(tr['y']),n_val=len(te['y']),state='signed_v'))
        atomic_json(checkpoint, dict(signature=signature, rows=current))
        rows.extend(current)
        print(f'legacy {tag} split {split}: {len(te["y"])} test steps [{time.monotonic()-start:.1f}s]',flush=True)
    r = pd.DataFrame(rows)
    verdict = 'pilot_only'
    if len(recordings) == 60:
        old = pd.read_csv(P.CONTROLS_RESULTS/'augment_markov.csv')
        old = old[(old.state=='signed_v') & old.model.isin(models)]
        compare = r.merge(old,on=['split','model','state'],suffixes=('_new','_old'),validate='one_to_one')
        for c in ('train_ll','val_ll','n_ctx','n_train','n_val'):
            compare[f'delta_{c}']=compare[f'{c}_new']-compare[f'{c}_old']
        compare['environment']=json.dumps(environment())
        compare['alpha']=.5
        compare['units']='nats/step'
        compare['implementation']=VERSION
        compare['seed_rule']='numpy.default_rng(100 + split), 80/20 vehicles'
        compare.to_csv(P.out(P.CONTROLS_RESULTS/'absorption_highd_legacy_reproduction.csv'),index=False)
        maxdiff = compare[[f'delta_{c}' for c in ('train_ll','val_ll','n_ctx','n_train','n_val')]].abs().max().max()
        verdict = 'match' if maxdiff < 1e-10 else 'FAILED'
        print(f'Legacy reproduction: {verdict}, max absolute difference {maxdiff:g}',flush=True)
    atomic_json(cached,dict(signature=signature,environment=environment(),rows=rows,
                           verdict=verdict,elapsed_seconds=time.monotonic()-start))
    if verdict=='FAILED':
        raise RuntimeError('Legacy reproduction gate failed. Inspect and report before main analysis.')
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['noise','prepare','legacy-check'])
    parser.add_argument('--recording', default='01')
    args = parser.parse_args()
    recordings = ([p.name[:2] for p in sorted(P.HIGHD_TRACKS.glob('*_tracks.csv'))]
                  if args.recording=='all' else [args.recording])
    if args.command=='noise':
        noise(args.recording)
    elif args.command=='prepare':
        for rec in recordings:
            prepare_recording(rec)
    else:
        legacy_check(recordings)
