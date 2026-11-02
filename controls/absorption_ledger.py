"""T09 absorption ledger, CPU, with atomic resumable units.

Implemented system: highd. Run after the noise and legacy reproduction checks:
  python controls/absorption_ledger.py highd --recording 01 --pilot
  python controls/absorption_ledger.py highd
Ctrl+C interrupts; rerun the exact command to resume completed units.
Pilot outputs are separate and cannot be mistaken for full-system results.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import paths as P
import absorption_highd as hd
import geolife_robust as gr
import heading_state as hs
import highd_augment as ha

VERSION = 't09-ledger-highd-v1'
_table_cache = None
_base_cache = {}


def seed(quantity, group, k):
    return zlib.crc32(f'T09|highd|{quantity}|{group}|{k}'.encode())


def cached_prob(y, contexts, train, evaluate, K):
    """Fit counts once per layer; return conditional count lookups for all betas.

    Reuses the existing sparse lookup routine. Caching avoids re-sorting the
    same training counts for every smoothing strength; it is new functionality.
    """
    uni = np.bincount(y[train], minlength=K).astype(float)
    base = (uni[y[evaluate]] + 1) / (uni.sum() + K)
    layers = []
    for ctx in contexts:
        joint, nj = np.unique(ctx[train]*K+y[train],return_counts=True)
        keys, nc = np.unique(ctx[train],return_counts=True)
        layers.append((gr.lookup(joint,nj,ctx[evaluate]*K+y[evaluate]),
                       gr.lookup(keys,nc,ctx[evaluate])))
    return base, layers


def tune_and_score(y, contexts, inner, validation, train, test):
    base, layers = cached_prob(y,contexts,inner,validation,20)
    betas = []
    for numerator,denominator in layers:
        best = max(hs.GRID,key=lambda b: np.log2((numerator+b*base)/(denominator+b)).mean())
        betas.append(best)
        base = (numerator+best*base)/(denominator+best)
    base, layers = cached_prob(y,contexts,train,test,20)
    for (numerator,denominator),beta in zip(layers,betas):
        base = (numerator+beta*base)/(denominator+beta)
    return -np.log2(base), betas


def chained(s, other, radices):
    out = [s]
    for col,radix in zip(other,radices):
        out.append(out[-1]*radix+col)
    return out


def build_table(S, mode, split, k):
    """Same outer vehicle permutation as highd_augment; per-track inner 80/20.

    Each k has its own fixed test targets. Backward/center comparisons share
    frame indices, although their numerical target definitions differ.
    """
    uid = S.uid.to_numpy()
    fr = S.frame.to_numpy()
    uids, first, count = np.unique(uid,return_index=True,return_counts=True)
    if not np.all(np.diff(uid)>=0):
        raise ValueError('States must be sorted by recording, vehicle, frame')
    perm = np.random.default_rng(100+split).permutation(S.uid.unique())
    train_ids = perm[:int(.8*len(perm))]
    outer = np.isin(uid,train_ids)
    # inner_split cuts each trajectory in sampled time and removes crossing windows.
    rank = np.arange(len(S))-np.repeat(first,count)
    cut = np.repeat(np.rint(.8*count).astype(np.int64),count)
    i = np.arange(k,len(S)-1)
    good = (uid[i-k]==uid[i+1]) & (fr[i+1]-fr[i-k]==5*(k+1))
    for name in ('center','backward2'):
        values = S[name].to_numpy()
        good &= np.isfinite(values[i+1]) & np.isfinite(S['a_'+name].to_numpy()[i])
        for lag in range(k+1):
            good &= np.isfinite(values[i-lag])
    i = i[good]
    train, test = outer[i], ~outer[i]
    inner = train & (rank[i]+1 < cut[i])
    validation = train & (rank[i]-k >= cut[i])
    v = S[mode].to_numpy()
    a = S['a_'+mode].to_numpy()
    # Existing highD convention: quantiles on outer training vehicles only.
    e20 = ha.qedges(v[outer],20)
    e100 = ha.qedges(v[outer],100)
    ea = ha.qedges(a[outer],20)
    vb = ha.qbins(v,e20)
    s,y = vb[i],vb[i+1]
    hist = [vb[i-lag] for lag in range(1,k+1)]
    fine = ha.qbins(v[i],e100)
    acc = ha.qbins(a[i],ea)
    threshold = float(np.nanquantile(np.abs(a[outer]),.9))
    recent = S.lane_recent.to_numpy()[i]
    return dict(s=s,y=y,hist=hist,fine=fine,acc=acc,train=train,test=test,
                inner=inner,validation=validation,uid=uid[i],
                groups={'all':np.ones(len(i),bool),'lane_recent':recent==1,
                        'lane_other':recent==0,'hard_accel':np.abs(a[i])>=threshold,
                        'other_accel':np.abs(a[i])<threshold},
                edges={'v20':e20.tolist(),'v100':e100.tolist(),'a20':ea.tolist()},
                accel_threshold=threshold,indices=i)


def aggregate_units(uid, losses, groups):
    unique,inv = np.unique(uid,return_inverse=True)
    result = {}
    for name,mask in groups.items():
        count = np.bincount(inv[mask],minlength=len(unique))
        result[name]=dict(uid=unique.tolist(),n=count.tolist(),
                          sums={m:np.bincount(inv[mask],weights=value[mask],minlength=len(unique)).tolist()
                                for m,value in losses.items()})
    return result


def score_unit(S, mode, split, k, variable, repeat, pilot_tag):
    global _table_cache, _base_cache
    start=time.monotonic()
    path=hd.WORK/f'{pilot_tag}{mode}_{variable}_k{k}_r{repeat}_split{split}.json'
    signature=dict(version=VERSION,mode=mode,split=split,k=k,variable=variable,repeat=repeat,
                   recording_tag=pilot_tag or 'all',beta_grid=hs.GRID)
    if path.exists():
        saved=json.loads(path.read_text())
        # JSON turns tuple into list.
        if json.dumps(saved['signature'],sort_keys=True)!=json.dumps(signature,sort_keys=True):
            raise ValueError('Unit signature changed; archive before rerun.')
        return saved
    table_key=(id(S),mode,split,k)
    if _table_cache is None or _table_cache[0]!=table_key:
        _table_cache=(table_key,build_table(S,mode,split,k))
        _base_cache={}
    T=_table_cache[1]
    s,y=T['s'],T['y']
    var=T[variable].copy()
    if repeat>=0:
        rng=np.random.default_rng(seed('shuffle',f'{mode}|{variable}|split{split}|r{repeat}',k))
        var=gr.strat_shuffle(var,s,rng,np.argsort(s,kind='stable'))
        # Exact stratum-value histogram preservation is a gate.
        radix=100 if variable=='fine' else 20
        if not np.array_equal(np.bincount(s*radix+var,minlength=20*radix),
                              np.bincount(s*radix+T[variable],minlength=20*radix)):
            raise AssertionError('Within-state shuffle did not preserve label counts')
    radix=100 if variable=='fine' else 20
    contexts={'S':[s], 'H':chained(s,T['hist'],[20]*k),
              'V':chained(s,[var],[radix]),
              'VH':chained(s,[var]+T['hist'],[radix]+[20]*k)}
    losses,betas={},{}
    for name,chain in contexts.items():
        if name in ('S','H') and name in _base_cache:
            losses[name],betas[name]=_base_cache[name]
        else:
            losses[name],betas[name]=tune_and_score(y,chain,T['inner'],T['validation'],T['train'],T['test'])
            if name in ('S','H'):
                _base_cache[name]=(losses[name],betas[name])
    g={name:mask[T['test']] for name,mask in T['groups'].items()}
    result=dict(signature=signature,environment=hd.environment(),betas=betas,
                n_train=int(T['train'].sum()),n_inner=int(T['inner'].sum()),
                n_validation=int(T['validation'].sum()),n_test=int(T['test'].sum()),
                edges=T['edges'],accel_threshold=T['accel_threshold'],
                grouped=aggregate_units(T['uid'][T['test']],losses,g),
                elapsed_seconds=time.monotonic()-start,
                target_definition='(x[t]-x[t-2])/0.08' if mode=='backward2' else '(x[t+1]-x[t-1])/0.08',
                upstream_smoothing_limit='Backward window does not remove possible upstream use of future data.')
    hd.atomic_json(path,result)
    print(f'{path.stem}: {result["n_test"]} test rows, {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def summarize(units, pilot_tag):
    """Paired 200-resample vehicle bootstrap, with shared draws for failure gaps.

    Repeated outer splits contribute their test-step losses for the same vehicle;
    the vehicle, rather than an individual test row or split, is resampled.
    This is conditional on the fitted models (training is not bootstrapped).
    """
    rows=[]
    keys=sorted({(u['signature']['mode'],u['signature']['variable'],u['signature']['k']) for u in units})
    for mode,var,k in keys:
        subset=[u for u in units if (u['signature']['mode'],u['signature']['variable'],u['signature']['k'])==(mode,var,k)]
        repeats=sorted({u['signature']['repeat'] for u in subset if u['signature']['repeat']>=0})
        truth=[u for u in subset if u['signature']['repeat']==-1]
        ids=np.unique(np.concatenate([u['grouped']['all']['uid'] for u in truth]))
        groups=list(truth[0]['grouped'])
        def collect(items,group,model):
            sums=np.zeros(len(ids))
            counts=np.zeros(len(ids))
            for u in items:
                g=u['grouped'][group]
                ix=np.searchsorted(ids,g['uid'])
                sums[ix]+=g['sums'][model]
                counts[ix]+=g['n']
            return sums,counts
        matrices={}
        for group in groups:
            model_sums={name:collect(truth,group,name)[0] for name in ('S','H','V','VH')}
            count=collect(truth,group,'S')[1]
            shuffled=[]
            for repeat in repeats:
                these=[u for u in subset if u['signature']['repeat']==repeat]
                rr,n=collect(these,group,'VH'); vv,nv=collect(these,group,'V')
                if not np.array_equal(n,count) or not np.array_equal(nv,count):
                    raise AssertionError('Truth and shuffled test unit counts differ')
                shuffled.append(rr-vv) # lossVH-lossV = -residual gain
            # Six sufficient sums; the last is shuffled residual loss difference.
            mean_diff=np.mean(shuffled,axis=0) if shuffled else np.full(len(ids),np.nan)
            matrices[group]=np.column_stack([count,*[model_sums[n] for n in ('S','H','V','VH')],mean_diff])
        def derive(t):
            n,s,h,v,vh,sd=t
            if n<=0:
                return {m:np.nan for m in ('G_H','G_V','residual','Q','A_true','A_shuffled','A_net')}
            gh=(s-h)/n; gv=(s-v)/n; residual=(v-vh)/n
            q=gv/gh if gh!=0 else np.nan
            at=1-residual/gh if gh!=0 else np.nan
            ash=1+sd/n/gh if gh!=0 else np.nan
            return dict(G_H=gh,G_V=gv,residual=residual,Q=q,A_true=at,A_shuffled=ash,A_net=at-ash)
        point={group:derive(mat.sum(axis=0)) for group,mat in matrices.items()}
        bootstrap={group:{m:[] for m in point[group]} for group in groups}
        rng=np.random.default_rng(seed('paired_metrics',f'{mode}|{var}|all_groups',k))
        for _ in range(200):
            multiplicity=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
            for group,mat in matrices.items():
                metrics=derive(multiplicity@mat)
                for name,value in metrics.items():
                    bootstrap[group][name].append(value)
        for group in groups:
            row=dict(system='highd',mode=mode,variable=var,k=k,group=group,
                     **point[group],n_test=int(matrices[group][:,0].sum()),
                     n_vehicles=int((matrices[group][:,0]>0).sum()),shuffle_repeats=len(repeats),
                     bootstrap_repeats=200,units='bits/step',implementation=VERSION,
                     bootstrap_seed=seed('paired_metrics',f'{mode}|{var}|all_groups',k),
                     interval_scope='vehicle resampling conditional on fitted models; shared draws across groups',
                     environment=json.dumps(hd.environment()),
                     beta_grid='/'.join(map(str,hs.GRID)),
                     split='pilot: one 80/20 vehicle split, seed100' if pilot_tag else 'three 80/20 vehicle splits, seeds100/101/102',
                     velocity_formula='(x[t]-x[t-2])/0.08' if mode=='backward2' else '(x[t+1]-x[t-1])/0.08',
                     acceleration_formula='(v[t]-v[t-5])/0.2',sampling='frame%5==0; dt=0.2s',
                     history_span_seconds=.2*k+.08,
                     past_span_seconds=.2*k+(.08 if mode=='backward2' else .04),
                     future_span_seconds=0 if mode=='backward2' else .04,
                     shuffle_seed_rule='crc32(T09|highd|shuffle|mode|variable|split{split}|r{repeat}|{k})',
                     shuffle_scope='all eligible prediction rows, within current 20-bin state, train+test together',
                     upstream_smoothing_limit='Backward window does not remove possible upstream use of future data.',
                     residual_memory_verdict='not_evaluated_CMI_and_negative_calibration_pending')
            for name,vals in bootstrap[group].items():
                vals=np.asarray(vals)
                if np.isfinite(vals).all():
                    row[name+'_lo'],row[name+'_hi']=np.quantile(vals,[.025,.975])
                else:
                    row[name+'_lo']=row[name+'_hi']=np.nan
            gaps=np.asarray(bootstrap[group]['A_net'])-np.asarray(bootstrap['all']['A_net'])
            row['A_net_minus_all']=point[group]['A_net']-point['all']['A_net']
            if np.isfinite(gaps).all():
                row['A_net_minus_all_lo'],row['A_net_minus_all_hi']=np.quantile(gaps,[.025,.975])
            row['history_denominator_resolved']=bool(row['G_H_lo']>0) if np.isfinite(row['G_H_lo']) else False
            rows.append(row)
    suffix='_center' if {u['signature']['mode'] for u in units}=={'center'} else ''
    dest=P.CONTROLS_RESULTS/(f'absorption_highd_{pilot_tag.rstrip("_")}{suffix}.csv' if pilot_tag else f'absorption_highd{suffix}.csv')
    if not pilot_tag and dest.exists():
        previous=pd.read_csv(dest)
        new_keys={(r['mode'],r['variable'],r['k'],r['group']) for r in rows}
        previous=previous[[tuple(v) not in new_keys for v in previous[['mode','variable','k','group']].to_numpy()]]
        table=pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    else:
        table=pd.DataFrame(rows)
    tmp=dest.with_suffix('.csv.tmp')
    table.sort_values(['mode','variable','k','group']).to_csv(P.out(tmp),index=False)
    tmp.replace(dest)
    print(f'saved {dest.name}: {len(table)} rows; residual CMI/calibration not yet evaluated',flush=True)
    return table


def compare_windows():
    """Report paired descriptive differences; do not rewrite paper conclusions."""
    sources=[P.CONTROLS_RESULTS/'absorption_highd.csv',P.CONTROLS_RESULTS/'absorption_highd_center.csv']
    if not all(p.exists() for p in sources):
        raise RuntimeError('Both window analyses must finish first.')
    new,old=[pd.read_csv(p) for p in sources]
    new=new[new['mode']=='backward2']; old=old[old['mode']=='center']
    keys=['system','variable','k','group']
    compare=new.merge(old,on=keys,suffixes=('_backward2','_center'),validate='one_to_one')
    # Acceleration-defined groups change membership when the velocity window
    # changes. Only the all-steps and lane groups share identical frame sets.
    compare['same_target_frames']=compare['group'].isin(['all','lane_recent','lane_other'])
    paired=compare['same_target_frames']
    if not (compare.loc[paired,'n_test_backward2']==compare.loc[paired,'n_test_center']).all():
        raise AssertionError('Window comparison must use identical target frames.')
    metrics=['G_H','G_V','residual','Q','A_true']
    for metric in metrics:
        compare['delta_'+metric]=compare[metric+'_backward2']-compare[metric+'_center']
        compare.loc[~paired,'delta_'+metric]=np.nan
    compare['interpretation']='Same frames, different numerical velocity targets and train-only bin edges; descriptive sensitivity, not a same-target model comparison.'
    compare.loc[~paired,'interpretation']='Acceleration threshold groups have different membership across windows; no paired difference reported.'
    path=P.CONTROLS_RESULTS/'absorption_highd_window_comparison.csv'
    tmp=path.with_suffix('.csv.tmp'); compare.to_csv(P.out(tmp),index=False); tmp.replace(path)
    print(f'saved {path.name}: {len(compare)} rows, {int(paired.sum())} with paired frames',flush=True)


def assess_highd():
    """Attach only the completed primary-variable, longest-history diagnostic."""
    path=P.CONTROLS_RESULTS/'absorption_highd.csv'
    models=pd.read_csv(path)
    cmi=pd.read_csv(P.CONTROLS_RESULTS/'absorption_highd_residual.csv')
    negative=pd.read_csv(P.CONTROLS_RESULTS/'absorption_highd_negative.csv')
    if len(negative)!=90 or not (negative.groupby('split').rep.nunique()==30).all():
        raise RuntimeError('The prespecified 3 x 30 negative replicates are not complete')
    primary=models[(models.variable=='fine')&(models.k==3)&(models.group=='all')]
    if len(primary)!=1 or int(primary.iloc[0].shuffle_repeats)!=5:
        raise RuntimeError('The complete primary model comparison is missing')
    p=primary.iloc[0]
    diagnostic=cmi[cmi.group=='all'].sort_values('split')
    if len(diagnostic)!=3:
        raise RuntimeError('All three observed split diagnostics are required')
    calibrated=bool((diagnostic.calibration=='no_detected_inflation').all())
    detected=bool(calibrated and (diagnostic.p<.05).all() and p.residual_lo>.01)
    verdict=('conditional_sequence_dependence_detected_at_k3' if detected else
             ('needs_user_review_negative_calibration' if not calibrated else 'not_detected_under_current_method'))
    idx=primary.index[0]
    models.loc[idx,'residual_memory_verdict']=verdict
    models.loc[idx,'excess_CMI']=float(np.average(diagnostic.excess,weights=diagnostic.n_test))
    models.loc[idx,'p_max_over_three_splits']=diagnostic.p.max()
    models.loc[idx,'excess_CMI_interval_status']='vehicle-bootstrap interval not yet computed'
    models.loc[idx,'negative_calibration_scope']='all steps, primary fine variable, k=3 only'
    tmp=path.with_suffix('.csv.tmp'); models.to_csv(tmp,index=False); tmp.replace(path)
    groups=models[(models.variable=='fine')&(models.k==3)&(models.group!='all')]
    failures=[]
    for _,r in groups.iterrows():
        if r.A_net_minus_all<=-.2 and r.A_net_minus_all_hi<0:
            failures.append(r['group'])
    assessment=dict(system='highd',task_status='in_progress',
                    primary_absorption_lower_ci_at_least_half=bool(p.A_net_lo>=.5),
                    primary={k:float(p[k]) for k in ('G_H','G_V','residual','residual_lo','residual_hi','Q',
                                                    'A_true','A_shuffled','A_net','A_net_lo','A_net_hi')},
                    residual_verdict=verdict,diagnostics=diagnostic.fillna('original sparse sort').to_dict(orient='records'),
                    failure_groups_under_prespecified_rule=failures,
                    coverage='highD models complete; residual test only fine/k3/all is negatively calibrated; CMI intervals not yet computed',
                    interpretation='Dependence in these processed velocity states; upstream smoothing limitation remains. No claim about physical driver memory.',
                    environment=hd.environment())
    hd.atomic_json(P.CONTROLS_RESULTS/'absorption_highd_assessment.json',assessment)
    print(json.dumps(assessment,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system',choices=['highd','summarize-highd','compare-highd-windows','assess-highd'])
    parser.add_argument('--recording',default='all')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--mode',choices=['backward2','center'],default='backward2')
    parser.add_argument('--k',type=int,choices=[1,2,3],nargs='+',default=[1,2,3])
    parser.add_argument('--variable',choices=['fine','acc'],nargs='+',default=['fine','acc'])
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.system=='compare-highd-windows':
        compare_windows()
        return
    if args.system=='assess-highd':
        assess_highd()
        return
    if args.system=='summarize-highd':
        units=[json.loads(p.read_text()) for p in sorted(hd.WORK.glob('backward2_*_k*_r*_split*.json'))]
        if len(units)!=108:
            raise RuntimeError(f'Full main analysis incomplete: {len(units)}/108 units')
        summarize(units,'')
        centers=[json.loads(p.read_text()) for p in sorted(hd.WORK.glob('center_*_k*_r*_split*.json'))]
        if len(centers)!=18:
            raise RuntimeError(f'Center analysis incomplete: {len(centers)}/18 units')
        summarize(centers,'')
        compare_windows()
        if (P.CONTROLS_RESULTS/'absorption_highd_negative.csv').exists():
            assess_highd()
        return
    if not args.pilot and args.recording!='all':
        parser.error('Single-recording models must be explicitly marked --pilot')
    if not args.pilot:
        gate=hd.WORK/'legacy_check_all.json'
        if not gate.exists() or json.loads(gate.read_text())['verdict']!='match':
            raise RuntimeError('Full legacy reproduction gate must pass first')
    records=([p.name[:2] for p in sorted(P.HIGHD_TRACKS.glob('*_tracks.csv'))]
             if args.recording=='all' else [args.recording])
    S=hd.load_states(records)
    tag=f'pilot_{args.recording}_' if args.pilot else ''
    units=[]
    for k in args.k:
        for split in range(1 if args.pilot else 3):
            for variable in args.variable:
                for repeat in range(-1,args.repeats):
                    units.append(score_unit(S,args.mode,split,k,variable,repeat,tag))
    summarize(units,tag)
    if (not args.pilot and args.mode=='backward2' and args.repeats==5
            and set(args.k)=={1,2,3} and set(args.variable)=={'fine','acc'}
            and (P.CONTROLS_RESULTS/'absorption_highd_negative.csv').exists()):
        assess_highd()


if __name__=='__main__':
    main()
