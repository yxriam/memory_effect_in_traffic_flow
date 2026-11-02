"""T09 residual CMI, primary fine-speed variable, k=3, and matched null checks.

python controls/absorption_highd_cmi.py --recording 01 --pilot
python controls/absorption_highd_cmi.py
CPU; Ctrl+C then rerun resumes per observed/negative replicate. No positive control.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import paths as P
import absorption_highd as hd
import absorption_ledger as ledger
import geolife_robust as gr
import highd_augment as ha
from null_calibration import Chain

VERSION='t09-highd-residual-fine-k3-v1'


def cmi(hist,cond,y,rng):
    # Compact contingency counts implement exactly gr.cond_H, but avoid
    # sorting the same large test table twice for each permutation.
    # gr's pair_id otherwise also needs K >= max(multi-step history)+1.
    _,c=np.unique(cond,return_inverse=True)
    _,h=np.unique(hist,return_inverse=True)
    nc,nh=int(c.max())+1,int(h.max())+1
    # Target labels may be compacted separately within each condition. This
    # preserves the order of nonzero contingency counts and all entropies.
    pair_keys,pair_inv=np.unique(c*20+y,return_inverse=True)
    per_condition=np.bincount(pair_keys//20,minlength=nc)
    first_pair=np.r_[0,np.cumsum(per_condition)[:-1]]
    local_y=pair_inv-first_pair[c]
    ny=int(per_condition.max())
    if nc*nh*ny>64_000_000:
        radix=max(20,int(hist.max(initial=0))+1)
        real,null,p,_,_=gr.cmi_with_null(hist,cond,y,radix,hs_permutations(),rng)
        return dict(plugin=real,null_mean=null,excess=real-null,p=p,n_test=len(y),entropy_engine='existing sparse sort')
    hcond=gr.cond_H(c,y,20)
    def full_entropy(hh):
        counts=np.bincount((c*nh+hh)*ny+local_y,minlength=nc*nh*ny)
        margins=counts.reshape(nc*nh,ny).sum(axis=1)
        return gr._H(counts[counts>0])-gr._H(margins[margins>0])
    real=hcond-full_entropy(h)
    order=np.argsort(c,kind='stable')
    boundaries=np.r_[0,np.flatnonzero(np.diff(c[order]))+1,len(c)]
    sorted_h=h[order]
    null=[]
    for _ in range(hs_permutations()):
        # Exactly the same random numbers and stable within-stratum order as
        # strat_shuffle's two-key lexsort, without globally sorting both keys.
        random_order=rng.random(len(h))[order]
        sorted_shuffle=np.empty_like(h)
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            sorted_shuffle[lo:hi]=sorted_h[lo:hi][np.argsort(random_order[lo:hi],kind='stable')]
        shuffled=np.empty_like(h); shuffled[order]=sorted_shuffle
        null.append(hcond-full_entropy(shuffled))
    null=np.asarray(null)
    p=float((1+(null>=real).sum())/(len(null)+1))
    return dict(plugin=real,null_mean=float(null.mean()),excess=float(real-null.mean()),p=p,n_test=len(y),
                entropy_engine='compact contingency counts + existing entropy/strat_shuffle')


def hs_permutations():
    from heading_state import NPERM
    return NPERM


def histories(cols):
    result=np.zeros(len(cols[0]),np.int64)
    for col in cols:
        result=result*20+col
    return result


def sequences(codes,S,mask):
    """Split at missing states, cars, or sampled-frame gaps; keep original indices."""
    uid=S.uid.to_numpy(); fr=S.frame.to_numpy()
    ix=np.flatnonzero(mask & (codes>=0))
    split=np.flatnonzero((uid[ix[1:]]!=uid[ix[:-1]]) | (fr[ix[1:]]-fr[ix[:-1]]!=5))+1
    indices=np.split(ix,split)
    return [codes[i] for i in indices if len(i)>=2], [i for i in indices if len(i)>=2]


def simulate_aligned(chain,templates,indices,n,rng):
    """Use the existing first-order generator; vectorize storage across cars.

    Trip lengths/initial states match the observed templates. A chain reaching
    a state without a fitted successor terminates as in null_calibration.py.
    """
    out=np.full(n,-1,np.int64)
    lengths=np.array([len(t) for t in templates])
    alive=np.ones(len(templates),bool)
    current=np.array([t[0] for t in templates])
    for j,ix in enumerate(indices):
        out[ix[0]]=current[j]
    for step in range(1,int(lengths.max())):
        use=np.flatnonzero(alive & (lengths>step))
        if not len(use):
            break
        nxt,can=chain.step1(current[use],rng)
        alive[use[~can]]=False
        use=use[can]; nxt=nxt[can]
        current[use]=nxt
        positions=np.fromiter((indices[j][step] for j in use),dtype=np.int64,count=len(use))
        out[positions]=nxt
    return out


def run(S,split,pilot,negative_reps):
    start=time.monotonic()
    T=ledger.build_table(S,'backward2',split,3)
    prefix=f'cmi_{pilot}' if pilot else 'cmi_all'
    key=f'{prefix}_fine_k3_split{split}'
    observed_path=hd.WORK/f'{key}_observed.json'
    signature=dict(version=VERSION,mode='backward2',variable='fine',k=3,split=split,
                   permutations=hs_permutations(),pilot=pilot)
    test=T['test']; cond=T['s'][test]*100+T['fine'][test]
    hist=histories([col[test] for col in T['hist']]); y=T['y'][test]
    if observed_path.exists():
        observed=json.loads(observed_path.read_text())
        if observed['signature']!=signature:
            raise ValueError('Observed CMI cache signature differs')
    else:
        observed=dict(signature=signature,environment=hd.environment(),groups={})
        for group,mask in T['groups'].items():
            m=mask[test]
            rng=np.random.default_rng(ledger.seed('residual_CMI',f'fine|split{split}|{group}',3))
            observed['groups'][group]=cmi(hist[m],cond[m],y[m],rng)
        hd.atomic_json(observed_path,observed)
        print(f'{key} observed: {observed["groups"]["all"]} [{time.monotonic()-start:.1f}s]',flush=True)
    # A first-order Markov chain on (S,V) has true residual CMI zero. The
    # simulated target is the next state's S component; histories are past S.
    values=S.backward2.to_numpy(); uid=S.uid.to_numpy()
    uids=S.uid.unique(); perm=np.random.default_rng(100+split).permutation(uids)
    outer=np.isin(uid,perm[:int(.8*len(perm))])
    e20=np.asarray(T['edges']['v20']); e100=np.asarray(T['edges']['v100'])
    valid=np.isfinite(values)
    codes=ha.qbins(values,e20)*100+ha.qbins(values,e100)
    codes[~valid]=-1
    training,_=sequences(codes,S,outer)
    chain=Chain(training,int(codes.max())+1)
    templates,template_ix=sequences(codes,S,~outer)
    ix=T['indices'][test]
    negatives=[]
    for rep in range(negative_reps):
        path=hd.WORK/f'{key}_negative{rep}.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['signature']!=signature or result['rep']!=rep:
                raise ValueError('Negative CMI cache signature differs')
            negatives.append(result)
            continue
        repstart=time.monotonic()
        rng=np.random.default_rng(ledger.seed('negative_Markov',f'fine|split{split}|rep{rep}',3))
        generated=simulate_aligned(chain,templates,template_ix,len(S),rng)
        take=(generated[ix+1]>=0)
        for lag in range(4):
            take &= generated[ix-lag]>=0
        j=ix[take]
        gy=generated[j+1]//100; gc=generated[j]
        gh=histories([generated[j-lag]//100 for lag in (1,2,3)])
        result=dict(signature=signature,environment=hd.environment(),rep=rep,
                    seed=ledger.seed('negative_Markov',f'fine|split{split}|rep{rep}',3),
                    **cmi(gh,gc,gy,rng),true_residual_cmi=0.,
                    generator='first-order chain on (20-bin current velocity,100-bin fine velocity), fitted to training vehicles',
                    elapsed_seconds=time.monotonic()-repstart)
        hd.atomic_json(path,result); negatives.append(result)
        print(f'{key} negative {rep}: p={result["p"]:.4g}, n={len(j)} [{time.monotonic()-repstart:.1f}s]',flush=True)
    rejected=sum(n['p']<.05 for n in negatives)
    n=len(negatives)
    tail=sum(math.comb(n,j)*.05**j*.95**(n-j) for j in range(rejected,n+1)) if n else None
    z=1.959963984540054
    if n:
        rate=rejected/n; den=1+z*z/n
        center=(rate+z*z/(2*n))/den
        radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        interval=[center-radius,center+radius]
    else:
        interval=[None,None]
    summary=dict(signature=signature,environment=hd.environment(),observed=observed['groups'],
                 negative_reps=n,negative_rejections=rejected,false_positive_rate=rejected/n if n else None,
                 false_positive_wilson95=interval,inflation_binomial_tail_p=tail,
                 calibration='pilot_only' if pilot else ('needs_user_review' if tail is not None and tail<.05 else 'no_detected_inflation'),
                 limitation='30 negative replicates provide a broad FPR interval; no claim that nominal 5% is proven.',
                 elapsed_seconds=time.monotonic()-start)
    hd.atomic_json(hd.WORK/f'{key}_summary.json',summary)
    return summary,negatives


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',default='all')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--negative-reps',type=int,default=30)
    args=parser.parse_args()
    if args.recording!='all' and not args.pilot:
        parser.error('A single recording requires --pilot')
    if not args.pilot and args.negative_reps!=30:
        parser.error('The prespecified full calibration uses exactly 30 replicates per split')
    records=([p.name[:2] for p in sorted(P.HIGHD_TRACKS.glob('*_tracks.csv'))]
             if args.recording=='all' else [args.recording])
    S=hd.load_states(records)
    rows=[]; negative_rows=[]
    for split in range(1 if args.pilot else 3):
        r,negative=run(S,split,args.recording if args.pilot else '',args.negative_reps)
        for group,metrics in r['observed'].items():
            rows.append(dict(split=split,group=group,**metrics,negative_reps=r['negative_reps'],
                             false_positive_rate=r['false_positive_rate'],false_positive_lo=r['false_positive_wilson95'][0],
                             false_positive_hi=r['false_positive_wilson95'][1],calibration=r['calibration'],
                             inflation_binomial_tail_p=r['inflation_binomial_tail_p'],
                             calibration_scope='all steps, primary fine variable, k=3 only; not group-specific',
                             implementation=VERSION,environment=json.dumps(hd.environment()),units='bits',
                             seed_rule='crc32(T09|highd|quantity|group|k)',permutations=hs_permutations()))
        for item in negative:
            negative_rows.append(dict(split=split,**{k:v for k,v in item.items() if k not in ('signature','environment')},
                                      implementation=VERSION,mode='backward2',variable='fine',k=3,
                                      permutations=hs_permutations(),
                                      environment=json.dumps(item['environment'])))
    suffix=f'_pilot_{args.recording}' if args.pilot else ''
    for label,table in [('residual',rows),('negative',negative_rows)]:
        path=P.CONTROLS_RESULTS/f'absorption_highd_{label}{suffix}.csv'
        tmp=path.with_suffix('.csv.tmp'); pd.DataFrame(table).to_csv(P.out(tmp),index=False); tmp.replace(path)


if __name__=='__main__':
    main()
