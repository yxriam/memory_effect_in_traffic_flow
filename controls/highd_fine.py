import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
import numpy as np, pandas as pd, highd_augment as h
S = pd.read_pickle(P.CONTROLS_WORK / 'highd_states.pkl')
uid=S.uid.to_numpy(); fr=S.frame.to_numpy(); v=S.v.to_numpy(); a=S.a.to_numpy()
n=len(S); i=np.arange(2,n-1)
ok=(uid[i-2]==uid[i+1])&(fr[i+1]-fr[i-2]==15)&np.isfinite(a[i]); i=i[ok]
rows=[]
for sp in range(3):
    rng=np.random.default_rng(100+sp); u=rng.permutation(np.unique(uid)); tr=set(u[:int(.8*len(u))])
    istr=np.isin(uid[i],list(tr)); itr, iva = i[istr], i[~istr]
    vt=v[np.isin(uid,list(tr))]; at=a[np.isin(uid,list(tr))]
    E={k:h.qedges(vt,k) for k in (20,40,100,400)}; EA={k:h.qedges(at,k) for k in (7,20)}
    def st(ix):
        d=dict(y=h.qbins(v[ix+1],E[20]))
        for k in (20,40,100,400): d[f'v0_{k}']=h.qbins(v[ix],E[k])
        d['v1_20']=h.qbins(v[ix-1],E[20]); d['v1_100']=h.qbins(v[ix-1],E[100])
        for k in (7,20): d[f'a_{k}']=h.qbins(a[ix],EA[k])
        return d
    T,V=st(itr),st(iva)
    radix={'v0_20':20,'v0_40':40,'v0_100':100,'v0_400':400,'v1_20':20,'v1_100':100,'a_7':7,'a_20':20}
    M={'V1 (20)':['v0_20'],'V2 (20,20)':['v0_20','v1_20'],
       'V1 (40)':['v0_40'],'V1 (100)':['v0_100'],'V1 (400)':['v0_400'],
       'V2 (100,20)':['v0_100','v1_20'],'V2 (100,100)':['v0_100','v1_100'],
       'VA (100,a20)':['v0_100','a_20'],'VA (20,a20)':['v0_20','a_20'],'VA2 (100,a20,v1_20)':['v0_100','a_20','v1_20']}
    for name,cols in M.items():
        ltr,lva,nc=h.eval_model(T,V,cols,radix,20,alpha=0.5)
        rows.append(dict(split=sp,model=name,train=ltr,val=lva,nctx=nc))
    print(sp,flush=True)
R=pd.DataFrame(rows); R.to_csv(P.out(P.CONTROLS_RESULTS / 'fine_resolution.csv'),index=False)
print(R.groupby('model')[['train','val','nctx']].mean().sort_values('val').round(4))
