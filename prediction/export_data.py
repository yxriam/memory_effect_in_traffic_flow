"""Export token sequences with fixed train/val/test splits for chapter 5 (prediction) and 6 (networks).
Each dataset -> npz with: tokens (int32, concatenated), offsets (len n_seq+1), unit (int32 per seq),
part (int8 per seq: 0 train, 1 val, 2 test), vocab_size, meta json.
GeoLife : global 0.001 deg (~100 m) cells, consecutive repeats collapsed; each trip split in time
          80/20 (cache split); the last 12.5 % of the training part is validation (70/10/20 overall).
highD   : signed longitudinal speed at 0.2 s in 20 equal-frequency bins (edges from training vehicles);
          vehicles split 70/10/20 at random (seed 0).
taxi    : per-trajectory 20x20 grid (own bounding box), run-length compressed, split in time 70/10/20."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repository root
from common import paths as P
_sys.path.insert(0, str(P.CONTROLS))  # geolife_robust.py
import sys, os, glob, json, numpy as np, pandas as pd
import geolife_robust as gr
OUT = str(P.PRED_DATA); os.makedirs(OUT, exist_ok=True)

def save(name, seqs, meta):
    keys = np.unique(np.concatenate([s for s, _, _ in seqs]))
    enc = {k: i for i, k in enumerate(keys.tolist())}
    toks = np.concatenate([np.searchsorted(keys, s) for s, _, _ in seqs]).astype(np.int32)
    off = np.r_[0, np.cumsum([len(s) for s, _, _ in seqs])].astype(np.int64)
    np.savez_compressed(f'{OUT}/{name}.npz', tokens=toks, offsets=off, unit=np.array([u for _, u, _ in seqs], np.int32),
                        part=np.array([p for _, _, p in seqs], np.int8), vocab_size=len(keys), keys=keys)
    meta.update(n_seq=len(seqs), n_tokens=int(len(toks)), vocab=int(len(keys)),
                tokens_by_part={p: int(sum(len(s) for s, _, q in seqs if q == i)) for i, p in enumerate(['train', 'val', 'test'])})
    json.dump(meta, open(f'{OUT}/{name}.json', 'w'), indent=1); print(name, meta)

def split3(seq):
    n = len(seq); a = int(n * 0.7); b = int(n * 0.8)
    return seq[:a], seq[a:b], seq[b:]

def geolife(g=0.001):
    seqs = []
    for f in sorted(glob.glob(os.path.join(gr.CACHE, '*.npz'))):
        u = int(os.path.basename(f)[:3]); d = np.load(f, allow_pickle=True)
        for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
            tr = gr.to_states(np.asarray(tla, float), np.asarray(tlo, float), g)
            te = gr.to_states(np.asarray(sla, float), np.asarray(slo, float), g)
            if len(tr) >= 3:
                c = int(round(len(tr) * 0.875)); a, v = tr[:c], tr[c:]
                if len(a) >= 2: seqs.append((a, u, 0))
                if len(v) >= 2: seqs.append((v, u, 1))
            if len(te) >= 2: seqs.append((te, u, 2))
    save('geolife_100m', seqs, dict(dataset='GeoLife', state='global 0.001 deg cell, repeats collapsed', split='per trip in time 70/10/20'))

def highd(nb=20):
    S = pd.read_pickle(P.CONTROLS_WORK / 'highd_states.pkl')  # written by controls/highd_augment.py
    uid = S.uid.to_numpy(); fr = S.frame.to_numpy(); v = S.v.to_numpy()
    brk = np.r_[True, (uid[1:] != uid[:-1]) | (np.diff(fr) != 5)]; seg = np.cumsum(brk) - 1
    vid = np.unique(uid); rng = np.random.default_rng(0); perm = rng.permutation(vid)
    part_of = {x: (0 if i < .7 * len(vid) else 1 if i < .8 * len(vid) else 2) for i, x in enumerate(perm)}
    trmask = np.array([part_of[x] == 0 for x in uid])
    e = np.quantile(v[trmask], np.linspace(0, 1, nb + 1)); e[0], e[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(e, v, side='right') - 1, 0, nb - 1)
    starts = np.where(brk)[0]; ends = np.r_[starts[1:], len(v)]
    seqs = [(b[s:t], int(uid[s] % 10**9), part_of[uid[s]]) for s, t in zip(starts, ends) if t - s >= 3]
    save(f'highd_v{nb}', seqs, dict(dataset='highD', state=f'signed speed, {nb} equal-frequency bins (train vehicles), dt=0.2 s', split='vehicles 70/10/20, seed 0'))

def taxi(files, g=20):
    seqs = []
    for k, f in enumerate(files):
        try: d = pd.read_csv(f, header=None, usecols=[2, 3])
        except Exception: continue
        lon, lat = d[2].to_numpy(float), d[3].to_numpy(float)
        if len(lon) < 10: continue
        def idx(x):
            lo, hi = x.min(), x.max(); return np.clip(((x - lo) / (hi - lo + 1e-12) * g).astype(int), 0, g - 1)
        c = idx(lon) * g + idx(lat); c = c[np.r_[True, c[1:] != c[:-1]]]
        if len(c) < 20: continue
        uid = int(os.path.basename(f).split('_')[1])
        for p, s in enumerate(split3(c)):
            if len(s) >= 2: seqs.append((s, uid, p))
    save(f'taxi_traj{g}', seqs, dict(dataset='Shanghai taxi', state=f'per-trajectory {g}x{g} grid, run-length compressed', split='per trajectory in time 70/10/20'))

def geolife_bj(g=0.005, box=(39.6, 40.3, 115.9, 116.9)):
    """Pooled benchmark version: Beijing box, ~500 m cells; fixes outside the box end the sequence piece."""
    seqs = []
    for f in sorted(glob.glob(os.path.join(gr.CACHE, '*.npz'))):
        u = int(os.path.basename(f)[:3]); d = np.load(f, allow_pickle=True)
        for tla, tlo, sla, slo in zip(d['train_trips_lat'], d['train_trips_lon'], d['test_trips_lat'], d['test_trips_lon']):
            pieces = []
            for la, lo in ((tla, tlo), (sla, slo)):
                la, lo = np.asarray(la, float), np.asarray(lo, float)
                m = (la > box[0]) & (la < box[1]) & (lo > box[2]) & (lo < box[3])
                pieces.append(gr.to_states(la[m], lo[m], g))
            tr, te = pieces
            if len(tr) >= 3:
                c = int(round(len(tr) * 0.875)); a, v = tr[:c], tr[c:]
                if len(a) >= 2: seqs.append((a, u, 0))
                if len(v) >= 2: seqs.append((v, u, 1))
            if len(te) >= 2: seqs.append((te, u, 2))
    save('geolife_bj500m', seqs, dict(dataset='GeoLife', state='Beijing box 39.6-40.3N 115.9-116.9E, 0.005 deg cell, repeats collapsed', split='per trip in time 70/10/20'))

if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'geolife': geolife()
    elif what == 'geolife_bj': geolife_bj()
    elif what == 'highd': highd(20); highd(100)
    elif what == 'taxi': taxi(sorted(glob.glob(os.path.join(sys.argv[2] if len(sys.argv) > 2 else str(P.TAXI_RAW), '*'))))
