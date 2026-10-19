"""Memory-efficient replacement for the GLOBAL row and summary of 02_memory_flow_metrics.py
(the dict-of-Counter version needs >7 GB for 16.5 M pooled transitions). Same definitions, numpy encoding."""
import importlib.util, glob, numpy as np, pandas as pd
from pathlib import Path
spec = importlib.util.spec_from_file_location('m2', str(Path(__file__).with_name('02_memory_flow_metrics.py'))); m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
base = Path(__file__).resolve().parent; out = base / 'outputs_clean' / '02_memory_flow'
df = pd.read_csv(out / 'memory_flow_metrics.csv'); keep = set(df.user.astype(str).str.zfill(3))
A, B, C, n_states, n_trips, allst = [], [], [], 0, 0, []
for f in sorted((base / 'cache_clean').glob('*.npz')):
    if f.stem not in keep: continue
    for s in m2.load_full_trip_sequences(f, 0.0001, True):
        k = np.array([a * 10_000_000 + b for a, b in s], dtype=np.int64)
        n_trips += 1; n_states += len(k); allst.append(np.unique(k))
        A.append(k[:-2]); B.append(k[1:-1]); C.append(k[2:])
a, b, c = [np.concatenate(x) for x in (A, B, C)]
_, inv = np.unique(np.concatenate([a, b, c]), return_inverse=True); n = len(a)
a, b, c = inv[:n].astype(np.int64), inv[n:2*n].astype(np.int64), inv[2*n:].astype(np.int64); K = int(inv.max()) + 1
def H(ctx, y):
    _, cj = np.unique(ctx * K + y, return_counts=True); _, cc = np.unique(ctx, return_counts=True)
    ent = lambda x: -(x / x.sum() * np.log2(x / x.sum())).sum()
    return ent(cj) - ent(cc)
hist = np.unique(a * K + b, return_inverse=True)[1].astype(np.int64)
h1, h2 = H(b, c), H(hist, c)
# expected two-step return: for each trigram, P1(next=prev | cur) and P2(next=prev | prev,cur)
def cond_prob(ctx, y, q_ctx, q_y):
    kj, nj = np.unique(ctx * K + y, return_counts=True); kc, nc = np.unique(ctx, return_counts=True)
    def look(keys, cnt, q):
        i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1); return np.where(keys[i] == q, cnt[i], 0)
    return look(kj, nj, q_ctx * K + q_y) / look(kc, nc, q_ctx)
p1 = cond_prob(b, c, b, a).mean(); p2 = cond_prob(hist, c, hist, a).mean()
row = dict(user='GLOBAL', grid_size_deg=0.0001, collapse_repeats=1, n_trips=n_trips, n_states=n_states,
           n_unique_states=int(len(np.unique(np.concatenate(allst)))), n_bigrams=n_states - n_trips, n_trigrams=n,
           n_current_states=int(len(np.unique(b))), n_second_order_histories=int(hist.max() + 1),
           first_order_entropy_bits=h1, second_order_entropy_bits=h2, entropy_reduction_bits=h1 - h2,
           entropy_reduction_pct=(h1 - h2) / h1 * 100, empirical_two_step_return=float((c == a).mean()),
           first_order_expected_two_step_return=p1, second_order_expected_two_step_return=p2,
           second_minus_first_return=p2 - p1, second_over_first_return=p2 / p1)
pd.DataFrame([row]).to_csv(out / 'memory_flow_global.csv', index=False); print(pd.Series(row))
summary = dict(users=len(df), grid_size_deg=0.0001, collapse_repeats=1,
    first_order_entropy_median=df.first_order_entropy_bits.median(), second_order_entropy_median=df.second_order_entropy_bits.median(),
    entropy_reduction_bits_median=df.entropy_reduction_bits.median(), entropy_reduction_pct_median=df.entropy_reduction_pct.median(),
    empirical_two_step_return_median=df.empirical_two_step_return.median(),
    first_order_expected_return_median=df.first_order_expected_two_step_return.median(),
    second_order_expected_return_median=df.second_order_expected_two_step_return.median(),
    second_minus_first_return_median=df.second_minus_first_return.median(),
    positive_entropy_reduction_users=int((df.entropy_reduction_bits > 0).sum()), positive_return_lift_users=int((df.second_minus_first_return > 0).sum()))
pd.DataFrame([summary]).to_csv(out / 'memory_flow_summary.csv', index=False); print(pd.Series(summary))
