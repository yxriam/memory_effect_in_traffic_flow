"""History-length ablation for the chapter 5 LSTM.

Every target s[t] (t >= 1) of a piece is predicted from exactly the last k symbols of the same piece
(fewer at the start of a piece; left-padded), so the model can use at most k steps of memory.
k = 1 is a neural first-order Markov chain. Same splits, same scored targets and metrics as the
n-gram models and neural_ch5.py. Each training epoch uses a random subset of training targets.
--model transformer: the same ablation with the chapter 5 Transformer (2 layers, 4 heads, width 256) reading the
k-symbol window (padding masked); its output at the most recent position predicts the target.

  python neural_hist.py --data data/taxi_traj20.npz --hist 2
  python neural_hist.py --data data/taxi_traj20.npz --hist 2 --model transformer
Writes results/neural/hist/hist_<dataset>_k<k>_s<seed>.json (LSTM) or hist_transformer_<dataset>_k<k>_s<seed>.json
and a log line per epoch.
"""
import argparse, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--hist', type=int, required=True)
p.add_argument('--model', choices=['lstm', 'transformer'], default='lstm'); p.add_argument('--heads', type=int, default=4)
p.add_argument('--emb', type=int, default=128); p.add_argument('--hid', type=int, default=256)
p.add_argument('--layers', type=int, default=2); p.add_argument('--dropout', type=float, default=0.2)
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--batch', type=int, default=512)
p.add_argument('--epochs', type=int, default=25); p.add_argument('--patience', type=int, default=2)
p.add_argument('--per_epoch', type=int, default=300000, help='training targets drawn per epoch')
p.add_argument('--max_val', type=int, default=150000, help='validation targets (fixed random subset)')
p.add_argument('--seed', type=int, default=0); p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'neural', 'hist'))
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(a.data))[0]
d = np.load(a.data); tok, off, part = d['tokens'].astype(np.int64), d['offsets'], d['part']; V = int(d['vocab_size'])
PAD, K = V, a.hist
tag = f'hist_{name}_k{K}_s{a.seed}' if a.model == 'lstm' else f'hist_transformer_{name}_k{K}_s{a.seed}'
skip_key = f'{name} {K} {a.seed}' if a.model == 'lstm' else f'transformer {name} {K} {a.seed}'
skip = os.path.join(a.out, 'SKIP.txt')   # runs listed here ('<dataset> <k> <seed>', 'transformer <dataset> <k> <seed>') exit at once
if os.path.exists(skip) and skip_key in [l.strip() for l in open(skip)]:
    print(f'{tag}: skipped (listed in SKIP.txt)', flush=True); raise SystemExit(0)

def targets(which):
    """Global positions of all targets (t >= 1 inside each piece) and the start of their piece."""
    pos, st = [], []
    for i in np.where(part == which)[0]:
        n = off[i + 1] - off[i]
        if n < 2: continue
        pos.append(np.arange(off[i] + 1, off[i + 1])); st.append(np.full(n - 1, off[i]))
    return np.concatenate(pos), np.concatenate(st)

def batch_xy(pos, st):
    idx = pos[:, None] - np.arange(K, 0, -1)[None, :]          # the K positions before each target
    x = np.where(idx >= st[:, None], tok[np.maximum(idx, 0)], PAD)
    return torch.from_numpy(x), torch.from_numpy(tok[pos])

class LSTMLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V + 1, a.emb, padding_idx=PAD)
        s.rnn = nn.LSTM(a.emb, a.hid, a.layers, batch_first=True, dropout=a.dropout if a.layers > 1 else 0)
        s.drop = nn.Dropout(a.dropout); s.out = nn.Linear(a.hid, V)
    def forward(s, x):
        h, _ = s.rnn(s.drop(s.emb(x))); return s.out(s.drop(h[:, -1]))

class TransHist(nn.Module):
    """Transformer over the k-symbol window (as TransLM in neural_ch5.py); the last position is the most recent symbol."""
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V + 1, a.hid, padding_idx=PAD); s.pos = nn.Embedding(K, a.hid)
        layer = nn.TransformerEncoderLayer(a.hid, a.heads, 4 * a.hid, a.dropout, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, a.layers); s.norm = nn.LayerNorm(a.hid); s.out = nn.Linear(a.hid, V)
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(K, device=x.device))[None]
        h = s.enc(h, src_key_padding_mask=(x == PAD))          # the most recent position is never padding
        return s.out(s.norm(h[:, -1]))

def run_eval(model, pos, st, topk=True):
    model.eval(); nll, c1, c5 = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(pos), 2048):
            x, y = batch_xy(pos[i:i + 2048], st[i:i + 2048]); x, y = x.to(dev), y.to(dev)
            lg = model(x).float(); nll += F.cross_entropy(lg, y, reduction='sum').item()
            if topk:
                t5 = lg.topk(5, -1).indices; c1 += int((t5[:, 0] == y).sum()); c5 += int((t5 == y[:, None]).any(-1).sum())
    n = len(pos); return dict(bits=nll / n / math.log(2), n=n, acc1=c1 / n, acc5=c5 / n)

ptr, str_ = targets(0); pva, sva = targets(1); pte, ste = targets(2)
if len(pva) > a.max_val:
    j = np.sort(rng.choice(len(pva), a.max_val, replace=False)); pva, sva = pva[j], sva[j]
print(f'{tag}: V={V} targets train/val/test = {len(ptr)}/{len(pva)}/{len(pte)} device={dev}', flush=True)
model = (LSTMLM() if a.model == 'lstm' else TransHist()).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
best, bad, hist, t0 = float('inf'), 0, [], time.time(); ckpt = os.path.join(a.out, tag + '.pt')
for ep in range(a.epochs):
    model.train(); j = rng.choice(len(ptr), min(a.per_epoch, len(ptr)), replace=False); tot, nb = 0.0, 0
    for i in range(0, len(j), a.batch):
        x, y = batch_xy(ptr[j[i:i + a.batch]], str_[j[i:i + a.batch]]); x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x), y); opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item(); nb += 1
    v = run_eval(model, pva, sva, topk=False)['bits']
    hist.append(dict(epoch=ep, train_loss=tot / nb, val_bits=v, sec=time.time() - t0)); print(hist[-1], flush=True)
    if v < best - 1e-4: best, bad = v, 0; torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
        if bad >= a.patience: break
model.load_state_dict(torch.load(ckpt, map_location=dev))
res = dict(dataset=name, model=a.model, hist=K, seed=a.seed, val_bits=best, test=run_eval(model, pte, ste), args=vars(a), history=hist)
json.dump(res, open(os.path.join(a.out, tag + '.json'), 'w'), indent=1)
print('TEST', res['test'], flush=True)
