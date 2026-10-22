"""Chapter 5 neural baselines (LSTM / Transformer) on exactly the same splits and metrics as the n-gram models.

Data: *.npz from export_data.py (tokens, offsets, part: 0 train / 1 val / 2 test / 3 unused, vocab_size).
Every sequence piece is predicted from its own start: the target at position t >= 1 sees only s[0..t-1]
of the same piece (as for the n-gram models). Metrics on the test pieces: bits per step (cross-entropy / ln 2),
top-1 and top-5 accuracy. Early stopping on validation bits.

Usage (GPU recommended):
  python neural_ch5.py --data data/taxi_traj20.npz --model lstm
  python neural_ch5.py --data data/taxi_traj20.npz --model transformer
Writes results/neural/neural_<dataset>_<model>.json and a training log.
"""
import argparse, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from keep_awake import keep_awake; keep_awake()  # Windows: no idle sleep while training (see keep_awake.py)
import resume                                     # epoch-level checkpoints: an interrupted run continues (resume.py)

p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--model', choices=['lstm', 'transformer'], required=True)
p.add_argument('--ctx', type=int, default=128, help='window length (tokens) used for training and evaluation')
p.add_argument('--emb', type=int, default=128); p.add_argument('--hid', type=int, default=256)
p.add_argument('--layers', type=int, default=2); p.add_argument('--heads', type=int, default=4)
p.add_argument('--dropout', type=float, default=0.2); p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--batch', type=int, default=128); p.add_argument('--epochs', type=int, default=30)
p.add_argument('--patience', type=int, default=3); p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'neural')); p.add_argument('--tag', default='', help='suffix for output files, e.g. _s1')
p.add_argument('--stop_after', type=int, default=0, help='testing: exit after this many epochs, as if interrupted')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(a.data))[0]
d = np.load(a.data); tok, off, part = d['tokens'].astype(np.int64), d['offsets'], d['part']; V = int(d['vocab_size'])
PAD = V  # padding index (never a target)
if V > 2000 and a.batch > 32: a.batch = 32   # large vocabularies: keep the output layer within a small GPU's memory

def windows(which):
    """Cut every piece of the given part into windows of length ctx+1 with stride ctx/2.
    Returns inputs X, targets Y (-100 = ignored). Each target is scored exactly once, with the longest
    available history inside the piece (up to ctx-1 tokens for targets after the first window)."""
    W = a.ctx; S = W // 2; X, Y = [], []
    for i in np.where(part == which)[0]:
        s = tok[off[i]:off[i + 1]]
        if len(s) < 2: continue
        start = 0
        while True:
            seg = s[start:start + W + 1]
            x = np.full(W, PAD, dtype=np.int64); y = np.full(W, -100, dtype=np.int64)
            x[:len(seg) - 1] = seg[:-1]; y[:len(seg) - 1] = seg[1:]
            if start > 0: y[:W - S] = -100          # these targets were scored by the previous window
            X.append(x); Y.append(y)
            if start + W + 1 >= len(s): break
            start += S
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

class LSTMLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V + 1, a.emb, padding_idx=PAD)
        s.rnn = nn.LSTM(a.emb, a.hid, a.layers, batch_first=True, dropout=a.dropout if a.layers > 1 else 0)
        s.drop = nn.Dropout(a.dropout); s.out = nn.Linear(a.hid, V)
    def forward(s, x):
        h, _ = s.rnn(s.drop(s.emb(x))); return s.out(s.drop(h))

class TransLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V + 1, a.hid, padding_idx=PAD); s.pos = nn.Embedding(a.ctx, a.hid)
        layer = nn.TransformerEncoderLayer(a.hid, a.heads, 4 * a.hid, a.dropout, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, a.layers); s.norm = nn.LayerNorm(a.hid); s.out = nn.Linear(a.hid, V)
        s.register_buffer('mask', torch.triu(torch.full((a.ctx, a.ctx), float('-inf')), 1))
    def forward(s, x):
        L = x.shape[1]; h = s.emb(x) + s.pos(torch.arange(L, device=x.device))[None]
        return s.out(s.norm(s.enc(h, mask=s.mask[:L, :L])))

def run_eval(model, X, Y, topk=True):
    model.eval(); nll, n, c1, c5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(X), a.batch * 2):
            x, y = X[i:i + a.batch * 2].to(dev), Y[i:i + a.batch * 2].to(dev)
            lg = model(x).float(); m = y != -100
            nll += F.cross_entropy(lg[m], y[m], reduction='sum').item(); n += int(m.sum())
            if topk:
                t5 = lg[m].topk(5, -1).indices; yy = y[m][:, None]
                c1 += int((t5[:, :1] == yy).sum()); c5 += int((t5 == yy).any(-1).sum())
    return dict(bits=nll / n / math.log(2), n=n, acc1=c1 / n, acc5=c5 / n)

Xtr, Ytr = windows(0); Xva, Yva = windows(1); Xte, Yte = windows(2)
print(f'{name}: V={V} windows train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)} device={dev}', flush=True)
model = (LSTMLM() if a.model == 'lstm' else TransLM()).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
best, bad, hist, t0 = float('inf'), 0, [], time.time()
ckpt = os.path.join(a.out, f'neural_{name}_{a.model}{a.tag}.pt')
rfile = os.path.join(a.out, f'neural_{name}_{a.model}{a.tag}.resume.pt'); start, stop = 0, False
c = resume.load(rfile, model, opt, vars(a))
if c is not None: start, best, bad, stop, hist, t0 = c['epoch'] + 1, c['best'], c['bad'], c['stop'], c['hist'], time.time() - c['elapsed']
for ep in range(start, a.epochs):
    if stop: break
    model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
    for i in range(0, len(Xtr), a.batch):
        idx = perm[i:i + a.batch]; x, y = Xtr[idx].to(dev), Ytr[idx].to(dev)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
    v = run_eval(model, Xva, Yva, topk=False)['bits']
    hist.append(dict(epoch=ep, train_loss=tot / max(1, len(Xtr) // a.batch), val_bits=v, sec=time.time() - t0)); print(hist[-1], flush=True)
    if v < best - 1e-4: best, bad = v, 0; torch.save(model.state_dict(), ckpt)
    else: bad += 1
    stop = bad >= a.patience
    resume.save(rfile, model, opt, dict(args=vars(a), epoch=ep, best=best, bad=bad, stop=stop, hist=hist, elapsed=time.time() - t0))
    if a.stop_after and ep + 1 - start >= a.stop_after: print(f'stopped after epoch {ep} (--stop_after)', flush=True); raise SystemExit(3)
model.load_state_dict(torch.load(ckpt, map_location=dev))
res = dict(dataset=name, model=a.model, val_bits=best, test=run_eval(model, Xte, Yte), args=vars(a), history=hist,
           n_params=sum(p.numel() for p in model.parameters()), env=resume.environment())
json.dump(res, open(os.path.join(a.out, f'neural_{name}_{a.model}{a.tag}.json'), 'w'), indent=1); resume.remove(rfile)
print('TEST', res['test'])
