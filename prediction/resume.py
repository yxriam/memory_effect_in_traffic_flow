"""Epoch-level checkpoints for neural_hist.py and neural_ch5.py, so that an interrupted run continues where it stopped.

After every epoch the training state is written atomically (write to <file>.tmp, then rename) to <tag>.resume.pt:
model and optimizer state, epoch, best validation value, early-stopping counter and stop flag, history, elapsed time and
every random-number state the scripts use (torch CPU and CUDA, numpy's global state and the script's numpy Generator).
On start, a resume file written with the same arguments is loaded and training continues with the next epoch, drawing the
same random numbers as an uninterrupted run; results then agree up to the usual non-determinism of GPU kernels.
Exception: the cuDNN LSTM keeps the random state of its inter-layer dropout inside the process (seeded once on first
use), which cannot be saved, so a resumed LSTM run differs slightly (GPU test 2026-10-22: 0.002 bits/step after 4 short
epochs; the spread over seeds is about 0.01). The Transformer has no such hidden state.
A resume file written with different arguments is ignored (and overwritten). The scripts delete it after writing their
result JSON. Arguments listed in IGNORE (testing switches) are not compared.
"""
import os
import numpy as np
import torch

IGNORE = ('stop_after',)


def _same_args(a, b):
    strip = lambda d: {k: v for k, v in d.items() if k not in IGNORE}
    return strip(a) == strip(b)


def _rng_state(gen):
    s = dict(torch=torch.get_rng_state(), np_global=np.random.get_state())
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    if gen is not None:
        s['np_gen'] = gen.bit_generator.state
    return s


def _set_rng_state(s, gen):
    torch.set_rng_state(s['torch'])
    np.random.set_state(s['np_global'])
    if 'cuda' in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])
    if gen is not None and 'np_gen' in s:
        gen.bit_generator.state = s['np_gen']


def save(path, model, opt, state, gen=None):
    """state: dict with args, epoch, best, bad, stop, hist, elapsed."""
    tmp = path + '.tmp'
    try:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=_rng_state(gen), **state), tmp)
        os.replace(tmp, path)
    except OSError as e:                    # e.g. the file is briefly locked by a virus scanner: keep training
        print(f'could not write resume file ({e}); training continues', flush=True)


def load(path, model, opt, args, gen=None):
    """Load a matching resume file into model, optimizer and random states; return its state dict, or None."""
    if not os.path.exists(path):
        return None
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                       # torch < 1.13 has no weights_only
        c = torch.load(path, map_location='cpu')
    except Exception as e:                  # unreadable (e.g. power cut during an old-style write): start again
        print(f'resume file {os.path.basename(path)} unreadable ({e}); starting from epoch 0', flush=True)
        return None
    if not _same_args(c.get('args', {}), args):
        print(f'resume file {os.path.basename(path)} has different arguments; starting from epoch 0', flush=True)
        return None
    model.load_state_dict(c['model'])
    opt.load_state_dict(c['opt'])
    _set_rng_state(c['rng'], gen)
    print(f'resumed after epoch {c["epoch"]} (best val {c["best"]:.4f} bits, {c["elapsed"]:.0f} s so far)', flush=True)
    return c


def remove(path):
    """Delete this run's own resume file after its result has been written."""
    for p in (path, path + '.tmp'):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def environment():
    """Software and hardware of the run, stored in the result JSON."""
    import platform
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
               device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or 'cpu',
               platform=platform.platform())
    return env
