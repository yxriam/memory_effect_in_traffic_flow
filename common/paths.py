"""One place for every file location used by controls/ and prediction/.

Raw data defaults to the git-ignored data folders inside this repository
(highD/data/, taxi/data/Taxi_070220/, geolife/data_raw/, geolife/cache_clean/).
If your data lives somewhere else, copy local_paths.example.json to local_paths.json
(git-ignored) in the repository root and edit it, or set an environment variable
MEMFX_<KEY>, e.g. MEMFX_HIGHD_TRACKS.

    python -m common.paths      # prints every location and whether it exists
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS = {
    'highd_tracks': ROOT / 'highD' / 'data',                                   # 01_tracks.csv ... 60_tracks.csv
    'highd_full': ROOT / 'highD' / 'data_full',                               # original highD release: *_tracks.csv with velocities, accelerations, preceding vehicle; *_tracksMeta.csv
    'taxi_raw': ROOT / 'taxi' / 'data' / 'Taxi_070220',                        # 4 316 per-taxi files
    'geolife_raw': ROOT / 'geolife' / 'data_raw' / 'Data',                     # GeoLife 1.3 user folders 000 ... 181
    'geolife_trips_by_mode': ROOT / 'geolife' / 'data_raw' / 'trips_by_mode_data',  # <mode>/plt/*.plt
    'geolife_cache': ROOT / 'geolife' / 'cache_clean',                         # 000.npz ... from geolife/01_preprocess_no_leakage.py
    'airports_seq': ROOT / 'airports' / 'data' / 'sequences_split',            # <YYYYMM>/<YYYYMM>_<n>.txt from airports/airports_sequences_split.py
}


def _load():
    cfg = dict(_DEFAULTS)
    f = ROOT / 'local_paths.json'
    if f.exists():
        for k, v in json.loads(f.read_text(encoding='utf-8')).items():
            if k.startswith('_'):
                continue
            if k not in cfg:
                raise KeyError('unknown key %r in %s (allowed: %s)' % (k, f, ', '.join(cfg)))
            cfg[k] = Path(v)
    for k in cfg:
        env = os.environ.get('MEMFX_' + k.upper())
        if env:
            cfg[k] = Path(env)
    return cfg


_CFG = _load()

# raw data (outside git)
HIGHD_TRACKS = _CFG['highd_tracks']
HIGHD_FULL = _CFG['highd_full']
TAXI_RAW = _CFG['taxi_raw']
GEOLIFE_RAW = _CFG['geolife_raw']
GEOLIFE_TRIPS_BY_MODE = _CFG['geolife_trips_by_mode']
GEOLIFE_CACHE = _CFG['geolife_cache']
AIRPORTS_SEQ = _CFG['airports_seq']

# code and committed results inside the repository
GEOLIFE_PIPELINE = ROOT / 'geolife'                 # v1 GeoLife scripts (04_mode_hurst_memory.py is reused)
GEOLIFE_OUTPUTS = GEOLIFE_PIPELINE / 'outputs_clean'
TAXI_RESULTS = ROOT / 'taxi'                        # memory_*.csv written by taxi/taxi_memory_analysis.ipynb

CONTROLS = ROOT / 'controls'
CONTROLS_RESULTS = CONTROLS / 'results'             # tables (committed)
CONTROLS_FIGURES = CONTROLS / 'figures' / 'output'  # figures (committed)
CONTROLS_WORK = CONTROLS / 'work'                   # large intermediate caches (git-ignored)

PREDICTION = ROOT / 'prediction'
PRED_DATA = PREDICTION / 'data'                     # *.npz token sequences (git-ignored), *.json metadata (committed)
PRED_RESULTS = PREDICTION / 'results'
PRED_NGRAM = PRED_RESULTS / 'ngram'                 # n-gram and memory-network tables
PRED_NEURAL = PRED_RESULTS / 'neural'               # LSTM / Transformer runs (hist/ = history-length ablation)
PRED_INFOMAP = PRED_RESULTS / 'infomap'             # official Infomap runs
PRED_FIGURES = PREDICTION / 'figures' / 'output'
INFOMAP_INPUT = PREDICTION / 'infomap_input'        # exported .net files (git-ignored, regenerable)


def out(path):
    """Create the parent folder of an output file and return the path as a string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


_PATTERN = {'highd_tracks': '*_tracks.csv', 'highd_full': '*_tracksMeta.csv', 'taxi_raw': '*', 'geolife_raw': '[0-9][0-9][0-9]',
            'geolife_trips_by_mode': '*/plt/*.plt', 'geolife_cache': '*.npz', 'airports_seq': '*/*_*.txt'}


def check():
    """Print every data location, how many expected files it holds, and where the setting came from."""
    f = ROOT / 'local_paths.json'
    print('repository:', ROOT)
    print('local_paths.json:', 'found' if f.exists() else 'not found (using the repository data folders)')
    for k in _DEFAULTS:
        p = _CFG[k]
        n = len(list(p.glob(_PATTERN[k]))) if p.exists() else 0
        print('  %-22s %-8s %6d files   %s' % (k, 'ok' if n else 'MISSING', n, p))


if __name__ == '__main__':
    check()
