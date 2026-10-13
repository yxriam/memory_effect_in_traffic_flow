"""Split BTS DB1B itinerary files into airport-code sequences grouped by path length.

Each line of a `db1b.public.YYYYMM.asc` file describes one itinerary; the three-letter
upper-case tokens on the line are its airports in travel order. The script writes
`<out_dir>/<YYYYMM>/<YYYYMM>_<n>.txt`, one file per number of airports n, each line being the
space-separated airport codes of one itinerary. These files feed markov_chain.py,
city_markov_sankey.py and lstm_rnn_predict.ipynb.

Usage:
    python airports_sequences_split.py data/original_data/db1b.public.202401.asc [more.asc ...] \
        --out-dir data/sequences_split
"""
import argparse
import os
import re

AIRPORT_CODE = re.compile(r'\b[A-Z]{3}\b')


def split_airport_sequences(input_path, out_dir):
    year_month = os.path.basename(input_path).split('.')[2]        # db1b.public.202401.asc -> 202401
    target_dir = os.path.join(out_dir, year_month)
    os.makedirs(target_dir, exist_ok=True)
    handles = {}
    try:
        with open(input_path, 'r') as fin:
            for line in fin:
                codes = AIRPORT_CODE.findall(line)
                if not codes:
                    continue
                n = len(codes)
                if n not in handles:
                    handles[n] = open(os.path.join(target_dir, f"{year_month}_{n}.txt"), 'w')
                handles[n].write(' '.join(codes) + '\n')
    finally:
        for h in handles.values():
            h.close()
    print(f"{input_path}: wrote {len(handles)} files to {target_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='DB1B .asc files')
    ap.add_argument('--out-dir', default='data/sequences_split')
    args = ap.parse_args()
    for p in args.inputs:
        split_airport_sequences(p, args.out_dir)
