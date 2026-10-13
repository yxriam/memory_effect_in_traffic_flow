from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


State = Tuple[int, int]


def state_sequence(lat: np.ndarray, lon: np.ndarray, grid_size: float) -> list[State]:
    lat_bin = np.floor(lat / grid_size).astype(np.int64)
    lon_bin = np.floor(lon / grid_size).astype(np.int64)
    return list(zip(lat_bin.tolist(), lon_bin.tolist()))


def load_cache(path: Path) -> dict[str, list[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    return {
        "train_lat": list(data["train_trips_lat"]),
        "train_lon": list(data["train_trips_lon"]),
        "test_lat": list(data["test_trips_lat"]),
        "test_lon": list(data["test_trips_lon"]),
    }


def build_first_order(seqs: Iterable[list[State]]) -> dict[State, Counter]:
    chain: dict[State, Counter] = defaultdict(Counter)
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            chain[a][b] += 1
    return chain


def build_second_order(seqs: Iterable[list[State]]) -> dict[tuple[State, State], Counter]:
    chain: dict[tuple[State, State], Counter] = defaultdict(Counter)
    for seq in seqs:
        for i in range(2, len(seq)):
            chain[(seq[i - 2], seq[i - 1])][seq[i]] += 1
    return chain


def top(counter: Counter | None) -> State | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def safe_div(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def evaluate(
    test_seqs: list[list[State]],
    first: dict[State, Counter],
    second: dict[tuple[State, State], Counter],
) -> dict[str, float | int]:
    first_ok = first_total = first_seen = 0
    first_on_second_ok = first_on_second_total = 0
    second_ok = second_total = second_seen = 0
    second_fb_ok = second_fb_total = 0

    for seq in test_seqs:
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            pred1 = top(first.get(seq[i - 1]))
            first_total += 1
            if pred1 is not None:
                first_seen += 1
            if pred1 == seq[i]:
                first_ok += 1

        for i in range(2, len(seq)):
            pred1 = top(first.get(seq[i - 1]))
            pred2 = top(second.get((seq[i - 2], seq[i - 1])))
            first_on_second_total += 1
            second_total += 1
            second_fb_total += 1

            if pred1 == seq[i]:
                first_on_second_ok += 1
            if pred2 is not None:
                second_seen += 1
            if pred2 == seq[i]:
                second_ok += 1

            pred_fb = pred2 if pred2 is not None else pred1
            if pred_fb == seq[i]:
                second_fb_ok += 1

    return {
        "first_acc_all": safe_div(first_ok, first_total),
        "first_coverage": safe_div(first_seen, first_total),
        "first_acc_second_horizon": safe_div(first_on_second_ok, first_on_second_total),
        "second_acc_all_no_fallback": safe_div(second_ok, second_total),
        "second_coverage": safe_div(second_seen, second_total),
        "second_acc_all_fallback": safe_div(second_fb_ok, second_fb_total),
        "first_eval_n": first_total,
        "second_eval_n": second_total,
    }


def process_user(cache_file: Path, grid_size: float) -> dict[str, float | int | str] | None:
    cached = load_cache(cache_file)
    train_seqs = [
        state_sequence(lat, lon, grid_size)
        for lat, lon in zip(cached["train_lat"], cached["train_lon"])
        if len(lat) >= 3
    ]
    test_seqs = [
        state_sequence(lat, lon, grid_size)
        for lat, lon in zip(cached["test_lat"], cached["test_lon"])
        if len(lat) >= 3
    ]
    if not train_seqs or not test_seqs:
        return None

    first = build_first_order(train_seqs)
    second = build_second_order(train_seqs)
    metrics = evaluate(test_seqs, first, second)
    return {
        "user": cache_file.stem,
        "grid_size": grid_size,
        "train_trips": len(train_seqs),
        "test_trips": len(test_seqs),
        "first_states_seen": len(first),
        "second_histories_seen": len(second),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Hold-out Markov evaluation for GeoLife.")
    parser.add_argument("--cache-dir", type=Path, default=base / "cache_clean")
    parser.add_argument("--out-dir", type=Path, default=base / "outputs_clean" / "02_markov")
    parser.add_argument("--grid-size", type=float, default=0.0001)
    parser.add_argument("--max-users", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cache_dir.glob("*.npz"))
    if args.max_users is not None:
        files = files[: args.max_users]

    rows = []
    skipped = []
    for file in tqdm(files, desc="markov users"):
        row = process_user(file, args.grid_size)
        if row is None:
            skipped.append({"user": file.stem, "reason": "no valid train/test state sequences"})
        else:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "markov_holdout_metrics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.out_dir / "markov_holdout_skipped.csv", index=False)

    if not df.empty:
        summary = {
            "users": len(df),
            "first_acc_all_median": df["first_acc_all"].median(),
            "second_acc_no_fallback_median": df["second_acc_all_no_fallback"].median(),
            "second_acc_fallback_median": df["second_acc_all_fallback"].median(),
            "first_coverage_median": df["first_coverage"].median(),
            "second_coverage_median": df["second_coverage"].median(),
            "second_fallback_beats_first_users": int(
                (df["second_acc_all_fallback"] > df["first_acc_second_horizon"]).sum()
            ),
        }
        pd.DataFrame([summary]).to_csv(args.out_dir / "markov_holdout_summary.csv", index=False)
        print(pd.Series(summary))
    print(f"metrics: {args.out_dir / 'markov_holdout_metrics.csv'}")


if __name__ == "__main__":
    main()
