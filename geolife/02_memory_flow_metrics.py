from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import log2
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


State = Tuple[int, int]
History = Tuple[State, State]


def state_sequence(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_size: float,
    collapse_repeats: bool,
) -> list[State]:
    lat_bin = np.floor(np.asarray(lat, dtype=float) / grid_size).astype(np.int64)
    lon_bin = np.floor(np.asarray(lon, dtype=float) / grid_size).astype(np.int64)
    seq = list(zip(lat_bin.tolist(), lon_bin.tolist()))
    if not collapse_repeats or not seq:
        return seq

    collapsed = [seq[0]]
    for state in seq[1:]:
        if state != collapsed[-1]:
            collapsed.append(state)
    return collapsed


def load_full_trip_sequences(
    cache_file: Path,
    grid_size: float,
    collapse_repeats: bool,
) -> list[list[State]]:
    data = np.load(cache_file, allow_pickle=True)
    seqs: list[list[State]] = []
    for train_lat, train_lon, test_lat, test_lon in zip(
        data["train_trips_lat"],
        data["train_trips_lon"],
        data["test_trips_lat"],
        data["test_trips_lon"],
    ):
        lat = np.concatenate([np.asarray(train_lat, dtype=float), np.asarray(test_lat, dtype=float)])
        lon = np.concatenate([np.asarray(train_lon, dtype=float), np.asarray(test_lon, dtype=float)])
        mask = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[mask]
        lon = lon[mask]
        if len(lat) < 3:
            continue
        seq = state_sequence(lat, lon, grid_size, collapse_repeats)
        if len(seq) >= 3:
            seqs.append(seq)
    return seqs


def entropy(counter: Counter[State]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return float("nan")
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * log2(p)
    return value


def weighted_entropy(counters: dict[object, Counter[State]]) -> float:
    total = sum(sum(counter.values()) for counter in counters.values())
    if total <= 0:
        return float("nan")
    return float(
        sum((sum(counter.values()) / total) * entropy(counter) for counter in counters.values())
    )


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def build_memory_metrics(
    label: str,
    seqs: Iterable[list[State]],
    grid_size: float,
    collapse_repeats: bool,
    min_trigrams: int,
) -> dict[str, float | int | str] | None:
    first_given_current: dict[State, Counter[State]] = defaultdict(Counter)
    second_given_history: dict[History, Counter[State]] = defaultdict(Counter)
    history_counts: Counter[History] = Counter()
    state_set: set[State] = set()

    n_trips = 0
    n_states = 0
    n_bigrams = 0
    n_trigrams = 0
    empirical_returns = 0

    for seq in seqs:
        if len(seq) < 3:
            continue
        n_trips += 1
        n_states += len(seq)
        n_bigrams += len(seq) - 1
        n_trigrams += len(seq) - 2
        state_set.update(seq)
        for idx in range(2, len(seq)):
            prev_state = seq[idx - 2]
            current_state = seq[idx - 1]
            next_state = seq[idx]
            history = (prev_state, current_state)
            first_given_current[current_state][next_state] += 1
            second_given_history[history][next_state] += 1
            history_counts[history] += 1
            if next_state == prev_state:
                empirical_returns += 1

    if n_trigrams < min_trigrams:
        return None

    h_first = weighted_entropy(first_given_current)
    h_second = weighted_entropy(second_given_history)
    entropy_reduction = h_first - h_second

    first_return_expected = 0.0
    second_return_expected = 0.0
    for (prev_state, current_state), count in history_counts.items():
        first_counter = first_given_current[current_state]
        second_counter = second_given_history[(prev_state, current_state)]
        first_return_expected += count * safe_div(first_counter[prev_state], sum(first_counter.values()))
        second_return_expected += count * safe_div(second_counter[prev_state], sum(second_counter.values()))
    first_return_expected = safe_div(first_return_expected, n_trigrams)
    second_return_expected = safe_div(second_return_expected, n_trigrams)
    empirical_return_rate = safe_div(empirical_returns, n_trigrams)

    return {
        "user": label,
        "grid_size_deg": grid_size,
        "collapse_repeats": int(collapse_repeats),
        "n_trips": n_trips,
        "n_states": n_states,
        "n_unique_states": len(state_set),
        "n_bigrams": n_bigrams,
        "n_trigrams": n_trigrams,
        "n_current_states": len(first_given_current),
        "n_second_order_histories": len(second_given_history),
        "first_order_entropy_bits": h_first,
        "second_order_entropy_bits": h_second,
        "entropy_reduction_bits": entropy_reduction,
        "entropy_reduction_pct": safe_div(entropy_reduction, h_first) * 100.0,
        "empirical_two_step_return": empirical_return_rate,
        "first_order_expected_two_step_return": first_return_expected,
        "second_order_expected_two_step_return": second_return_expected,
        "second_minus_first_return": second_return_expected - first_return_expected,
        "second_over_first_return": safe_div(second_return_expected, first_return_expected),
    }


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Descriptive first- and second-order memory-flow metrics for GeoLife."
    )
    parser.add_argument("--cache-dir", type=Path, default=base / "cache_clean")
    parser.add_argument("--out-dir", type=Path, default=base / "outputs_clean" / "02_memory_flow")
    parser.add_argument("--grid-size", type=float, default=0.0001)
    parser.add_argument("--min-trigrams", type=int, default=50)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--keep-repeats",
        action="store_true",
        help="Keep consecutive duplicate grid cells instead of compressing trajectories into paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    collapse_repeats = not args.keep_repeats

    files = sorted(args.cache_dir.glob("*.npz"))
    if args.max_users is not None:
        files = files[: args.max_users]

    rows = []
    skipped = []
    global_sequences: list[list[State]] = []

    for file in tqdm(files, desc="memory-flow users"):
        seqs = load_full_trip_sequences(file, args.grid_size, collapse_repeats)
        row = build_memory_metrics(
            file.stem,
            seqs,
            args.grid_size,
            collapse_repeats,
            args.min_trigrams,
        )
        if row is None:
            skipped.append(
                {
                    "user": file.stem,
                    "reason": f"fewer than {args.min_trigrams} valid trigrams",
                    "n_sequences": len(seqs),
                }
            )
            continue
        rows.append(row)
        global_sequences.extend(seqs)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "memory_flow_metrics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.out_dir / "memory_flow_skipped.csv", index=False)

    global_row = build_memory_metrics(
        "GLOBAL",
        global_sequences,
        args.grid_size,
        collapse_repeats,
        args.min_trigrams,
    )
    if global_row is not None:
        pd.DataFrame([global_row]).to_csv(args.out_dir / "memory_flow_global.csv", index=False)

    if not df.empty:
        summary = {
            "users": len(df),
            "grid_size_deg": args.grid_size,
            "collapse_repeats": int(collapse_repeats),
            "first_order_entropy_median": df["first_order_entropy_bits"].median(),
            "second_order_entropy_median": df["second_order_entropy_bits"].median(),
            "entropy_reduction_bits_median": df["entropy_reduction_bits"].median(),
            "entropy_reduction_pct_median": df["entropy_reduction_pct"].median(),
            "empirical_two_step_return_median": df["empirical_two_step_return"].median(),
            "first_order_expected_return_median": df["first_order_expected_two_step_return"].median(),
            "second_order_expected_return_median": df["second_order_expected_two_step_return"].median(),
            "second_minus_first_return_median": df["second_minus_first_return"].median(),
            "positive_entropy_reduction_users": int((df["entropy_reduction_bits"] > 0).sum()),
            "positive_return_lift_users": int((df["second_minus_first_return"] > 0).sum()),
        }
        pd.DataFrame([summary]).to_csv(args.out_dir / "memory_flow_summary.csv", index=False)
        print(pd.Series(summary))

    print(f"metrics: {args.out_dir / 'memory_flow_metrics.csv'}")


if __name__ == "__main__":
    main()
