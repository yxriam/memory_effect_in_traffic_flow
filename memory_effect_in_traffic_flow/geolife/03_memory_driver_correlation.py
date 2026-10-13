from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


State = Tuple[int, int]
EARTH_LAT_M = 110540.0
EARTH_LON_M = 111320.0


def state_sequence(lat: np.ndarray, lon: np.ndarray, grid_size: float) -> list[State]:
    lat_bin = np.floor(np.asarray(lat, dtype=float) / grid_size).astype(np.int64)
    lon_bin = np.floor(np.asarray(lon, dtype=float) / grid_size).astype(np.int64)
    return list(zip(lat_bin.tolist(), lon_bin.tolist()))


def collapse_repeats(seq: list[State]) -> list[State]:
    if not seq:
        return seq
    collapsed = [seq[0]]
    for state in seq[1:]:
        if state != collapsed[-1]:
            collapsed.append(state)
    return collapsed


def step_distances_m(seq: list[State], grid_size: float) -> np.ndarray:
    if len(seq) < 2:
        return np.asarray([], dtype=float)
    lat = np.asarray([s[0] * grid_size for s in seq], dtype=float)
    lon = np.asarray([s[1] * grid_size for s in seq], dtype=float)
    lat0 = float(np.nanmean(lat))
    dx = np.diff(lon) * EARTH_LON_M * np.cos(np.deg2rad(lat0))
    dy = np.diff(lat) * EARTH_LAT_M
    return np.sqrt(dx * dx + dy * dy)


def turn_count(seq: list[State], grid_size: float, min_step_m: float, angle_deg: float) -> int:
    if len(seq) < 3:
        return 0
    lat = np.asarray([s[0] * grid_size for s in seq], dtype=float)
    lon = np.asarray([s[1] * grid_size for s in seq], dtype=float)
    lat0 = float(np.nanmean(lat))
    x = lon * EARTH_LON_M * np.cos(np.deg2rad(lat0))
    y = lat * EARTH_LAT_M
    vx = np.diff(x)
    vy = np.diff(y)
    step = np.sqrt(vx * vx + vy * vy)
    if len(step) < 2:
        return 0
    dot = vx[:-1] * vx[1:] + vy[:-1] * vy[1:]
    denom = step[:-1] * step[1:]
    valid = (step[:-1] >= min_step_m) & (step[1:] >= min_step_m) & (denom > 0)
    cosang = np.zeros_like(dot, dtype=float)
    cosang[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.degrees(np.arccos(cosang[valid]))
    return int(np.sum(angles >= angle_deg))


def return_rates(seq: list[State]) -> tuple[int, int, int, int]:
    two_total = max(0, len(seq) - 2)
    three_total = max(0, len(seq) - 3)
    two = sum(1 for i in range(2, len(seq)) if seq[i] == seq[i - 2])
    three = sum(1 for i in range(3, len(seq)) if seq[i] == seq[i - 3])
    return two, two_total, three, three_total


def load_user_sequences(cache_file: Path, grid_size: float) -> list[tuple[list[State], list[State]]]:
    data = np.load(cache_file, allow_pickle=True)
    out: list[tuple[list[State], list[State]]] = []
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
        raw_seq = state_sequence(lat, lon, grid_size)
        path_seq = collapse_repeats(raw_seq)
        if len(path_seq) >= 3:
            out.append((raw_seq, path_seq))
    return out


def process_user(
    cache_file: Path,
    grid_size: float,
    min_step_m: float,
    angle_deg: float,
) -> dict[str, float | int | str] | None:
    pairs = load_user_sequences(cache_file, grid_size)
    if not pairs:
        return None

    raw_points = 0
    path_states = 0
    unique_states: set[State] = set()
    turns = 0
    turn_den = 0
    two = two_den = three = three_den = 0
    all_steps: list[np.ndarray] = []
    efficiencies = []
    trip_lengths = []

    for raw_seq, path_seq in pairs:
        raw_points += len(raw_seq)
        path_states += len(path_seq)
        unique_states.update(path_seq)
        turns += turn_count(path_seq, grid_size, min_step_m, angle_deg)
        turn_den += max(0, len(path_seq) - 2)
        t2, d2, t3, d3 = return_rates(path_seq)
        two += t2
        two_den += d2
        three += t3
        three_den += d3
        steps = step_distances_m(path_seq, grid_size)
        if len(steps):
            all_steps.append(steps)
            path_len = float(np.sum(steps))
            direct = float(step_distances_m([path_seq[0], path_seq[-1]], grid_size).sum())
            if path_len > 0:
                efficiencies.append(direct / path_len)
                trip_lengths.append(path_len)

    steps_cat = np.concatenate(all_steps) if all_steps else np.asarray([], dtype=float)
    dwell_ratio = 1.0 - (path_states / raw_points) if raw_points else np.nan
    unique_ratio = len(unique_states) / path_states if path_states else np.nan
    history_counter = Counter()
    for _, path_seq in pairs:
        for i in range(2, len(path_seq)):
            history_counter[(path_seq[i - 2], path_seq[i - 1])] += 1

    return {
        "user": cache_file.stem,
        "n_trips_feature": len(pairs),
        "raw_points": raw_points,
        "path_states": path_states,
        "unique_states_feature": len(unique_states),
        "unique_state_ratio": unique_ratio,
        "dwell_ratio": dwell_ratio,
        "turn_count": turns,
        "turn_rate": turns / turn_den if turn_den else np.nan,
        "two_step_return_rate_feature": two / two_den if two_den else np.nan,
        "three_step_return_rate": three / three_den if three_den else np.nan,
        "mean_step_m": float(np.mean(steps_cat)) if len(steps_cat) else np.nan,
        "median_step_m": float(np.median(steps_cat)) if len(steps_cat) else np.nan,
        "mean_trip_length_m": float(np.mean(trip_lengths)) if trip_lengths else np.nan,
        "mean_trip_efficiency": float(np.mean(efficiencies)) if efficiencies else np.nan,
        "second_order_history_reuse": float(np.mean(list(history_counter.values()))) if history_counter else np.nan,
    }


def correlation_table(df: pd.DataFrame, rows: list[str], cols: list[str], method: str) -> pd.DataFrame:
    out = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for row in rows:
        for col in cols:
            sub = df[[row, col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3 or sub[row].nunique() < 2 or sub[col].nunique() < 2:
                out.loc[row, col] = np.nan
            else:
                out.loc[row, col] = sub[row].corr(sub[col], method=method)
    return out


def mutual_information_table(df: pd.DataFrame, rows: list[str], cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for row in rows:
        for col in cols:
            sub = df[[row, col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 5 or sub[row].nunique() < 2 or sub[col].nunique() < 2:
                out.loc[row, col] = np.nan
                continue
            x = sub[[row]].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            out.loc[row, col] = float(mutual_info_regression(x, y, random_state=0)[0])
    return out


def plot_heatmaps(tables: dict[str, pd.DataFrame], out_file: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), dpi=180)
    for ax, (name, table) in zip(axes.ravel(), tables.items()):
        values = table.to_numpy(dtype=float)
        if name == "Mutual information":
            vmax = np.nanpercentile(values, 95) if np.isfinite(values).any() else 1.0
            vmin = 0.0
            cmap = "YlOrRd"
        else:
            vmax = 1.0
            vmin = -1.0
            cmap = "RdBu_r"
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks(np.arange(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(table.shape[0]))
        ax.set_yticklabels(table.index, fontsize=8)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("GeoLife driver features vs memory-flow metrics", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Correlation analysis for GeoLife memory drivers.")
    parser.add_argument("--cache-dir", type=Path, default=base / "cache_clean")
    parser.add_argument("--memory-flow-dir", type=Path, default=base / "outputs_clean" / "02_memory_flow")
    parser.add_argument("--out-dir", type=Path, default=base / "outputs_clean" / "03_correlation_analysis")
    parser.add_argument("--fig-dir", type=Path, default=base / "outputs_clean" / "figures")
    parser.add_argument("--grid-size", type=float, default=0.0001)
    parser.add_argument("--turn-angle-deg", type=float, default=45.0)
    parser.add_argument("--min-step-m", type=float, default=8.0)
    parser.add_argument("--max-users", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.cache_dir.glob("*.npz"))
    if args.max_users is not None:
        files = files[: args.max_users]

    rows = []
    skipped = []
    for file in tqdm(files, desc="correlation features"):
        row = process_user(file, args.grid_size, args.min_step_m, args.turn_angle_deg)
        if row is None:
            skipped.append({"user": file.stem, "reason": "no valid paths"})
        else:
            rows.append(row)

    features = pd.DataFrame(rows)
    features.to_csv(args.out_dir / "geolife_user_driver_features.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.out_dir / "geolife_user_driver_skipped.csv", index=False)

    memory = pd.read_csv(args.memory_flow_dir / "memory_flow_metrics.csv", dtype={"user": str})
    df = features.merge(memory, on="user", how="inner")
    df.to_csv(args.out_dir / "geolife_driver_memory_merged.csv", index=False)

    driver_cols = [
        "dwell_ratio",
        "turn_rate",
        "unique_state_ratio",
        "two_step_return_rate_feature",
        "three_step_return_rate",
        "mean_step_m",
        "median_step_m",
        "mean_trip_length_m",
        "mean_trip_efficiency",
        "second_order_history_reuse",
    ]
    target_cols = [
        "first_order_entropy_bits",
        "second_order_entropy_bits",
        "entropy_reduction_bits",
        "entropy_reduction_pct",
        "empirical_two_step_return",
        "second_minus_first_return",
    ]

    tables = {
        "Pearson": correlation_table(df, driver_cols, target_cols, "pearson"),
        "Spearman": correlation_table(df, driver_cols, target_cols, "spearman"),
        "Kendall": correlation_table(df, driver_cols, target_cols, "kendall"),
        "Mutual information": mutual_information_table(df, driver_cols, target_cols),
    }
    for name, table in tables.items():
        safe = name.lower().replace(" ", "_")
        table.to_csv(args.out_dir / f"correlation_{safe}.csv")

    plot_heatmaps(tables, args.fig_dir / "fig_geolife_correlation_heatmaps.png")

    strongest = []
    for name, table in tables.items():
        for row in table.index:
            for col in table.columns:
                val = table.loc[row, col]
                if pd.notna(val):
                    strongest.append(
                        {
                            "method": name,
                            "driver": row,
                            "memory_metric": col,
                            "value": val,
                            "abs_value": abs(val),
                        }
                    )
    strongest_df = pd.DataFrame(strongest).sort_values(["method", "abs_value"], ascending=[True, False])
    strongest_df.to_csv(args.out_dir / "strongest_correlations.csv", index=False)

    print(f"users merged: {len(df)}")
    print(f"features: {args.out_dir / 'geolife_user_driver_features.csv'}")
    print(f"figure: {args.fig_dir / 'fig_geolife_correlation_heatmaps.png'}")
    print(strongest_df.groupby("method").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
