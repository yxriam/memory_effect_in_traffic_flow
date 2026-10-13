from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SEQ_LEN_DEFAULT = 5
TRAIN_FRAC_DEFAULT = 0.8


def looks_like_geolife_datetime(date_text: str, time_text: str) -> bool:
    return (
        len(date_text) == 10
        and date_text[4] == "-"
        and date_text[7] == "-"
        and len(time_text) == 8
        and time_text[2] == ":"
        and time_text[5] == ":"
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    base = Path(__file__).resolve().parent
    root = project_root()
    for candidate in [base / "data_raw" / "Data", root / "Data"]:
        if candidate.exists():
            return candidate
    return base / "data_raw" / "Data"


def read_plt(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    lats = []
    lons = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                if not looks_like_geolife_datetime(parts[5], parts[6]):
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                except ValueError:
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                lats.append(lat)
                lons.append(lon)
    except OSError:
        return None
    if not lats:
        return None
    return np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32)


def process_user(
    user_dir: Path,
    seq_len: int,
    train_frac: float,
    max_trips: int | None,
) -> dict[str, object] | None:
    min_trip_points = seq_len * 2 + 2
    files = sorted((user_dir / "Trajectory").glob("*.plt"))
    if max_trips is not None:
        files = files[:max_trips]

    train_lat: list[np.ndarray] = []
    train_lon: list[np.ndarray] = []
    test_lat: list[np.ndarray] = []
    test_lon: list[np.ndarray] = []

    for file in files:
        trip = read_plt(file)
        if trip is None:
            continue
        lat, lon = trip
        n = len(lat)
        if n < min_trip_points:
            continue

        cut = int(n * train_frac)
        if cut < seq_len + 1 or n - cut < seq_len + 1:
            if n >= 2 * (seq_len + 1):
                cut = n - (seq_len + 1)
            else:
                continue

        train_lat.append(lat[:cut])
        train_lon.append(lon[:cut])
        test_lat.append(lat[cut:])
        test_lon.append(lon[cut:])

    if not train_lat or not test_lat:
        return None

    all_train_lat = np.concatenate(train_lat)
    all_train_lon = np.concatenate(train_lon)
    return {
        "train_trips_lat": np.array(train_lat, dtype=object),
        "train_trips_lon": np.array(train_lon, dtype=object),
        "test_trips_lat": np.array(test_lat, dtype=object),
        "test_trips_lon": np.array(test_lon, dtype=object),
        "lat_min_train": float(all_train_lat.min()),
        "lat_max_train": float(all_train_lat.max()),
        "lon_min_train": float(all_train_lon.min()),
        "lon_max_train": float(all_train_lon.max()),
        "n_train_trips": len(train_lat),
        "n_test_trips": len(test_lat),
        "n_train_points": int(len(all_train_lat)),
        "n_test_points": int(sum(len(x) for x in test_lat)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GeoLife without train/test leakage.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "cache_clean")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC_DEFAULT)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-trips-per-user", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted([p for p in args.data_dir.iterdir() if p.is_dir()])
    if args.max_users is not None:
        user_dirs = user_dirs[: args.max_users]

    rows = []
    for user_dir in tqdm(user_dirs, desc="preprocess users"):
        result = process_user(user_dir, args.seq_len, args.train_frac, args.max_trips_per_user)
        if result is None:
            rows.append({"user": user_dir.name, "status": "skipped", "reason": "no valid trips"})
            continue
        np.savez_compressed(args.cache_dir / f"{user_dir.name}.npz", **result)
        rows.append(
            {
                "user": user_dir.name,
                "status": "ok",
                "n_train_trips": result["n_train_trips"],
                "n_test_trips": result["n_test_trips"],
                "n_train_points": result["n_train_points"],
                "n_test_points": result["n_test_points"],
                "lat_range_train": result["lat_max_train"] - result["lat_min_train"],
                "lon_range_train": result["lon_max_train"] - result["lon_min_train"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.cache_dir / "_summary.csv", index=False)
    ok = int((summary["status"] == "ok").sum()) if not summary.empty else 0
    print(f"cached users: {ok}/{len(summary)}")
    print(f"summary: {args.cache_dir / '_summary.csv'}")


if __name__ == "__main__":
    main()
