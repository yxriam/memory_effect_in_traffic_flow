from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm


EARTH_R = 6371000.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_mode_data_dir() -> Path:
    base = Path(__file__).resolve().parent
    root = project_root()
    for candidate in [base / "data_raw" / "trips_by_mode_data", root / "trips_by_mode" / "data"]:
        if candidate.exists():
            return candidate
    return base / "data_raw" / "trips_by_mode_data"


def read_plt(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
            except ValueError:
                continue
            rows.append((parts[5] + " " + parts[6], lat, lon))
    df = pd.DataFrame(rows, columns=["datetime", "lat", "lon"])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon"])
    return df[["datetime", "lat", "lon"]].sort_values("datetime").reset_index(drop=True)


def split_by_gap(df: pd.DataFrame, gap_sec: float, min_points: int) -> list[pd.DataFrame]:
    if df.empty:
        return []
    deltas = df["datetime"].diff().dt.total_seconds().fillna(0).to_numpy()
    starts = [0] + (np.where(deltas > gap_sec)[0]).tolist()
    ends = starts[1:] + [len(df)]
    segments = []
    for start, end in zip(starts, ends):
        seg = df.iloc[start:end].reset_index(drop=True)
        if len(seg) >= min_points:
            segments.append(seg)
    return segments


def resample_segment(seg: pd.DataFrame, dt_sec: float) -> pd.DataFrame | None:
    if len(seg) < 2:
        return None
    t0 = seg["datetime"].iloc[0]
    ts = (seg["datetime"] - t0).dt.total_seconds().to_numpy()
    keep = np.r_[True, np.diff(ts) > 0]
    ts = ts[keep]
    lat = seg["lat"].to_numpy()[keep]
    lon = seg["lon"].to_numpy()[keep]
    if len(ts) < 2 or ts[-1] < dt_sec:
        return None
    new_ts = np.arange(0, ts[-1] + 1e-6, dt_sec)
    if len(new_ts) < 4:
        return None
    return pd.DataFrame(
        {
            "t": new_ts,
            "lat": np.interp(new_ts, ts, lat),
            "lon": np.interp(new_ts, ts, lon),
        }
    )


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    x = np.deg2rad(lon - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_R
    return x, y


def velocity(seg_rs: pd.DataFrame, dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = latlon_to_xy(seg_rs["lat"].to_numpy(), seg_rs["lon"].to_numpy())
    vx = np.diff(x) / dt_sec
    vy = np.diff(y) / dt_sec
    speed = np.sqrt(vx * vx + vy * vy)
    return vx, vy, speed


def hurst_rs(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 32 or np.std(x) == 0:
        return float("nan")
    max_k = max(8, len(x) // 4)
    windows = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max_k), 12)).astype(int))
    vals = []
    used = []
    for win in windows:
        nseg = len(x) // win
        if nseg < 2:
            continue
        rs_vals = []
        for i in range(nseg):
            part = x[i * win : (i + 1) * win]
            centered = part - np.mean(part)
            walk = np.cumsum(centered)
            spread = np.max(walk) - np.min(walk)
            std = np.std(part, ddof=1)
            if std > 0:
                rs_vals.append(spread / std)
        if rs_vals:
            vals.append(float(np.mean(rs_vals)))
            used.append(win)
    if len(vals) < 3:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log(used), np.log(vals))
    return float(slope)


def vector_acf(vx: np.ndarray, vy: np.ndarray, max_lag: int) -> np.ndarray | None:
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    denom = np.sum(vx * vx + vy * vy)
    if len(vx) < max_lag + 2 or denom <= 0:
        return None
    cx = signal.correlate(vx, vx, mode="full", method="fft")
    cy = signal.correlate(vy, vy, mode="full", method="fft")
    mid = len(vx) - 1
    acf = (cx[mid : mid + max_lag + 1] + cy[mid : mid + max_lag + 1]) / denom
    if acf[0] != 0:
        acf = acf / acf[0]
    return acf


def memory_kernel_from_acf(acf: np.ndarray, dt_sec: float) -> np.ndarray:
    c = np.asarray(acf, dtype=float)
    kernel = np.zeros_like(c)
    if len(c) < 2 or c[0] == 0:
        return kernel
    for n in range(len(c) - 1):
        rhs = -(c[n + 1] - c[n]) / (dt_sec * dt_sec)
        conv = 0.0
        for m in range(n):
            conv += kernel[m] * c[n - m]
        kernel[n] = (rhs - conv) / c[0]
    return kernel


def fit_power_alpha(y: np.ndarray, dt_sec: float, min_lag: int, max_lag: int) -> float:
    y = np.asarray(y, dtype=float)
    lags = np.arange(len(y), dtype=float) * dt_sec
    idx = np.arange(len(y))
    mask = (idx >= min_lag) & (idx <= max_lag) & np.isfinite(y) & (np.abs(y) > 0)
    if int(mask.sum()) < 4:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log(lags[mask]), np.log(np.abs(y[mask])))
    return float(-slope)


def analyze_mode_file(
    mode: str,
    file: Path,
    gap_sec: float,
    min_points: int,
    dt_sec: float,
    max_lag: int,
    max_segments: int | None,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    try:
        raw = read_plt(file)
    except Exception:
        return [], []
    segments = split_by_gap(raw, gap_sec, min_points)
    if max_segments is not None:
        segments = segments[:max_segments]

    rows = []
    acfs = []
    for seg_idx, seg in enumerate(segments):
        seg_rs = resample_segment(seg, dt_sec)
        if seg_rs is None:
            continue
        vx, vy, speed = velocity(seg_rs, dt_sec)
        if len(speed) < max(32, max_lag + 2):
            continue
        acf = vector_acf(vx, vy, max_lag)
        if acf is not None:
            acfs.append(acf)
        rows.append(
            {
                "mode": mode,
                "file": file.name,
                "segment_index": seg_idx,
                "resample_sec": dt_sec,
                "n_points": len(seg_rs),
                "duration_s": float(seg_rs["t"].iloc[-1]),
                "mean_speed_mps": float(np.mean(speed)),
                "median_speed_mps": float(np.median(speed)),
                "hurst_rs": hurst_rs(speed),
            }
        )
    return rows, acfs


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Mode-level Hurst and memory analysis.")
    parser.add_argument("--mode-data-dir", type=Path, default=default_mode_data_dir())
    parser.add_argument("--out-dir", type=Path, default=base / "outputs_clean" / "04_mode_memory")
    parser.add_argument("--gap-sec", type=float, default=1800.0)
    parser.add_argument("--resample-sec", type=float, default=5.0)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-lag", type=int, default=200)
    parser.add_argument("--fit-min-lag", type=int, default=2)
    parser.add_argument("--fit-max-lag", type=int, default=80)
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--max-files-per-mode", type=int, default=None)
    parser.add_argument("--max-segments-per-file", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = sorted([p for p in args.mode_data_dir.iterdir() if p.is_dir()])
    if args.max_modes is not None:
        mode_dirs = mode_dirs[: args.max_modes]

    segment_rows = []
    summary_rows = []
    acf_rows = []
    kernel_rows = []

    for mode_dir in tqdm(mode_dirs, desc="modes"):
        mode = mode_dir.name
        files = sorted((mode_dir / "plt").glob("*.plt"))
        if args.max_files_per_mode is not None:
            files = files[: args.max_files_per_mode]

        mode_acfs = []
        mode_segments = []
        for file in files:
            rows, acfs = analyze_mode_file(
                mode,
                file,
                args.gap_sec,
                args.min_points,
                args.resample_sec,
                args.max_lag,
                args.max_segments_per_file,
            )
            segment_rows.extend(rows)
            mode_segments.extend(rows)
            mode_acfs.extend(acfs)

        if mode_acfs:
            acf_mean = np.nanmean(np.vstack(mode_acfs), axis=0)
            kernel = memory_kernel_from_acf(acf_mean, args.resample_sec)
            acf_alpha = fit_power_alpha(acf_mean, args.resample_sec, args.fit_min_lag, args.fit_max_lag)
            kernel_alpha = fit_power_alpha(kernel, args.resample_sec, args.fit_min_lag, args.fit_max_lag)
            for lag, value in enumerate(acf_mean):
                acf_rows.append({"mode": mode, "lag": lag, "time_s": lag * args.resample_sec, "acf": value})
            for lag, value in enumerate(kernel):
                kernel_rows.append({"mode": mode, "lag": lag, "time_s": lag * args.resample_sec, "kernel": value})
        else:
            acf_alpha = float("nan")
            kernel_alpha = float("nan")

        seg_df = pd.DataFrame(mode_segments)
        summary_rows.append(
            {
                "mode": mode,
                "resample_sec": args.resample_sec,
                "max_lag": args.max_lag,
                "fit_min_lag": args.fit_min_lag,
                "fit_max_lag": args.fit_max_lag,
                "files": len(files),
                "segments": len(seg_df),
                "hurst_rs_median": seg_df["hurst_rs"].median() if not seg_df.empty else np.nan,
                "mean_speed_mps_median": seg_df["mean_speed_mps"].median() if not seg_df.empty else np.nan,
                "acf_power_alpha": acf_alpha,
                "kernel_power_alpha": kernel_alpha,
            }
        )

    pd.DataFrame(segment_rows).to_csv(args.out_dir / "mode_segment_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "mode_memory_summary.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(args.out_dir / "mode_acf_curves.csv", index=False)
    pd.DataFrame(kernel_rows).to_csv(args.out_dir / "mode_kernel_curves.csv", index=False)
    print(f"summary: {args.out_dir / 'mode_memory_summary.csv'}")


if __name__ == "__main__":
    main()
