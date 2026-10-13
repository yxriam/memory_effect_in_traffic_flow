from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MODE_SCRIPT = BASE / "04_mode_hurst_memory.py"


def load_mode_module():
    spec = importlib.util.spec_from_file_location("mode_hurst_memory", MODE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_candidate_seconds(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("candidate seconds must be positive")
        values.append(value)
    if not values:
        raise ValueError("no candidate seconds provided")
    return sorted(set(values))


def sec_label(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}s"
    return f"{value:g}s".replace(".", "p")


def evenly_sample(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None or len(paths) <= limit:
        return paths
    idx = np.linspace(0, len(paths) - 1, limit)
    keep = sorted({int(round(x)) for x in idx})
    return [paths[i] for i in keep]


def lag_settings(dt_sec: float, max_lag_seconds: float, fit_min_seconds: float, fit_max_seconds: float) -> tuple[int, int, int]:
    max_lag = max(4, int(math.ceil(max_lag_seconds / dt_sec)))
    fit_min_lag = max(1, int(math.ceil(fit_min_seconds / dt_sec)))
    fit_max_lag = max(fit_min_lag + 1, int(math.floor(fit_max_seconds / dt_sec)))
    fit_max_lag = min(fit_max_lag, max_lag)
    return max_lag, fit_min_lag, fit_max_lag


def raw_gap_profile(mode_data_dir: Path, candidates: list[float], sample_files_per_mode: int, max_gaps_per_mode: int) -> pd.DataFrame:
    rows = []
    fmt = "%Y-%m-%d %H:%M:%S"
    for mode_dir in sorted([p for p in mode_data_dir.iterdir() if p.is_dir()]):
        files = evenly_sample(sorted((mode_dir / "plt").glob("*.plt")), sample_files_per_mode)
        gaps = []
        files_used = 0
        for file in files:
            times = []
            try:
                with file.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        parts = line.strip().split(",")
                        if len(parts) < 7:
                            continue
                        try:
                            float(parts[0])
                            float(parts[1])
                            times.append(datetime.strptime(parts[5] + " " + parts[6], fmt))
                        except Exception:
                            continue
            except OSError:
                continue
            if len(times) < 3:
                continue
            times = sorted(set(times))
            local_gaps = [
                (times[i] - times[i - 1]).total_seconds()
                for i in range(1, len(times))
            ]
            local_gaps = [x for x in local_gaps if 0 < x <= 300]
            if local_gaps:
                gaps.extend(local_gaps)
                files_used += 1
            if len(gaps) >= max_gaps_per_mode:
                gaps = gaps[:max_gaps_per_mode]
                break
        if not gaps:
            continue
        arr = np.asarray(gaps, dtype=float)
        row = {
            "mode": mode_dir.name,
            "files_sampled": files_used,
            "gap_count": int(len(arr)),
            "gap_p25_s": float(np.percentile(arr, 25)),
            "gap_median_s": float(np.median(arr)),
            "gap_p75_s": float(np.percentile(arr, 75)),
            "gap_p90_s": float(np.percentile(arr, 90)),
        }
        for dt in candidates:
            row[f"pct_gap_le_{sec_label(dt)}"] = float((arr <= dt).mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_candidate(
    mode_module,
    mode_data_dir: Path,
    out_dir: Path,
    dt_sec: float,
    gap_sec: float,
    min_points: int,
    max_lag_seconds: float,
    fit_min_seconds: float,
    fit_max_seconds: float,
    max_modes: int | None,
    max_files_per_mode: int | None,
    max_segments_per_file: int | None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_lag, fit_min_lag, fit_max_lag = lag_settings(dt_sec, max_lag_seconds, fit_min_seconds, fit_max_seconds)

    mode_dirs = sorted([p for p in mode_data_dir.iterdir() if p.is_dir()])
    if max_modes is not None:
        mode_dirs = mode_dirs[:max_modes]

    segment_rows = []
    summary_rows = []
    acf_rows = []
    kernel_rows = []

    for mode_dir in tqdm(mode_dirs, desc=f"dt={dt_sec:g}s", leave=False):
        mode = mode_dir.name
        files = evenly_sample(sorted((mode_dir / "plt").glob("*.plt")), max_files_per_mode)
        mode_acfs = []
        mode_segments = []
        for file in files:
            rows, acfs = mode_module.analyze_mode_file(
                mode,
                file,
                gap_sec,
                min_points,
                dt_sec,
                max_lag,
                max_segments_per_file,
            )
            segment_rows.extend(rows)
            mode_segments.extend(rows)
            mode_acfs.extend(acfs)

        if mode_acfs:
            acf_mean = np.nanmean(np.vstack(mode_acfs), axis=0)
            kernel = mode_module.memory_kernel_from_acf(acf_mean, dt_sec)
            acf_alpha = mode_module.fit_power_alpha(acf_mean, dt_sec, fit_min_lag, fit_max_lag)
            kernel_alpha = mode_module.fit_power_alpha(kernel, dt_sec, fit_min_lag, fit_max_lag)
            for lag, value in enumerate(acf_mean):
                acf_rows.append({"mode": mode, "candidate_sec": dt_sec, "lag": lag, "time_s": lag * dt_sec, "acf": value})
            for lag, value in enumerate(kernel):
                kernel_rows.append({"mode": mode, "candidate_sec": dt_sec, "lag": lag, "time_s": lag * dt_sec, "kernel": value})
        else:
            acf_alpha = float("nan")
            kernel_alpha = float("nan")

        seg_df = pd.DataFrame(mode_segments)
        summary_rows.append(
            {
                "mode": mode,
                "candidate_sec": dt_sec,
                "max_lag": max_lag,
                "fit_min_lag": fit_min_lag,
                "fit_max_lag": fit_max_lag,
                "files": len(files),
                "segments": len(seg_df),
                "hurst_rs_median": seg_df["hurst_rs"].median() if not seg_df.empty else np.nan,
                "mean_speed_mps_median": seg_df["mean_speed_mps"].median() if not seg_df.empty else np.nan,
                "acf_power_alpha": acf_alpha,
                "kernel_power_alpha": kernel_alpha,
            }
        )

    pd.DataFrame(segment_rows).to_csv(out_dir / "mode_segment_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "mode_memory_summary.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(out_dir / "mode_acf_curves.csv", index=False)
    pd.DataFrame(kernel_rows).to_csv(out_dir / "mode_kernel_curves.csv", index=False)
    return pd.DataFrame(summary_rows)


def rank_stability(candidate_summary: pd.DataFrame, value_col: str) -> dict[float, float]:
    pivot = candidate_summary.pivot_table(index="mode", columns="candidate_sec", values=value_col, aggfunc="median")
    reference = pivot.median(axis=1, skipna=True)
    scores = {}
    for candidate in pivot.columns:
        pair = pd.concat([pivot[candidate], reference], axis=1).dropna()
        if len(pair) < 3:
            scores[float(candidate)] = float("nan")
        else:
            scores[float(candidate)] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
    return scores


def choose_candidate(
    candidates: list[float],
    summaries: pd.DataFrame,
    gaps: pd.DataFrame,
    min_native_coverage_pct: float,
    min_segment_coverage: float,
    min_finite_ratio: float,
) -> tuple[float, pd.DataFrame]:
    total_segments = summaries.groupby("candidate_sec")["segments"].sum()
    best_segments = max(float(total_segments.max()), 1.0)
    hurst_rank = rank_stability(summaries, "hurst_rs_median")
    kernel_rank = rank_stability(summaries, "kernel_power_alpha")

    rows = []
    for dt in candidates:
        sub = summaries[summaries["candidate_sec"] == dt]
        gap_col = f"pct_gap_le_{sec_label(dt)}"
        native_cov = float(gaps[gap_col].median()) if gap_col in gaps and not gaps.empty else float("nan")
        seg_cov = float(total_segments.get(dt, 0.0) / best_segments)
        finite_kernel = float(sub["kernel_power_alpha"].notna().mean()) if not sub.empty else 0.0
        finite_hurst = float(sub["hurst_rs_median"].notna().mean()) if not sub.empty else 0.0
        h_stab = hurst_rank.get(float(dt), float("nan"))
        k_stab = kernel_rank.get(float(dt), float("nan"))
        eligible = (
            native_cov >= min_native_coverage_pct
            and seg_cov >= min_segment_coverage
            and finite_kernel >= min_finite_ratio
        )
        stability_bonus = np.nanmean([h_stab, k_stab])
        if not np.isfinite(stability_bonus):
            stability_bonus = 0.0
        score = (
            0.45 * min(native_cov / 100.0, 1.0)
            + 0.30 * seg_cov
            + 0.15 * finite_kernel
            + 0.10 * max(stability_bonus, 0.0)
            - 0.01 * math.log2(dt)
        )
        rows.append(
            {
                "candidate_sec": dt,
                "native_gap_coverage_median_pct": native_cov,
                "segment_coverage_vs_best": seg_cov,
                "finite_kernel_ratio": finite_kernel,
                "finite_hurst_ratio": finite_hurst,
                "hurst_rank_stability": h_stab,
                "kernel_rank_stability": k_stab,
                "eligible": bool(eligible),
                "selection_score": float(score),
                "total_segments": int(total_segments.get(dt, 0)),
            }
        )

    decision = pd.DataFrame(rows).sort_values("candidate_sec")
    eligible_rows = decision[decision["eligible"]]
    if not eligible_rows.empty:
        selected = float(eligible_rows["candidate_sec"].min())
    else:
        selected = float(decision.sort_values(["selection_score", "candidate_sec"], ascending=[False, True]).iloc[0]["candidate_sec"])
    return selected, decision


def run_final(mode_data_dir: Path, final_out_dir: Path, dt_sec: float, args: argparse.Namespace) -> None:
    max_lag, fit_min_lag, fit_max_lag = lag_settings(dt_sec, args.max_lag_seconds, args.fit_min_seconds, args.fit_max_seconds)
    cmd = [
        sys.executable,
        str(MODE_SCRIPT),
        "--mode-data-dir",
        str(mode_data_dir),
        "--out-dir",
        str(final_out_dir),
        "--gap-sec",
        str(args.gap_sec),
        "--resample-sec",
        str(dt_sec),
        "--min-points",
        str(args.min_points),
        "--max-lag",
        str(max_lag),
        "--fit-min-lag",
        str(fit_min_lag),
        "--fit-max-lag",
        str(fit_max_lag),
    ]
    print("final rerun:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for GeoLife resampling interval and automatic final rerun.")
    parser.add_argument("--mode-data-dir", type=Path, default=BASE / "data_raw" / "trips_by_mode_data")
    parser.add_argument("--out-dir", type=Path, default=BASE / "outputs_clean" / "04_resample_sensitivity")
    parser.add_argument("--final-out-dir", type=Path, default=BASE / "outputs_clean" / "04_mode_memory")
    parser.add_argument("--candidate-seconds", default="1,2,5,10")
    parser.add_argument("--gap-sec", type=float, default=1800.0)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-lag-seconds", type=float, default=1000.0)
    parser.add_argument("--fit-min-seconds", type=float, default=10.0)
    parser.add_argument("--fit-max-seconds", type=float, default=400.0)
    parser.add_argument("--sensitivity-max-modes", type=int, default=None)
    parser.add_argument("--sensitivity-max-files-per-mode", type=int, default=8)
    parser.add_argument("--sensitivity-max-segments-per-file", type=int, default=None)
    parser.add_argument("--gap-sample-files-per-mode", type=int, default=30)
    parser.add_argument("--max-gaps-per-mode", type=int, default=100000)
    parser.add_argument("--min-native-coverage-pct", type=float, default=95.0)
    parser.add_argument("--min-segment-coverage", type=float, default=0.85)
    parser.add_argument("--min-finite-ratio", type=float, default=0.70)
    parser.add_argument("--skip-final-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = parse_candidate_seconds(args.candidate_seconds)
    mode_module = load_mode_module()

    gaps = raw_gap_profile(args.mode_data_dir, candidates, args.gap_sample_files_per_mode, args.max_gaps_per_mode)
    gaps.to_csv(args.out_dir / "raw_gap_profile.csv", index=False)

    all_summaries = []
    for dt in candidates:
        cand_dir = args.out_dir / f"dt_{sec_label(dt)}"
        summary = run_candidate(
            mode_module,
            args.mode_data_dir,
            cand_dir,
            dt,
            args.gap_sec,
            args.min_points,
            args.max_lag_seconds,
            args.fit_min_seconds,
            args.fit_max_seconds,
            args.sensitivity_max_modes,
            args.sensitivity_max_files_per_mode,
            args.sensitivity_max_segments_per_file,
        )
        all_summaries.append(summary)

    summaries = pd.concat(all_summaries, ignore_index=True)
    summaries.to_csv(args.out_dir / "candidate_mode_memory_summary.csv", index=False)
    selected, decision = choose_candidate(
        candidates,
        summaries,
        gaps,
        args.min_native_coverage_pct,
        args.min_segment_coverage,
        args.min_finite_ratio,
    )
    decision.to_csv(args.out_dir / "resample_selection_summary.csv", index=False)

    max_lag, fit_min_lag, fit_max_lag = lag_settings(selected, args.max_lag_seconds, args.fit_min_seconds, args.fit_max_seconds)
    metadata = {
        "selected_resample_sec": selected,
        "rule": "choose the smallest candidate with median native-gap coverage, segment coverage, and finite-kernel ratios above thresholds; otherwise choose the highest score",
        "thresholds": {
            "min_native_coverage_pct": args.min_native_coverage_pct,
            "min_segment_coverage": args.min_segment_coverage,
            "min_finite_ratio": args.min_finite_ratio,
        },
        "max_lag_seconds": args.max_lag_seconds,
        "fit_window_seconds": [args.fit_min_seconds, args.fit_max_seconds],
        "final_lags": {
            "max_lag": max_lag,
            "fit_min_lag": fit_min_lag,
            "fit_max_lag": fit_max_lag,
        },
        "sensitivity_sample": {
            "max_modes": args.sensitivity_max_modes,
            "max_files_per_mode": args.sensitivity_max_files_per_mode,
            "max_segments_per_file": args.sensitivity_max_segments_per_file,
        },
        "raw_gap_profile": {
            "sample_files_per_mode": args.gap_sample_files_per_mode,
            "max_gaps_per_mode": args.max_gaps_per_mode,
        },
    }
    (args.out_dir / "selected_resample_interval.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nselection summary:")
    print(decision.to_string(index=False))
    print(f"\nselected resample interval: {selected:g}s")
    print(f"selection metadata: {args.out_dir / 'selected_resample_interval.json'}")

    if not args.skip_final_rerun:
        run_final(args.mode_data_dir, args.final_out_dir, selected, args)


if __name__ == "__main__":
    main()
