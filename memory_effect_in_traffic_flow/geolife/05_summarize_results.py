from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import json


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4g}"


def summarize_memory_flow(base: Path) -> list[str]:
    path = base / "02_memory_flow" / "memory_flow_metrics.csv"
    if not path.exists():
        return ["Memory flow: no output found."]
    df = pd.read_csv(path)
    if df.empty:
        return ["Memory flow: empty output."]
    lines = [
        "Memory flow:",
        f"  users = {len(df)}",
        f"  first-order entropy median = {fmt_num(df['first_order_entropy_bits'].median())} bits",
        f"  second-order entropy median = {fmt_num(df['second_order_entropy_bits'].median())} bits",
        f"  entropy reduction median = {fmt_num(df['entropy_reduction_bits'].median())} bits",
        f"  entropy reduction pct median = {fmt_num(df['entropy_reduction_pct'].median())}%",
        f"  empirical two-step return median = {fmt_num(df['empirical_two_step_return'].median())}",
        f"  first-order expected return median = {fmt_num(df['first_order_expected_two_step_return'].median())}",
        f"  second-order expected return median = {fmt_num(df['second_order_expected_two_step_return'].median())}",
        f"  positive entropy-reduction users = {int((df['entropy_reduction_bits'] > 0).sum())}/{len(df)}",
    ]
    global_path = base / "02_memory_flow" / "memory_flow_global.csv"
    if global_path.exists():
        global_df = pd.read_csv(global_path)
        if not global_df.empty:
            row = global_df.iloc[0]
            lines.extend(
                [
                    "  global pooled flow:",
                    f"    H1 = {fmt_num(row['first_order_entropy_bits'])} bits",
                    f"    H2 = {fmt_num(row['second_order_entropy_bits'])} bits",
                    f"    reduction = {fmt_num(row['entropy_reduction_bits'])} bits",
                    f"    two-step return first/second = {fmt_num(row['first_order_expected_two_step_return'])}/{fmt_num(row['second_order_expected_two_step_return'])}",
                ]
            )
    return lines


def summarize_mode(base: Path) -> list[str]:
    path = base / "04_mode_memory" / "mode_memory_summary.csv"
    if not path.exists():
        return ["Mode memory: no output found."]
    df = pd.read_csv(path)
    if df.empty:
        return ["Mode memory: empty output."]
    lines = ["Mode memory:"]
    selection_path = base / "04_resample_sensitivity" / "selected_resample_interval.json"
    if selection_path.exists():
        try:
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            lines.append(f"  selected resample interval = {fmt_num(selected.get('selected_resample_sec'))} s")
        except Exception:
            pass
    cols = ["mode", "segments", "hurst_rs_median", "acf_power_alpha", "kernel_power_alpha"]
    for _, row in df[cols].sort_values("mode").iterrows():
        lines.append(
            "  "
            + f"{row['mode']}: segments={int(row['segments'])}, "
            + f"H={fmt_num(row['hurst_rs_median'])}, "
            + f"acf_alpha={fmt_num(row['acf_power_alpha'])}, "
            + f"kernel_alpha={fmt_num(row['kernel_power_alpha'])}"
        )
    return lines


def summarize_correlation(base: Path) -> list[str]:
    path = base / "03_correlation_analysis" / "strongest_correlations.csv"
    if not path.exists():
        return ["Correlation analysis: no output found."]
    df = pd.read_csv(path)
    if df.empty:
        return ["Correlation analysis: empty output."]
    lines = ["Correlation analysis:"]
    for method in ["Pearson", "Spearman", "Kendall", "Mutual information"]:
        sub = df[df["method"] == method].head(3)
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            lines.append(
                "  "
                + f"{method}: {row['driver']} vs {row['memory_metric']} = {fmt_num(row['value'])}"
            )
    return lines


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarize clean pipeline outputs.")
    parser.add_argument("--base-out-dir", type=Path, default=base / "outputs_clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for block in [
        summarize_memory_flow(args.base_out_dir),
        summarize_correlation(args.base_out_dir),
        summarize_mode(args.base_out_dir),
    ]:
        lines.extend(block)
        lines.append("")
    report = "\n".join(lines).rstrip() + "\n"
    out = args.base_out_dir / "summary_report.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
