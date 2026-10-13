from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent / "outputs_clean"
NEW_DIR = BASE / "04_mode_memory"          # final 5 s run written by 04a_resample_sensitivity_select.py
OLD_DIR = BASE / "04_mode_memory_old_5s"
OUT_DIR = BASE / "04_mode_memory_compare_5s"

ORDER = ["motorcycle", "boat", "train", "run", "car", "bike", "airplane", "taxi", "subway", "walk", "bus"]
LABELS = {
    "motorcycle": "Motorcycle",
    "boat": "Boat",
    "train": "Train",
    "run": "Run",
    "car": "Car",
    "bike": "Bike",
    "airplane": "Airplane",
    "taxi": "Taxi",
    "subway": "Subway",
    "walk": "Walk",
    "bus": "Bus",
}
GROUPS = [
    ("Motorized Transport", ["car", "taxi", "bus"]),
    ("Public / Rail / Water", ["boat", "train", "subway"]),
    ("Human Powered", ["bike", "run", "walk"]),
]
COLORS = {
    "car": "#c83f32",
    "taxi": "#2f78ad",
    "bus": "#d8a64a",
    "boat": "#c83f32",
    "train": "#2f78ad",
    "subway": "#d8a64a",
    "bike": "#c83f32",
    "run": "#2f78ad",
    "walk": "#d8a64a",
}
BOX_COLORS = [
    "#1f5a92",
    "#2b6f9f",
    "#337aa8",
    "#3f88b4",
    "#4f95bc",
    "#65a3c4",
    "#7fb2ca",
    "#9ec4d5",
    "#bad3e2",
    "#d5e3ec",
    "#ecf2f6",
]
TAU = "\u03c4"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.edgecolor": "#c7c7c7",
            "axes.linewidth": 1.0,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )


def load_summary(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "mode_memory_summary.csv")


def load_segments(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "mode_segment_metrics.csv")


def load_kernel(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "mode_kernel_curves.csv")


def write_comparison_table() -> None:
    new = load_summary(NEW_DIR)
    old = load_summary(OLD_DIR)
    cols = ["mode", "files", "segments", "hurst_rs_median", "acf_power_alpha", "kernel_power_alpha"]
    merged = new[cols].merge(old[cols], on="mode", suffixes=("_new", "_old"))
    for col in ["files", "segments", "hurst_rs_median", "acf_power_alpha", "kernel_power_alpha"]:
        merged[col + "_diff"] = merged[col + "_new"] - merged[col + "_old"]
    merged.to_csv(OUT_DIR / "new_vs_old_5s_summary.csv", index=False)


def mode_data_for_hurst(folder: Path) -> tuple[list[np.ndarray], list[str]]:
    segments = load_segments(folder)
    summary = load_summary(folder).set_index("mode")
    data = []
    labels = []
    for mode in ORDER:
        data.append(segments.loc[segments["mode"] == mode, "hurst_rs"].dropna().to_numpy())
        n_files = int(summary.loc[mode, "files"]) if mode in summary.index else 0
        labels.append(f"{LABELS[mode]}\n(N={n_files})")
    return data, labels


def draw_hurst_boxplot(ax: plt.Axes, folder: Path, title: str, point_limit: int = 160) -> None:
    data, labels = mode_data_for_hurst(folder)
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.3},
        whiskerprops={"color": "#333333", "linewidth": 1.1},
        capprops={"color": "#333333", "linewidth": 1.1},
        boxprops={"color": "#333333", "linewidth": 1.1},
    )
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(BOX_COLORS[min(i, len(BOX_COLORS) - 1)])
        patch.set_alpha(0.92)

    rng = np.random.default_rng(20260529)
    for idx, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        plot_values = values if len(values) <= point_limit else rng.choice(values, point_limit, replace=False)
        ax.scatter(
            np.full(len(plot_values), idx) + rng.normal(0, 0.045, len(plot_values)),
            plot_values,
            s=9,
            color="#4b6478",
            alpha=0.34,
            linewidths=0,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1.4)
    ax.set_ylim(0.35, 1.1)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=18, pad=14)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_hurst_single(folder: Path, title: str, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    draw_hurst_boxplot(ax, folder, title)
    ax.text(
        len(ORDER) + 0.35,
        1.012,
        r"Deterministic Trend ($H=1.0$)",
        ha="right",
        va="bottom",
        fontsize=11,
        style="italic",
    )
    ax.text(
        len(ORDER) + 0.35,
        0.512,
        r"Random Walk ($H=0.5$)",
        ha="right",
        va="bottom",
        fontsize=11,
        color="#777777",
        style="italic",
    )
    ax.set_ylabel(r"Hurst Exponent ($H$)", fontsize=15)
    ax.set_xlabel("Transportation Mode", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def plot_hurst_compare(out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=180, sharey=True)
    draw_hurst_boxplot(axes[0], NEW_DIR, "New data, 5 s", point_limit=120)
    draw_hurst_boxplot(axes[1], OLD_DIR, "Old data, 5 s", point_limit=120)
    axes[0].set_ylabel(r"Hurst Exponent ($H$)", fontsize=15)
    fig.supxlabel("Transportation Mode", fontsize=15, y=0.03)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < 4:
        return None
    slope, intercept, *_ = stats.linregress(np.log(x[mask]), np.log(y[mask]))
    xx = np.geomspace(float(x[mask].min()), float(x[mask].max()), 120)
    yy = np.exp(intercept) * xx**slope
    return float(-slope), xx, yy


def draw_kernel_panel(ax: plt.Axes, curves: pd.DataFrame, summary: pd.DataFrame, modes: list[str], x_max: float) -> None:
    for mode in modes:
        sub = curves[(curves["mode"] == mode) & (curves["time_s"] > 0) & (curves["time_s"] <= x_max)].copy()
        if sub.empty:
            continue
        x = sub["time_s"].to_numpy(float)
        y = np.abs(sub["kernel"].to_numpy(float))
        y[y <= 0] = np.nan
        color = COLORS[mode]
        ax.plot(x, y, color=color, alpha=0.30, marker="o", markersize=4, linewidth=1.8)
        fit_mask = (x >= 10) & (x <= min(100, x_max))
        fit = fit_line(x[fit_mask], y[fit_mask])
        if fit is not None:
            _, xx, yy = fit
            ax.plot(xx, yy, color=color, linestyle="--", linewidth=2.2)
        alpha = summary.loc[mode, "kernel_power_alpha"] if mode in summary.index else np.nan
        label = f"{LABELS[mode]} (\u03b1={alpha:.2f})" if np.isfinite(alpha) else LABELS[mode]
        ax.plot([], [], color=color, linestyle="--", linewidth=2.2, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5, x_max)
    ax.set_ylim(1e-6, 1e-1)
    ax.grid(True, which="major", color="#b6b6b6", alpha=0.40, linewidth=0.8)
    ax.legend(frameon=False, fontsize=14, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_kernel_single(folder: Path, title_suffix: str, out_file: Path, x_max: float = 100) -> None:
    curves = load_kernel(folder)
    summary = load_summary(folder).set_index("mode")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=180, sharey=True)
    for ax, (title, modes) in zip(axes, GROUPS):
        draw_kernel_panel(ax, curves, summary, modes, x_max)
        ax.set_title(title, fontsize=24, pad=14)
        ax.set_xlabel(f"Lag Time {TAU} (s)", fontsize=18)
    axes[0].set_ylabel(f"Memory Magnitude |K({TAU})|", fontsize=18)
    fig.suptitle(title_suffix, fontsize=18, y=1.04)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_compare(out_file: Path, x_max: float = 100) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=180, sharey=True)
    for row, (folder, row_title) in enumerate([(NEW_DIR, "New data, 5 s"), (OLD_DIR, "Old data, 5 s")]):
        curves = load_kernel(folder)
        summary = load_summary(folder).set_index("mode")
        for col, (title, modes) in enumerate(GROUPS):
            ax = axes[row, col]
            draw_kernel_panel(ax, curves, summary, modes, x_max)
            if row == 0:
                ax.set_title(title, fontsize=22, pad=12)
            ax.text(0.02, 0.95, row_title, transform=ax.transAxes, ha="left", va="top", fontsize=13)
            if row == 1:
                ax.set_xlabel(f"Lag Time {TAU} (s)", fontsize=16)
    axes[0, 0].set_ylabel(f"Memory Magnitude |K({TAU})|", fontsize=17)
    axes[1, 0].set_ylabel(f"Memory Magnitude |K({TAU})|", fontsize=17)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    write_comparison_table()
    plot_hurst_single(NEW_DIR, "Hurst Exponent by Transportation Mode (New data, 5 s)", OUT_DIR / "fig_hurst_5s_new_original_style.png")
    plot_hurst_single(OLD_DIR, "Hurst Exponent by Transportation Mode (Old data, 5 s)", OUT_DIR / "fig_hurst_5s_old_original_style.png")
    plot_hurst_compare(OUT_DIR / "fig_hurst_5s_new_vs_old_original_style.png")
    plot_kernel_single(NEW_DIR, "Memory Kernel Decay (New data, 5 s)", OUT_DIR / "fig_memory_kernel_5s_new_original_style.png")
    plot_kernel_single(OLD_DIR, "Memory Kernel Decay (Old data, 5 s)", OUT_DIR / "fig_memory_kernel_5s_old_original_style.png")
    plot_kernel_compare(OUT_DIR / "fig_memory_kernel_5s_new_vs_old_original_style.png")
    print(f"figures: {OUT_DIR}")


if __name__ == "__main__":
    main()
