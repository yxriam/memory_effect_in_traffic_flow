from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent / "outputs_clean"
MODE_DIR = BASE / "04_mode_memory"         # final 5 s run written by 04a_resample_sensitivity_select.py
OUT_DIR = BASE / "05_decay_narrative_figures"

GROUPS = {
    "Fast decay": ["bus", "walk"],
    "Moderate decay": ["car", "bike", "airplane"],
    "Slow decay": ["train", "taxi", "subway"],
}
LOW_SAMPLE = ["run", "boat"]
MODE_LABEL = {
    "airplane": "Airplane",
    "bike": "Bike",
    "boat": "Boat",
    "bus": "Bus",
    "car": "Car",
    "run": "Run",
    "subway": "Subway",
    "taxi": "Taxi",
    "train": "Train",
    "walk": "Walk",
}
COLORS = {
    "Fast decay": "#c83f32",
    "Moderate decay": "#2f78ad",
    "Slow decay": "#3a946a",
    "Low sample": "#9a9a9a",
}
LINE_COLORS = {
    "bus": "#c83f32",
    "walk": "#e07a5f",
    "car": "#2f78ad",
    "bike": "#5aa0d6",
    "airplane": "#9ecae1",
    "train": "#3a946a",
    "taxi": "#73b77b",
    "subway": "#a6dba0",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.edgecolor": "#bdbdbd",
            "axes.linewidth": 1.0,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_summary() -> pd.DataFrame:
    return pd.read_csv(MODE_DIR / "mode_memory_summary.csv")


def load_kernel() -> pd.DataFrame:
    return pd.read_csv(MODE_DIR / "mode_kernel_curves.csv")


def fit_power_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 10) & (x <= 400) & (y > 0)
    if int(mask.sum()) < 4:
        return None
    slope, intercept, *_ = stats.linregress(np.log(x[mask]), np.log(y[mask]))
    xx = np.geomspace(float(x[mask].min()), float(x[mask].max()), 160)
    yy = np.exp(intercept) * xx**slope
    return xx, yy


def rolling_median(values: np.ndarray, window: int = 3) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def plot_alpha_bars(summary: pd.DataFrame) -> Path:
    main_order = GROUPS["Fast decay"] + GROUPS["Moderate decay"] + GROUPS["Slow decay"]
    plot_order = main_order + LOW_SAMPLE
    summary = summary.set_index("mode")

    labels = []
    values = []
    segments = []
    colors = []
    group_labels = []
    for mode in plot_order:
        row = summary.loc[mode]
        labels.append(MODE_LABEL[mode])
        values.append(float(row["kernel_power_alpha"]))
        segments.append(int(row["segments"]))
        if mode in GROUPS["Fast decay"]:
            colors.append(COLORS["Fast decay"])
            group_labels.append("Fast")
        elif mode in GROUPS["Moderate decay"]:
            colors.append(COLORS["Moderate decay"])
            group_labels.append("Moderate")
        elif mode in GROUPS["Slow decay"]:
            colors.append(COLORS["Slow decay"])
            group_labels.append("Slow")
        else:
            colors.append(COLORS["Low sample"])
            group_labels.append("Low sample")

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=220)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="#333333", linewidth=0.8, alpha=0.92)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Memory-kernel decay exponent (kernel alpha)", fontsize=13)
    ax.set_title("Memory Decay by Transportation Mode (5 s resampling)", fontsize=17, pad=14)
    ax.set_xlim(0, 1.75)
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.text(1.0, -0.78, "alpha = 1 reference", ha="center", va="bottom", fontsize=10, color="#555555")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value, nseg, group in zip(bars, values, segments, group_labels):
        text = f"{value:.3f}  (n={nseg})"
        if group == "Low sample":
            text += "  low sample"
        ax.text(value + 0.025, bar.get_y() + bar.get_height() / 2, text, ha="left", va="center", fontsize=10)

    for yline in [1.5, 4.5, 7.5]:
        ax.axhline(yline, color="#cfcfcf", linewidth=1.0)

    ax.text(1.66, 0.5, "Fast decay", ha="right", va="center", fontsize=11, color=COLORS["Fast decay"], weight="bold")
    ax.text(1.66, 3.0, "Moderate", ha="right", va="center", fontsize=11, color=COLORS["Moderate decay"], weight="bold")
    ax.text(1.66, 6.0, "Slow decay", ha="right", va="center", fontsize=11, color=COLORS["Slow decay"], weight="bold")
    ax.text(1.66, 8.5, "Exploratory only", ha="right", va="center", fontsize=11, color="#777777", weight="bold")

    fig.text(
        0.12,
        0.02,
        "Motorcycle has no valid segment and is omitted. Boat and run are shown only as low-sample exploratory results.",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    out = OUT_DIR / "fig_kernel_alpha_decay_groups_clear.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_kernel_group(ax: plt.Axes, curves: pd.DataFrame, summary: pd.DataFrame, modes: list[str], title: str) -> None:
    summary = summary.set_index("mode")
    for mode in modes:
        sub = curves[(curves["mode"] == mode) & (curves["time_s"] > 0) & (curves["time_s"] <= 400)].copy()
        if sub.empty:
            continue
        x = sub["time_s"].to_numpy(float)
        y = np.abs(sub["kernel"].to_numpy(float))
        y[y <= 0] = np.nan
        color = LINE_COLORS[mode]
        ax.plot(x, y, color=color, alpha=0.22, marker="o", markersize=3.0, linewidth=1.0)
        ax.plot(x, rolling_median(y, 3), color=color, alpha=0.86, linewidth=2.2)
        fit = fit_power_line(x, y)
        if fit is not None:
            xx, yy = fit
            ax.plot(xx, yy, color=color, linestyle="--", linewidth=2.0)
        alpha = float(summary.loc[mode, "kernel_power_alpha"])
        nseg = int(summary.loc[mode, "segments"])
        ax.plot([], [], color=color, linewidth=2.6, label=f"{MODE_LABEL[mode]}: alpha={alpha:.3f}, n={nseg}")

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5, 400)
    ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which="major", color="#bcbcbc", linewidth=0.8, alpha=0.45)
    ax.grid(True, which="minor", color="#d8d8d8", linewidth=0.4, alpha=0.22)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_kernel_groups(summary: pd.DataFrame, curves: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=220, sharey=True)
    titles = [
        "Fast decay\nbus, walk",
        "Moderate decay\ncar, bike, airplane",
        "Slow decay\ntaxi, subway, train",
    ]
    for ax, title, modes in zip(axes, titles, GROUPS.values()):
        draw_kernel_group(ax, curves, summary, modes, title)
        ax.set_xlabel("Lag time (s)", fontsize=12)
    axes[0].set_ylabel("Memory-kernel magnitude |K(t)|", fontsize=13)
    fig.suptitle("Memory Kernel Curves Grouped by Decay Speed (5 s resampling)", fontsize=18, y=1.03)
    fig.text(
        0.11,
        0.01,
        "Faint points/lines show raw kernel values; solid lines show a 3-point rolling median for visual readability; dashed lines show the power-law fit.",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = OUT_DIR / "fig_memory_kernel_decay_groups_clear.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    summary = load_summary()
    curves = load_kernel()
    alpha_fig = plot_alpha_bars(summary)
    kernel_fig = plot_kernel_groups(summary, curves)
    summary.to_csv(OUT_DIR / "mode_memory_summary_used_for_decay_figures.csv", index=False)
    print(alpha_fig)
    print(kernel_fig)


if __name__ == "__main__":
    main()
