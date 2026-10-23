# -*- coding: utf-8 -*-
"""Figure: completing the current state versus a longer history (airline itineraries and taxi), drawn from the result files.

Inputs:  airports/results/airport_memory_cache.json (airport_memory.py), controls/results/heading_state_taxi_200m_device.csv
Outputs: airports/figures/fig_airline_memory.png (report, 2 x 2) and fig_airline_memory_slide.png (slide, panels a and b)

    python airports/fig_airline_memory.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "msyh" in f.lower():
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
RED, TEAL, NAVY = "#E66D50", "#299D8F", "#264653"          # palette of the other report figures
TEAL_L, RED_L = "#94CEC7", "#F2B6A7"

c = json.load(open(HERE / "results" / "airport_memory_cache.json"))
Q = [("202409", "2024 年第 3 季度"), ("202412", "2024 年第 4 季度")]
air = {q: c[f"{q}|airport"] for q, _ in Q}
taxi = pd.read_csv(ROOT / "controls" / "results" / "heading_state_taxi_200m_device.csv")
taxi = taxi[(taxi.heading == "device") & (taxi.visits == "all")].iloc[0]


def lab(ax, s):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(-36, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", va="bottom", ha="left")


def panel_history(ax, inline=False):
    xs = [0, 1, 2, 3, 4.3]
    for sub, col, name in [("ge2", TEAL, "前面至少 2 站的步"), ("ge4", RED, "前面至少 4 站的步")]:
        for (q, qn), ls, mf in zip(Q, ["-", "--"], [col, "white"]):
            d = air[q]["ce_curve"][sub]
            y = [d[f"order{L}"] for L in range(1, 5)]
            ax.plot(xs[:4], y, ls=ls, color=col, lw=1.2, marker="o", ms=4.5, mfc=mf, mec=col)
            ax.plot([xs[4]], [d["origin_current"]], marker="*", ms=11, color=col, mfc=mf, mec=col, ls="")
    ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "3", "4", "起点＋\n当前机场"])
    ax.set_xlabel("看的历史（站数，含当前机场）")
    ax.set_ylabel("留出交叉熵（bits/步）" if inline else "下一站的留出交叉熵（bits/步）")
    ax.set_ylim(1.0, 7.0)
    ax.axvline(3.65, color="#BBBBBB", lw=0.6, ls=":")
    if inline:      # the slide has no caption: name the two step sets next to their lines
        ax.text(1.05, 6.45, "前面至少 4 站的步", color=RED, fontsize=9, va="bottom")
        ax.text(1.05, 3.75, "前面至少 2 站的步", color=TEAL, fontsize=9, va="top")


def panel_taxi(ax):
    vals = [taxi["G2_tuned"], taxi["GA_heading+side_tuned"], taxi["R_heading+side_tuned"]]
    names = ["前一格子", "方向＋\n格内位置", "补入后\n前一格子"]
    bars = ax.bar(range(3), vals, 0.6, color=[NAVY, TEAL, RED])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_ylabel("加入后的留出增益（bits/步）"); ax.set_ylim(0, 0.5)


def panel_cmi(ax):
    keys = [("I2", "前一站"), ("I3", "前第二站"), ("IO3", "行程起点\n（三站之外）")]
    x = np.arange(3); w = 0.36
    for j, ((q, qn), col) in enumerate(zip(Q, [TEAL, TEAL_L])):
        v = [air[q][k]["excess"] for k, _ in keys]
        bars = ax.bar(x + (j - 0.5) * w, v, w, color=col, label=qn)
        for b, vv in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, vv + 0.03, f"{vv:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys])
    ax.set_ylabel("超出零模型的 CMI（bits/步）"); ax.set_ylim(0, 2.8)


def panel_return(ax):
    x = np.arange(2); w = 0.36
    for j, ((q, qn), col) in enumerate(zip(Q, [RED, RED_L])):
        v = [air[q]["return_rate"] * 100, air[q]["return_rate_first_order"] * 100]
        bars = ax.bar(x + (j - 0.5) * w, v, w, color=col, label=qn)
        for b, vv in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, vv + 0.6, f"{vv:.0f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(["实际", "一阶链给出"])
    ax.set_ylabel("A→B→A 返程步的比例（%）"); ax.set_ylim(0, 42)


W = 14.7 / 2.54
fig, axs = plt.subplots(2, 2, figsize=(W, W * 0.88), gridspec_kw=dict(width_ratios=[1.35, 1], hspace=0.62, wspace=0.42))
panel_history(axs[0, 0]); lab(axs[0, 0], "a")
panel_taxi(axs[0, 1]); lab(axs[0, 1], "b")
panel_cmi(axs[1, 0]); lab(axs[1, 0], "c")
panel_return(axs[1, 1]); lab(axs[1, 1], "d")
fig.subplots_adjust(left=0.11, right=0.985, top=0.925, bottom=0.11)     # room above the panel letters
(HERE / "figures").mkdir(exist_ok=True)
fig.savefig(HERE / "figures" / "fig_airline_memory.png", dpi=300)

# slide version: a and b, stacked, for the conclusion slide
fig, axs = plt.subplots(2, 1, figsize=(4.6, 4.2), gridspec_kw=dict(hspace=0.75))
panel_history(axs[0], inline=True)
axs[0].set_title("航空行程：起点作输入 vs 加长历史", fontsize=10, loc="left")
panel_taxi(axs[1])
axs[1].set_title("出租车：补入方向 vs 前一格子", fontsize=10, loc="left")
fig.subplots_adjust(left=0.17, right=0.98, top=0.93, bottom=0.12)
fig.savefig(HERE / "figures" / "fig_airline_memory_slide.png", dpi=300)
print("saved figures")
