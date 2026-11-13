#!/usr/bin/env python3
"""Figures for the short-selling article.

fig1_payoff.png — long vs short payoff asymmetry (two panels)
fig2_pnl.png   — famous short-seller P&L bars (Tesla 2020/2022, market 2023.11)

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

DBLUE, DRED, DGREEN = "#2288CC", "#BB3333", "#117733"


def fig1_payoff():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    P = np.linspace(0, 400, 400)  # stock price, entry = 100

    ax = axes[0]
    ax.plot(P, P - 100, color=DBLUE, lw=1.8)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(100, color="#bbbbbb", lw=0.7, ls=":")
    ax.fill_between(P, -100, P - 100, where=(P - 100 > 0), color="#eaf4ea", zorder=0)
    ax.text(300, 250, "盈利无上限", fontsize=10, color=DGREEN, ha="center")
    ax.text(300, -78, "亏损封底 −100%", fontsize=9.5, color="#555555", ha="center")
    ax.set_title("(a) 做多:亏有限,赚无限", fontsize=10.5, pad=6)

    ax = axes[1]
    ax.plot(P, 100 - P, color=DRED, lw=1.8)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(100, color="#bbbbbb", lw=0.7, ls=":")
    ax.fill_between(P, 0, 100 - P, where=(100 - P < 0), color="#fdeaea", zorder=0)
    ax.text(300, -250, "亏损无下限", fontsize=10, color=DRED, ha="center")
    ax.text(300, 78, "盈利封顶 +100%", fontsize=9.5, color="#555555", ha="center")
    ax.set_title("(b) 做空:赚有限,亏无限", fontsize=10.5, pad=6)

    for ax in axes:
        ax.set_xlabel("股价(入场价 = 100)")
        ax.set_ylim(-400, 320)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("盈亏")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_payoff.png"); plt.close(fig)


def fig2_pnl():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["特斯拉空头\n2020(股价+740%)", "特斯拉空头\n2022", "全市场空头\n2023.11 单月"]
    vals = [-40.1, 15.0, -43.2]
    colors = [DRED, DGREEN, DRED]
    bars = ax.bar(range(3), vals, 0.5, color=colors, ec="white", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (2 if v > 0 else -2),
                f"{v:+.0f} 亿美元", ha="center",
                va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#555555", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("账面盈亏(十亿美元)")
    ax.set_ylim(-60, 30)
    ax.set_title("著名的空头账本:亏的月份以几十亿计,赚的年份凤毛麟角", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_pnl.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_payoff(); print("fig1 ok")
    fig2_pnl(); print("fig2 ok")
    print("->", OUT)
