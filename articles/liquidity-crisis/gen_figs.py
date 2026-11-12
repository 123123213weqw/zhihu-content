#!/usr/bin/env python3
"""Figures for the liquidity-crisis article (popular style, academic rendering).

fig1_run.png       — bank-run self-fulfilling loop (village store story)
fig2_money.png     — where the money went: transferred vs burned (waterfall)

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

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

DBLUE, DRED, DGREEN, DORANGE = "#2288CC", "#BB3333", "#117733", "#B8860B"


def rrect(ax, x, y, w, h, ec, fc="white", lw=1.3, z=2):
    ax.add_patch(Rectangle((x, y), w, h, ec=ec, fc=fc, lw=lw, zorder=z))


def arr(ax, x1, y1, x2, y2, color="#333333", lw=1.6, ms=17):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=4))


# ================================================================ fig 1
def fig1_run():
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    nodes = [
        (5.0, 8.5, "有人喊:\n\"老板要跑路!\"", DORANGE),
        (8.0, 5.4, "所有人冲去退卡\n(先到先得)", DBLUE),
        (5.0, 2.3, "现金不够\n被迫贱卖资产", DRED),
        (2.0, 5.4, "越卖越便宜\n店\"真的\"要倒了", DRED),
    ]
    for x, y, s, c in nodes:
        rrect(ax, x - 1.75, y - 1.05, 3.5, 2.1, c, "white")
        ax.text(x, y, s, ha="center", va="center", fontsize=10, linespacing=1.6)

    seq = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for a, b in seq:
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        ang = math.atan2(yb - ya, xb - xa)
        ox, oy = 0.5 * math.sin(ang), -0.5 * math.cos(ang)
        arr(ax, xa + 1.55 * math.cos(ang) + ox, ya + 1.55 * math.sin(ang) + oy,
            xb - 1.55 * math.cos(ang) + ox, yb - 1.55 * math.sin(ang) + oy, "#888888")

    ax.text(5.0, 5.4, "谣言\n自我实现", ha="center", va="center",
            fontsize=12, color=DRED, fontweight="bold", linespacing=1.6)
    ax.text(5.0, 0.5, "挤兑:店不是死于\"不值钱\",是死于\"大家同时要钱\"",
            ha="center", fontsize=9.5, color="#555555")
    fig.savefig(f"{OUT}/fig1_run.png"); plt.close(fig)


# ================================================================ fig 2
def fig2_money():
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    # 100 yuan lost -> split
    rrect(ax, 0.3, 3.4, 2.6, 1.5, "#555555", "#f0f0f0")
    ax.text(1.6, 4.15, "你亏掉的", ha="center", va="center", fontsize=10.5)
    ax.text(1.6, 3.75, "100 块", ha="center", va="center", fontsize=12,
            fontweight="bold")

    rrect(ax, 4.4, 5.0, 2.5, 1.4, DGREEN, "#eaf4ea")
    ax.text(5.65, 5.9, "先跑的人", ha="center", va="center", fontsize=9.5)
    ax.text(5.65, 5.42, "拿走", ha="center", va="center", fontsize=10.5, fontweight="bold")
    rrect(ax, 7.5, 5.0, 2.2, 1.4, DGREEN, "#eaf4ea")
    ax.text(8.6, 5.9, "做空者", ha="center", va="center", fontsize=9.5)
    ax.text(8.6, 5.42, "赚走", ha="center", va="center", fontsize=10.5, fontweight="bold")
    rrect(ax, 4.4, 3.2, 2.5, 1.4, DGREEN, "#eaf4ea")
    ax.text(5.65, 4.1, "抄底买家", ha="center", va="center", fontsize=9.5)
    ax.text(5.65, 3.62, "贱买", ha="center", va="center", fontsize=10.5, fontweight="bold")
    rrect(ax, 7.5, 3.2, 2.2, 1.4, DGREEN, "#eaf4ea")
    ax.text(8.6, 4.1, "收抵押方", ha="center", va="center", fontsize=9.5)
    ax.text(8.6, 3.62, "收走", ha="center", va="center", fontsize=10.5, fontweight="bold")
    rrect(ax, 6.0, 1.2, 3.7, 1.5, DRED, "#fdeaea")
    ax.text(7.85, 2.0, "恐慌烧掉的(谁都没拿到)", ha="center", va="center",
            fontsize=10, color=DRED)
    ax.text(7.85, 1.55, "急卖折价 · 价值蒸发", ha="center", va="center",
            fontsize=9.5, color="#777777")

    arr(ax, 2.9, 4.4, 4.35, 5.5, DGREEN, lw=1.3, ms=13)
    arr(ax, 2.9, 4.4, 4.35, 4.0, DGREEN, lw=1.3, ms=13)
    arr(ax, 2.9, 4.1, 7.45, 5.5, DGREEN, lw=1.0, ms=11)
    arr(ax, 2.9, 4.0, 7.45, 3.9, DGREEN, lw=1.0, ms=11)
    arr(ax, 2.9, 3.7, 5.95, 1.95, DRED, lw=1.6, ms=15)

    ax.text(9.9, 0.35, "左:转移(零和)  右下:烧毁(负和,总财富真变少了)",
            ha="right", fontsize=8.5, color="#777777")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    fig.savefig(f"{OUT}/fig2_money.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_run(); print("fig1 ok")
    fig2_money(); print("fig2 ok")
    print("->", OUT)
