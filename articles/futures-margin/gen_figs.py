#!/usr/bin/env python3
"""Figures for the futures-margin article (academic style).

fig1_collateral.png  — bank loan vs futures margin funding structure
fig2_waterfall.png   — default waterfall (four layers of loss absorption)
fig3_spiral.png      — deleveraging spiral (causal loop)
fig4_solvency.png    — solvency x liquidity 2x2 quadrant

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

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

BLUE, RED, GREEN, ORANGE, GREY = "#4477AA", "#EE6677", "#228833", "#CCBB44", "#BBBBBB"
DBLUE, DRED, DGREEN = "#2288CC", "#BB3333", "#117733"


def rrect(ax, x, y, w, h, ec, fc="white", lw=1.2, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def arr(ax, x1, y1, x2, y2, color="#333333", lw=1.4, ms=16, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


# ================================================================ fig 1
def fig1_collateral():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # (a) bank loan
    ax = axes[0]
    ax.set_title("(a) 银行贷款买资产", fontsize=10.5, pad=8)
    rrect(ax, 2.2, 0.8, 5.6, 2.6, "#555555", "#e8e8e8")
    ax.text(5.0, 2.1, "自有资金 1 万", ha="center", va="center", fontsize=9.5)
    rrect(ax, 2.2, 3.7, 5.6, 5.4, DRED, "#fdeaea")
    ax.text(5.0, 6.4, "银行借款 9 万\n(真实垫付的资金,付利息)", ha="center",
            va="center", fontsize=9.5, linespacing=1.6)
    arr(ax, 8.0, 6.4, 9.3, 6.4, DRED)
    ax.text(9.45, 6.4, "9 万\n利息", fontsize=8.5, color=DRED, va="center",
            ha="left", linespacing=1.5)
    rrect(ax, 0.6, 4.6, 1.3, 1.8, "#555555", "white")
    ax.text(1.25, 5.5, "银行", ha="center", va="center", fontsize=9, rotation=90)
    ax.text(5.0, 9.6, "资产 10 万(买断,归你)", ha="center", fontsize=9.5)

    # (b) futures
    ax = axes[1]
    ax.set_title("(b) 期货合约", fontsize=10.5, pad=8)
    rrect(ax, 2.2, 0.8, 5.6, 2.6, DGREEN, "#eaf4ea")
    ax.text(5.0, 2.1, "保证金 1 万(押金)", ha="center", va="center", fontsize=9.5)
    rrect(ax, 2.2, 3.7, 5.6, 5.4, "#999999", "white", ls=(0, (4, 3)))
    ax.text(5.0, 6.4, "名义价值 9 万\n(无人出资,不付利息)", ha="center",
            va="center", fontsize=9.5, linespacing=1.6, color="#777777")
    ax.text(5.0, 9.6, "合约 10 万(立约,非买断)", ha="center", fontsize=9.5)

    fig.suptitle("同样『1 万撬动 10 万』,资金结构完全不同:借款要付息,担保不付息",
                 fontsize=11, y=1.02)
    fig.savefig(f"{OUT}/fig1_collateral.png"); plt.close(fig)


# ================================================================ fig 2
def fig2_waterfall():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")

    layers = [
        ("第 1 层", "违约者保证金与抵押品", "¥10,000 被直接吞掉", DBLUE, 7.4),
        ("第 2 层", "清算会员(期货公司)法人资金", "承接客户穿仓缺口", "#3b7ea1", 5.0),
        ("第 3 层", "违约保障基金 + 清算所资本", "全体会员共同缴纳的风险池", "#7aa6c2", 2.6),
        ("第 4 层", "系统性风险", "损失分摊、连锁违约、流动性危机", DRED, 0.3),
    ]
    for i, (tag, name, desc, color, y) in enumerate(layers):
        w = 11.5 - i * 1.2
        rrect(ax, 1.2, y, w, 1.9, color, "white", lw=1.4)
        ax.text(1.7, y + 1.32, tag, fontsize=8.5, color=color)
        ax.text(1.7, y + 0.82, name, fontsize=10, fontweight="bold", color="#222222")
        ax.text(1.7, y + 0.32, desc, fontsize=8.5, color="#666666")
        if i < 3:
            arr(ax, 12.9, y, 12.9, y - 0.45, color, lw=1.6, ms=15)
    ax.text(13.15, 5.0, "吸收不完,\n逐层下渗", fontsize=8.5, color="#888888",
            rotation=90, va="center", ha="left")

    ax.text(0.4, 9.7, "Default Waterfall:一笔 2 万元穿仓缺口的兜底顺序",
            fontsize=11, fontweight="bold")
    fig.savefig(f"{OUT}/fig2_waterfall.png"); plt.close(fig)


# ================================================================ fig 3
def fig3_spiral():
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    nodes = [
        (5.0, 8.3, "资产价格下跌", DBLUE),
        (8.2, 5.0, "触发追加保证金\n(Margin Call)", DRED),
        (5.0, 1.7, "为筹现金,被迫抛售\n其他资产", DRED),
        (1.8, 5.0, "抛售压低更多资产价格\n抵押品贬值", DBLUE),
    ]
    for x, y, s, c in nodes:
        rrect(ax, x - 1.85, y - 1.0, 3.7, 2.0, c, "white", lw=1.5)
        ax.text(x, y, s, ha="center", va="center", fontsize=9.5, linespacing=1.6)

    seq = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for a, b in seq:
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        # arc arrows between adjacent nodes
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        import math
        ang = math.atan2(yb - ya, xb - xa)
        ox, oy = 0.55 * math.sin(ang), -0.55 * math.cos(ang)
        arr(ax, xa + 1.4 * math.cos(ang) + ox, ya + 1.4 * math.sin(ang) + oy,
            xb - 1.4 * math.cos(ang) + ox, yb - 1.4 * math.sin(ang) + oy,
            DRED, lw=1.8, ms=18)

    ax.text(5.0, 5.0, "Deleveraging\nSpiral", ha="center", va="center",
            fontsize=13, color=DRED, fontweight="bold", linespacing=1.5)
    ax.text(5.0, 0.35, "循环自我强化:传染不需要逻辑,只需要现金",
            ha="center", fontsize=9, color="#666666")
    fig.savefig(f"{OUT}/fig3_spiral.png"); plt.close(fig)


# ================================================================ fig 4
def fig4_solvency():
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)

    # quadrant fills
    ax.fill_between([-2.2, 0], 0, 2.2, color="#fdeaea", zorder=0)
    ax.fill_between([0, 2.2], 0, 2.2, color="#eaf4ea", zorder=0)
    ax.fill_between([-2.2, 0], -2.2, 0, color="#f2f2f2", zorder=0)
    ax.fill_between([0, 2.2], -2.2, 0, color="#fff8e6", zorder=0)
    ax.axhline(0, color="#555555", lw=1.2)
    ax.axvline(0, color="#555555", lw=1.2)

    labels = [
        (1.1, 1.55, "健康", "资产足、现金足", DGREEN),
        (1.1, -1.5, "流动性危机", "资产 > 负债,但变不了现\n(贝尔斯登, 2008)", DRED),
        (-1.1, 1.55, "资不抵债边缘", "现金暂时够,算总账要完\n(Solvency 危机)", "#996600"),
        (-1.1, -1.5, "双重危机", "既没钱也没资产", "#666666"),
    ]
    for x, y, t, d, c in labels:
        ax.text(x, y + 0.18, t, ha="center", va="center", fontsize=12,
                fontweight="bold", color=c)
        ax.text(x, y - 0.34, d, ha="center", va="center", fontsize=8.5,
                color="#555555", linespacing=1.5)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("流动性(现在拿得出现金吗)→", fontsize=10.5)
    ax.set_ylabel("偿付能力(最终资产够还债吗)→", fontsize=10.5)
    ax.set_title("最危险的象限:有偿付能力,没有流动性", fontsize=11.5, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/fig4_solvency.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_collateral(); print("fig1 ok")
    fig2_waterfall(); print("fig2 ok")
    fig3_spiral(); print("fig3 ok")
    fig4_solvency(); print("fig4 ok")
    print("->", OUT)


# ================================================================ fig 5
def fig5_leverage():
    """(a) sqrt-T risk slicing: daily settlement vs hold-to-expiry buffer;
    (b) equity curve of a 10x position: margin-call line vs bankruptcy."""
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    # (a) sigma * sqrt(T) vs daily sigma
    ax = axes[0]
    T = np.linspace(1, 100, 300)
    sigma = 1.0  # daily vol in %
    ax.plot(T, sigma * np.sqrt(T), color=DBLUE, lw=1.6,
            label=r"$\sigma_{\mathrm{daily}}\sqrt{T}$(无每日结算需覆盖)")
    ax.axhline(sigma, color=DGREEN, lw=1.4, ls="--",
               label=r"$\sigma_{\mathrm{daily}}$(每日结算只需覆盖)")
    ax.axhline(10, color=DRED, lw=1.0, ls=":",
               label="10% 保证金线")
    ax.annotate(r"$\sqrt{90}\approx 9.5\times$", xy=(90, sigma * np.sqrt(90)),
                xytext=(48, 4.2), fontsize=9, color=DBLUE,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=DBLUE))
    ax.set_xlabel("持仓天数 $T$")
    ax.set_ylabel("需覆盖的波动(%)")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=7.2, loc="lower right")
    ax.set_title("(a) 每日结算把风险切成单日", fontsize=10, pad=6)

    # (b) equity vs adverse move
    ax = axes[1]
    x = np.linspace(0, 14, 300)          # adverse move in %
    equity = 1 - x / 10                   # 10x leverage, margin=1
    ax.plot(x, equity, color="#333333", lw=1.6, label="账户权益(保证金倍数)")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axhline(0.7, color=DGREEN, lw=1.2, ls="--", label="维持保证金线(0.7)")
    mc_x = 10 - 7                         # equity = 0.7 at x = 3%
    ax.axvline(mc_x, color=DGREEN, lw=1.0, ls=(0, (3, 2)))
    ax.axvline(10, color=DRED, lw=1.0, ls=(0, (3, 2)))
    ax.fill_between(x, 0, 1.05, where=x > 10, color="#fdeaea", zorder=0)
    ax.text(12.0, 0.88, "穿仓区\n(权益为负)", ha="center", fontsize=8,
            color=DRED, linespacing=1.5)
    ax.text(mc_x - 0.3, 0.06, "强平点\n逆向 3%", ha="right", fontsize=8,
            color=DGREEN, linespacing=1.5)
    ax.text(10.35, -0.42, "保证金\n归零", ha="left", fontsize=8,
            color="#888888", linespacing=1.5)
    ax.set_xlabel("价格逆向幅度(%)")
    ax.set_ylabel("权益 / 初始保证金")
    ax.set_xlim(0, 14); ax.set_ylim(-0.75, 1.15)
    ax.legend(fontsize=7.2, loc="upper right")
    ax.set_title("(b) 10 倍杠杆的强平与穿仓", fontsize=10, pad=6)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(f"{OUT}/fig5_leverage.png"); plt.close(fig)
