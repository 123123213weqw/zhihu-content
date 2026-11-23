#!/usr/bin/env python3
"""Figures for the loop-transformer article (GLM-5.4).

fig_ablation.png  -- (copied, NOT generated) official 30B iso-parameter ablation table from the GLM-5.4 blog
fig1_anatomy.png  -- 78 physical layers -> 106 executed layers: prelude / loop x2 / coda
fig2_gate.png     -- gated progressive update: x2 = x1 + b2 (h2 - x1), as a convex mix of three signals
fig3_axes.png     -- three orthogonal axes: MoE (width), linear attention (sequence), loop (depth)

Run:  python3 gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

DBLUE, DRED, DGREEN, DGRAY, DPURP = "#2288CC", "#BB3333", "#117733", "#666666", "#995588"


def box(ax, x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=8.6, bold=False, tc="#222222", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.4, ls="-", ms=11):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls))


def fig1_anatomy():
    """78 physical layers -> 106 executed. Only the middle 28 layers are executed twice."""
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 28.0); ax.set_ylim(0, 14.0); ax.axis("off")

    ax.text(14.0, 13.55, "GLM-5.4:78 层权重,一次前向走 106 层", ha="center", fontsize=14, fontweight="bold")
    ax.text(14.0, 12.8, "参数一个没多,只是中间 28 层跑了两遍", ha="center", fontsize=10.5, color="#555555")

    U = 0.075          # data units per layer
    BASE = 2.6         # y of layer 1 (stacks grow upward)
    W = 3.4
    PRE = ("前奏", 26, "#cfe0f2", DBLUE, "#174a73")
    LOOP = ("循环段", 28, "#f2d9d9", DRED, "#7a1f1f")
    POST = ("尾声", 24, "#e5dcc4", "#a08a4a", "#5c4c1c")

    def block(xc, y, spec, label=None, bold=False, lw=1.4, fc=None):
        name, n, f, ec, tc = spec
        hh = n * U
        ax.add_patch(plt.Rectangle((xc - W / 2, y), W, hh - 0.06, fc=fc or f, ec=ec, lw=lw))
        ax.text(xc, y + (hh - 0.06) / 2, label or f"{name} {n} 层", ha="center", va="center",
                fontsize=9.6, color=tc, fontweight="bold" if bold else "normal")

    def column(xc, title, tc, loop_passes):
        ax.text(xc, 11.9, title, ha="center", fontsize=11.5, fontweight="bold", color=tc)
        y = BASE
        block(xc, y, PRE); y += 26 * U
        yloop = []
        for p in range(loop_passes):
            yloop.append(y)
            lab = f"循环段 {28} 层" if loop_passes == 1 else f"循环段 {28} 层 · 第 {p + 1} 遍"
            block(xc, y, LOOP, label=lab, bold=True,
                  fc=("#fbeaea" if p == 1 else None)); y += 28 * U
        block(xc, y, POST); y += 24 * U
        total = int(round((y - BASE) / U))
        ax.text(xc, y + 0.22, f"一次前向共 {total} 层", ha="center", fontsize=10.5,
                color="#333333", fontweight="bold")
        # 层号标注
        ax.text(xc - W / 2 - 0.2, BASE + 13 * U, "L1–26", ha="right", va="center", fontsize=8.2, color="#777777")
        ax.text(xc - W / 2 - 0.2, yloop[0] + 14 * U, "L27–54", ha="right", va="center", fontsize=8.2, color="#777777")
        ax.text(xc - W / 2 - 0.2, y - 12 * U, "L55–78", ha="right", va="center", fontsize=8.2, color="#777777")
        return yloop

    xa, xb = 4.6, 15.2
    column(xa, "基座 GLM-5:不循环", "#555555", 1)
    yloop = column(xb, "GLM-5.4:循环段跑两遍", DRED, 2)

    # 同一套权重:右侧方括号把两遍连起来
    (ya, yb) = yloop
    xbr = xb + W / 2 + 0.22
    y_bot, y_top = ya + 0.06, yb + 28 * U - 0.11          # 第 1 遍的底 到 第 2 遍的顶
    ax.plot([xbr, xbr + 0.3, xbr + 0.3, xbr], [y_bot, y_bot, y_top, y_top], color=DRED, lw=1.7)
    ax.text(xbr + 0.45, (y_bot + y_top) / 2, "同一套权重\n没有新增参数",
            ha="left", va="center", fontsize=10, color=DRED, fontweight="bold", linespacing=1.5)

    # 中间箭头
    ymid = BASE + 39 * U
    ax.text((xa + xb) / 2, ymid + 1.35, "改造:\n中间 28 层\n跑两遍", ha="center", va="center", fontsize=9.6,
            color="#333333", linespacing=1.5, bbox=dict(fc="white", ec="#999999", pad=4))
    ax.add_patch(FancyArrowPatch((xa + W / 2 + 0.15, ymid), (xb - W / 2 - 0.95, ymid),
                                 arrowstyle="-|>", mutation_scale=14, color="#777777", lw=1.6))

    # 右侧:账
    bx, bw = 20.4, 7.2
    box(ax, bx, 2.6, bw, 8.6, "", fc="#fbfbfb", ec="#bbbbbb", lw=1.0)
    ax.text(bx + bw / 2, 10.55, "账", ha="center", fontsize=11.5, fontweight="bold")
    c1, c2 = bx + 4.4, bx + 6.1
    ax.text(bx + 0.35, 9.9, "基座  →  GLM-5.4", ha="left", fontsize=9, color="#777777")
    rows = [("参数层数", "78", "78", DGRAY),
            ("执行深度", "78", "106", DRED),
            ("每 token 计算量", "×1.00", "×1.36", DRED),
            ("循环段占执行次数", "—", "52.8%", DRED)]
    yy = 8.95
    for name, a, b, c in rows:
        ax.text(bx + 0.35, yy, name, ha="left", va="center", fontsize=10, color="#333333")
        ax.text(c1, yy, a, ha="right", va="center", fontsize=10, color=DGRAY)
        ax.text(c2, yy, b, ha="right", va="center", fontsize=12.5, color=c, fontweight="bold")
        ax.text((c1 + c2) / 2 - 0.1, yy, "→", ha="center", va="center", fontsize=10, color="#999999")
        yy -= 1.25
    ax.text(bx + bw / 2, 3.35, "26 + 28×2 + 24 = 106\n28×2 / 106 = 52.8%", ha="center", va="center",
            fontsize=9.2, color="#666666", linespacing=1.6)

    # 底:decode 时的访存
    ax.text(14.0, 1.5, "decode 时,每个 token 都要走完这一趟:", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(14.0, 0.75, "循环段那 28 层的权重从显存读 1 次、用 2 次 —— 多出来的是计算,不是访存",
            ha="center", fontsize=10, color=DRED, fontweight="bold")
    fig.savefig(f"{OUT}/fig1_anatomy.png"); plt.close(fig)


def fig2_gate():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.22})

    # ---- (a) 融合公式的示意 ----
    ax = axes[0]
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(6.1, 9.55, "(a) 门控渐进更新:最终状态是三路信号的加权和", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(6.1, 8.95, r"$x_2 = (1-\beta_1)(1-\beta_2)\,x_0 + \beta_1(1-\beta_2)\,h_1 + \beta_2\,h_2$",
            ha="center", fontsize=12.5)
    box(ax, 0.4, 6.4, 2.4, 1.1, "原始输入 $x_0$", fc="#f5f0e6", ec="#b09a6a", fs=9)
    box(ax, 0.4, 4.6, 2.4, 1.1, "第 1 遍结论 $h_1$", fc="#f2d9d9", ec=DRED, fs=9)
    box(ax, 0.4, 2.8, 2.4, 1.1, "第 2 遍结论 $h_2$", fc="#fbeaea", ec=DRED, fs=9)
    box(ax, 7.6, 4.4, 3.6, 1.9, "最终状态 $x_2$\n送入尾声 24 层", fc="#e8f3fb", ec=DBLUE, fs=9.6, bold=True)
    arrow(ax, 2.85, 6.95, 7.55, 5.75); arrow(ax, 2.85, 5.15, 7.55, 5.35); arrow(ax, 2.85, 3.35, 7.55, 4.95)
    ax.text(5.0, 6.85, r"权重 $(1-\beta_1)(1-\beta_2)$", ha="center", fontsize=8.4, color="#7a6a3a")
    ax.text(5.0, 5.35, r"权重 $\beta_1(1-\beta_2)$", ha="center", fontsize=8.4, color=DRED)
    ax.text(5.0, 3.75, r"权重 $\beta_2$", ha="center", fontsize=8.4, color=DRED)
    ax.text(6.1, 1.55, r"β 是逐 token 的门,模型自己学。$\beta_1$ 小 → 第一遍写得保守;$\beta_2$ 大 → 第二遍完整写入。",
            ha="center", fontsize=8.8, color="#444444")

    # ---- (b) 两个门的典型行为:示意 ----
    ax = axes[1]
    b1 = np.linspace(0.0, 1.0, 200)
    b2 = np.linspace(0.0, 1.0, 200)
    B1, B2 = np.meshgrid(b1, b2)
    W2 = B2                      # 第二遍结论在最终状态里的权重
    cs = ax.contourf(B1, B2, W2, levels=np.linspace(0, 1, 11), cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
    ax.contour(B1, B2, W2, levels=[0.5], colors="k", linewidths=0.9, linestyles="--")
    ax.plot([0.25], [0.8], "o", ms=9, color="k", zorder=5)
    # 官方只给了定性的分工(第一遍门值小、第二遍开大;难 token 第一遍写得更多),没有公布数值。
    # 下面的点只用来示意方向,不是实测值。
    ax.plot([0.25], [0.8], "o", ms=9, color="k", zorder=5)
    ax.text(0.635, 0.925, "区域 ①:第一遍保守、第二遍放开\n(官方描述的分工;点的位置仅示意)",
            ha="center", va="center", fontsize=8.4, linespacing=1.45,
            bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))
    ax.plot([0.6], [0.5], "s", ms=8, color="k", zorder=5)
    ax.text(0.31, 0.33, "区域 ②:$\\beta_1$ 更大\n(难 token:第一遍就写得多)",
            ha="center", va="center", fontsize=8.4, linespacing=1.45,
            bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))
    ax.set_xlabel(r"第一遍的门 $\beta_1$"); ax.set_ylabel(r"第二遍的门 $\beta_2$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(b) 颜色 = 第 2 遍结论在最终状态里的占比;两个点只示意方向,非实测值", fontsize=10, fontweight="bold", pad=8)
    cb = fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1]); cb.ax.tick_params(labelsize=8)
    fig.savefig(f"{OUT}/fig2_gate.png"); plt.close(fig)


def fig3_axes():
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_xlim(0, 18.4); ax.set_ylim(0, 10.4); ax.axis("off")
    ax.text(9.2, 9.95, "三个正交的轴:改的是不同维度,可以叠着用", ha="center", fontsize=12, fontweight="bold")

    cols = [
        ("MoE", "改宽度", "每层放很多专家\n每个 token 只走几个", "参数多\n每 token 算得少", DPURP, "#f2d9ec"),
        ("线性注意力", "改序列方向", "上下文压成\n固定大小的状态", "上下文长\n状态不涨", DGREEN, "#eef7ee"),
        ("循环深度", "改深度方向", "同一批层\n多走几遍", "参数不变\n每 token 算得多", DRED, "#fbeaea"),
    ]
    for i, (name, what, how, trade, ec, fc) in enumerate(cols):
        x = 0.6 + i * 6.1
        box(ax, x, 2.3, 5.0, 6.7, "", fc=fc, ec=ec, lw=1.5)
        ax.text(x + 2.5, 8.15, name, ha="center", fontsize=13, fontweight="bold", color=ec)
        ax.text(x + 2.5, 7.35, what, ha="center", fontsize=9.6, color="#555555")
        ax.text(x + 2.5, 5.75, how, ha="center", va="center", fontsize=10, color="#222222", linespacing=1.6)
        ax.text(x + 2.5, 3.35, trade, ha="center", va="center", fontsize=10, color=ec,
                fontweight="bold", linespacing=1.6)

    ax.text(9.2, 1.2, "Geiping et al.:循环深度擅长学习推理模式,MoE 擅长高效地存储和检索复杂信息。",
            ha="center", fontsize=9.6, color="#444444", style="italic")
    fig.savefig(f"{OUT}/fig3_axes.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_anatomy()
    fig2_gate()
    fig3_axes()
    print("done ->", OUT)
