#!/usr/bin/env python3
"""Figures for the compute-pipeline article.

fig1_pipeline.png — 5-stage pipeline + 4 warps staggered so EX is busy every cycle
fig2_latency.png — 1-warp serial chain vs 4-warp rotation + concurrency ladder (log)
fig3_fma.png     — FMA datapath block diagram (Booth -> compressor tree -> align c -> CPA -> normalize)

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

DBLUE, DRED, DGREEN, DGRAY = "#2288CC", "#BB3333", "#117733", "#666666"


def fig1_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6),
                             gridspec_kw={"height_ratios": [1, 1.5]})

    # ---- (a) 5-stage chain -------------------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    stages = [("IF", "取指"), ("ID", "译码"), ("IS", "发射"), ("EX", "执行"), ("WB", "写回")]
    descs = ["从指令缓存\n取 SASS 指令", "拆成控制信号\n定执行单元", "读寄存器堆\n取操作数", "FMA 电路\n真正算数", "结果回寄存器\n记分板解锁"]
    w, gap, y0, h = 1.45, 0.32, 1.35, 0.85
    for i, ((en, zh), d) in enumerate(zip(stages, descs)):
        x = 0.6 + i * (w + gap)
        color = DBLUE if en != "EX" else DRED
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.06",
                             fc="#e8f3fb" if en != "EX" else "#fbeaea",
                             ec=color, lw=1.4)
        ax.add_patch(box)
        ax.text(x + w/2, y0 + h - 0.26, en, ha="center", fontsize=12, fontweight="bold", color=color)
        ax.text(x + w/2, y0 + h/2 - 0.16, zh, ha="center", fontsize=9, color="#333333")
        ax.text(x + w/2, y0 - 0.42, d, ha="center", va="top", fontsize=7.3, color="#666666", linespacing=1.4)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.04, y0 + h/2), (x + w + gap - 0.04, y0 + h/2),
                                         arrowstyle="-|>", mutation_scale=11, color="#888888", lw=1.1))
    ax.text(5.0, 2.95, "(a) 一条指令的五级旅程:延迟 5 拍,吞吐每拍 1 条",
            ha="center", fontsize=10.5, fontweight="bold")

    # ---- (b) staggered warps ------------------------------------------------
    ax = axes[1]
    n_warp, n_cyc = 4, 9
    ax.set_xlim(0, n_cyc + 1.4); ax.set_ylim(-0.4, n_warp + 1.15); ax.axis("off")
    stage_names = ["IF", "ID", "IS", "EX", "WB"]
    cmap = {"IF": "#d8e8f5", "ID": "#c2d9ee", "IS": "#a9c9e6", "EX": "#f2b8b8", "WB": "#e3d9c0"}
    ecmap = {"IF": "#7fb3d8", "ID": "#5f9cc9", "IS": "#3f85bb", "EX": DRED, "WB": "#b09a6a"}
    for wi in range(n_warp):
        for s in range(5):
            c = 1 + wi + s
            if c > n_cyc:
                continue
            ax.add_patch(plt.Rectangle((c + 0.06, wi + 0.06), 0.88, 0.88,
                                       fc=cmap[stage_names[s]], ec=ecmap[stage_names[s]], lw=1.0))
            ax.text(c + 0.5, wi + 0.5, stage_names[s], ha="center", va="center",
                    fontsize=8.5, color="#222222" if s != 3 else DRED,
                    fontweight="bold" if s == 3 else "normal")
        ax.text(0.82, wi + 0.5, f"W{wi}", ha="right", va="center", fontsize=10, fontweight="bold")
    for c in range(1, n_cyc + 1):
        ax.text(c + 0.5, n_warp + 0.08, str(c), ha="center", fontsize=8.5, color="#555555")
    ax.text(0.0, n_warp + 0.08, "时钟拍", fontsize=8.5, color="#555555")
    # EX busy marker
    ax.annotate("", xy=(8.06, -0.18), xytext=(4.06, -0.18),
                arrowprops=dict(arrowstyle="-|>", color=DRED, lw=1.6))
    ax.text(6.1, -0.40, "EX 每拍都被占满 → 吞吐 = 峰值", ha="center", fontsize=9.5,
            color=DRED, fontweight="bold")
    ax.text(5.7, n_warp + 0.75, "(b) 四个 warp 错开一拍进入,同一条流水线上并行推进",
            ha="center", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_pipeline.png"); plt.close(fig)


def fig2_latency():
    fig = plt.figure(figsize=(7.6, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.52, wspace=0.28)

    # ---- (a) 1 warp serial chain -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    n = 13
    for c in range(1, n + 1):
        if (c - 1) % 4 == 0:
            ax.add_patch(plt.Rectangle((c + 0.06, 0.56), 0.88, 0.88, fc="#f2b8b8", ec=DRED, lw=1.1))
            ax.text(c + 0.5, 1.0, "FMA", ha="center", va="center", fontsize=8.5, color=DRED, fontweight="bold")
        else:
            ax.add_patch(plt.Rectangle((c + 0.06, 0.56), 0.88, 0.88, fc="#f3f3f3", ec="#cccccc", lw=0.8, ls=":"))
    for c in range(1, n + 1):
        ax.text(c + 0.5, 0.30, str(c), ha="center", fontsize=8, color="#555555")
    ax.text(0.7, 1.0, "发射口", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(n / 2 + 0.5, 1.85, "(a) 单 warp 串行链:每 4 拍发 1 条", ha="center", fontsize=10, fontweight="bold")
    ax.text(n / 2 + 0.5, 1.55, "3/4 的拍在等结果,利用率 25%", ha="center", fontsize=9, color=DRED)
    ax.set_xlim(-1.2, n + 1); ax.set_ylim(0.05, 2.1); ax.axis("off")

    # ---- (b) 4 warps rotation ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    labels = ["W0", "W1", "W2", "W3"]
    for c in range(1, n + 1):
        wi = (c - 1) % 4
        y = 3 - wi
        ax.add_patch(plt.Rectangle((c + 0.06, y + 0.10), 0.88, 0.80, fc="#cfe6f7", ec=DBLUE, lw=1.1))
        ax.text(c + 0.5, y + 0.5, labels[wi], ha="center", va="center", fontsize=8.5, color=DBLUE, fontweight="bold")
    for c in range(1, n + 1):
        ax.text(c + 0.5, 0.02, str(c), ha="center", fontsize=8, color="#555555")
    ax.text(0.7, 3.5, "发射口", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(n / 2 + 0.5, 5.15, "(b) 四 warp 轮转:每拍都有就绪的 warp", ha="center", fontsize=10, fontweight="bold")
    ax.text(n / 2 + 0.5, 4.72, "W0 的结果第 5 拍回来,正好接上", ha="center", fontsize=9, color=DGREEN)
    ax.set_xlim(-1.2, n + 1); ax.set_ylim(-0.25, 5.5); ax.axis("off")

    # ---- (c) concurrency ladder ---------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ops = ["FMA\n(寄存器)", "Shared\nMemory", "L2\n命中", "HBM\n往返"]
    need = [4, 30, 200, 600]
    colors = [DGREEN, "#d9a441", "#d97742", DRED]
    bars = ax.bar(range(4), need, 0.55, color=colors, ec="white", lw=0.6)
    for b, v in zip(bars, need):
        ax.text(b.get_x() + b.get_width()/2, v * 1.15, f"需要 {v} 条在途指令",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(16, color=DBLUE, lw=1.6, ls="--")
    ax.text(3.42, 22, "每个调度器 16 个 warp\n能提供的在途指令", fontsize=8.8, color=DBLUE,
            ha="right", va="bottom", linespacing=1.5)
    ax.set_yscale("log"); ax.set_ylim(2, 1600)
    ax.set_xticks(range(4)); ax.set_xticklabels(ops, fontsize=9.5)
    ax.set_ylabel("藏住延迟所需的并发(条,对数轴)")
    ax.set_title("(c) 延迟阶梯:FMA 藏得住,DRAM 藏不住——这就是 Tiling 存在的理由", fontsize=10.5, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig2_latency.png"); plt.close(fig)


def fig3_fma():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 17.6); ax.set_ylim(0, 9.4); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=8.6, bold=False, tc="#222222"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color="#888888"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=color, lw=1.2))

    # operands (full FP32: exponent + significand)
    box(0.3, 6.7, 2.0, 0.9, "a:尾数 24 位\n指数 8 位", fc="#f5f0e6", ec="#b09a6a", bold=True)
    box(0.3, 5.2, 2.0, 0.9, "b:尾数 24 位\n指数 8 位", fc="#f5f0e6", ec="#b09a6a", bold=True)
    box(0.3, 2.6, 2.0, 0.9, "尾数 c\n(加数)", fc="#f5f0e6", ec="#b09a6a", bold=True)

    # multiply path
    box(3.0, 5.7, 2.3, 1.4, "Booth 编码\n(基 4)\n部分积 ~13 个", fs=8.2)
    arrow(2.3, 7.15, 3.0, 6.65); arrow(2.3, 5.65, 3.0, 6.15)
    ax.text(2.65, 7.35, "尾数", fontsize=7.5, color="#999999")
    box(6.0, 5.7, 2.4, 1.4, "Wallace 树\n4:2 压缩器\n→ carry-save", fs=8.2)
    arrow(5.3, 6.4, 6.0, 6.4)
    box(6.0, 4.15, 2.4, 1.0, "48 位积\n(S, C 两数)", fc="#eef7ee", ec=DGREEN, fs=8.2)
    arrow(7.2, 5.7, 7.2, 5.15)

    # addend path
    box(3.0, 2.45, 2.3, 1.2, "对齐移位器\n按指数差移位", fs=8.2)
    arrow(2.3, 3.05, 3.0, 3.05)
    box(6.0, 2.45, 2.4, 1.2, "3:2 压缩\n并入树中", fc="#f2d9ec", ec="#995588", fs=8.2)
    arrow(5.3, 3.05, 6.0, 3.05)
    arrow(7.2, 3.65, 7.2, 4.15, color="#995588")

    # CPA + normalize
    box(9.3, 4.6, 2.5, 1.6, "进位传播加法\nCPA ~53 位\n(一次算完)", fs=8.2)
    arrow(8.4, 4.9, 9.3, 5.2)
    box(12.6, 4.6, 2.2, 1.6, "前导零检测\n+ 规格化移位", fs=8.2)
    arrow(11.8, 5.4, 12.6, 5.4)
    box(15.4, 4.6, 1.9, 1.6, "舍入\n(RNE)\n→ 结果", fc="#eef7ee", ec=DGREEN, bold=True, fs=8.6)
    arrow(14.8, 5.4, 15.4, 5.4)

    # fused note
    ax.text(6.9, 1.55, "Fused 的含义:乘积不做最终加法、不中间舍入,\n加数 c 直接并入压缩树——乘与加融合成一步",
            ha="center", fontsize=9, color="#995588", linespacing=1.5)
    ax.add_patch(FancyArrowPatch((7.2, 2.15), (7.2, 2.45), arrowstyle="-|>",
                                 mutation_scale=10, color="#995588", lw=1.0, linestyle="--"))

    # exponent side path
    box(3.0, 7.6, 2.3, 0.9, "指数相加\n8 位", fc="#f3f3f3", ec="#999999", fs=8.2, tc="#555555")
    arrow(2.3, 7.5, 3.0, 7.9, color="#aaaaaa")
    ax.text(2.62, 7.78, "指数", fontsize=7.5, color="#999999")
    arrow(4.15, 7.6, 4.15, 7.1, color="#aaaaaa")
    ax.text(5.35, 8.05, "指数通路(小):指数相加、对齐量计算", fontsize=8.2, color="#777777", ha="left")

    # pipeline cuts
    for x, lab in [(2.85, "拍 1 |"), (5.85, "拍 2 |"), (8.9, "拍 3 |"), (12.3, "拍 4 |")]:
        ax.plot([x, x], [1.9, 8.85], color="#bbbbbb", lw=0.9, ls="--")
        ax.text(x - 0.12, 8.85, lab, fontsize=8.5, color="#888888", ha="right", va="bottom")
    ax.text(8.8, 0.9, "整条通路切成流水级,一拍推进一段:延迟 ~4 拍,吞吐每拍 1 个结果",
            ha="center", fontsize=9.5, color="#333333", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_fma.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_latency()
    fig3_fma()
    print("done ->", OUT)
