#!/usr/bin/env python3
"""Figures for the tensor-core-generations article.

fig1_gap.png   -- the 17% vs 98% gap: same BF16 GEMM on one B200, step by step
fig2_gens.png  -- five generations of MMA instruction: fewer issuing threads, accumulator moves out
fig3_tmem.png  -- Blackwell SM datapath: SMEM -> Tensor Core -> TMEM, registers freed
fig4_b300.png  -- H100 / B200 / B300 dense peak per GPU: gap only shows up in FP4
fig5_fp4.png   -- MXFP4 vs NVFP4 block scaling

All numbers in fig1/fig4/fig5 are from the sources cited in the article.
Run:  python3 gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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

DBLUE, DRED, DGREEN, DGRAY, DPURP, DGOLD = "#2288CC", "#BB3333", "#117733", "#666666", "#995588", "#b8922e"


def box(ax, x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=8.8, bold=False, tc="#222222", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.5, ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def fig1_gap():
    # gau-nernst, "tcgen05 for dummies": B200, M=N=K=4096, BF16
    labels = ["v1\n朴素写法", "v2\n用 TMA", "v3\n多级缓冲", "v4\nwarp 专门化", "v5\n2-SM MMA", "v6\nTMEM 双缓冲", "cuBLAS"]
    vals = [254, 688, 940, 1209, 1302, 1476, 1507]
    colors = [DRED, "#d97742", "#d9a441", DGOLD, "#7aa35a", DGREEN, DGRAY]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bars = ax.bar(range(len(vals)), vals, 0.62, color=colors, ec="white", lw=0.8)
    for i, (b, v) in enumerate(zip(bars, vals)):
        pct = v / 1507 * 100
        ax.text(b.get_x() + b.get_width() / 2, v + 22, f"{v}\n{pct:.0f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if i in (0, 5, 6) else "normal", linespacing=1.35)
    # 首尾连线
    ax.annotate("", xy=(6.3, 1507), xytext=(-0.3, 254),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.2, ls="--"))
    ax.text(3.0, 700, "同一块 B200,同一个峰值\n只改喂数据的方式:5.8 倍", ha="center", va="center",
            fontsize=10.5, color="#333333", fontweight="bold", linespacing=1.6,
            bbox=dict(fc="white", ec="#cccccc", pad=5))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("实测吞吐(TFLOPS)")
    ax.set_ylim(0, 1750)
    ax.set_title("同一块 B200 上,同一个 BF16 矩阵乘(4096³)逐步优化的实测", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig1_gap.png"); plt.close(fig)


def fig2_gens():
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.set_xlim(0, 25.2); ax.set_ylim(0, 10.8); ax.axis("off")
    ax.text(12.6, 10.45, "五代矩阵乘指令:发指令的线程越来越少,累加器越搬越远", ha="center",
            fontsize=12.5, fontweight="bold")

    gens = [
        ("Volta", "2017", "mma", "8 线程", "8×8×4", "寄存器", "寄存器", "#9ec5e3"),
        ("Ampere", "2020", "mma.sync", "warp(32)", "m16n8k16", "寄存器", "寄存器", "#7fb3d8"),
        ("Hopper", "2022", "wgmma", "warpgroup(128)", "m64nNk16", "A 可在 SMEM", "寄存器", "#5f9cc9"),
        ("Blackwell", "2024", "tcgen05.mma", "1 线程", "最大 128×256×16", "SMEM 或 TMEM", "TMEM", "#3f85bb"),
    ]
    x0, w, gap = 0.8, 5.6, 0.65
    for i, (name, year, instr, who, shape, operands, accum, c) in enumerate(gens):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.6), w, 9.1, boxstyle="round,pad=0.08", fc="#fafcff", ec=c, lw=1.6))
        ax.add_patch(Rectangle((x, 8.3), w, 1.4, fc=c, ec="none"))
        ax.text(x + w / 2, 9.0, f"{name}  ({year})", ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="white")
        ax.text(x + w / 2, 7.55, instr, ha="center", va="center", fontsize=10.5, fontweight="bold",
                family="monospace", color="#222222")
        rows = [("谁发指令", who), ("形状", shape), ("操作数", operands), ("累加器", accum)]
        yy = 6.35
        for k, (lab, val) in enumerate(rows):
            is_last = k >= 2
            ax.text(x + 0.3, yy, lab, ha="left", va="center", fontsize=9, color="#777777")
            ax.text(x + w - 0.3, yy, val, ha="right", va="center", fontsize=9.6,
                    fontweight="bold" if (is_last and i == 3) else "normal",
                    color=DRED if (is_last and i == 3) else "#222222")
            yy -= 1.25
    # 底部两条箭头
    ax.annotate("", xy=(25.0, 0.25), xytext=(0.8, 0.25), arrowprops=dict(arrowstyle="-|>", color=DRED, lw=1.4))
    ax.text(12.6, -0.35, "线程越来越少参与   ·   累加器从寄存器一路搬到专门的内存", ha="center",
            fontsize=9.8, color=DRED, fontweight="bold")
    fig.savefig(f"{OUT}/fig2_gens.png"); plt.close(fig)


def fig3_tmem():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"wspace": 0.1})

    def draw(ax, title, tmem):
        ax.set_xlim(0, 14.2); ax.set_ylim(0, 10.6); ax.axis("off")
        ax.text(7.1, 10.2, title, ha="center", fontsize=11, fontweight="bold")
        # 显存 → SMEM
        box(ax, 0.3, 7.5, 3.0, 1.6, "显存\n(HBM)", fc="#f3f3f3", ec="#999999", fs=9.5, bold=True)
        box(ax, 4.3, 7.5, 3.4, 1.6, "Shared Memory\n(操作数 A、B)", fc="#eef7ee", ec=DGREEN, fs=9.2, bold=True)
        arrow(ax, 3.35, 8.3, 4.25, 8.3)
        ax.text(3.8, 8.75, "TMA", ha="center", fontsize=8.4, color="#555555")
        # Tensor Core
        box(ax, 4.3, 4.2, 3.4, 1.7, "Tensor Core", fc="#fbeaea", ec=DRED, fs=10.5, bold=True)
        arrow(ax, 6.0, 7.45, 6.0, 5.95)
        if tmem:
            box(ax, 8.6, 4.2, 3.3, 1.7, "TMEM\n(累加器,256 KB)", fc="#f2d9ec", ec=DPURP, fs=9.4, bold=True)
            arrow(ax, 7.75, 5.05, 8.55, 5.05, color=DPURP)
            arrow(ax, 8.55, 4.6, 7.75, 4.6, color=DPURP)
            box(ax, 3.0, 0.6, 5.0, 1.6, "线程寄存器\n只做收尾:加偏置、量化、写回", fc="#f3f3f3", ec="#aaaaaa", fs=8.6, tc="#777777")
            arrow(ax, 9.9, 4.15, 7.6, 2.25, color=DPURP, lw=1.2)
            # 标签放在箭头的右下方,不压线
            ax.text(9.75, 2.85, "算完后用\ntcgen05.ld\n搬回寄存器", ha="left", va="center", fontsize=8.2,
                    color=DPURP, linespacing=1.45,
                    bbox=dict(fc="white", ec="none", pad=2))
        else:
            box(ax, 8.6, 4.2, 3.3, 1.7, "线程寄存器\n(累加器)", fc="#f2d9ec", ec=DPURP, fs=9.4, bold=True)
            arrow(ax, 7.75, 5.05, 8.55, 5.05, color=DPURP)
            arrow(ax, 8.55, 4.6, 7.75, 4.6, color=DPURP)
            ax.text(6.1, 1.45, "累加器占满寄存器 → 能同时驻留的 warp 变少", ha="center", fontsize=9,
                    color=DPURP, fontweight="bold")

    draw(axes[0], "Hopper(wgmma):累加器在线程寄存器里", tmem=False)
    draw(axes[1], "Blackwell(tcgen05.mma):累加器搬进 TMEM", tmem=True)
    fig.savefig(f"{OUT}/fig3_tmem.png"); plt.close(fig)


def fig4_b300():
    # NVIDIA DGX 整机规格 / 8,全部 dense
    gpus = ["H100", "B200", "B300"]
    fp8 = [1.979, 4.5, 4.5]
    fp4 = [0.0, 9.0, 13.5]
    bf16 = [0.989, 2.25, 2.25]
    x = np.arange(len(gpus)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar(x - w, bf16, w, label="FP16 / BF16", color=DBLUE, ec="white")
    b2 = ax.bar(x, fp8, w, label="FP8", color=DRED, ec="white")
    b3 = ax.bar(x + w, fp4, w, label="FP4", color=DGREEN, ec="white")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            if h == 0:
                ax.text(b.get_x() + b.get_width() / 2, 0.12, "不支持", ha="center", va="bottom",
                        fontsize=7.8, color="#999999", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.15, f"{h:.1f}", ha="center", va="bottom",
                        fontsize=8.8, fontweight="bold")
    # 标出 B200→B300 唯一的变化
    ax.annotate("", xy=(2 + w, 13.5), xytext=(1 + w, 9.0),
                arrowprops=dict(arrowstyle="-|>", color=DGREEN, lw=1.6, ls="--"))
    ax.text(1.5 + w, 11.6, "+50%", ha="center", fontsize=10, color=DGREEN, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=2))
    ax.set_xticks(x); ax.set_xticklabels(gpus, fontsize=11)
    ax.set_ylabel("每 GPU 稠密峰值(PFLOPS)")
    ax.set_ylim(0, 15.5)
    ax.legend(loc="upper left", ncol=3, fontsize=9.5)
    ax.set_title("B200 → B300:FP8 和 BF16 没变,只有 FP4 涨了", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig4_b300.png"); plt.close(fig)


def fig5_fp4():
    """Block scaling: 128 values. MXFP4 = 4 blocks of 32; NVFP4 = 8 blocks of 16."""
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 4.4), gridspec_kw={"hspace": 0.75})
    rng = np.random.default_rng(7)
    vals = rng.uniform(0.3, 1.0, 128)
    N = 128

    def draw(ax, block, scale_fmt, title, color):
        ax.set_xlim(-0.6, N - 0.4); ax.set_ylim(-1.35, 1.75); ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=22, loc="left")
        for i in range(N):
            ax.add_patch(Rectangle((i - 0.42, -0.6), 0.84, 1.2, fc=plt.cm.YlGnBu(vals[i]), ec="white", lw=0.4))
        # 组之间的分隔线
        for k in range(0, N + 1, block):
            ax.plot([k - 0.5, k - 0.5], [-0.62, 0.62], color="#555555", lw=0.7, ls=":")
        # 每组上方一条括线 + 标签
        for k in range(0, N, block):
            ax.plot([k - 0.45, k + block - 0.55], [0.78, 0.78], color=color, lw=1.6)
            ax.plot([k - 0.45, k - 0.45], [0.78, 0.9], color=color, lw=1.6)
            ax.plot([k + block - 0.55, k + block - 0.55], [0.78, 0.9], color=color, lw=1.6)
        nb = N // block
        ax.text(-0.5, 1.45, f"{N} 个数,分成 {nb} 组,每组 {block} 个数共用 1 个缩放因子({scale_fmt})",
                ha="left", va="center", fontsize=9.6, color=color, fontweight="bold")

    draw(axes[0], 32, "E8M0:只能是 2 的整数次幂", "MXFP4", DGRAY)
    draw(axes[1], 16, "E4M3:一种 FP8", "NVFP4(另外每个张量还有一个 FP32 的二级缩放)", DPURP)
    fig.savefig(f"{OUT}/fig5_fp4.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_gap()
    fig2_gens()
    fig3_tmem()
    fig4_b300()
    fig5_fp4()
    print("done ->", OUT)
