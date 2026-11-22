#!/usr/bin/env python3
"""Figures for the attention-on-GPU article.

fig1_roofline.png — H100 roofline: prefill vs decode vs linear-attention positions
fig2_kv.png       — KV cache evolution across attention generations (log bars)
fig3_chunk.png    — naive serial recurrence vs chunked parallel scan

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

DBLUE, DRED, DGREEN, DGRAY, DORANGE = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742"

PEAK = 989.0   # TFLOPS FP16 dense
BW = 3.35      # TB/s HBM


def fig1_roofline():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ai = np.logspace(-1, 4, 300)
    attain = np.minimum(PEAK, ai * BW)
    ax.plot(ai, attain, color="#333333", lw=2.0, label="H100 屋顶线")
    ax.axvline(PEAK / BW, color="#999999", lw=1.0, ls="--")
    ax.text(PEAK / BW * 1.15, 1.6, "拐点 ≈ 295 FLOP/B", fontsize=9, color="#666666", rotation=90, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))

    pts = [
        (1.0,   1 * BW, DRED,   "Decode GEMV\nbatch=1 (AI=1)", (12, 8)),
        (64.0,  64 * BW, DORANGE, "Decode\nbatch=64", (12, -34)),
        (100.0, min(PEAK, 100 * BW), "#d9a441", "线性注意力\n分块扫描", (26, 18)),
        (500.0, PEAK, DBLUE,  "Prefill GEMM\n(AI≈500)", (-16, -44)),
    ]
    for x, y, c, lab, off in pts:
        ax.plot(x, y, "o", ms=8, color=c, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=off, fontsize=8.6, color=c,
                    ha="left" if off[0] > 0 else "right", linespacing=1.4,
                    bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.5), zorder=6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("算术强度(FLOP / Byte,对数轴)")
    ax.set_ylabel("可达算力(TFLOPS,对数轴)")
    ax.set_xlim(0.1, 3e4); ax.set_ylim(0.5, 3000)
    ax.text(0.13, 700, "内存受限区\n(带宽的囚徒)", fontsize=9.5, color=DRED, linespacing=1.5)
    ax.text(3500, 700, "计算受限区\n(算力拉满)", fontsize=9.5, color=DGREEN, ha="right", linespacing=1.5)
    ax.axhspan(0.5, 3.35, xmax=0.06, color="#f5e3e3", zorder=0)
    ax.text(1.0, 2.2, "decode 上限 ≈ 239 tok/s (7B)", fontsize=8.2, color=DRED)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_roofline.png"); plt.close(fig)


def fig2_kv():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = ["MHA\n(Llama-2-7B)", "GQA\n(Llama-3-8B, 8 组 KV)", "MLA\n(DeepSeek 压缩)", "GDN/KDA\n(固定状态)"]
    # 每 token KV bytes
    mha = 524_288
    gqa = 131_072
    mla = 70_000          # ~70 KB/token(DeepSeek-V2/V3 报告)
    gdn = 2 * 1024 * 1024  # 2 MB/层状态,28 层 ≈ 57 MB 总量(固定,不随 N 增长)
    N = 32768
    vals = [mha * N / 1e9, gqa * N / 1e9, mla * N / 1e9, gdn * 28 / 1e9]
    colors = [DRED, DORANGE, "#d9a441", DGREEN]
    bars = ax.bar(range(4), vals, 0.55, color=colors, ec="white", lw=0.6)
    txts = ["17.2 GB", "4.3 GB", "2.3 GB", "57 MB\n(与长度无关)"]
    for b, t in zip(bars, txts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.18, t,
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(0.02, 60)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("32K 上下文的 KV / 状态总量(GB,对数轴)")
    ax.set_title("四代注意力:从 17 GB 到 57 MB,KV 的 300 倍瘦身\n"
                 "(GDN 状态按 32 头 × 128×128 × 4B × 28 层估算)",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_kv.png"); plt.close(fig)


def fig4_chunked():
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), gridspec_kw={"height_ratios": [1.15, 1]})

    # ---- (a) 批发 vs 零售:权重复用 -----------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 4.6); ax.axis("off")
    # W 大方块
    ax.add_patch(FancyBboxPatch((0.8, 0.7), 3.2, 3.2, boxstyle="round,pad=0.05",
                                fc="#f5f0e6", ec="#b09a6a", lw=1.4))
    ax.text(2.4, 2.3, "权重 W\n$d \\times d$", ha="center", va="center", fontsize=11, fontweight="bold")
    # X 多列(prefill)与单列(decode)
    ax.add_patch(plt.Rectangle((6.2, 0.7), 3.0, 3.2, fc="#e8f3fb", ec=DBLUE, lw=1.2))
    ax.text(7.7, 2.3, "X\n$T$ 列 token", ha="center", va="center", fontsize=9.5)
    ax.text(5.1, 2.3, "$\\times$", ha="center", va="center", fontsize=15, color="#888888")
    ax.add_patch(plt.Rectangle((11.6, 1.75), 0.55, 1.1, fc="#f2d9d9", ec=DRED, lw=1.2))
    ax.text(11.87, 1.3, "1 列", ha="center", fontsize=9, color=DRED)
    ax.text(10.4, 2.3, "$\\times$", ha="center", va="center", fontsize=15, color="#888888")
    ax.text(7.7, 4.15, "(a) 同一份权重:prefill 批发(T 列),decode 零售(1 列)",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(7.7, 0.25, "每个 $w_{ij}$ 被复用 T 次 → AI ≈ T:右上 = 站上屋顶,右下 = 趴在墙根",
            ha="center", fontsize=9, color="#333333")

    # ---- (b) chunked prefill 混排 -------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 3.6); ax.axis("off")
    for i, x in enumerate((0.9, 5.7, 10.5)):
        ax.add_patch(FancyBboxPatch((x, 0.8), 3.8, 1.8, boxstyle="round,pad=0.06",
                                    fc="#eef7ee", ec=DGREEN, lw=1.3))
        ax.add_patch(plt.Rectangle((x + 0.25, 1.05), 2.6, 1.3, fc="#e8f3fb", ec=DBLUE, lw=1.0))
        ax.text(x + 1.55, 1.7, f"prefill 块 {i+1}\n512 token", ha="center", va="center", fontsize=8.6)
        ax.add_patch(plt.Rectangle((x + 3.0, 1.05), 0.6, 1.3, fc="#f2d9d9", ec=DRED, lw=1.0))
        ax.text(x + 3.68, 1.7, "decode", ha="left", va="center", fontsize=7.6, color=DRED)
        ax.text(x + 1.9, 2.85, f"step {i+1}:同一 batch", ha="center", fontsize=8.6, color=DGREEN)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + 3.9, 1.7), (x + 4.6, 1.7), arrowstyle="-|>",
                                         mutation_scale=11, color="#888888", lw=1.2))
    ax.text(7.8, 3.35, "(b) Chunked prefill:长 prompt 切块逐段算,decode token 每步搭车",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(7.8, 0.25, "decode 的权重复用:1 次 → 512+ 次,从墙根被拽上屋顶",
            ha="center", fontsize=9.5, color=DRED, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_chunked.png"); plt.close(fig)


def fig3_chunk():
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0), gridspec_kw={"height_ratios": [1, 1.25]})

    # ---- (a) naive serial recurrence ----------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 3.4); ax.axis("off")
    for i in range(8):
        x = 0.5 + i * 1.75
        ax.add_patch(FancyBboxPatch((x, 1.1), 1.35, 1.0, boxstyle="round,pad=0.05",
                                    fc="#f2d9d9", ec=DRED, lw=1.1))
        ax.text(x + 0.675, 1.78, f"$t_{{{i+1}}}$", ha="center", va="center", fontsize=9)
        ax.text(x + 0.675, 1.40, "$S \\leftarrow S + vk^\\top$", ha="center", va="center", fontsize=7.0)
        if i < 7:
            ax.add_patch(FancyArrowPatch((x + 1.40, 1.6), (x + 1.72, 1.6),
                                         arrowstyle="-|>", mutation_scale=10, color=DRED, lw=1.2))
    ax.text(7.8, 2.9, "(a) 朴素递推:一步接一步,串行深度 = N", ha="center",
            fontsize=10.5, fontweight="bold")
    ax.text(7.8, 2.55, "小算子、时间维依赖,GPU 的两大盲区", ha="center", fontsize=9, color=DRED)

    # ---- (b) chunked parallel scan -------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 4.1); ax.axis("off")
    chunk_defs = [
        (0.5,  "#e8f3fb", DBLUE, "块 1\n$t_1$–$t_{64}$"),
        (5.3,  "#e8f3fb", DBLUE, "块 2\n$t_{65}$–$t_{128}$"),
        (10.1, "#e8f3fb", DBLUE, "块 3\n$t_{129}$–$t_{192}$"),
    ]
    for x, fc, ec, lab in chunk_defs:
        ax.add_patch(FancyBboxPatch((x, 0.9), 4.0, 1.7, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.4))
        ax.text(x + 2.0, 2.32, lab, ha="center", fontsize=9.5, fontweight="bold")
        # 内部 GEMM 子块(竖排三行)
        for j, gm in enumerate(["GEMM 打分", "外积堆叠", "状态修正"]):
            y = 1.78 - j * 0.38
            ax.add_patch(FancyBboxPatch((x + 0.25, y - 0.14), 3.5, 0.30,
                                        boxstyle="round,pad=0.02", fc="white", ec=ec, lw=0.8))
            ax.text(x + 2.0, y, gm, ha="center", va="center", fontsize=8.2)
    # 块间状态箭头
    for x0 in (4.55, 9.35):
        ax.add_patch(FancyArrowPatch((x0, 1.75), (x0 + 0.5, 1.75), arrowstyle="-|>",
                                     mutation_scale=11, color=DGREEN, lw=1.6))
        ax.text(x0 + 0.25, 2.0, "$S$", ha="center", fontsize=10, color=DGREEN, fontweight="bold")
    ax.text(7.8, 3.55, "(b) 分块扫描:块内全部整理成 GEMM 进 Tensor Core,块间只传一个状态",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(7.8, 3.2, "串行深度 N → N/64,剩下的一点点延迟用 warp 轮转藏掉",
            ha="center", fontsize=9, color=DGREEN)
    ax.text(7.8, 0.35, "同一份数学,换一种切法——这就是「重新矩阵化」",
            ha="center", fontsize=9.5, color="#333333", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_chunk.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_roofline()
    fig2_kv()
    fig3_chunk()
    fig4_chunked()
    print("done ->", OUT)
