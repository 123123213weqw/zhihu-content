#!/usr/bin/env python3
"""Figures for the ptx-ir article. All numbers come from the real CUDA 12.9 / V100 runs cited in the article.

fig1_pipeline.png -- compile time (CUDA C -> PTX -> SASS, both packed in a fatbin) vs run time (use SASS, or JIT the PTX)
fig2_vregs.png    -- matmul: 160 virtual registers in PTX vs 32 physical registers after ptxas
fig3_compat.png   -- which GPUs the same PTX file can be assembled for (tested with ptxas)

Run:  python3 gen_figs.py   -> ./images/
"""
from __future__ import annotations
import os
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
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
DBLUE, DRED, DGREEN, DGRAY, DPURP, DGOLD = "#2288CC", "#BB3333", "#117733", "#666666", "#995588", "#b8922e"


def box(ax, x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=9.8, bold=False, tc="#222222", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.6, ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls))


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 7.6))
    ax.set_xlim(0, 24.8); ax.set_ylim(0, 15.2); ax.axis("off")
    ax.text(12.4, 14.7, "一个 CUDA 程序的两种形态:SASS 给现在的 GPU 用,PTX 留给以后的 GPU", ha="center", fontsize=13, fontweight="bold")

    # ---- 编译时 ----
    ax.text(0.6, 13.55, "编译时(nvcc)", ha="left", fontsize=11, fontweight="bold", color=DGRAY)
    xs = [(0.8, "CUDA C\n人写的", "#f3f3f3", "#999999", False),
          (7.2, "PTX\n虚拟机的指令集", "#eef7ee", DGREEN, True),
          (13.6, "SASS\n这块 GPU 的机器码", "#fbeaea", DRED, True)]
    w, h, y = 5.0, 2.0, 10.8
    for x, t, fc, ec, b in xs:
        box(ax, x, y, w, h, t, fc=fc, ec=ec, bold=b)
    arrow(ax, 5.85, y + h / 2, 7.15, y + h / 2); ax.text(6.5, y + h / 2 + 0.38, "cicc", ha="center", fontsize=9, color="#555555")
    arrow(ax, 12.25, y + h / 2, 13.55, y + h / 2); ax.text(12.9, y + h / 2 + 0.38, "ptxas", ha="center", fontsize=9, color="#555555")
    box(ax, 0.8, 8.1, 17.8, 1.7, "fatbin:PTX 和 SASS 一起打包,嵌进你的程序", fc="#f5f0e6", ec="#b09a6a", fs=10.2, bold=True)
    arrow(ax, 9.7, 10.75, 9.7, 9.85)
    arrow(ax, 16.1, 10.75, 16.1, 9.85)
    box(ax, 19.4, 8.1, 4.6, 5.5, "这里打包的\n是两样东西:\n\nSASS 让现在的\nGPU 不用等\n\nPTX 留给未来\n的 GPU", fc="#fbfbfb", ec="#bbbbbb", fs=9.4, lw=1.0)
    arrow(ax, 18.65, 8.95, 19.35, 9.6, color="#999999", lw=1.1)

    # ---- 运行时 ----
    ax.text(0.6, 7.35, "运行时(程序启动)", ha="left", fontsize=11, fontweight="bold", color=DGRAY)
    box(ax, 7.4, 5.2, 8.0, 1.5, "这台 GPU 有现成的 SASS 吗?", fc="#f2d9ec", ec=DPURP, fs=10.4, bold=True)
    arrow(ax, 9.7, 8.05, 9.7, 6.75)
    box(ax, 0.8, 1.9, 9.0, 1.9, "有\n直接加载 SASS,不用等", fc="#eef7ee", ec=DGREEN, fs=10, bold=True)
    box(ax, 14.2, 1.9, 9.8, 1.9, "没有\n驱动把 PTX 现场编译成 SASS(慢一些,\n结果会被缓存)", fc="#fbeaea", ec=DRED, fs=10, bold=True)
    arrow(ax, 9.0, 5.15, 6.2, 3.85, color=DGREEN); ax.text(7.0, 4.55, "有", fontsize=10, color=DGREEN, fontweight="bold")
    arrow(ax, 13.8, 5.15, 18.6, 3.85, color=DRED); ax.text(16.4, 4.55, "没有", fontsize=10, color=DRED, fontweight="bold")
    arrow(ax, 5.3, 1.85, 11.9, 1.0); arrow(ax, 19.1, 1.85, 12.9, 1.0)
    ax.text(12.4, 0.55, "GPU 执行", ha="center", va="center", fontsize=12, fontweight="bold", color="#333333")
    fig.savefig(f"{OUT}/fig1_pipeline.png"); plt.close(fig)


def fig2_vregs():
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(0, 250); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(125, 5.25, "分块矩阵乘:PTX 申请了 160 个虚拟寄存器,ptxas 优化之后只占 32 个物理寄存器", ha="center",
            fontsize=12, fontweight="bold")
    segs = [(105, "浮点 %f  105", "#7fb3d8"), (39, "32 位整数 %r  39", "#5f9cc9"), (13, "64 位 %rd  13", "#3f85bb"), (3, "谓词 3", "#9ec5e3")]
    # 上条:PTX
    x0 = 4; y1 = 3.5; hh = 1.0
    ax.text(x0, y1 + hh + 0.32, "PTX 里的虚拟寄存器(要多少有多少)", ha="left", fontsize=10, color=DGREEN, fontweight="bold")
    x = x0
    for n, lab, c in segs:
        ax.add_patch(Rectangle((x, y1), n, hh, fc=c, ec="white", lw=1.2))
        x += n
    ax.text(x0 + 105 / 2, y1 + hh / 2, "浮点 105", ha="center", va="center", fontsize=9.4, color="white", fontweight="bold")
    ax.text(x0 + 105 + 39 / 2, y1 + hh / 2, "整数 39", ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    ax.text(x0 + 160 + 3, y1 + hh / 2, "后面两段:64 位 13,谓词 3", ha="left", va="center", fontsize=9, color="#333333")
    ax.text(x0 + 160, y1 + hh + 0.32, "共 160", ha="right", va="center", fontsize=10.5, fontweight="bold", color=DGREEN)
    # 箭头
    ax.add_patch(FancyArrowPatch((84, y1 - 0.1), (84, y1 - 0.75), arrowstyle="-|>", mutation_scale=16, color=DRED, lw=2))
    ax.text(87, y1 - 0.42, "ptxas:一个值用完了,寄存器腾给下一个值", ha="left", va="center", fontsize=10, color=DRED, fontweight="bold")
    # 下条:SASS
    y2 = 0.9
    ax.text(x0, y2 + hh + 0.32, "SASS 里的物理寄存器", ha="left", fontsize=10, color=DRED, fontweight="bold")
    ax.add_patch(Rectangle((x0, y2), 32, hh, fc=DRED, ec="white", lw=1.2))
    ax.text(x0 + 32 + 2, y2 + hh / 2, "共 32  ← 第 7 篇的表:每线程 32 个寄存器,每个 SM 可驻留 64 个 warp", ha="left", va="center",
            fontsize=10, fontweight="bold", color=DRED)
    fig.savefig(f"{OUT}/fig2_vregs.png"); plt.close(fig)


def fig3_compat():
    rows = ["PTX 的目标是 sm_70", "PTX 的目标是 sm_80"]
    cols = ["sm_60", "sm_70", "sm_80", "sm_90"]
    cells = [["失败", "成功", "成功", "成功"], ["未测", "失败", "成功", "成功"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.set_xlim(0, 5.65); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.text(2.85, 2.45, "同一份 PTX,ptxas 能把它汇编成哪些架构的 SASS", ha="center", fontsize=12, fontweight="bold")
    ax.text(3.57, 2.08, "汇编成的架构 →", ha="center", fontsize=9.6, color="#666666")
    for j, c in enumerate(cols):
        ax.text(1.55 + j + 0.5, 1.82, c, ha="center", va="center", fontsize=10.5, fontweight="bold")
    col = {"成功": ("#eef7ee", DGREEN), "失败": ("#fbeaea", DRED), "未测": ("#f3f3f3", "#999999")}
    for i, r in enumerate(rows):
        y = 1.42 - i * 0.62
        ax.text(0.02, y + 0.25, r, ha="left", va="center", fontsize=10.2)
        for j, v in enumerate(cells[i]):
            fc, ec = col[v]
            ax.add_patch(FancyBboxPatch((1.55 + j + 0.06, y), 0.88, 0.5, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.4))
            ax.text(1.55 + j + 0.5, y + 0.25, v, ha="center", va="center", fontsize=10, color=ec, fontweight="bold")
    ax.text(2.85, 0.12, "规律:PTX 只能往上汇编,不能往下", ha="center", fontsize=10.5, color="#333333", fontweight="bold")
    fig.savefig(f"{OUT}/fig3_compat.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline(); fig2_vregs(); fig3_compat()
    print("done ->", OUT)
