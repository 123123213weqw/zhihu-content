#!/usr/bin/env python3
"""Figures for the what-is-tvm article.

fig1_stack.png -- TVM's IR stack: model -> Relax -> TensorIR (S-TIR) -> target code -> vendor compiler; TIRx as a sibling kernel-level entry
fig2_block.png -- the same block wrapped by three different loop nests (computation unchanged)

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


def box(ax, x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=9.6, bold=False, tc="#222222", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.6, ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls))


def fig1_stack():
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    ax.set_xlim(0, 22.4); ax.set_ylim(0, 14.8); ax.axis("off")
    ax.text(11.2, 14.3, "TVM 的编译栈:模型往下翻译,一层比一层靠近硬件", ha="center", fontsize=13, fontweight="bold")

    cx, w, h = 7.4, 10.2, 1.75
    rows = [
        (12.0, "模型(PyTorch、ONNX 等)", "#f3f3f3", "#999999", False),
        (9.5,  "Relax —— 图级\n一个算子一个节点,边是数据流", "#eef7ee", DGREEN, True),
        (7.0,  "TensorIR(S-TIR)—— 张量程序级\n带标注的块 + 循环嵌套,可以被调度", "#e8f3fb", DBLUE, True),
        (4.5,  "目标代码\nCUDA C、LLVM IR 等", "#f5f0e6", "#b09a6a", False),
        (2.0,  "厂商的编译器\nNVIDIA: CUDA C → PTX → SASS", "#f2d9d9", DRED, False),
        (-0.5, "", "", "", False),
    ]
    box(ax, cx - w / 2, 12.0, w, h, rows[0][1], fc=rows[0][2], ec=rows[0][3])
    box(ax, cx - w / 2, 9.5, w, h, rows[1][1], fc=rows[1][2], ec=rows[1][3], bold=True)
    box(ax, cx - w / 2, 7.0, w, h, rows[2][1], fc=rows[2][2], ec=rows[2][3], bold=True)
    box(ax, cx - w / 2, 4.5, w, h, rows[3][1], fc=rows[3][2], ec=rows[3][3])
    box(ax, cx - w / 2, 2.0, w, h, rows[4][1], fc=rows[4][2], ec=rows[4][3])
    ax.text(cx, 0.75, "GPU 执行", ha="center", va="center", fontsize=12, fontweight="bold", color="#333333")

    for y in (12.0, 9.5, 7.0, 4.5, 2.0):
        arrow(ax, cx, y, cx, y - 0.75)
    # 两个箭头上的标注
    ax.text(cx + 0.6, 6.62, "调度:改循环,不改计算", ha="left", va="center", fontsize=9.2, color=DRED, fontweight="bold")
    ax.text(cx + 0.6, 1.62, "下一篇", ha="left", va="center", fontsize=9.2, color=DRED)

    # TIRx:并列的内核级入口
    box(ax, 15.2, 6.6, 6.8, 2.55, "TIRx —— 硬件原生的内核语言\n流水线、同步由人写\ntile 级操作交给编译器",
        fc="#f2d9ec", ec=DPURP, fs=9.2, bold=False)
    arrow(ax, 15.2, 7.3, 12.6, 5.5, color=DPURP, ls="--")
    ax.text(19.2, 4.95, "新一代,和 TensorIR 并列", ha="center", fontsize=8.6, color=DPURP)

    # 右侧括注:哪些是 TVM 的
    xb = 0.5
    ax.plot([xb, xb, xb], [7.1, 11.25, 11.25], color=DGREEN, lw=1.6)
    ax.plot([xb, xb], [7.1, 11.25], color=DGREEN, lw=1.6)
    ax.text(xb - 0.25, 9.2, "TVM 自己的\n中间表示", ha="right", va="center", fontsize=9.6, color=DGREEN,
            fontweight="bold", linespacing=1.5)
    ax.plot([xb, xb], [2.1, 6.15], color="#999999", lw=1.6)
    ax.text(xb - 0.25, 4.1, "TVM 生成\n或调用", ha="right", va="center", fontsize=9.6, color="#777777", linespacing=1.5)
    fig.savefig(f"{OUT}/fig1_stack.png"); plt.close(fig)


def fig2_block():
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 25.2); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.text(12.6, 10.7, "同一个块,外面套三种循环:块里的计算一个字没改", ha="center", fontsize=13, fontweight="bold")

    panels = [
        ("① 逐个算", "for i in 128:\n  for j in 128:\n    for k in 128:\n      BLOCK C", "#f3f3f3", "#999999"),
        ("② 分块算", "for i0 in 4:\n  for j0 in 4:\n    for i1 in 32:\n      for j1 in 32:\n        for k in 128:\n          BLOCK C", "#e8f3fb", DBLUE),
        ("③ 拆给线程", "for by in grid.y:\n  for bx in grid.x:\n    for ty in block.y:\n      for tx in block.x:\n        for k in 128:\n          BLOCK C", "#eef7ee", DGREEN),
    ]
    x = 0.6
    for title, code, fc, ec in panels:
        w = 7.8
        ax.add_patch(FancyBboxPatch((x, 3.1), w, 6.6, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, 9.05, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=ec if ec != "#999999" else "#444444")
        ax.text(x + 0.5, 8.35, code, ha="left", va="top", fontsize=10.2, family="monospace", color="#222222", linespacing=1.75)
        x += w + 0.6

    box(ax, 2.4, 0.5, 20.4, 2.0, "", fc="#fbeaea", ec=DRED, lw=1.6)
    ax.text(12.6, 2.05, "块 C(三种套法里都是这一段)", ha="center", va="center", fontsize=10, fontweight="bold", color=DRED)
    ax.text(12.6, 1.2, "vi, vj, vk = axis.remap(\"SSR\", [i, j, k])        C[vi, vj] += A[vi, vk] * B[vk, vj]",
            ha="center", va="center", fontsize=10.6, family="monospace", color="#222222")
    fig.savefig(f"{OUT}/fig2_block.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_stack(); fig2_block()
    print("done ->", OUT)
