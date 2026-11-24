#!/usr/bin/env python3
"""Figures for the gemm-libraries article (teaching version: two figures).

fig1_layers.png    -- who decides each of the five layers of a GEMM, for each of the five approaches (author's synthesis)
fig2_selection.png -- why the "selection" layer matters: FP16 GEMM on two RTX cards, cuBLAS default vs exhaustively tuned vs Triton best-of-16
                      (every number is a measurement made on 2026-11-24; see the article's references)

Run:  python3 gen_figs.py   -> ./images/
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

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
BLUE, RED, GREEN, GRAY, PURP, GOLD, ORANGE = "#2288CC", "#BB3333", "#117733", "#666666", "#995588", "#b8922e", "#d97742"


def fig2_selection():
    sizes = [512, 1024, 2048, 4096, 8192]
    panels = [
        ("RTX 4080", [44.7, 77.7, 83.5, 92.5, 101.0], [23.1, 68.5, 88.5, 99.9, 103.7], [100.1, 101.5], 120),
        ("RTX 4090 D", [20.7, 89.2, 132.7, 140.6, 142.6], [12.5, 87.4, 133.2, 151.8, 152.6], [150.3, 151.2], 178),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, (name, cub, tri, tuned, ymax) in zip(axes, panels):
        ax.plot(sizes, cub, "o-", color=RED, lw=2.0, ms=7, label="cuBLAS(默认,启发式选 kernel)")
        ax.plot(sizes, tri, "s-", color=GREEN, lw=2.0, ms=7, label="Triton(16 组配置里最好的一组)")
        ax.plot([sizes[3], sizes[4]], tuned, "D", color=RED, mfc="white", ms=9, mew=2.0, ls="none",
                label="cuBLAS(穷举调优之后)")
        for s_, c_, t_ in zip(sizes, cub, tri):
            r = t_ / c_
            ax.annotate(f"{r:.2f} 倍", (s_, max(c_, t_)), textcoords="offset points", xytext=(0, 11), ha="center",
                        fontsize=9, fontweight="bold", color=GREEN if r >= 1 else RED,
                        bbox=dict(fc="white", ec="none", pad=1.0))
        ax.set_xscale("log", base=2); ax.set_xticks(sizes); ax.set_xticklabels([str(v) for v in sizes])
        ax.minorticks_off()
        ax.set_xlabel("矩阵大小(N × N × N)"); ax.set_ylabel("TFLOPS")
        ax.set_ylim(0, ymax); ax.set_xlim(430, 9700)
        ax.set_title(name + " 上的半精度矩阵乘", fontsize=11.5, fontweight="bold", pad=10)
        ax.legend(loc="upper left", fontsize=9.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].text(9200, 6, "数字:Triton 是 cuBLAS 默认的几倍(绿色:超过,红色:不到)", fontsize=8.6, color="#555555", ha="right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_selection.png"); plt.close(fig)


def fig1_layers():
    rows = ["① 算法\n要不要切开、换成 GEMV", "② 分块\ntile、流水级数、遍历顺序", "③ 线程和寄存器\n一个线程算几个输出",
            "④ 指令\nTensor Core、异步拷贝", "⑤ 选择\n这个形状用哪个 kernel"]
    cols = ["手写 CUDA", "cuBLAS", "CUTLASS", "Triton", "TVM"]
    YOU, PICK, VENDOR, COMP, MACHINE = "you", "pick", "vendor", "compiler", "machine"
    cells = [
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("你", YOU), ("你写计算", YOU)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("你", YOU), ("规则或搜索", MACHINE)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("编译器", COMP), ("规则或搜索", MACHINE)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("编译器", COMP), ("规则或搜索", MACHINE)],
        [("你", YOU), ("启发式·可穷举", MACHINE), ("自己解决", PICK), ("自己调优", PICK), ("搜索", MACHINE)],
    ]
    style = {
        YOU: ("#dbeaf7", BLUE), PICK: ("#eef5fb", "#4a8fc8"), VENDOR: ("#e4e4e4", "#666666"),
        COMP: ("#e2f0e2", GREEN), MACHINE: ("#f0e0ea", PURP),
    }
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0, 25.2); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.text(12.6, 12.0, "一个矩阵乘的五层决定,五种做法里各由谁来定", ha="center", fontsize=13, fontweight="bold")
    x0, w, h, ytop = 7.0, 3.56, 1.86, 10.1
    for j, c in enumerate(cols):
        ax.text(x0 + j * w + w / 2, ytop + 0.32, c, ha="center", va="center", fontsize=11.5, fontweight="bold")
    for i, (r, row) in enumerate(zip(rows, cells)):
        y = ytop - (i + 1) * h
        ax.text(0.4, y + h / 2, r, ha="left", va="center", fontsize=9.8, color="#222222", linespacing=1.5)
        for j, (txt, kind) in enumerate(row):
            fc, ec = style[kind]
            ax.add_patch(FancyBboxPatch((x0 + j * w + 0.08, y + 0.1), w - 0.16, h - 0.2, boxstyle="round,pad=0.02",
                                        fc=fc, ec=ec, lw=1.3))
            ax.text(x0 + j * w + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.8, color=ec, fontweight="bold")
    # 图例
    ly = 0.45
    legend = [("你", YOU, "你自己写"), ("你挑", PICK, "你从现成的里挑,或自己解决"), ("厂商", VENDOR, "厂商定好,改不了"),
              ("编译器", COMP, "编译器替你定"), ("搜索", MACHINE, "机器来选:启发式、规则、搜索")]
    lx = 0.4
    for txt, kind, desc in legend:
        fc, ec = style[kind]
        ax.add_patch(FancyBboxPatch((lx, ly - 0.22), 0.9, 0.46, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2))
        ax.text(lx + 1.08, ly, desc, ha="left", va="center", fontsize=8.8, color="#444444")
        lx += 1.08 + len(desc) * 0.25 + 0.65
    fig.savefig(f"{OUT}/fig1_layers.png"); plt.close(fig)



if __name__ == "__main__":
    fig1_layers(); fig2_selection()
    print("done ->", OUT)
