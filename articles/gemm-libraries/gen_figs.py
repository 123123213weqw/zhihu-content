#!/usr/bin/env python3
"""Figures for the gemm-libraries article. Every number is a measurement made on 2026-11-24 (see the article's references).

fig1_fp32.png   -- FP32 4096^3 GEMM on V100 and RTX 4090 D: naive / tiled / register-tiled / Triton / cuBLAS, with each card's ceiling
fig2_fp16.png   -- FP16 GEMM on RTX 4090 D vs size: cuBLAS default, cuBLAS exhaustive-tuned, Triton best-of-16
fig3_layers.png -- who decides each of the five layers, for each of the five approaches

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


def fig1_fp32():
    labels = ["朴素", "分块", "分块加\n寄存器分块", "Triton\n(真 FP32)", "cuBLAS"]
    colors = ["#c9c9c9", "#9ec5e3", BLUE, GREEN, RED]
    panels = [
        ("Tesla V100", [1.9, 3.0, 9.1, None, 12.6], [14, 21, 65, None, 89], 14.1),
        ("RTX 4090 D", [4.6, 4.1, 25.7, 37.5, 49.1], [6, 6, 35, 51, 67], 73.5),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, (name, vals, pcts, ceil) in zip(axes, panels):
        x = np.arange(len(labels))
        for i, (v, c, pc) in enumerate(zip(vals, colors, pcts)):
            if v is None:
                ax.text(i, ceil * 0.04, "编译不过", ha="center", va="bottom", fontsize=8.6, color="#999999")
                continue
            ax.bar(i, v, 0.62, color=c, ec="white", lw=0.8)
            ax.text(i, v + ceil * 0.025, f"{v:.1f}\n上限的 {pc}%", ha="center", va="bottom", fontsize=8.6,
                    fontweight="bold" if i == 4 else "normal", color="#222222", linespacing=1.35,
                    bbox=dict(fc="white", ec="none", pad=1.0))
        ax.axhline(ceil, color="#444444", lw=1.1, ls="--")
        ax.text(-0.42, ceil, f"上限 {ceil:.1f} TFLOPS", ha="left", va="bottom", fontsize=8.8,
                color="#444444", bbox=dict(fc="white", ec="none", pad=1.5))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_title(name, fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, ceil * 1.26); ax.set_ylabel("TFLOPS")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("单精度矩阵乘 4096×4096×4096:同一个算法,五种做法", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_fp32.png"); plt.close(fig)


def fig2_fp16():
    sizes = [512, 1024, 2048, 4096, 8192]
    cub = [20.696, 89.241, 132.731, 140.611, 142.560]
    tri = [12.466, 87.362, 133.170, 151.813, 152.601]
    tuned_x, tuned_y = [3, 4], [150.33, 151.17]
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.plot(sizes, cub, "o-", color=RED, lw=2.0, ms=7, label="cuBLAS(默认,启发式选 kernel)")
    ax.plot(sizes, tri, "s-", color=GREEN, lw=2.0, ms=7, label="Triton(16 组配置里最好的一组)")
    ax.plot(tuned_x and [sizes[i] for i in tuned_x], tuned_y, "D", color=RED, mfc="white", ms=9, mew=2.0,
            ls="none", label="cuBLAS(穷举调优之后)")
    for s, c, t in zip(sizes, cub, tri):
        r = t / c
        ax.annotate(f"{r:.2f} 倍", (s, max(c, t)), textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=9, fontweight="bold", color=GREEN if r >= 1 else RED)
    ax.text(9200, 7, "数字:Triton 是 cuBLAS 默认的几倍(绿色:超过,红色:不到)", fontsize=8.8, color="#555555", ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("矩阵大小(N × N × N)"); ax.set_ylabel("TFLOPS")
    ax.set_ylim(0, 182); ax.set_xlim(430, 9700)
    ax.set_title("RTX 4090 D 上的半精度矩阵乘", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig2_fp16.png"); plt.close(fig)


def fig3_layers():
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
    fig.savefig(f"{OUT}/fig3_layers.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_fp32(); fig2_fp16(); fig3_layers()
    print("done ->", OUT)
