#!/usr/bin/env python3
"""Figures for the MXMACA / AP article.

fig1_ap_vs_sm.png   MXMACA's AP next to NVIDIA's SM: what sits in each, and where every number comes from
fig2_wave.png       the 64-lane wave (4 rows of 16 lanes, quads of 4) and what a 32-bit mask covers on it
fig3_toolchain.png  CUDA's and MXMACA's compile paths side by side
fig4_budget.png     registers per thread for one wave that computes output tiles of different sizes

Numbers come from bench/calc_summary.json (written by calc.py).
Run:  python3 gen_figs.py  ->  ./images/
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, ORANGE, GOLD, LIGHT = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#b8922e", "#e6e6e6"
PURP, DARK, PLUM = "#995588", "#333333", "#5b2a86"
calc = json.load(open(os.path.join(HERE, "bench", "calc_summary.json")))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rbox(ax, x, y, w, h, fc, ec="#8a8a8a", lw=0.9, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw))


# ---------------------------------------------------------------------------------------------- fig1
def panel(ax, title, subtitle, unit_word, unit_lines, units, regs, mem1, mem2, occupancy, unit_fc):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    rbox(ax, 1, 3, 98, 94, "#fbfbfb", ec=GRAY, lw=1.1, r=1.2)
    ax.text(50, 92.5, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=DARK)
    ax.text(50, 86.5, subtitle, ha="center", va="center", fontsize=8.2, color=GRAY)
    for i in range(4):
        x = 5.5 + i * 22.5
        rbox(ax, x, 68, 20, 14, unit_fc, ec=DARK)
        ax.text(x + 10, 78.2, f"{unit_word} {i + 1}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=DARK)
        ax.text(x + 10, 72.2, unit_lines, ha="center", va="center", fontsize=7.4, color=DARK, linespacing=1.35)
    ax.text(20.3, 63, "运算单元", ha="right", va="center", fontsize=7.6, color=GRAY)
    rbox(ax, 22, 58.5, 73, 9, "#f1f1f1", ec="#9a9a9a")
    ax.text(58.5, 63, units, ha="center", va="center", fontsize=7.9, color=DARK)
    ax.text(20.3, 47.5, "寄存器", ha="right", va="center", fontsize=7.6, color=GRAY)
    rbox(ax, 22, 41.5, 73, 12, "#ead9f3", ec="#9a9a9a")
    ax.text(58.5, 47.5, regs, ha="center", va="center", fontsize=7.5, color=DARK, linespacing=1.4)
    rbox(ax, 5.5, 26, 89.5, 11, "#d9efd9", ec="#9a9a9a")
    ax.text(50.25, 31.5, mem1, ha="center", va="center", fontsize=7.9, color=DARK)
    rbox(ax, 5.5, 12, 89.5, 11, "#dcebf7", ec="#9a9a9a")
    ax.text(50.25, 17.5, mem2, ha="center", va="center", fontsize=7.9, color=DARK)
    ax.text(50, 7, occupancy, ha="center", va="center", fontsize=8.4, color=DARK, fontweight="bold")


def fig1_ap_vs_sm():
    c = calc["compare"]
    n, m = c["nvidia_h100_sm"], c["metax_ap_n260"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 4.5))
    panel(a, "NVIDIA 的 SM (H100)", f"{n['sub_units']} 个分区,{n['waves']} 条线程束,每条 {n['lanes_per_wave']} 个线程",
          "分区", f"warp 调度器\n{calc['wave']['nvidia_warp_lanes']} 个 lane",
          "128 个 CUDA Core,4 个 Tensor Core",
          "64K 个 32 位寄存器 (256 KB)\n所有驻留的线程一起分,每个线程最多 255 个",
          "L1 缓存和共享内存:256 KB,两者的比例可配",
          f"每个线程块最多 {n['threads_per_block_max']} 个线程",
          f"{n['waves']} 条线程束 × {n['lanes_per_wave']} 个线程 = {n['threads']} 个线程", "#dcebf7")
    panel(b, "沐曦的 AP (加速处理器)", f"{m['sub_units']} 个 PEU,{m['waves']} 条线程束,每条 {m['lanes_per_wave']} 个线程",
          "PEU", f"{calc['ap']['waves_per_peu']} 条线程束\n{m['lanes_per_wave']} 个 lane",
          "标量,矢量,张量三类运算单元,超标量发射",
          "分两类:mtreg 和 streg,总数没有公布\n-maxrregcount 和 -maxsregcount 分别限",
          "工作组共享内存 64 KB:32 个 bank,每个 4 字节",
          "L1 缓存:指令和数据,常量,矢量数据",
          f"{m['waves']} 条线程束 × {m['lanes_per_wave']} 个线程 = {m['threads']} 个线程", "#f7e3d3")
    fig.text(0.5, 0.005, "沐曦一侧:线程束 64 个线程出自官方编程模型;每个 AP 4 个 PEU,32 条线程束,64 KB 共享内存出自曦思 N260 上 macainfo 的输出,\n"
             "N260 和 C500 是同一代架构。NVIDIA 一侧出自本专栏《GPU 的心脏:SM 内部解剖》。", ha="center", va="bottom", fontsize=7.4, color=GRAY, linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.1, wspace=0.05)
    save(fig, "fig1_ap_vs_sm.png")


# ---------------------------------------------------------------------------------------------- fig2
def fig2_wave():
    w = calc["wave"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.12, 1]))
    # left: 64 lanes, 4 rows of 16
    a.set_xlim(-15, 22)
    a.set_ylim(-4.6, 21.5)
    a.axis("off")
    tint = ["#cfe3f5", "#f9dcc8", "#cfe9d3", "#e6d6ee"]
    edge = [BLUE, ORANGE, GREEN, PURP]
    for i in range(w["rows"]):
        y = (3 - i) * 4.2
        for j in range(w["lanes_per_row"]):
            a.add_patch(Rectangle((j * 1.28, y), 1.14, 3.0, fc=tint[i], ec="none"))
            a.text(j * 1.28 + 0.57, y + 1.5, str(i * 16 + j), ha="center", va="center", fontsize=5.6, color=DARK)
        for j in range(0, 17, 4):                    # quad boundaries, every 4 lanes
            a.plot([j * 1.28 - 0.075, j * 1.28 - 0.075], [y - 0.25, y + 3.25], color=edge[i], lw=1.3)
        a.text(-0.9, y + 1.5, f"row {i}\n线程 {i * 16} 到 {i * 16 + 15}", ha="right", va="center", fontsize=7.6, color=edge[i], linespacing=1.3)
    a.text(10.2, 19.6, "每个 row 16 个 lane", ha="center", va="center", fontsize=8.0, color=DARK)
    a.text(1.3, 16.4, "每 4 个 lane 一组", ha="left", va="center", fontsize=7.2, color=GRAY)
    a.text(10.2, -2.6, "一条线程束 64 个线程 = 4 个 row × 16 个 lane", ha="center", va="center", fontsize=8.4, color=DARK, fontweight="bold")
    # right: masks
    b.set_xlim(-27, 66)
    b.set_ylim(-4.6, 21.5)
    b.axis("off")
    def cells(y, n, colors, h=3.0, step=1.0):
        for k in range(n):
            b.add_patch(Rectangle((k * step, y), step - 0.08, h, fc=colors(k), ec="white", lw=0.4))
    b.text(-1.2, 18.0, "CUDA 的掩码\n0xffffffff,32 位", ha="right", va="center", fontsize=7.6, color=DARK, linespacing=1.3)
    cells(16.5, 32, lambda k: BLUE, step=2.0)
    b.text(-1.2, 10.6, "MXMACA 的掩码\n0xffffffffffffffff,64 位", ha="right", va="center", fontsize=7.6, color=DARK, linespacing=1.3)
    cells(9.1, 64, lambda k: BLUE if k < 32 else ORANGE)
    b.text(-1.2, 3.2, "把 32 位的掩码\n用在 64 个 lane 上", ha="right", va="center", fontsize=7.6, color=RED, linespacing=1.3)
    cells(1.7, 64, lambda k: BLUE if k < 32 else "#eeeeee")
    for k in range(32, 64):                          # hatch the lanes the 32-bit mask leaves out
        b.add_patch(Rectangle((k, 1.7), 0.92, 3.0, fc="none", ec="#bbbbbb", lw=0.4, hatch="////"))
    for y in (16.5, 9.1, 1.7):
        b.plot([32, 32], [y - 0.5, y + 3.5], color=RED, lw=1.0, linestyle=(0, (4, 3)))
    b.text(16, -1.4, "线程 0 到 31", ha="center", va="center", fontsize=7.4, color=DARK)
    b.text(48, -1.4, "线程 32 到 63", ha="center", va="center", fontsize=7.4, color=DARK)
    b.text(64, -3.6, "灰色斜线:32 位掩码盖不到的另一半", ha="right", va="center", fontsize=7.2, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03, wspace=0.03)
    save(fig, "fig2_wave.png")


# ---------------------------------------------------------------------------------------------- fig3
def row(ax, y, label, label_color, steps, note):
    n = len(steps)
    w, gap = 17.2, 3.5
    for i, (head, body) in enumerate(steps):
        x = i * (w + gap)
        rbox(ax, x, y, w, 17, "#fbfbfb", ec=DARK if i in (1, 3) else "#8a8a8a")
        ax.text(x + w / 2, y + 13, head, ha="center", va="center", fontsize=8.3, fontweight="bold", color=label_color)
        ax.text(x + w / 2, y + 6.4, body, ha="center", va="center", fontsize=7.1, color=DARK, linespacing=1.35)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, y + 8.5), (x + w + gap - 0.5, y + 8.5), arrowstyle="-|>", mutation_scale=9, color=GRAY, lw=1.0))
    ax.text(0, y + 20.6, label, ha="left", va="center", fontsize=9.2, fontweight="bold", color=label_color)
    ax.text(0, y - 3.2, note, ha="left", va="center", fontsize=7.6, color=GRAY)


def fig3_toolchain():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(-1, 100)
    ax.set_ylim(16, 87)
    ax.axis("off")
    row(ax, 64, "CUDA", GREEN,
        [("x.cu", "源文件"), ("nvcc", "编译"), ("PTX", "文本格式\n指令集文档公开"), ("ptxas 或驱动", "再编译一次"), ("SASS", "机器码\n不公开")],
        "可以写内联汇编,写的就是 PTX;换一代卡,旧的 PTX 还能由驱动再编一次。")
    row(ax, 22, "MXMACA", PLUM,
        [("x.maca", "源文件"), ("mxcc", "clang 前端\n设备库是 .bc"), ("LLVM bitcode", "二进制格式\n同大版本通用"), ("驱动,加载时编译", "后端 metaxgpu"), ("XCORE 机器码", "指令集\n不公开")],
        "不支持内联汇编,要用硬件特性只能走编译器内建函数;按 xcore1000 编的代码,同一大版本的卡都能跑。")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03)
    save(fig, "fig3_toolchain.png")


# ---------------------------------------------------------------------------------------------- fig4
def fig4_budget():
    b = calc["budget"]
    keys = [k for k, v in b.items() if isinstance(v, dict) and "total" in v]
    order = sorted(keys, key=lambda k: b[k]["total"])          # smallest at the bottom
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ys = range(len(order))
    for y, k in zip(ys, order):
        d = b[k]
        left = 0
        for part, color, name in (("acc", BLUE, "累加器"), ("a", ORANGE, "A 的片段"), ("b", GREEN, "B 的片段")):
            ax.barh(y, d[part], left=left, height=0.58, color=color, label=name if y == 0 else None)
            if d[part] >= 12:
                ax.text(left + d[part] / 2, y, str(d[part]), ha="center", va="center", fontsize=7.6, color="white")
            left += d[part]
        ax.text(left + 5, y, f"共 {d['total']} 个", ha="left", va="center", fontsize=8.2, color=DARK, fontweight="bold")
    lim = b["mtreg_limit_in_kernel_comment"]
    ax.axvline(lim, color=RED, lw=1.0, linestyle=(0, (5, 4)))
    ax.text(lim + 4, 1.5, f"{lim} 个:沐曦算子库注释里给的上限", ha="left", va="center", fontsize=7.6, color=RED)
    ax.set_yticks(list(ys), [f"一条线程束算 {b[k]['rows']} × {b[k]['cols']} 的块" for k in order], fontsize=8.4)
    ax.set_xlabel("每个线程的 32 位寄存器数 (BF16,只数主循环里的累加器和 A,B 片段)", fontsize=8.4)
    ax.set_xlim(0, 345)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.legend(loc="lower right", fontsize=8.2, ncol=3)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig4_budget.png")


# ---------------------------------------------------------------------------------------------- fig5
def fig5_bridge():
    """Where MXMACA substitutes its own parts under an unchanged CUDA project (cu-bridge's own design diagram)."""
    rows = [
        ("nvcc hello.cu", "cucc:一个脚本,转手调用 mxcc"),
        ("make / cmake", "make_maca / cmake_maca,工具链换成沐曦的"),
        ("#include <cuda_runtime.h>", "cu-bridge 的同名头文件:转去包含 mc_runtime.h"),
        ("cudaMalloc, cudaMemcpy, ...", "#define cudaMalloc wcudaMalloc,封装函数再调 mcMalloc"),
        ("-lcublas -lcudnn -lnccl", "mcBLAS, mcDNN, MCCL"),
        ("PyTorch, vLLM, Triton 的源码", "用这套工具链重新编一遍:mcPyTorch, vllm-metax, mcTriton"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 94, "CUDA 的项目一行不改,沐曦在六处换上自己的零件", ha="center", va="center", fontsize=10, fontweight="bold", color=DARK)
    ax.text(23, 86.5, "CUDA 项目里的", ha="center", va="center", fontsize=8.4, color=GREEN, fontweight="bold")
    ax.text(74, 86.5, "沐曦换上去的", ha="center", va="center", fontsize=8.4, color=PLUM, fontweight="bold")
    h, gap = 10.4, 2.2
    for i, (left, right) in enumerate(rows):
        y = 82.5 - (i + 1) * (h + gap)
        rbox(ax, 2, y, 42, h, "#d9efd9", ec=GREEN)
        ax.text(23, y + h / 2, left, ha="center", va="center", fontsize=8.0, color=DARK)
        ax.add_patch(FancyArrowPatch((45.2, y + h / 2), (49.8, y + h / 2), arrowstyle="-|>", mutation_scale=10, color=GRAY, lw=1.1))
        rbox(ax, 50.5, y, 47.5, h, "#f7e3d3", ec=PLUM)
        ax.text(74.25, y + h / 2, right, ha="center", va="center", fontsize=7.7, color=DARK)
    ax.text(50, 2.2, "前五行出自 cu-bridge 项目的原理图和使用指南,第六行出自沐曦各仓库的说明。", ha="center", va="center", fontsize=7.4, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "fig5_bridge.png")


if __name__ == "__main__":
    fig1_ap_vs_sm()
    fig2_wave()
    fig3_toolchain()
    fig4_budget()
    fig5_bridge()
