#!/usr/bin/env python3
"""Figures for the MetaX C500 article.

fig1_lineage.png   the XCORE architecture lineage and the products built on each generation (sources: the prospectus)
fig2_block.png     the chip's block diagram as drawn in the prospectus (1-1-122), redrawn with annotations
fig3_mma.png       register layout of one 16x16x16 matrix instruction on a 64-lane wavefront (source: a third-party probe)
fig4_plateau.png   GEMM time against the number of rows, measured on a real C500 (source: the SpecPilot reproduction log)
fig5_compare.png   C500 on paper and as measured, next to A100 and H20
fig6_stack.png     the MXMACA stack in the four layers of the inquiry reply (8-1-89)
fig7_hierarchy.png thread hierarchy (grid, workgroup, 64-lane wavefront) and the memory hierarchy

Numbers come from bench/calc_summary.json (written by calc.py).
Run:  python3 gen_figs.py  ->  ./images/
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
BLUE, RED, GREEN, GRAY, ORANGE, PURP, GOLD, LIGHT = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#995588", "#b8922e", "#e6e6e6"
DARK, PLUM = "#333333", "#5b2a86"
calc = json.load(open(os.path.join(HERE, "bench", "calc_summary.json")))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def box(ax, x, y, w, h, text, fc, ec, tc="white", fs=9.0, rot=0, lw=1.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, rotation=rot,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw, mutation_scale=11, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------------------------- fig1
def fig1_lineage():
    rows = [  # (label, note, events: (year, text, color, dy, ha))
        ("XCORE 0.5", "曦思 N100:推理,视频编解码;HBM2e 16 GB",
         [(2022.0, "2022\nN100 推出", GRAY, 0.3, "center"), (2023.3, "2023-04\nN100 量产", GRAY, 0.3, "center")]),
        ("XCORE 1.0", "曦云 C500、C550:训推一体;曦思 N260。C500:HBM2e 64 GB",
         [(2023.45, "2023-06 C500 回片", BLUE, 0.3, "right"), (2024.15, "2024-02 C500 量产", BLUE, 0.3, "left")]),
        ("XCORE 1.5", "曦云 C600、曦思 N300:国产供应链,增加 FP8 张量指令",
         [(2025.3, "2025-04 C600 已完成流片", GREEN, 0.3, "center")]),
        ("XCORE 2.0", "曦彩 G100:图形渲染",
         [(2025.3, "2025-04 完成 GPU IP 设计和验证", PURP, 0.3, "center")]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (label, note, events) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([2020.72, 2025.9], [y, y], color=LIGHT, lw=6, solid_capstyle="round", zorder=0)
        ax.text(2020.45, y + 0.17, label, ha="right", va="center", fontsize=10, fontweight="bold", color=DARK)
        ax.text(2020.45, y - 0.24, note, ha="right", va="center", fontsize=7.6, color=GRAY)
        for year, text, c, dy, ha in events:
            off = -5 if ha == "right" else (5 if ha == "left" else 0)
            ax.plot(year, y, "o", color=c, ms=8, zorder=3)
            ax.annotate(text, (year, y), xytext=(off, 7), textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=c, linespacing=1.3)
    ax.axvline(2020.72, color=GRAY, lw=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(2020.55, 3.6, "2020-09 沐曦成立", ha="right", va="bottom", fontsize=8, color=DARK)
    ax.set_xlim(2017.7, 2026.5)
    ax.set_ylim(-0.6, 4.1)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025], ["2021", "2022", "2023", "2024", "2025"])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig1_lineage.png")


# ---------------------------------------------------------------------------------------------- fig2
def fig2_block():
    """Redrawn from the prospectus's figure (1-1-122). Notes sit outside the die; leader lines never cross a block."""
    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    ax.add_patch(Rectangle((13, 4), 92, 96, fc="#faf7fb", ec=PLUM, lw=1.6))
    # outer ring
    box(ax, 17, 90, 28, 7, "PCIe 5.0", PLUM, PLUM, fs=10, bold=True)
    box(ax, 49, 90, 52, 7, "MetaXLink", PLUM, PLUM, fs=10, bold=True)
    box(ax, 49, 4, 52, 7, "MetaXLink", PLUM, PLUM, fs=10, bold=True)
    box(ax, 17, 4, 20, 7, "Media Engine", PLUM, PLUM, fs=9)
    box(ax, 39, 4, 9.5, 7, "DMA", PLUM, PLUM, fs=9)
    for xp, xc in ((17, 26), (93.5, 85.5)):
        for y0, y1 in ((56, 85), (16, 47)):
            box(ax, xp, y0, 7.5, y1 - y0, "HBM PHY", PLUM, PLUM, fs=9.5, rot=90)
            box(ax, xc, y0, 6.5, y1 - y0, "Controller", PLUM, PLUM, fs=8.2, rot=90)
    # fabric + core
    box(ax, 36, 15, 46, 72, "", "#f4f0f7", GREEN, lw=1.2)
    ax.text(59, 85.0, "High Speed Fabric", ha="center", va="center", fontsize=8.5, color=GREEN)
    box(ax, 40, 76.5, 38, 6, "Command Engine", PLUM, PLUM, fs=10, bold=True)
    apc = "#b9a6cf"
    for ytop in (72.0, 34.6):
        for col in range(4):
            x0 = 40.5 + col * 9.9
            ax.add_patch(Rectangle((x0, ytop - 17.4), 8.7, 17.4, fc="white", ec="#cdbfd9", lw=0.8))
            for row in range(3):
                yy = ytop - 0.6 - row * 5.6
                box(ax, x0 + 1.0, yy - 4.2, 6.7, 4.2, "AP", apc, apc, fs=8.5)
            ax.plot([x0 + 4.35] * 3, [ytop - 15.0, ytop - 15.8, ytop - 16.6], "o", color="#8d7aa6", ms=1.8)
    box(ax, 40.5, 44.6, 37.4, 6.2, "L2 Cache", "#4caf2a", "#4caf2a", fs=10.5, bold=True)
    # notes outside the die
    ax.text(59, 118.5, "计算单元 (AP):第三方在真卡上读到 104 个;图上只画了示意", ha="center", va="center", fontsize=8.6, color=DARK)
    ax.annotate("4 个 HBM PHY\n= 4 个 HBM 堆栈\n64 GB,每个 16 GB", xy=(17, 84), xytext=(-5, 107), fontsize=8.5, color=DARK, ha="left", va="center",
                linespacing=1.35, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkB=2))
    ax.annotate("MetaXLink\n7 个高速接口\n2 到 64 卡的互连拓扑", xy=(101, 93.5), xytext=(124, 107), fontsize=8.5, color=DARK, ha="right", va="center",
                linespacing=1.35, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkB=2))
    ax.annotate("媒体引擎\n视频编解码", xy=(19, 4), xytext=(-5, -7), fontsize=8.5, color=DARK, ha="left", va="center",
                linespacing=1.35, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkB=2))
    ax.set_xlim(-7, 126)
    ax.set_ylim(-13, 122)
    ax.axis("off")
    save(fig, "fig2_block.png")


# ---------------------------------------------------------------------------------------------- fig3
def fig3_mma():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    shades = ["#cfe3f3", "#f6d6b8", "#cde8d3", "#e2d3ec"]
    for row in range(16):
        for col in range(16):
            a.add_patch(Rectangle((col, 15 - row), 1, 1, fc=shades[row // 4], ec="white", lw=0.8))
    a.set_xlim(-0.3, 16.3)
    a.set_ylim(-0.3, 16.3)
    a.set_aspect("equal")
    a.set_xticks([]), a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("一条指令的累加器:16 × 16 = 256 个 FP32", fontsize=9.5, pad=6)
    a.text(8, -1.5, "列 = r (0 到 15)", ha="center", va="center", fontsize=8.5, color=GRAY)
    a.text(-1.6, 8, "行 = k_grp × 4 + elem", ha="center", va="center", rotation=90, fontsize=8.5, color=GRAY)
    a.annotate("", xy=(16.6, 3.9), xytext=(16.6, 0.1), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9), annotation_clip=False)
    a.text(16.9, 2.0, "k_grp = 0", ha="left", va="center", fontsize=8, color=GRAY, clip_on=False)
    a.text(16.9, 10.0, "k_grp = 1, 2, 3\n同样 4 行一组", ha="left", va="center", fontsize=8, color=GRAY, clip_on=False)
    # right: one lane
    b.axis("off")
    b.set_xlim(0, 10), b.set_ylim(0, 10)
    box(b, 0.2, 6.6, 9.6, 2.9, "", "#f4f0f7", PLUM, lw=1.3)
    b.text(0.6, 8.9, "一个线程里的片段 (整条波前 64 个线程, 人手一份)", ha="left", va="center", fontsize=8.6, color=DARK)
    for i, (lab, sub, c) in enumerate((("A 片段", "4 个 BF16", BLUE), ("B 片段", "4 个 BF16", ORANGE), ("D 累加器", "4 个 FP32", GREEN))):
        x = 0.6 + i * 3.15
        box(b, x, 6.95, 2.9, 1.55, f"{lab}\n{sub}", c, c, fs=8.6)
    lines = [
        "线程编号 = k_grp × 16 + r,负责第 r 列里连续的 4 行",
        "一条指令:16 × 16 × 16,做 4096 次乘加",
        "累加器 64 个线程 × 4 个 = 256 个,正好铺满左边的 16 × 16",
        "输入 A、B 各 64 个线程 × 4 个 = 256 个数,各是一张 16 × 16",
        "算完落在 FP32 里,不丢精度",
    ]
    for i, t in enumerate(lines):
        b.text(0.4, 5.6 - i * 0.95, "· " + t, ha="left", va="center", fontsize=8.6, color=DARK)
    b.text(0.4, 0.5, "数据来自第三方在真卡上的探针,不是沐曦公开的文档", ha="left", va="center", fontsize=7.8, color=GRAY)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_mma.png")


# ---------------------------------------------------------------------------------------------- fig4
def fig4_plateau():
    p = calc["plateau"]
    ms = p["m"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    for ax, (key, title) in zip(axes, (("gate_up", "gate_up 17408×5120,178.3 MB"), ("o_proj", "o_proj 5120×6144,62.9 MB"))):
        d = p[key]
        us = np.array(d["us"])
        ax.step(ms, us, where="mid", color=BLUE, lw=1.4)
        ax.plot(ms, us, "o", color=BLUE, ms=4)
        for m_, u_ in ((1, us[0]), (32, us[6]), (48, us[7]), (128, us[10]), (192, us[11])):
            if m_ == 1:    # under its point, clear of the M = 4..32 cluster
                ax.annotate(f"{u_:.0f} µs", (m_, u_), xytext=(-2, -7), textcoords="offset points", ha="left", va="top", fontsize=7.6, color=BLUE)
            else:
                ax.annotate(f"{u_:.0f} µs", (m_, u_), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=7.6, color=BLUE)
        for x in (40, 160):
            ax.axvline(x, color=RED, lw=0.9, linestyle=(0, (5, 4)))
        ax.set_xticks([1, 32, 48, 64, 96, 128, 192, 256], ["1", "32", "48", "64", "96", "128", "192", "256"], fontsize=7.6)
        ax.set_xlim(-10, 270)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("验证的行数 M")
        ax.set_ylim(0, us.max() * 1.32)
        clean(ax)
    axes[0].set_ylabel("矩阵乘的耗时 (µs),BF16,mcBLAS")
    fig.suptitle("真卡上量出来的矩阵乘 (Qwen3.8-27B 的两个投影层,mcBLAS)。红色虚线:耗时跳一截的地方", fontsize=9.5, y=1.02)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_plateau.png")


# ---------------------------------------------------------------------------------------------- fig5
def fig5_compare():
    c = calc["compare"]
    panels = [  # (title, key, unit, measured value for C500 or None, digits)
        ("BF16 / FP16 稠密算力 (TFLOPS)\n实测:大矩阵乘的峰值", "bf16_tflops", "", calc["gemm"]["bf16_measured_tflops"], "{:.0f}"),

        ("显存带宽 (TB/s)\n实测:读权重的最高速度", "hbm_tbps", "", round(calc["hbm"]["measured_gbps"]["gemm_read_m1_lm_head"] / 1000, 2), "{:.2f}"),
        ("显存容量 (GB)", "memory_gb", "", None, "{:.0f}"),
        ("卡间互联 (GB/s)", "interconnect_gbps", "", None, "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.4))
    for ax, (title, key, unit, meas, fmt) in zip(axes, panels):
        vals = [c[key]["c500"], c[key]["a100"], c[key]["h20"]]
        cols = [BLUE, GREEN, GOLD]
        bars = ax.bar([0, 1, 2], vals, width=0.62, color=cols)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=8.2, color=DARK)
        if meas is not None:
            ax.bar([0], [meas], width=0.62, color="white", ec=RED, lw=1.4, hatch="////", fc=(0, 0, 0, 0))
            ax.text(0, meas * 0.5, "实测\n" + fmt.format(meas), ha="center", va="center", fontsize=7.8, color=RED, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=0.8))
        ax.set_xticks([0, 1, 2], ["C500", "A100", "H20"], fontsize=8.6)
        ax.set_title(title, fontsize=8.4, linespacing=1.4)
        ax.set_yticks([])
        ax.set_ylim(0, max(vals + ([meas] if meas else [])) * 1.2)
        clean(ax)
    fig.suptitle("纸面上的 C500 (第三方转述的峰值) 对 A100 和 H20;红框是 C500 在真卡上量到的", fontsize=9.5, y=1.04)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig5_compare.png")


# ---------------------------------------------------------------------------------------------- fig6, fig7
def _box_width(text):
    """Width of an item box in axes units (100 units across a 7.2 in figure). Calibrated on Noto Serif CJK JP at 7 pt:
    a Latin character is about 0.69 em wide, a CJK character 1 em; 2.4 units of padding in total."""
    latin = sum(1 for ch in text if ord(ch) <= 0x2E80)
    return 0.93 * latin + 1.35 * (len(text) - latin) + 2.4


def _layout(items, avail, gap=1.0):
    """One row if it fits, otherwise the two-row split that minimises the wider row (in order, nothing dropped).
    Whatever is left over in a row is spread over its boxes in proportion to their text width."""
    one = [_box_width(it) for it in items]
    if sum(one) + gap * (len(items) - 1) <= avail:
        return [(items, _spread(one, avail, gap))]
    best = None
    for k in range(1, len(items)):
        a, b = items[:k], items[k:]
        wa, wb = _spread([_box_width(it) for it in a], avail, gap), _spread([_box_width(it) for it in b], avail, gap)
        need = max(sum(_box_width(it) for it in a) + gap * (k - 1), sum(_box_width(it) for it in b) + gap * (len(b) - 1))
        if best is None or need < best[0]:
            best = (need, [(a, wa), (b, wb)])
    return best[1]


def _spread(widths, avail, gap):
    spare = avail - sum(widths) - gap * (len(widths) - 1)
    return [w + max(spare, 0) * w / sum(widths) for w in widths]


def fig6_stack():
    """The MXMACA stack in the four layers of the inquiry reply (8-1-89), from the hardware up."""
    layers = [  # bottom to top: (title, color, items)
        ("软件驱动层", "#e3e3e3", ["内核模式驱动", "用户态驱动", "内存访问与页表", "虚拟化和容器", "性能分析:mcProfiler,mcTracer,cycle-trace", "代码调试"]),
        ("编程语言层", "#ead9f3", ["mxcc (基于 LLVM 12)", "cucc:编译 CUDA 源码", "cu-bridge,make_maca,cmake_maca", "MXMACA C++ / Fortran", "mcTriton"]),
        ("数学库及通信库层", "#f7e3d3", ["mcBLAS", "mcDNN", "mcFFT", "mcSPARSE", "mcSOLVER", "mcRAND", "mcThrust,mcCUB", "MCCL", "mcTlass,mcoplib,mcFlashAttention"]),
        ("算法应用层", "#dcebf7", ["PyTorch,TensorFlow,JAX", "vLLM,SGLang,LMDeploy", "Megatron-LM,DeepSpeed", "PaddlePaddle", "FlashAttention,FlashInfer,FlashMLA", "1,000 多个开源模型"]),
    ]
    h, gap, left, right = 11.0, 2.0, 24.0, 98.5
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    total = 9.5 + len(layers) * (h + gap) + 4.0
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total + 2)
    # hardware
    y0 = 2.0
    ax.add_patch(FancyBboxPatch((1, y0), 98, 9.0, boxstyle="round,pad=0.02,rounding_size=0.8", fc=PLUM, ec=PLUM))
    ax.text(50, y0 + 4.5, "硬件:XCORE 1.0 · 计算单元 · LDS 与两级缓存 · 4 颗 HBM2e 共 64 GB · MetaXLink 与 PCIe 5.0",
            ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")
    y = y0 + 9.0 + gap
    for title, color, items in layers:
        ax.add_patch(Rectangle((1, y), 98, h, fc=color, ec="#9a9a9a", lw=0.9))
        ax.text(3.0, y + h / 2, title, ha="left", va="center", fontsize=9.2, fontweight="bold", color=DARK)
        layout = _layout(items, right - left)
        box_h = (h - 0.8 - 0.6 * (len(layout) - 1)) / len(layout)
        for r, (row_items, widths) in enumerate(layout):
            x = left
            by = y + h - 0.4 - (r + 1) * box_h - r * 0.6
            for it, w in zip(row_items, widths):
                ax.add_patch(FancyBboxPatch((x, by), w, box_h, boxstyle="round,pad=0.02,rounding_size=0.5", fc="white", ec="#8a8a8a", lw=0.8))
                ax.text(x + w / 2, by + box_h / 2, it, ha="center", va="center", fontsize=7.0, color=DARK)
                x += w + 1.0
        y += h + gap
    ax.text(1.5, y + 1.4, "分层照问询回复里的四层。对上呈现 CUDA 风格的接口:核函数不用改,主机 API 把前缀 cuda 换成 mc。",
            ha="left", va="center", fontsize=8.4, color=RED)
    ax.axis("off")
    save(fig, "fig6_stack.png")


def fig7_hierarchy():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    a.set_xlim(0, 10), a.set_ylim(0, 10)
    a.axis("off")
    # nesting: grid > workgroup > wavefront > thread
    a.add_patch(FancyBboxPatch((0.3, 0.4), 9.4, 9.2, boxstyle="round,pad=0.02,rounding_size=0.4", fc="#f6f6f6", ec=GRAY, lw=1.1))
    a.text(0.7, 9.1, "网格 (ND-Range)", ha="left", va="center", fontsize=9.5, fontweight="bold", color=DARK)
    for i, (x, lab) in enumerate(((0.9, "工作组 1"), (3.75, "工作组 2"), (6.6, "工作组 …"))):
        a.add_patch(FancyBboxPatch((x, 0.9), 2.5, 7.5, boxstyle="round,pad=0.02,rounding_size=0.3", fc="white", ec=BLUE, lw=1.1))
        a.text(x + 1.25, 7.9, lab, ha="center", va="center", fontsize=8.6, color=BLUE, fontweight="bold")
        for j in range(3 if i < 2 else 1):
            yj = 6.3 - j * 1.9
            a.add_patch(FancyBboxPatch((x + 0.25, yj - 1.5), 2.0, 1.5, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#dcebf7", ec=DARK, lw=0.9))
            a.text(x + 1.25, yj - 0.3, "波前 " + str(j + 1), ha="center", va="center", fontsize=8.2, color=DARK)
            for k in range(6):
                a.add_patch(Rectangle((x + 0.45 + k * 0.27, yj - 1.3), 0.2, 0.42, fc=BLUE, ec="none"))
        if i == 2:
            a.plot([x + 1.25] * 3, [2.2, 1.8, 1.4], "o", color=BLUE, ms=2.0)
    a.set_ylim(-1.3, 10)
    a.text(5.0, -0.75, "波前是硬件调度的单位,每个 64 个线程;\n一个工作组里的线程共用一块 LDS。", ha="center", va="center", fontsize=8.2, color=GRAY, linespacing=1.5)
    # right: memory hierarchy
    b.set_xlim(0, 10), b.set_ylim(0, 10)
    b.axis("off")
    rows = [  # (name, size, note, color)
        ("寄存器", "每个线程私有", "放线程自己的变量", "#ead9f3"),
        ("LDS (共享内存)", "上限 64 KB / 工作组", "32 个 bank,组内线程共用", "#d9efd9"),
        ("缓存 (L1、L2)", "容量未公开", "L1 在单元内,L2 共享", "#dcebf7"),
        ("HBM2e", "64 GB", "4 颗,单通道 3.6 Gbps", "#f7e3d3"),
        ("主机内存", "—", "走 PCIe 5.0 或统一内存", "#e6e6e6"),
    ]
    for i, (name, size, note, color) in enumerate(rows):
        y = 8.3 - i * 1.75
        b.add_patch(FancyBboxPatch((0.3, y - 1.25), 9.4, 1.5, boxstyle="round,pad=0.02,rounding_size=0.3", fc=color, ec="#8a8a8a", lw=0.9))
        b.text(0.7, y - 0.25, name, ha="left", va="center", fontsize=9.2, fontweight="bold", color=DARK)
        b.text(0.7, y - 0.95, size, ha="left", va="center", fontsize=8.2, color=DARK)
        b.text(4.6, y - 0.5, note, ha="left", va="center", fontsize=8.2, color=GRAY)
    b.text(5.0, 9.6, "越往下越大,也越慢", ha="center", va="center", fontsize=8.6, color=GRAY)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig7_hierarchy.png")


if __name__ == "__main__":
    fig1_lineage()
    fig2_block()
    fig3_mma()
    fig4_plateau()
    fig5_compare()
    fig6_stack()
    fig7_hierarchy()
