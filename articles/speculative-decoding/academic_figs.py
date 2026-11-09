#!/usr/bin/env python3
"""Academic-style figures for the speculative-decoding article.

Style: paper-Figure convention — serif (Noto Serif CJK + STIX math),
inward ticks, Paul Tol colorblind-safe palette, restrained fills,
(a)/(b) panel labels. All numbers quoted from papers are cited in captions.

  figA_membound.png  prefill vs decode per weight-read          [schematic]
  figB_timeline.png  autoregressive chain vs draft+verify       [schematic]
  figC_accept.png    accept / reject / rollback                 [schematic]
  figD_dflash.png    DFlash chunk mechanism                     [schematic]
  figR1_accept.png   acceptance length (DSpark Tab.1)           [data]
  figR2_decay.png    position-wise decay (DSpark Fig.2)         [data]
  figR3_speed.png    measured speedups, six generations         [data]

Run:  python academic_figs.py    -> ./images/
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Paul Tol (colorblind-safe)
BLUE, RED, GREEN, ORANGE, TEAL, GREY = "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#BBBBBB"
DBLUE, DRED, DGREEN, DORANGE = "#2288CC", "#BB3333", "#117733", "#AA9922"


def rrect(ax, x, y, w, h, ec, fc="white", lw=1.0, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def arr(ax, x1, y1, x2, y2, color="#22222 2".replace(" ", ""), lw=1.2, ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def cell(ax, x, y, s, ec="#333333", fc="white", fs=8.5, lw=0.9, tc="#111111", z=3):
    ax.add_patch(Rectangle((x, y), 1, 1, ec=ec, fc=fc, lw=lw, zorder=z))
    ax.text(x + .5, y + .5, s, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def panel(ax, label):
    ax.text(0.01, 0.02, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


# ================================================================ fig A
def figA_membound():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # (a) prefill
    ax = axes[0]
    rrect(ax, 1, 4.4, 8, 1.1, "#333333", "#e8e8e8")
    ax.text(5, 4.95, "权重读取 $\\|W\\|$(140 GB)", ha="center", va="center", fontsize=9)
    arr(ax, 5, 4.3, 5, 3.3)
    for i in range(9):
        ax.add_patch(Rectangle((0.8 + i * 0.92, 2.1), 0.72, 0.72, fc="#228833", ec="none"))
    ax.text(9.35, 2.45, "…", fontsize=11, va="center")
    ax.text(5, 1.2, "一次访存并行计算全部位置\n(≈$10^3$ tokens,compute-bound)", ha="center",
            va="center", fontsize=9, linespacing=1.5)
    ax.set_title("Prefill", fontsize=11)
    panel(ax, "(a)")

    # (b) decode
    ax = axes[1]
    rrect(ax, 1, 4.4, 8, 1.1, "#333333", "#e8e8e8")
    ax.text(5, 4.95, "权重读取 $\\|W\\|$(140 GB)", ha="center", va="center", fontsize=9)
    arr(ax, 5, 4.3, 5, 3.3, color=DRED)
    ax.add_patch(Rectangle((4.6, 2.1), 0.8, 0.8, fc="#228833", ec="none"))
    ax.text(5, 1.2, "一次访存仅计算 1 个位置\n(memory-bound,算力闲置 >95%)", ha="center",
            va="center", fontsize=9, linespacing=1.5, color=DRED)
    ax.set_title("Decode (batch = 1)", fontsize=11)
    panel(ax, "(b)")

    fig.suptitle("Decode 阶段的瓶颈:每生成一个 token 都需完整读取权重;速度上限 $\\approx$ 显存带宽 $\\div$ 权重大小",
                 fontsize=10.5, y=1.04)
    fig.savefig(f"{OUT}/figA_membound.png"); plt.close(fig)


# ================================================================ fig B
def figB_timeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    # (a) autoregressive
    ax.text(0.2, 7.5, "(a) 自回归:$x_{t}$ 的输出是 $x_{t+1}$ 的输入(串行)", fontsize=9.5)
    for i in range(4):
        x = 0.4 + i * 3.1
        rrect(ax, x, 5.2, 2.3, 1.6, "#333333", "white")
        ax.text(x + 1.15, 6.0, f"前向\n$x_{{{i+1}}}$", ha="center", va="center",
                fontsize=8.5, linespacing=1.4)
        if i:
            arr(ax, x - 0.8, 6.0, x - 0.05, 6.0, color=DRED, lw=1.1)
    ax.text(13.0, 6.0, "4 次访存\n→ 4 tokens", fontsize=8.5, va="center", ha="center",
            color=DRED, linespacing=1.5)

    # (b) speculative
    ax.text(0.2, 4.3, "(b) 投机解码:小模型连猜 $\\gamma$ 个,大模型单次前向验证 $\\gamma{+}1$ 个位置",
            fontsize=9.5)
    rrect(ax, 0.4, 1.4, 3.4, 2.2, BLUE, "#eef4fa")
    ax.text(2.1, 2.5, "draft\n小模型连猜 $\\gamma$ 个\n(读取小权重)", ha="center", va="center",
            fontsize=8.5, linespacing=1.5)
    arr(ax, 3.8, 2.5, 4.5, 2.5)
    rrect(ax, 4.5, 1.4, 4.2, 2.2, DBLUE, "#e8f0f8")
    ax.text(6.6, 2.5, "verify\n大模型一次前向\n并行验证 $\\gamma{+}1$ 位置", ha="center", va="center",
            fontsize=8.5, linespacing=1.5)
    arr(ax, 8.7, 2.5, 9.4, 2.5)
    for i, (s, c) in enumerate([("$\\checkmark$", DGREEN), ("$\\checkmark$", DGREEN),
                                ("重采样", DORANGE)]):
        rrect(ax, 9.4 + i * 1.1, 1.85, 1.0, 1.3, c)
        ax.text(9.9 + i * 1.1, 2.5, s, ha="center", va="center", fontsize=8.5)
    ax.text(13.0, 2.5, "1 次访存\n→ 3 tokens", fontsize=8.5, va="center", ha="center",
            color=DGREEN, linespacing=1.5)

    fig.savefig(f"{OUT}/figB_timeline.png"); plt.close(fig)


# ================================================================ fig C
def figC_accept():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")

    toks = [("深度", DGREEN, "✓"), ("学习", DGREEN, "✓"), ("的", DGREEN, "✓"),
            ("本质", DRED, "✗"), ("能力", "#999999", "—")]
    for i, (t, c, v) in enumerate(toks):
        x = 0.5 + i * 2.4
        rrect(ax, x, 4.6, 2.0, 1.7, c, "white", lw=1.2)
        ax.text(x + 1.0, 5.75, f"draft $x_{{{i+1}}}$", ha="center", fontsize=8, color="#666666")
        ax.text(x + 1.0, 5.28, f"'{t}'", ha="center", fontsize=9)
        ax.text(x + 1.0, 4.95, v, ha="center", fontsize=9, color=c)
    # rollback cut
    ax.plot([4.7, 4.7], [4.2, 6.6], ls=(0, (3, 2)), lw=1.1, color=DRED)
    arr(ax, 4.7, 4.4, 4.7, 3.4, color=DORANGE)
    rrect(ax, 3.4, 2.0, 3.6, 1.2, DORANGE, "#fdf9e8")
    ax.text(5.2, 2.6, "按 $p_{\\mathrm{target}}$ 重采样", ha="center", va="center", fontsize=8.5)

    # accepted prefix brace
    ax.annotate("", xy=(0.6, 6.9), xytext=(6.9, 6.9),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=DGREEN))
    ax.text(3.75, 7.15, "接受前缀 $\\tau = 3$", ha="center", fontsize=9, color=DGREEN)

    ax.text(8.6, 3.1, "净产出 $\\tau{+}1 = 4$ tokens;\nrejection sampling 保证\n输出分布 $\\equiv p_{\\mathrm{target}}$(无损)",
            fontsize=9, va="center", linespacing=1.6,
            bbox=dict(boxstyle="square,pad=0.4", ec="#bbbbbb", fc="white", lw=0.8))

    fig.savefig(f"{OUT}/figC_accept.png"); plt.close(fig)


# ================================================================ fig D
def figD_dflash():
    """DFlash chunk mechanism: anchor + masked block -> one parallel pass ->
    draft block -> target verifies prefix -> last accepted token is next anchor."""
    fig, ax = plt.subplots(figsize=(7.6, 7.8))
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 20); ax.axis("off")

    # ---------- layer 1: sequence & chunking ----------
    ax.text(0.2, 19.4, "① 序列按块(chunk)推进:每块以 1 个已确认 token(anchor)开头",
            fontsize=9.5)
    for i in range(6):
        cell(ax, 0.5 + i * 1.05, 17.2, "$\\cdot$", ec="#999999", fs=8)
    cell(ax, 0.5 + 6 * 1.05, 17.2, "A", ec=DGREEN, fc="#eef7ee", fs=8.5)
    ax.text(7.3, 18.55, "anchor", fontsize=7.5, color=DGREEN, ha="center")
    for i in range(8):
        rrect(ax, 8.0 + i * 0.72, 17.2, 0.7, 1.0, "#999999", ls=(0, (2, 2)), lw=0.8)
    ax.text(10.9, 16.55, "下一块(待生成 $\\gamma=8$)", fontsize=7.5, ha="center", color="#666666")

    # ---------- layer 2: dflash drafter ----------
    arr(ax, 7, 16.9, 7, 15.6)
    rrect(ax, 0.4, 9.6, 13.7, 5.8, DBLUE, "white", lw=1.2)
    ax.text(0.7, 14.85, "② DFlash 草稿器(block diffusion):一次前向,并行输出整块",
            fontsize=9.5)
    # input row: anchor + masks (label centered above the row)
    cell(ax, 1.2, 12.6, "A", ec=DGREEN, fc="#eef7ee", fs=8.5)
    for i in range(8):
        cell(ax, 2.5 + i * 1.05, 12.6, "?", ec="#aaaaaa", fs=8.5, tc="#999999")
    ax.text(1.2, 13.8, "输入:anchor + $\\gamma$ 个掩码位", fontsize=7.5, color="#555555",
            ha="left", va="bottom")
    # parallel arrow band
    for i in range(8):
        arr(ax, 3.0 + i * 1.05, 12.4, 3.0 + i * 1.05, 11.1, color=DBLUE, lw=0.9, ms=9)
    # draft output row (label right-aligned to row center)
    for i in range(8):
        cell(ax, 2.5 + i * 1.05, 10.0, f"$d_{{{i+1}}}$", ec=DBLUE, fs=8.5)
    ax.text(2.2, 10.5, "草稿块", fontsize=8, color="#555555", va="center", ha="right")
    # KV injection (right side, inside frame with breathing room)
    rrect(ax, 12.3, 10.0, 1.55, 2.2, "#888888", "#f2f2f2", lw=0.9)
    ax.text(13.07, 11.1, "target\n隐状态\n(KV 注入)", fontsize=7.2, ha="center",
            va="center", linespacing=1.4)
    arr(ax, 12.25, 11.1, 11.75, 11.1, color="#888888")
    ax.text(12.55, 13.3, "同时输出\n(非自回归)", fontsize=7.5, color=DBLUE,
            va="center", ha="center", linespacing=1.5)

    # ---------- layer 3: target verify ----------
    arr(ax, 7, 9.5, 7, 8.7)
    rrect(ax, 0.4, 3.6, 13.7, 5.0, DRED, "white", lw=1.2)
    ax.text(0.7, 8.15, "③ target 单次前向验证:逐位置比对,首个分歧处截断",
            fontsize=9.5)
    for i in range(8):
        verdict = "✓" if i < 5 else ("✗" if i == 5 else "—")
        c = DGREEN if i < 5 else (DRED if i == 5 else "#bbbbbb")
        cell(ax, 1.2 + i * 1.05, 5.8, verdict, ec=c, fs=8.5)
    ax.text(1.2, 7.0, "判决", fontsize=7.5, color="#555555", ha="left", va="bottom")
    ax.plot([7.5, 7.5], [5.4, 7.3], ls=(0, (3, 2)), lw=1.1, color=DRED)
    arr(ax, 6.95, 5.55, 6.95, 4.75, color=DORANGE)
    cell(ax, 6.45, 3.85, "重采", ec=DORANGE, fc="#fdf9e8", fs=7.5)
    ax.text(8.6, 4.9, "前缀 5 接受,第 6 重采样,其后丢弃", fontsize=7.5,
            va="center", ha="left")

    # ---------- layer 4: append & loop ----------
    arr(ax, 6.85, 3.75, 6.85, 2.95, color=DORANGE)          # resample -> layer4
    ax.text(0.2, 2.35, "④ 确认 6 个 token 并入序列,末位 $A'$ 成为下一块的 anchor",
            fontsize=9.5)
    for i in range(6):
        cell(ax, 0.5 + i * 1.05, 0.7, "" if i < 5 else "$A'$", ec=DGREEN, fs=8)
    # true loop arrow back to layer 1
    ax.plot([8.6, 14.7], [1.2, 1.2], ls=(0, (4, 3)), lw=1.1, color=DGREEN)
    ax.plot([14.7, 14.7], [1.2, 17.7], ls=(0, (4, 3)), lw=1.1, color=DGREEN)
    arr(ax, 14.7, 17.7, 13.9, 17.7, color=DGREEN, lw=1.1)
    ax.text(12.2, 1.55, "回到 ①,继续下一块", fontsize=8.5, color=DGREEN)

    fig.savefig(f"{OUT}/figD_dflash.png"); plt.close(fig)


# ================================================================ fig R1
def figR1_accept():
    models = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
    eagle3 = [3.72, 3.80, 3.68]
    dflash = [4.29, 4.29, 4.26]
    dspark = [4.95, 4.96, 5.00]

    x = np.arange(3); w = 0.24
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for off, v, c, lab in [(-w, eagle3, GREY, "EAGLE-3"), (0, dflash, TEAL, "DFlash"),
                           (w, dspark, BLUE, "DSpark")]:
        ax.bar(x + off, v, w, color=c, ec="white", lw=0.5, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.07, f"{vi:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("接受长度 $\\tau$")
    ax.set_ylim(0, 5.7)
    ax.legend(fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.annotate("vs EAGLE-3\n+31% / +27% / +30%", xy=(2, 5.35), fontsize=7.5,
                color=DBLUE, ha="center")
    fig.savefig(f"{OUT}/figR1_accept.png"); plt.close(fig)


# ================================================================ fig R2
def figR2_decay():
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    series = [("DFlash·Code", 0.87, 0.78, DBLUE, "o"),
              ("DFlash·Chat", 0.72, 0.63, TEAL, "o"),
              ("EAGLE-3·Chat", 0.53, 0.74, GREY, "s"),
              ("DSpark·Math", 0.93, 0.92, DRED, "D")]
    for name, a, b, c, m in series:
        ax.plot([1, 8], [a, b], "--", color=c, lw=1.1)
        ax.scatter([1, 8], [a, b], color=c, marker=m, s=28, zorder=3)
        ax.annotate(f"{a:.2f}", (1, a), xytext=(-4, 6), textcoords="offset points",
                    fontsize=7.5, color=c)
        ax.annotate(f"{b:.2f}", (8, b), xytext=(5, -2), textcoords="offset points",
                    fontsize=7.5, color=c)
        ax.text(8.35, b + 0.005, name, fontsize=7.5, color=c, va="center")
    ax.set_xlim(0.5, 11.8); ax.set_ylim(0.45, 1.0)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("块内位置 $j$")
    ax.set_ylabel("条件接受率")
    fig.savefig(f"{OUT}/figR2_decay.png"); plt.close(fig)


# ================================================================ fig R3
def figR3_speed():
    rows = [
        ("Speculative Sampling", "2023", 2.0, 3.0, GREY, "Leviathan et al.(T5-XXL)"),
        ("Medusa", "2023", 2.2, 2.8, GREY, "Cai et al.(Vicuna-33B)"),
        ("EAGLE-2/3", "2024-25", 3.05, 4.3, TEAL, "Li et al."),
        ("DeepSeek-V3 MTP", "2024", 1.8, 1.8, GREEN, "技术报告(SGLang)"),
        ("Qwen3-27B MTP", "2025", 2.69, 2.69, GREEN, "Jetson AI Lab 实测"),
        ("DSpark(V4 生产)", "2026", 1.60, 1.85, BLUE, "同等吞吐 per-user +60~85%"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, (name, yr, lo, hi, c, src) in zip(ys, rows):
        left, width = (lo, hi - lo) if hi > lo else (lo - 0.045, 0.09)
        ax.barh(y, width, left=left, height=0.5, color=c, ec="white", lw=0.5)
        lab = f"{lo:.1f}×" if lo == hi else f"{lo:.1f}–{hi:.1f}×"
        ax.text(hi + 0.1, y + 0.08, lab, fontsize=8, va="center", color="#333333")
        ax.text(hi + 0.62, y + 0.08, src, fontsize=7, va="center", color="#888888")
        ax.text(0.06, y - 0.32, f"{yr}", fontsize=6.5, va="center", color="#aaaaaa")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 6.6)
    ax.set_xlabel("实测加速倍数(各论文原生口径,不可直接横比)")
    ax.axvline(1.0, color="#cccccc", lw=0.8, ls=":")
    fig.savefig(f"{OUT}/figR3_speed.png"); plt.close(fig)


if __name__ == "__main__":
    figA_membound(); print("figA ok")
    figB_timeline(); print("figB ok")
    figC_accept(); print("figC ok")
    figD_dflash(); print("figD ok")
    figR1_accept(); print("figR1 ok")
    figR2_decay(); print("figR2 ok")
    figR3_speed(); print("figR3 ok")
    print("->", OUT)
