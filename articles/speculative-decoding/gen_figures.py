#!/usr/bin/env python3
"""Concept figures for the speculative-decoding article (v2: big-type, readable at 690px).

Design rule: figure width ~9in @150dpi -> shown at Zhihu's ~690px body width,
fonts must stay >= 13pt so they remain readable after 0.5x downscale.

figA_membound.png   — prefill vs decode: what one full weight-read buys you
figB_timeline.png   — autoregressive t→t+1 chain vs draft+verify
figC_accept.png     — accept / reject / rollback

Run:  python gen_figures.py   (outputs to ./images/)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

cjk = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                        "WenQuanYi Micro Hei"]
            if n in {f.name for f in font_manager.fontManager.ttflist}), None)
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("CJK font:", cjk)

BLUE, LBLUE, GREEN, RED, ORANGE, GREY = "#1f77b4", "#aec7e8", "#2ca02c", "#d62728", "#ff7f0e", "#6b6b6b"


def rbox(ax, x, y, w, h, fc, ec, lw=2.0, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=lw, alpha=alpha))


def arrow(ax, x1, y1, x2, y2, color="#333", lw=2.4, ms=22):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw))


# ---------------------------------------------------------------- fig A
def figA_membound():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.6))
    for ax in axes:
        ax.set_xlim(0, 24); ax.set_ylim(0, 12.5); ax.axis("off")

    # ---- left: prefill ----
    ax = axes[0]
    ax.set_title("Prefill:一次访存 → 一整段", fontsize=16, color="#174a73", pad=8, fontweight="bold")
    rbox(ax, 2, 9.3, 20, 2.4, "#6baed6", "#2171b5")
    ax.text(12, 10.5, "读一遍全部权重(140 GB)", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    arrow(ax, 12, 9.1, 12, 7.6, "#2171b5")
    for i in range(9):
        rbox(ax, 2 + i * 2.25, 5.6, 1.9, 1.35, GREEN, "none")
    ax.text(20.6, 6.3, "…", fontsize=18, color=GREEN, ha="left", va="center")
    ax.text(12, 4.3, "≈1000 个 token 并行算出", ha="center", fontsize=13.5, color="#1b6b2b")
    rbox(ax, 2, 1.1, 20, 2.3, "#eaf5ea", "#2ca02c", lw=1.6)
    ax.text(12, 2.25, "算力打满(compute-bound)", ha="center", fontsize=14, color="#1b6b2b", fontweight="bold")

    # ---- right: decode ----
    ax = axes[1]
    ax.set_title("Decode:一次访存 → 1 个", fontsize=16, color="#a01717", pad=8, fontweight="bold")
    rbox(ax, 2, 9.3, 20, 2.4, "#6baed6", "#2171b5")
    ax.text(12, 10.5, "读一遍全部权重(140 GB)", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    arrow(ax, 12, 9.1, 12, 7.6, "#d62728")
    rbox(ax, 10.9, 5.6, 2.2, 1.5, GREEN, "none")
    ax.text(12, 3.9, "每生成 1 个 token 都要付一遍", ha="center", fontsize=13, color="#333")
    ax.text(12, 2.9, "同样的访存成本", ha="center", fontsize=13, color="#333")
    rbox(ax, 2, 0.7, 20, 2.1, "#fdeaea", "#d62728", lw=1.6)
    ax.text(12, 1.75, "算力闲置 > 95%(memory-bound)", ha="center", fontsize=14, color="#a01717", fontweight="bold")

    fig.suptitle("decode 快不起来的根源:瓶颈是访存,不是算力(batch=1)", fontsize=17)
    fig.text(0.5, 0.015, "生成速度上限 ≈ 显存带宽 ÷ 权重大小(跟算力没关系)",
             ha="center", fontsize=14,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec="#bbbbbb"))
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(f"{OUT}/figA_membound.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig B
def figB_timeline():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.set_xlim(0, 25); ax.set_ylim(0, 11.5); ax.axis("off")

    def block(x, y, w, h, fc, ec, lines, fs=13, tc="white", weight="normal"):
        rbox(ax, x, y, w, h, fc, ec)
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, linespacing=1.5)

    # ---- row 1 ----
    ax.text(0.4, 10.3, "传统自回归:token t 的输出 = t+1 的输入,只能一个等一个",
            fontsize=14.5, fontweight="bold")
    y1, bw, bh = 6.6, 4.9, 2.6
    for i in range(4):
        x = 0.8 + i * (bw + 1.2)
        block(x, y1, bw, bh, BLUE, "#174a73", ["大模型前向", f"产出 t+{i+1}"])
        if i:
            arrow(ax, x - 1.18, y1 + bh / 2, x - 0.04, y1 + bh / 2, "#d62728", lw=2.2, ms=16)
            ax.text(x - 0.6, y1 + bh + 0.28, "等", fontsize=11.5, color="#d62728", ha="center")
    ax.text(12.4, 5.35, "4 次完整访存 → 4 个 token", fontsize=14, color=RED,
            ha="center", fontweight="bold")

    # ---- row 2 ----
    ax.text(0.4, 4.15, "投机解码:小模型连猜 → 大模型一次并行验证", fontsize=14.5, fontweight="bold")
    y2 = 1.0
    block(0.8, y2, 6.9, 2.6, LBLUE, "#7fa8d0", ["小模型连猜 t+1 t+2 t+3", "(读小权重,极便宜)"], tc="#1a1a1a")
    arrow(ax, 7.75, y2 + 1.3, 8.55, y2 + 1.3)
    block(8.55, y2, 9.0, 2.6, BLUE, "#174a73", ["大模型一次前向", "并行验证 4 个位置"])
    arrow(ax, 17.6, y2 + 1.3, 18.4, y2 + 1.3)
    block(18.4, y2, 5.4, 1.22, GREEN, "#1b6b2b", ["接受 t+1、t+2,重采样出 t+3"], fs=12)
    block(18.4, y2 + 1.38, 5.4, 1.22, "#efefef", "#9a9a9a", ["t+4 起丢弃"], fs=12, tc="#6b6b6b")
    ax.text(21.1, 0.35, "1 次访存 → 3 个 token", fontsize=14, color=GREEN,
            ha="center", fontweight="bold")

    ax.set_title("投机解码:把闲置算力换成一次验证多个位置", fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figB_timeline.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig C
def figC_accept():
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    ax.set_xlim(0, 25); ax.set_ylim(0, 12); ax.axis("off")

    ax.text(0.5, 10.6, "验证规则:逐个对答案,第一个'想法不同'的地方就回滚",
            fontsize=15.5, fontweight="bold")

    toks = [
        ("t+1", "深度学习", "√ 接受", GREEN),
        ("t+2", "的", "√ 接受", GREEN),
        ("t+3", "本质", "√ 接受", GREEN),
        ("t+4", "是拟合", "× 拒绝", RED),
        ("t+5", "能力", "× 丢弃", GREY),
    ]
    x0, y0, w, h = 0.9, 5.9, 4.2, 3.4
    for i, (pos, tok, verdict, color) in enumerate(toks):
        x = x0 + i * (w + 0.55)
        faded = color == GREY
        rbox(ax, x, y0, w, h, "#fafafa", color, lw=2.6, alpha=0.55 if faded else 1.0)
        ax.text(x + w / 2, y0 + h - 0.8, pos, ha="center", fontsize=13, color="#888")
        ax.text(x + w / 2, y0 + h / 2 - 0.15, f"草稿:'{tok}'", ha="center", fontsize=14.5, color="#222")
        ax.text(x + w / 2, y0 + 0.75, verdict, ha="center", fontsize=14.5, color=color, fontweight="bold")

    xr = x0 + 3 * (w + 0.55)
    arrow(ax, xr + w / 2, y0 - 0.12, xr + w / 2, y0 - 1.35, ORANGE)
    rbox(ax, xr - 1.5, y0 - 3.5, w + 3.0, 1.9, "#fff3e0", ORANGE, lw=2.2)
    ax.text(xr + w / 2, y0 - 2.55, "按大模型分布重采样:'核心'", ha="center",
            fontsize=14, color="#b35a00", fontweight="bold")

    rbox(ax, 2.2, 0.25, 20.6, 1.8, "#f0f7ff", "#9dc3e6", lw=1.8)
    ax.text(12.5, 1.15, "本轮接受长度 3 → 净赚 4 个 token,输出分布 ≡ 大模型(无损)",
            ha="center", fontsize=14, color="#174a73", fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/figC_accept.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    figA_membound(); print("figA done")
    figB_timeline(); print("figB done")
    figC_accept(); print("figC done")
    print("all figures ->", OUT)
