#!/usr/bin/env python3
"""Concept figures for the speculative-decoding article.

figA_membound.png   — why decode is memory-bound (prefill vs decode per weight-read)
figB_timeline.png   — autoregressive t→t+1 chain vs draft+verify timeline
figC_accept.png     — accept/reject/rollback mechanism (rejection sampling)

Run:  python gen_figures.py   (outputs to ./images/)
"""

from __future__ import annotations

import os

import numpy as np
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


def weight_bar(ax, x, y, w, h, label="读一遍全部权重\n(70B ≈ 140 GB,成本相同)"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc="#6baed6", ec="#2171b5", lw=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)


def token_squares(ax, x0, y, n, size=0.9, gap=0.18, fc="#2ca02c", max_draw=None, label=None):
    max_draw = max_draw or n
    for i in range(max_draw):
        ax.add_patch(FancyBboxPatch((x0 + i * (size + gap), y), size, size * 0.62,
                                    boxstyle="round,pad=0.02", fc=fc, ec="none"))
    if n > max_draw:
        ax.text(x0 + max_draw * (size + gap) + 0.6, y + size * 0.31, "……", fontsize=13)
        ax.text(x0 + max_draw * (size + gap) + 2.6, y + size * 0.31, label or f"≈{n} tokens",
                fontsize=10, va="center")


# ---------------------------------------------------------------- fig A
def figA_membound():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, mode in zip(axes, ["Prefill(处理提示词)", "Decode(逐 token 生成)"]):
        ax.set_xlim(0, 24); ax.set_ylim(0, 10); ax.axis("off")
        weight_bar(ax, 3, 7.6, 17, 1.9)
        ax.add_patch(FancyArrowPatch((11.5, 7.4), (11.5, 5.9), arrowstyle="-|>",
                                     mutation_scale=22, color="#2171b5", lw=2))
        ax.set_title(mode, fontsize=13, pad=10)
    axes[0].text(12, 6.6, "一次访存,并行算出所有位置", ha="center", fontsize=10, color="#2171b5")
    token_squares(axes[0], 2.6, 4.6, 1000, max_draw=13, label="≈1000 tokens(整段 prompt)")
    axes[0].text(12, 3.1, "算力打满(compute-bound)", ha="center", fontsize=11,
                 color=GREEN, fontweight="bold")

    axes[1].text(12, 6.6, "一次访存,只算 1 个位置", ha="center", fontsize=10, color="#d62728")
    token_squares(axes[1], 10.6, 4.6, 1, size=0.9)
    axes[1].text(11.05, 3.1, "算力闲置 >95%(memory-bound)", ha="center", fontsize=11,
                 color=RED, fontweight="bold")

    fig.suptitle("decode 为什么快不起来:瓶颈不是算力,是访存(batch = 1)", fontsize=14.5)
    fig.text(0.5, 0.03,
             "生成速度上限 ≈ 显存带宽 ÷ 权重大小(与算力大小无关)—— 例如带宽 1 TB/s ÷ 140 GB ≈ 7 tokens/s",
             ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec="#cccccc"))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(f"{OUT}/figA_membound.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig B
def figB_timeline():
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.set_xlim(0, 26); ax.set_ylim(0, 11); ax.axis("off")

    def block(x, y, w, h, fc, ec, text, fs=9.5, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, text=None, color="#333", dy=0.45, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=18, color=color, lw=1.8, linestyle=ls))
        if text:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, text, ha="center", fontsize=8.5, color=color)

    # ---- row 1: autoregressive ----
    ax.text(0.4, 9.6, "① 传统自回归:每个 token 都要等上一个(token t 的输出 = t+1 的输入)",
            fontsize=11.5, fontweight="bold")
    bw, bh, y1 = 4.4, 2.2, 6.2
    labels = ["大模型前向\n→ 产出 t+1", "大模型前向\n→ 产出 t+2", "大模型前向\n→ 产出 t+3", "大模型前向\n→ 产出 t+4"]
    for i, lab in enumerate(labels):
        x = 0.8 + i * (bw + 1.1)
        block(x, y1, bw, bh, BLUE, "#174a73", lab)
        if i:
            arrow(x - 1.1, y1 + bh / 2, x, y1 + bh / 2, "串行依赖")
    ax.text(23.6, y1 + bh / 2, "= 4 次完整访存\n才得 4 个 token", fontsize=10,
            va="center", color=RED, fontweight="bold")

    # ---- row 2: speculative ----
    ax.text(0.4, 4.6, "② 投机解码:小模型连猜,大模型一次验(并行算 4 个位置)",
            fontsize=11.5, fontweight="bold")
    y2 = 0.9
    block(0.8, y2, 6.6, 2.2, LBLUE, "#7fa8d0", "draft:小模型连猜 t+1, t+2, t+3\n(读小模型权重,极便宜)", tc="#1a1a1a")
    arrow(7.4, y2 + 1.1, 8.2, y2 + 1.1)
    block(8.2, y2, 9.2, 2.2, BLUE, "#174a73", "verify:大模型一次前向\n并行算出 t+1…t+4 四个位置的概率")
    arrow(17.4, y2 + 1.1, 18.2, y2 + 1.1)
    block(18.2, y2, 2.4, 1.05, GREEN, "#1b6b2b", "t+1 √")
    block(20.8, y2, 2.4, 1.05, GREEN, "#1b6b2b", "t+2 √")
    block(18.2, y2 + 1.15, 2.4, 1.05, ORANGE, "#b35a00", "t+3 × 重采样", fs=9, tc="#5a2d00")
    block(20.8, y2 + 1.15, 2.4, 1.05, GREY, "#666", "t+4 × 丢弃", fs=9)
    ax.text(23.8, y2 + 1.1, "= 1 次大模型访存\n得 3 个 token", fontsize=10,
            va="center", color=GREEN, fontweight="bold")

    ax.set_title("同样的闲置算力,拿来一次验证多个位置 —— 这就是投机解码", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figB_timeline.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig C
def figC_accept():
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 26); ax.set_ylim(0, 11); ax.axis("off")

    ax.text(0.4, 9.8, "验证规则:从前往后逐个对答案,遇到第一个'想法不同'的 token 就回滚",
            fontsize=12.5, fontweight="bold")

    toks = [
        ("草稿 t+1\n'深度'", "√ 接受", GREEN),
        ("草稿 t+2\n'学习'", "√ 接受", GREEN),
        ("草稿 t+3\n'的'", "√ 接受", GREEN),
        ("草稿 t+4\n'本质'", "× 拒绝", RED),
        ("草稿 t+5\n'是'", "× 丢弃", GREY),
    ]
    x0, y0, w, h = 1.0, 5.4, 4.1, 3.0
    for i, (draft, verdict, color) in enumerate(toks):
        x = x0 + i * (w + 0.9)
        faded = color == GREY
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04",
                                    fc="#ffffff" if faded else "#f7f7f7", ec=color,
                                    lw=2.2, alpha=0.55 if faded else 1.0))
        ax.text(x + w / 2, y0 + h * 0.62, draft, ha="center", va="center", fontsize=10.5)
        ax.text(x + w / 2, y0 + h * 0.2, verdict, ha="center", fontsize=10.5,
                color=color, fontweight="bold")
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, y0 + h / 2), (x + w + 0.85, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=16, color="#555", lw=1.4))

    # resample replacement
    xr = x0 + 3 * (w + 0.9)
    ax.add_patch(FancyArrowPatch((xr + w / 2, y0 - 0.15), (xr + w / 2, y0 - 1.5),
                                 arrowstyle="-|>", mutation_scale=18, color=ORANGE, lw=2))
    ax.add_patch(FancyBboxPatch((xr - 0.4, y0 - 3.3), w + 0.8, 1.7, boxstyle="round,pad=0.04",
                                fc="#fff3e0", ec=ORANGE, lw=1.8))
    ax.text(xr + w / 2, y0 - 2.45,
            "按大模型分布重新采样 → '核心'", ha="center", fontsize=10.5, color="#b35a00")

    ax.text(13, 1.1,
            "本轮接受长度 = 3 → 产出 4 个 token(3 个接受 + 1 个重采样)\nrejection sampling 保证:最终输出分布 ≡ 直接用大模型生成(无损)",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f0f7ff", ec="#9dc3e6"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/figC_accept.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    figA_membound(); print("figA done")
    figB_timeline(); print("figB done")
    figC_accept(); print("figC done")
    print("all figures ->", OUT)
