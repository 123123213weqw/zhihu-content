#!/usr/bin/env python3
"""Figures for the weight-anatomy article.

fig2_anatomy.png — one transformer layer: attention group (1/3) vs gated MLP group (2/3)
fig3_journey.png — a token's journey: tokenizer -> embed -> 32 layers -> lm_head -> distribution
fig4_learning.png — pretraining loop: predict next-token distribution, CE loss, grads to all weights

Run:  python3 gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

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

DBLUE, DRED, DGREEN, DGRAY, DPURP = "#2288CC", "#BB3333", "#117733", "#666666", "#995588"


def box(ax, x, y, w, h, text, fc="#e8f3fb", ec=DBLUE, fs=8.4, bold=False, tc="#222222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#888888", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color=color, lw=lw))


def fig2_anatomy():
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.set_xlim(0, 16.8); ax.set_ylim(0, 10.4); ax.axis("off")

    ax.text(4.3, 9.9, "注意力组:67.1M / 层(占 1/3)", ha="center", fontsize=10.5, fontweight="bold", color=DBLUE)
    # h 输入
    box(ax, 0.3, 5.0, 1.5, 1.0, "$h$\n4096 维", fc="#f5f0e6", ec="#b09a6a", bold=True, fs=8.6)
    # W_Q W_K W_V
    for i, (name, desc) in enumerate([("$W_Q$ 16.8M", "→ 提问 q"),
                                      ("$W_K$ 16.8M", "→ 标签 k"),
                                      ("$W_V$ 16.8M", "→ 内容 v")]):
        y = 7.6 - i * 1.35
        box(ax, 2.6, y, 1.9, 1.05, name, fc="#e8f3fb", ec=DBLUE, fs=8.6)
        ax.text(4.75, y + 0.52, desc, fontsize=8.2, va="center", color="#444444")
        arrow(ax, 1.8, 5.5, 2.55, y + 0.52)
    # softmax attn
    box(ax, 5.9, 6.4, 2.3, 1.6, "$\\alpha_j = \\mathrm{softmax}$\n$\\frac{q \\cdot k_j}{\\sqrt{128}}$\n按比例取货 $\\sum \\alpha_j v_j$",
        fc="#eef7ee", ec=DGREEN, fs=7.8)
    for yy in (8.1, 6.75, 5.4):
        arrow(ax, 5.0, yy, 5.85, 7.2, color="#aaaaaa", lw=0.9)
    # W_O
    box(ax, 8.7, 6.65, 1.9, 1.1, "$W_O$ 16.8M\n合并 32 头", fc="#e8f3fb", ec=DBLUE, fs=8.4)
    arrow(ax, 8.2, 7.2, 8.65, 7.2)
    # 残差
    box(ax, 8.7, 4.9, 1.9, 0.9, "$+\\ h$(残差)", fc="#f3f3f3", ec="#999999", fs=8.6)
    arrow(ax, 9.65, 6.6, 9.65, 5.85)

    ax.text(13.0, 9.9, "门控 MLP 组:135.3M / 层(占 2/3)", ha="center", fontsize=10.5, fontweight="bold", color=DRED)
    # 输入
    box(ax, 10.9, 4.9, 1.5, 0.9, "$h'$", fc="#f5f0e6", ec="#b09a6a", bold=True, fs=9)
    # gate / up 两路
    box(ax, 13.0, 6.7, 2.2, 1.0, "$W_{gate}$ 45.1M\n→ SiLU 门", fc="#f2d9ec", ec=DPURP, fs=8.4)
    box(ax, 13.0, 5.2, 2.2, 1.0, "$W_{up}$ 45.1M\n→ 内容", fc="#f2d9ec", ec=DPURP, fs=8.4)
    arrow(ax, 12.4, 5.35, 12.95, 7.2); arrow(ax, 12.4, 5.35, 12.95, 5.7)
    box(ax, 15.5, 5.95, 1.0, 0.9, "$\\odot$", fc="white", ec=DPURP, bold=True, fs=11)
    arrow(ax, 15.2, 7.2, 15.45, 6.6); arrow(ax, 15.2, 5.7, 15.45, 6.2)
    box(ax, 13.0, 3.4, 2.2, 0.95, "$W_{down}$ 45.1M\n→ 降维回 4096", fc="#f2d9ec", ec=DPURP, fs=8.4)
    arrow(ax, 16.0, 5.9, 16.3, 5.5)
    arrow(ax, 15.5, 5.45, 15.25, 4.35)
    box(ax, 10.9, 3.4, 1.5, 0.95, "$+\\ h'$(残差)", fc="#f3f3f3", ec="#999999", fs=8.4)
    arrow(ax, 12.95, 3.87, 12.45, 3.87)

    # 参数占比条
    ax.add_patch(plt.Rectangle((0.4, 0.7), 4.6, 0.55, fc="#cfe0f2", ec=DBLUE, lw=1.0))
    ax.add_patch(plt.Rectangle((5.0, 0.7), 9.2, 0.55, fc="#f2d9d9", ec=DRED, lw=1.0))
    ax.text(2.7, 0.97, "注意力 67.1M(1/3)", ha="center", va="center", fontsize=8.8, color=DBLUE, fontweight="bold")
    ax.text(9.6, 0.97, "门控 MLP 135.3M(2/3)", ha="center", va="center", fontsize=8.8, color=DRED, fontweight="bold")
    ax.text(8.4, 0.28, "一层 202.4M × 32 层 + 嵌入 131M + 输出头 131M ≈ 6.74B",
            ha="center", fontsize=9.5, color="#333333", fontweight="bold")
    ax.text(8.4, 1.75, "注意力管「从上下文取信息」,门控 MLP 管「计算与知识存储」",
            ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_anatomy.png"); plt.close(fig)


def fig3_journey():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 18.6); ax.set_ylim(0, 9.6); ax.axis("off")

    # 流程:文本 → tokenizer → id → 嵌入 → 32层 → lm_head → 分布 → 采样
    steps = [
        (0.3, "文本\n「巴黎是法国的首」", "#f5f0e6", "#b09a6a"),
        (2.9, "切词 tokenizer\n→ token id", "#f3f3f3", "#999999"),
        (5.5, "嵌入查表\nE 的第 i 行", "#e8f3fb", DBLUE),
    ]
    for x, t, fc, ec in steps:
        box(ax, x, 6.6, 2.2, 1.3, t, fc=fc, ec=ec, fs=8.4)
    for x in (2.5, 5.1):
        arrow(ax, x, 7.25, x + 0.38, 7.25)

    # 32 层放大框
    box(ax, 8.1, 4.7, 5.4, 4.3, "", fc="#eef7ee", ec=DGREEN)
    ax.text(10.8, 8.5, "× 32 层", ha="center", fontsize=10, fontweight="bold", color=DGREEN)
    box(ax, 8.35, 7.0, 4.9, 1.1, "注意力:提问 q · 匹配 k · 取货 v", fc="white", ec=DBLUE, fs=8.4)
    box(ax, 8.35, 5.6, 4.9, 1.1, "门控 MLP:11008 个通道开关", fc="white", ec=DPURP, fs=8.4)
    ax.text(10.8, 5.15, "每层结束:残差相加", ha="center", fontsize=7.8, color="#666666")
    arrow(ax, 7.7, 7.25, 8.05, 7.25)

    # 输出头与分布
    box(ax, 14.0, 6.6, 2.0, 1.3, "输出头\n4096×32000", fc="#e8f3fb", ec=DBLUE, fs=8.2)
    arrow(ax, 13.55, 7.25, 13.95, 7.25)

    # 分布柱
    ax.add_patch(FancyBboxPatch((16.15, 5.3), 2.3, 3.4, boxstyle="round,pad=0.05",
                                fc="white", ec="#999999", lw=1.0))
    bars = [("首都", 2.55, DGREEN), ("城市", 0.62, DGRAY), ("最大", 0.30, DGRAY), ("其他", 0.22, "#bbbbbb")]
    for i, (lab, hgt, c) in enumerate(bars):
        x = 16.5 + i * 0.48
        ax.add_patch(plt.Rectangle((x, 6.05), 0.34, hgt, fc=c, ec="white", lw=0.5))
        ax.text(x + 0.17, 5.88, lab, ha="center", va="top", fontsize=7.4, color="#444444")
    ax.text(17.3, 8.95, "softmax 分布", ha="center", fontsize=8.4, fontweight="bold")
    arrow(ax, 16.0, 7.25, 16.1, 7.25)

    # 采样回路
    box(ax, 14.0, 3.6, 2.0, 1.1, "采样\nargmax / top-p", fc="#fbeaea", ec=DRED, fs=8.6)
    arrow(ax, 17.1, 5.25, 16.0, 4.75, color=DRED)
    # 回流箭头:沿下方绕行,不穿框
    ax.add_patch(FancyArrowPatch((13.95, 4.15), (1.45, 2.85), arrowstyle="-|>",
                                 mutation_scale=11, color=DRED, lw=1.2, linestyle="--"))
    ax.add_patch(FancyArrowPatch((1.45, 2.85), (1.45, 6.55), arrowstyle="-|>",
                                 mutation_scale=11, color=DRED, lw=1.2, linestyle="--"))
    ax.text(7.0, 3.05, "拼回开头,循环", fontsize=8.4, color=DRED, ha="center")
    ax.text(9.3, 1.95, "生成 = 每次挑 1 个 token(decode),「KV cache」复用历史 token 的 k、v 存货",
            ha="center", fontsize=8.6, color="#333333")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_journey.png"); plt.close(fig)


def fig4_learning():
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 17.2); ax.set_ylim(0, 9.4); ax.axis("off")

    # 输入文本
    box(ax, 0.3, 6.2, 2.6, 1.4, "训练文本\n「巴黎是法国的首 [?]」", fc="#f5f0e6", ec="#b09a6a", fs=8.4)
    # 模型
    box(ax, 3.6, 5.6, 2.6, 2.6, "模型\n67 亿参数", fc="#e8f3fb", ec=DBLUE, fs=9.2, bold=True)
    ax.text(4.9, 5.25, "推理时冻结,训练时每步都被更新", ha="center", fontsize=7.8, color="#666666")
    arrow(ax, 2.95, 6.9, 3.55, 6.9)
    # 分布
    ax.add_patch(FancyBboxPatch((6.9, 5.5), 2.6, 2.8, boxstyle="round,pad=0.05",
                                fc="white", ec="#999999", lw=1.0))
    for i, (lab, hgt, c) in enumerate([("首都", 0.62, DGRAY), ("城市", 0.3, "#bbbbbb"),
                                        ("最大", 0.15, "#bbbbbb"), ("…", 0.1, "#dddddd")]):
        x = 7.3 + i * 0.55
        ax.add_patch(plt.Rectangle((x, 6.0), 0.35, hgt, fc=c, ec="white", lw=0.5))
        ax.text(x + 0.17, 5.8, lab, ha="center", fontsize=6.8, color="#444444")
    ax.text(8.2, 8.6, "预测分布", ha="center", fontsize=9, fontweight="bold")
    arrow(ax, 6.25, 6.9, 6.85, 6.9)
    # 真实词
    box(ax, 9.9, 6.4, 2.2, 1.0, "真实词:\n「都」", fc="#eef7ee", ec=DGREEN, fs=8.8, bold=True)
    # 损失
    box(ax, 12.6, 6.3, 2.2, 1.2, "$L = -\\log P$\n(猜错就罚)", fc="#fbeaea", ec=DRED, fs=8.8)
    arrow(ax, 9.5, 6.9, 9.85, 6.9); arrow(ax, 12.1, 6.9, 12.55, 6.9)

    # 反向传播:从损失直接弯到更新框
    ax.annotate("", xy=(13.2, 3.8), xytext=(13.7, 6.25),
                arrowprops=dict(arrowstyle="-|>", color=DRED, lw=1.8,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(9.3, 4.5, "反向传播:算出每个权重的梯度", ha="center", fontsize=9.5, color=DRED, fontweight="bold")
    # 权重更新
    box(ax, 3.6, 2.6, 10.2, 1.15,
        "67 亿个数字,每个挪一小步:$w \\leftarrow w - \\eta \\cdot \\text{grad}$",
        fc="#f2d9ec", ec=DPURP, fs=9.6)
    # 循环:更新框左侧出发,沿左侧直角绕回文本框(与红色反向传播分家)
    ax.add_patch(FancyArrowPatch((3.55, 3.17), (1.5, 3.17), arrowstyle="-",
                                 color="#888888", lw=1.1, linestyle="--"))
    ax.add_patch(FancyArrowPatch((1.5, 3.17), (1.5, 6.15), arrowstyle="-|>",
                                 mutation_scale=11, color="#888888", lw=1.1, linestyle="--"))
    ax.text(2.55, 4.6, "下一个位置\n× 2 万亿 token", fontsize=8.6, color="#666666",
            ha="left", va="center", linespacing=1.5)
    ax.text(8.6, 0.9, "随机噪声 → 「懂语言」:没有任何一行代码写过规则,全靠这一圈转百万步",
            ha="center", fontsize=9.5, color="#333333", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_learning.png"); plt.close(fig)


if __name__ == "__main__":
    fig2_anatomy()
    fig3_journey()
    fig4_learning()
    print("done ->", OUT)
