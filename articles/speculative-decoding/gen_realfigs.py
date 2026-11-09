#!/usr/bin/env python3
"""Real-data figures for the speculative-decoding article.

All numbers are measured values quoted from papers — no simulation.
  figR1_accept.png  — acceptance length: EAGLE3 vs DFlash vs DSpark (DSpark Table 1)
  figR2_decay.png   — position-wise acceptance decay (DSpark Figure 2 anchor points)
  figR3_speed.png   — measured speedups across six generations (per-paper reported)

Run:  python gen_realfigs.py   (outputs to ./images/)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

cjk = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                        "WenQuanYi Micro Hei"]
            if n in {f.name for f in font_manager.fontManager.ttflist}), None)
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("CJK font:", cjk)

GREY, BLUE, LBLUE, ORANGE, GREEN = "#8c8c8c", "#1f77b4", "#7fb3d5", "#d97706", "#2ca02c"

# ---------------------------------------------------------------- fig R1
# DSpark paper Table 1 — acceptance length tau, averaged over 9 benchmarks
def figR1_accept():
    models = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
    eagle3 = [3.72, 3.80, 3.68]
    dflash = [4.29, 4.29, 4.26]
    dspark = [4.95, 4.96, 5.00]
    gain = ["+31%", "+27%", "+30%"]

    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for off, vals, color, name in [(-w, eagle3, GREY, "EAGLE-3"),
                                   (0, dflash, BLUE, "DFlash"),
                                   (w, dspark, ORANGE, "DSpark")]:
        bars = ax.bar(x + off, vals, w, color=color, label=name, edgecolor="white", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                    ha="center", fontsize=13, fontweight="bold" if color == ORANGE else "normal",
                    color="#222")
    for i, g in enumerate(gain):
        ax.annotate(g, xy=(x[i] + w, dspark[i] + 0.34), ha="center",
                    fontsize=13, color=ORANGE, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel("接受长度 τ(每轮平均接受的 token 数)", fontsize=13.5)
    ax.set_ylim(0, 5.9)
    ax.set_title("DSpark 接受长度实测:比 EAGLE-3 高 27-31%,比 DFlash 高 16-18%", fontsize=15)
    ax.legend(fontsize=13, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.005, "数据来源:DSpark 论文(arXiv:2607.05147)Table 1,GSM8K/MATH/AIME25/MBPP/HumanEval/LCB/MT-Bench/Alpaca/Arena-Hard 九个基准的平均",
             ha="center", fontsize=10, color="#666")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(f"{OUT}/figR1_accept.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig R2
# DSpark paper Figure 2 — per-position acceptance anchors on Qwen3-4B
def figR2_decay():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    series = [
        ("DFlash · Code", [0.87, 0.78], BLUE, "o"),
        ("DFlash · Chat", [0.72, 0.63], LBLUE, "o"),
        ("EAGLE-3 · Chat", [0.53, 0.74], GREY, "s"),
        ("DSpark · Math", [0.93, 0.92], ORANGE, "D"),
    ]
    xs = np.linspace(1, 8, 100)
    for name, (a, b), color, mk in series:
        ax.plot([1, 8], [a, b], "--", color=color, lw=1.8, alpha=0.75, zorder=2)
        ax.scatter([1, 8], [a, b], color=color, marker=mk, s=90, zorder=3)
        ax.annotate(f"{a:.2f}", (1, a), textcoords="offset points", xytext=(-2, 9),
                    fontsize=12.5, color=color, fontweight="bold", ha="right")
        ax.annotate(f"{b:.2f}", (8, b), textcoords="offset points", xytext=(4, 2),
                    fontsize=12.5, color=color, fontweight="bold", ha="left")
        ax.text(8.25, (a + b) / 2, name, fontsize=12.5, color=color, va="center")

    ax.set_xlim(0.4, 11.2); ax.set_ylim(0.42, 1.02)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xlabel("块内位置(草稿 token 序号)", fontsize=13.5)
    ax.set_ylabel("条件接受率", fontsize=13.5)
    ax.tick_params(labelsize=12.5)
    ax.set_title("并行草稿的\"后缀衰减\":块内越靠后,DFlash 越容易猜错", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.005, "端点数值为 DSpark 论文(arXiv:2607.05147)Figure 2 在 Qwen3-4B 上的实测;虚线为端点间连线示意。DSpark 的串行修正头使接受率全程平稳",
             ha="center", fontsize=10, color="#666")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(f"{OUT}/figR2_decay.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig R3
# measured speedups reported by each generation's own paper/deployment
def figR3_speed():
    rows = [
        # (name, year, lo, hi, color, source)
        ("Speculative Sampling\n(独立小模型)", "2023", 2.0, 3.0, GREY,  "Leviathan et al., T5-XXL"),
        ("Medusa\n(多头草稿)",                "2023", 2.2, 2.8, GREY,  "Cai et al., Vicuna-33B"),
        ("EAGLE-2/3\n(特征级草稿)",           "2024-25", 3.05, 4.3, BLUE, "Li et al. 论文实测"),
        ("DeepSeek-V3 MTP\n(原生头,SGLang)", "2024", 1.8, 1.8, GREEN, "DeepSeek-V3 技术报告"),
        ("Qwen3 27B MTP\n(Jetson AGX 实测)", "2025", 2.69, 2.69, GREEN, "Jetson AI Lab:13→35 tok/s"),
        ("DSpark\n(DeepSeek-V4 生产)",       "2026", 1.60, 1.85, ORANGE, "同等吞吐下单用户速度 +60~85%"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    ypos = np.arange(len(rows))[::-1]
    for y, (name, year, lo, hi, color, src) in zip(ypos, rows):
        ax.barh(y, hi - lo if hi > lo else 0.10, left=lo, height=0.52,
                color=color, alpha=0.88, edgecolor="white")
        label = f"{lo:.1f}×" if lo == hi else f"{lo:.1f}–{hi:.1f}×"
        ax.text(hi + 0.08, y, label, va="center", fontsize=13.5,
                fontweight="bold", color=color)
        ax.text(0.08, y + 0.36, f"{year}", va="center", fontsize=11, color="#999")
        ax.text(0.08, y - 0.02, name, va="center", fontsize=13, color="#222", linespacing=1.35)
        ax.text(hi + 0.95, y, src, va="center", fontsize=10.5, color="#777")

    ax.set_yticks([]); ax.set_xlim(0, 6.4)
    ax.set_xlabel("实测加速倍数(各方法论文/部署报告的原生数字)", fontsize=13.5)
    ax.set_title("六代投机解码的真实加速:论文实测,口径各异(注意)", fontsize=15)
    ax.axvline(1.0, color="#bbb", lw=1, ls=":")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=12.5)
    fig.text(0.5, 0.005, "口径说明:灰色=离线低并发基准;蓝色=离线基准;绿色=生产/端侧实测;橙色=生产环境同等吞吐下 per-user 加速。不同论文的基线与并发不同,不能直接横比",
             ha="center", fontsize=10, color="#666")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(f"{OUT}/figR3_speed.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    figR1_accept(); print("figR1 done")
    figR2_decay(); print("figR2 done")
    figR3_speed(); print("figR3 done")
    print("all figures ->", OUT)
