#!/usr/bin/env python3
"""fig1: three-color channel map — how mainland capital reaches crypto.

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

BLACK, GREY, WHITE = "#333333", "#8a8a8a", "#2288CC"
DBLUE = "#2288CC"


def fig1_channels():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")

    cols = [
        (0.3, BLACK, "黑色通道", "USDT 场外",
         ["人民币 ↔ USDT 直接置换", "境外 OTC 商家承接", "不经银行体系",
          "风险:帮信罪 / 洗钱罪", "资金链条不可见"]),
        (4.2, GREY, "灰色通道", "持牌券商",
         ["港币/美元直接买币", "富途 / 华盛等港牌券商", "80 港元起投,零佣金",
          "机构合规、用户身份模糊", "风险:政策收紧"]),
        (8.1, DBLUE, "白色通道", "香港 ETF / VATP",
         ["BTC / ETH 现货 ETF", "SFC 持牌交易平台", "稳定币条例 2025 生效",
          "OTC 发牌制度推进中", "风险:仅大盘,无山寨"]),
    ]
    for x, color, tag, name, items in cols:
        ax.add_patch(Rectangle((x, 0.5), 3.5, 8.6, fc="white", ec=color, lw=1.8))
        ax.add_patch(Rectangle((x, 7.9), 3.5, 1.2, fc=color, ec=color))
        ax.text(x + 1.75, 8.5, tag, ha="center", va="center", fontsize=12,
                color="white", fontweight="bold")
        ax.text(x + 1.75, 7.3, name, ha="center", va="center", fontsize=11,
                color=color, fontweight="bold")
        for i, it in enumerate(items):
            ax.text(x + 0.25, 6.6 - i * 0.82, "· " + it, fontsize=8.2,
                    va="center", color="#333333" if i < 3 else color)
        if x < 8:
            ax.add_patch(FancyArrowPatch((x + 3.55, 4.8), (x + 4.15, 4.8),
                                         arrowstyle="-|>", mutation_scale=18,
                                         color="#aaaaaa", lw=1.2))

    ax.text(6.0, 9.75, "合规度:黑 → 灰 → 白;每一层的\"价格\"是它收取的风险",
            ha="center", fontsize=10, color="#555555")
    fig.savefig(f"{OUT}/fig1_channels.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_channels(); print("fig1 ok")
    print("->", OUT)


def fig2_winners():
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["A股 1万以下账户\n(2016-19 上交所)", "A股 全市场\n(2025 口径)", "期货活跃散户\n5年存活盈利", "加密现货散户\n(BIS 2022)", "主动基金 15年\n跑赢标普(SPIVA)"]
    win = [0.1, 18.9, 5.0, 25.0, 10.5]
    err = [0, 0, 4.0, 4.0, 0]
    colors = ["#BB3333", "#EE6677", "#B8860B", "#888888", "#2288CC"]
    bars = ax.barh(range(5), win, xerr=[np.zeros(5), err], height=0.55,
                   color=colors, ec="white", lw=0.6,
                   error_kw=dict(ecolor="#555555", lw=1.0, capsize=3))
    texts = ["≈0.1%", "18.9%", "1-10%", "20-30%", "10.5%"]
    for i, (b, t) in enumerate(zip(bars, texts)):
        ax.text(win[i] + (err[i] + 1.5 if i in (2, 3) else 1.5), b.get_y() + b.get_height()/2,
                t, va="center", fontsize=9.5, fontweight="bold", color="#333333")
    ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("长期盈利面 / 跑赢比例(%)")
    ax.set_xlim(0, 40)
    ax.axvline(50, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title("散户长期盈利面:没有一个市场接近抛硬币", fontsize=11.5, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.savefig(f"{OUT}/fig2_winners.png"); plt.close(fig)
