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
