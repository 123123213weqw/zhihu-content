#!/usr/bin/env python3
"""fig1: founder power structure comparison — who can the board fire?

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

DBLUE, DRED, DGREEN, GREY = "#2288CC", "#BB3333", "#117733", "#888888"


def fig1_power():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    rows = [
        ("黄仁勋\nNVIDIA", "~3.5% 持股\n单一股权,一股一票", "无", "被赶走\n(2026)", DRED),
        ("乔布斯\nApple(1985)", "股权已稀释", "无", "被赶走\n(1985)", DRED),
        ("马斯克\nTesla(2018)", "~13% 持股\n一股一票", "部分", "被逼让出\n董事长(2018)", DORANGE := "#B8860B"),
        ("Altman\nOpenAI(2023)", "无股权", "人心+资本", "被解雇后\n一周杀回", DBLUE),
        ("扎克伯格\nMeta", "B 类股\n~61% 投票权", "绝对", "赶不走", DGREEN),
    ]
    # table header
    heads = ["创始人", "股权 / 投票权", "免死金牌", "结局"]
    xs = [0.2, 2.3, 5.4, 7.3]
    ws = [2.1, 3.1, 1.9, 2.5]
    y_head = 8.9
    for x, w, h in zip(xs, ws, heads):
        ax.add_patch(Rectangle((x, y_head), w, 0.75, fc="#eeeeee", ec="#bbbbbb", lw=0.8))
        ax.text(x + w / 2, y_head + 0.37, h, ha="center", va="center", fontsize=9.5)
    y = y_head
    for name, stake, shield, outcome, color in rows:
        y -= 1.62
        for x, w, txt, bold in [(xs[0], ws[0], name, True), (xs[1], ws[1], stake, False),
                                (xs[2], ws[2], shield, False), (xs[3], ws[3], outcome, True)]:
            ax.add_patch(Rectangle((x, y), w, 1.45, fc="white", ec="#cccccc", lw=0.8))
            ax.text(x + w / 2, y + 0.72, txt, ha="center", va="center",
                    fontsize=8.2, linespacing=1.5,
                    fontweight="bold" if bold else "normal",
                    color=color if bold and txt != name else "#222222")
    ax.text(5.0, 0.35, "权力来源决定结局:投票权 > 免死金牌;无投票权时,只剩'人心+不可替代性'——丑闻一来两者皆碎",
            ha="center", fontsize=8.5, color="#555555")
    ax.set_title("创始人能否被董事会赶走:五个案例的权力结构对照", fontsize=11.5, pad=10)
    fig.savefig(f"{OUT}/fig1_power.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_power(); print("fig1 ok")
    print("->", OUT)
