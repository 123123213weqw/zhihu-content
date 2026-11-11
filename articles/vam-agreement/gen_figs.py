#!/usr/bin/env python3
"""Figures for the VAM (valuation adjustment mechanism) article.

fig1_heatmap.png        — trigger probability heatmap over (g-mu, sigma), T=3
fig2_compensation.png   — expected compensation ratio E[C]/I vs promised g
fig3_execution.png      — nominal vs realized buyback returns / execution haircut

All curves computed from closed-form GBM formulas (verified numerically).

Run:  python gen_figs.py   -> ./images/
"""

from __future__ import annotations

import os
from math import erf as _scalar_erf

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

DBLUE, DRED, DGREEN, DORANGE = "#2288CC", "#BB3333", "#117733", "#B8860B"


try:
    from scipy.special import erf as _erf
except ImportError:
    _erf = np.vectorize(_scalar_erf)


def Phi(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def trigger_prob(mu, g, sigma, T):
    z = (g - mu + 0.5 * sigma ** 2) * np.sqrt(T) / sigma
    return Phi(z)


def expect_comp(mu, g, sigma, T):
    d2 = (g - mu + 0.5 * sigma ** 2) * np.sqrt(T) / sigma
    d1 = d2 - sigma * np.sqrt(T)
    return Phi(d2) - np.exp((mu - g) * T) * Phi(d1)


# ================================================================ fig 1
def fig1_heatmap():
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    T = 3.0
    dg = np.linspace(-0.10, 0.40, 250)          # g - mu
    sg = np.linspace(0.15, 0.60, 250)           # sigma
    DG, SG = np.meshgrid(dg, sg)
    PI = trigger_prob(0.3, 0.3 + DG, SG, T)     # mu=0.30, g = mu + (g-mu)

    cf = ax.contourf(DG * 100, SG * 100, PI, levels=np.arange(0.3, 1.01, 0.05),
                     cmap="RdYlGn_r", vmin=0.3, vmax=1.0)
    cs = ax.contour(DG * 100, SG * 100, PI, levels=[0.5, 0.7, 0.8, 0.9, 0.95],
                    colors="#333333", linewidths=0.7)
    ax.clabel(cs, fmt={0.5: "50%", 0.7: "70%", 0.8: "80%", 0.9: "90%", 0.95: "95%"},
              fontsize=7.5)

    pts = [
        (0, 30, "(g=μ: 仍 60%)", DGREEN),
        (20, 30, "市场常态\n(π≈88%)", DBLUE),
        (30, 30, "严苛对赌\n(π≈93%)", DRED),
    ]
    for x, y, label, c in pts:
        ax.plot(x, y, "*", color=c, ms=13, mec="white", mew=0.7, zorder=5)
        ax.annotate(label, (x, y), xytext=(x + 2.5, y + 3.5), fontsize=7.5,
                    color=c, linespacing=1.4)

    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("触发概率 $\\pi$", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("$g-\\mu$(承诺超出真实增长,百分点)")
    ax.set_ylabel("波动率 $\\sigma$(%)")
    ax.set_title("对赌触发概率地图($T=3$ 年,利润服从 GBM)", fontsize=11, pad=8)
    ax.tick_params(labelsize=8.5)
    fig.savefig(f"{OUT}/fig1_heatmap.png"); plt.close(fig)


# ================================================================ fig 2
def fig2_compensation():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    T = 3.0
    mu = 0.30
    g = np.linspace(0.10, 0.80, 300)
    for sigma, c in [(0.20, DGREEN), (0.30, DBLUE), (0.45, DRED)]:
        ax.plot(g * 100, expect_comp(mu, g, sigma, T), lw=1.5, color=c,
                label=f"$\\sigma$={sigma:.0%}")
    ax.axvline(mu * 100, color="#888888", ls=":", lw=0.9)
    ax.text(mu * 100 + 1, 0.02, "$g=\\mu$", fontsize=8, color="#888888")
    ax.axhline(0.205, color="#bbbbbb", lw=0.6, ls="--")
    ax.annotate("$g=\\mu$ 仍期望补偿 0.21", xy=(30, 0.205), xytext=(38, 0.10),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))
    ax.annotate("严苛对赌:0.34-0.60", xy=(52, expect_comp(mu, 0.52, 0.30, T)),
                xytext=(56, 0.25), fontsize=8, color=DRED,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=DRED))
    ax.set_xlabel("承诺增长率 $g$(%),$\\mu$=30%,$T$=3 年")
    ax.set_ylabel("期望补偿 $\\mathbb{E}[C]/I$")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8.5)
    ax.set_title("补偿条款的期权价值:期望拿回的投资比例", fontsize=11, pad=8)
    fig.savefig(f"{OUT}/fig2_compensation.png"); plt.close(fig)


# ================================================================ fig 3
def fig3_execution():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    labels = ["名义条款\n$(1+8\\%)^3$", "VC 实测\n(礼丰 2024)", "PE 实测\n(礼丰 2024)",
              "司法执行\n回款率", "创始人\n失信比例"]
    vals = [1.26, 1.20, 1.07, 0.06, 0.10]
    colors = [DGREEN, DBLUE, DBLUE, DRED, DRED]
    bars = ax.bar(range(5), vals, 0.55, color=colors, ec="white", lw=0.6)
    for b, v in zip(bars, vals):
        txt = f"{v:.2f}x" if v >= 1 else f"{v:.0%}"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, txt, ha="center",
                fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("倍数 / 比例")
    ax.set_ylim(0, 1.45)
    ax.axhline(1.0, color="#999999", lw=0.7, ls=":")
    ax.annotate("", xy=(0.45, 1.30), xytext=(2.55, 1.30),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=DRED))
    ax.text(1.5, 1.335, "执行折价 15-19 个点", fontsize=8.5, color=DRED, ha="center")
    ax.set_title("从条款到现实:回购对赌的层层折损", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{OUT}/fig3_execution.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_heatmap(); print("fig1 ok")
    fig2_compensation(); print("fig2 ok")
    fig3_execution(); print("fig3 ok")
    print("->", OUT)
