#!/usr/bin/env python3
"""Figures for the SpecPilot article (arXiv:2611.02501).

fig1_timeline.png    one decode step with one request on an H100: AR, stock and SpecPilot -- how long, how many tokens
fig2_tree.png        the toy lattice: the chain stock verifies, the six edges of a best-first tree, and where the target really goes
fig3_w12_window.png  W12 on a real weight row (Llama-3.1-8B-Instruct, layer 16 down_proj): exponent histogram with its
                     16-binade window; and how many escapes each row of that tensor has
fig4_promotions.png  cumulative gain of the 15 promotions on the H100 single-request line
fig5_platforms.png   AR / stock / SpecPilot on H100, RTX PRO 6000 and C500; and the same systems in the (c, tau) plane
fig6_rows.png        RTX 4090 D: time of a BF16 GEMM against the number of rows, for four real weight matrices (needs bench/w12_bench_4090d.json)
fig7_w12_speed.png   RTX 4090 D: five ways to read the weights in the same GEMM skeleton, relative to BF16 cuBLAS (same file)

Paper numbers come from bench/calc_summary.json (written by calc.py), real-weight data from bench/w12_stats.json,
the toy lattice from specpilot_toys.py.
Run:  python3 gen_figs.py  ->  ./images/
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

import specpilot_toys as toys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)
BENCH = os.environ.get("SPECPILOT_BENCH", os.path.join(HERE, "bench"))
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
BLUE, RED, GREEN, GRAY, ORANGE, PURP, GOLD, LIGHT = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#995588", "#b8922e", "#d9d9d9"
DARK = "#333333"
calc = json.load(open(os.path.join(BENCH, "calc_summary.json")))
stats = json.load(open(os.path.join(BENCH, "w12_stats.json")))
_bench_path = os.path.join(BENCH, "w12_bench_4090d.json")
bench = json.load(open(_bench_path)) if os.path.exists(_bench_path) else None
GROUP_COLOR = {"草稿模型": ORANGE, "树": GREEN, "W12": BLUE, "其它内核": GRAY, "两级验证": PURP}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------------------------- fig1
def fig1_timeline():
    """H100, one request, Qwen3.8-27B + DFlash2. The two columns on the right are tokens a step commits."""
    cm, two = calc["cost_model"]["h100"], calc["two_level"]
    parts, draft = two["parts_ms"], two["parts_ms"]["draft_x2"] / 2
    rows = [  # (label, [(ms, color, text inside the bar)], tokens a step commits)
        ("自回归", [(cm["t_ar_ms"], GRAY, "读全部权重")], 1.0),
        ("公开实现", [(cm["t_step_ms"], BLUE, "一趟验证")], cm["tau_stock"]),
        ("SpecPilot", [(draft, ORANGE, "草稿"), (parts["q_pass"], PURP, "Q pass"), (draft, ORANGE, "草稿"),
                       (parts["exact_pass"], GREEN, "精确验证"), (parts["glue"], LIGHT, "")], two["tokens_per_exact_pass"]["two_level"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    right = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (label, segs, tok) in enumerate(rows):
        y, left = len(rows) - 1 - i, 0.0
        for w, c, name in segs:
            ax.barh(y, w, left=left, height=0.5, color=c, edgecolor="white", linewidth=0.8)
            if w >= 4.5 and name:
                ax.text(left + w / 2, y, f"{name}\n{w:.1f} ms", ha="center", va="center", fontsize=7.8, color="white", linespacing=1.3)
            elif name:
                ax.annotate(f"草稿 {w:.2f}", (left + w / 2, y + 0.25), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.8, color=ORANGE)
            left += w
        ax.text(left + 0.4, y, f"{left:.1f} ms", va="center", fontsize=8, color=GRAY)
        ax.text(1.07, y, f"{tok:g}", transform=right, va="center", ha="center", fontsize=9.5)
        ax.text(1.19, y, f"{tok / left:.3f}", transform=right, va="center", ha="center", fontsize=9.5)
    ax.text(1.07, len(rows) - 0.42, "一步接受的\ntoken 数", transform=right, ha="center", va="bottom", fontsize=7.8, color=GRAY, linespacing=1.3)
    ax.text(1.19, len(rows) - 0.42, "每毫秒\n接受的 token", transform=right, ha="center", va="bottom", fontsize=7.8, color=GRAY, linespacing=1.3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.55, len(rows) - 0.3)
    ax.set_xlabel("时间 (ms)")
    clean(ax)
    save(fig, "fig1_timeline.png")


# ---------------------------------------------------------------------------------------------- fig2
def fig2_tree():
    """The toy lattice of specpilot_toys.demo_lattice(). A node is a (position, candidate); several paths share a node."""
    lat = toys.demo_lattice()
    cand, root_p, cond = lat["cand"], lat["root"], lat["cond"]
    log_edges = [[[math.log(v) for v in row] for row in layer] for layer in cond]
    tree = toys.best_first_tree([math.log(v) for v in root_p], log_edges, 6)
    chain, truth = (0, 0, 0), (1, 1, 1)                       # 公开实现验证的链 A-a-x;目标模型实际走的路 B-b-y
    y_top, y_bot = 1.25, 0.25
    pos = {(): (-0.75, (y_top + y_bot) / 2)}
    for d in range(len(cand)):
        for j in range(len(cand[d])):
            pos[(d, j)] = (float(d), y_top if j == 0 else y_bot)
    bonus = (len(cand) - 0.2, y_bot)                          # 目标模型在 y 之后白拿的那个 token

    def nodes_of(path):                                       # 一条路径经过的节点:根,然后是每个位置上的候选
        return [()] + [(d, c) for d, c in enumerate(path)]

    tree_edges, chain_edges, truth_edges = set(), set(), set()
    for p in tree:
        ns = nodes_of(p)
        tree_edges |= set(zip(ns, ns[1:]))
    for path, target in ((chain, chain_edges), (truth, truth_edges)):
        ns = nodes_of(path)
        target |= set(zip(ns, ns[1:]))

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    edges = [((), (0, j), root_p[j]) for j in range(len(cand[0]))]
    for d in range(1, len(cand)):
        edges += [((d - 1, i), (d, j), cond[d - 1][i][j]) for i in range(len(cand[d - 1])) for j in range(len(cand[d]))]
    for a, b, prob in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        if (a, b) in truth_edges:
            kw, t = dict(color=GREEN, lw=2.8, linestyle=(0, (4, 2.2))), 0.5
        elif (a, b) in chain_edges:
            kw, t = dict(color=DARK, lw=3.0), 0.5
        elif (a, b) in tree_edges:
            kw, t = dict(color=BLUE, lw=2.2), 0.5
        else:
            kw, t = dict(color="#cfcfcf", lw=0.9), (0.5 if y0 == y1 else 0.3)
        ax.annotate("", (x1 - 0.2, y1), (x0 + 0.2, y0), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, **kw), zorder=2)
        ux, uy = (x1 - x0) * 1.4, (y1 - y0) * 1.5                    # 屏幕上的方向 (两张图的横纵比例不同),用来求法向
        norm = math.hypot(ux, uy)
        ax.annotate(f"{prob:.1f}", (x0 + t * (x1 - x0), y0 + t * (y1 - y0)), xytext=(-9 * uy / norm, 9 * ux / norm),
                    textcoords="offset points", fontsize=7.4, color=GRAY, ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))
    for i, p in enumerate(tree):                              # 入选的先后:标在这条边的中点下方
        ns = nodes_of(p)
        (x0, y0), (x1, y1) = pos[ns[-2]], pos[ns[-1]]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.06, str(i + 1), fontsize=7.2, color="white", ha="center", va="top", zorder=4,
                bbox=dict(boxstyle="circle,pad=0.18", fc=BLUE, ec="white", lw=0.8))

    def node(xy, letter, fc, ec, tc, lw=1.4, ls="-"):
        ax.add_patch(plt.Circle(xy, 0.16, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=5))
        ax.text(xy[0], xy[1], letter, ha="center", va="center", fontsize=10.5, color=tc, fontweight="bold", zorder=6)

    node(pos[()], "P", "white", DARK, DARK)                   # P:上一步白拿的那个 token
    for d in range(len(cand)):
        for j in range(len(cand[d])):
            fc = GREEN if truth[d] == j else (DARK if chain[d] == j else "white")
            node(pos[(d, j)], cand[d][j], fc, fc, "white" if fc != "white" else "#b5b5b5", lw=1.2)
    ax.annotate("", (bonus[0] - 0.2, bonus[1]), (pos[(2, 1)][0] + 0.2, pos[(2, 1)][1]),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, linestyle=(0, (1.5, 2)), shrinkA=0, shrinkB=0), zorder=2)
    node(bonus, lat["truth"][3], "white", GREEN, GREEN, lw=1.3, ls="--")
    ax.text(bonus[0] + 0.23, bonus[1], "白拿", fontsize=8, color=GREEN, va="center")
    ax.text(pos[()][0], pos[()][1] - 0.36, "P:上一步的 token", fontsize=7.8, color=GRAY, ha="center", va="top")
    for d in range(len(cand)):
        ax.text(d, y_top + 0.3, f"位置 {d}", fontsize=8, color=GRAY, ha="center")
    ax.text(1.0, y_top + 0.62, "草稿模型每个位置给出两个候选,边上是条件概率", fontsize=8.2, color=GRAY, ha="center")
    handles = [plt.Line2D([], [], color=DARK, lw=3.0, label="公开实现验证的链:每一步取条件概率最大的候选"),
               plt.Line2D([], [], color=BLUE, lw=2.2, label="树里挑出来的 6 条边 (圆圈里的数字是入选的先后)"),
               plt.Line2D([], [], color=GREEN, lw=2.6, linestyle=(0, (4, 2.2)), label="目标模型实际走的路")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, -0.02), ncol=1, fontsize=8, handlelength=2.2)
    ax.set_xlim(-1.15, bonus[0] + 0.95)
    ax.set_ylim(-0.2, y_top + 0.8)
    ax.axis("off")
    save(fig, "fig2_tree.png")


# ---------------------------------------------------------------------------------------------- fig3
def fig3_w12_window():
    ex = stats["models"]["Llama-3.1-8B-Instruct"]["example"]
    worst = ex["rows"]["worst"]
    e = np.array(sorted(int(k) for k in worst["hist"]))
    cnt = np.array([worst["hist"][str(v)] for v in e], dtype=float)
    lo, hi = worst["E"] - 15, worst["E"]                       # 窗口 [E - 15, E],指数域带偏置 127
    outside = (e < lo) | (e > hi)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.45, 1]))
    a.bar(e - 127, cnt, width=0.78, color=[RED if m else BLUE for m in outside], edgecolor="white", linewidth=0.6)
    a.axvspan(lo - 0.5 - 127, hi + 0.5 - 127, color=BLUE, alpha=0.07, lw=0)
    a.text(lo - 127 + 4.2, cnt.max() * 0.97, "16 个 binade 的窗口", ha="center", va="top", fontsize=8.5, color=BLUE)
    a.annotate(f"窗口之外 {int(cnt[outside].sum())} 个,\n另存一张 escape 表", (e[0] - 127, cnt[0]), xytext=(e[0] - 127 + 1.3, cnt.max() * 0.5),
               fontsize=8, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    a.set_xlabel("一行权重里每个数落在哪两个 2 的整数次幂之间")
    a.set_ylabel("权重个数")
    a.set_title(f"down_proj 的第 {worst['row']} 行:14,336 个权重里 escape 最多的一行", fontsize=9)
    keys = sorted(int(k) for k in ex["esc_hist"])
    vals = [ex["esc_hist"][str(k)] for k in keys]
    b.bar(keys, vals, width=0.72, color=[BLUE if k == 0 else RED for k in keys], edgecolor="white", linewidth=0.6)
    b.set_yscale("log")
    b.set_xticks(keys)
    b.set_xlabel("一行里有几个 escape")
    b.set_ylabel("这样的行有几行")
    b.set_title("这一层的 4,096 行", fontsize=9)
    b.text(0.97, 0.95, f"{ex['rows_with_escape']:,} 行至少有 1 个\n最多的一行 {max(keys)} 个", transform=b.transAxes, fontsize=8, va="top", ha="right", color=GRAY)
    clean(a)
    clean(b)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_w12_window.png")


# ---------------------------------------------------------------------------------------------- fig4
def fig4_promotions():
    cum = calc["promotions_h100"]["cumulative"]                # [(name, category, cumulative gain)] in promotion order
    cats = [c[1] for c in cum]
    rounds = [1] * 3 + [2] * 6 + [3] * 6                       # 论文表 A3 的三轮
    xs = [0] + list(range(1, len(cum) + 1))
    ys = [1.0] + [c[2] for c in cum]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for r, c, lab in ((1, "#f3f3f3", "第 1 轮"), (2, "#e6e6e6", "第 2 轮"), (3, "#f3f3f3", "第 3 轮")):
        idx = [i + 1 for i, v in enumerate(rounds) if v == r]
        ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=c, lw=0, zorder=0)
        ax.text((idx[0] + idx[-1]) / 2, 2.47, lab, ha="center", va="top", fontsize=8.5, color=GRAY)
    ax.step(xs, ys, where="post", color="#555555", lw=1.2, zorder=1)
    for x, y, c in zip(xs[1:], ys[1:], cats):
        ax.plot(x, y, "o", color=GROUP_COLOR[c], ms=6.5, zorder=3)
    ax.axhline(2.13, color=RED, lw=0.9, linestyle=(0, (5, 3)), zorder=1)
    ax.text(15.45, 2.16, "密封验收\n2.13×", color=RED, fontsize=8, va="bottom", ha="left", linespacing=1.3)
    ax.text(15.45, 2.04, f"15 项的乘积\n{ys[-1]:.2f}×", color=DARK, fontsize=8, va="top", ha="left", linespacing=1.3)
    ax.annotate("FP8 草稿,块长 12:1.160×", (1, ys[1]), xytext=(1.5, 1.0), fontsize=7.6, color=GROUP_COLOR["草稿模型"], va="center",
                arrowprops=dict(arrowstyle="-", color=GROUP_COLOR["草稿模型"], lw=0.7))
    ax.annotate("树状验证:1.160×", (2, ys[2]), xytext=(3.3, 1.13), fontsize=7.6, color=GREEN, va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    ax.annotate("W12:1.076×", (8, ys[8]), xytext=(5.3, 1.4), fontsize=7.6, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.annotate("两级验证:1.037×", (13, ys[13]), xytext=(9.4, 2.22), fontsize=7.6, color=PURP, va="center",
                arrowprops=dict(arrowstyle="-", color=PURP, lw=0.7))
    handles = [plt.Line2D([], [], marker="o", lw=0, markerfacecolor=GROUP_COLOR[k], markersize=6.5, label=k)
               for k in ("草稿模型", "树", "W12", "两级验证", "其它内核")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=5, fontsize=8, handletextpad=0.2, columnspacing=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.96, 2.55)
    ax.set_yticks([1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], ["1.0", "1.1", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])
    ax.set_xlim(-0.4, 18.6)
    ax.set_xticks(range(0, 16))
    ax.set_xlabel("被控制器收下的改动 (按先后)")
    ax.set_ylabel("相对公开实现的加速比")
    clean(ax)
    save(fig, "fig4_promotions.png")


# ---------------------------------------------------------------------------------------------- fig5
def fig5_platforms():
    p = calc["platforms"]
    names = [("h100", "H100\n(论文)"), ("pro6000", "RTX PRO 6000\n(论文)"), ("c500", "C500\n(复现仓库)")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(len(names))
    for off, lab, c, key in ((-0.27, "自回归", GRAY, "ar"), (0.0, "公开实现", BLUE, "stock"), (0.27, "SpecPilot", GREEN, "final")):
        vals = [p[n][key] for n, _ in names]
        bars = a.bar(x + off, vals, width=0.25, color=c, label=lab)
        for rect, v in zip(bars, vals):
            a.text(rect.get_x() + rect.get_width() / 2, v + 6, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, color=c)
    cm = calc["cost_model"]
    for i, (n, _) in enumerate(names):
        c = cm[n]
        a.text(x[i], 556, f"公开实现 {c['S_stock']:.2f}×\nSpecPilot {c['S_final']:.2f}×", ha="center", va="top", fontsize=7.8, color=DARK, linespacing=1.35)
    a.set_xticks(x, [lab for _, lab in names], fontsize=8.5)
    a.set_ylim(0, 560)
    a.set_yticks([0, 100, 200, 300, 400, 500])
    a.set_ylabel("输出速度 (token/s)")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=3)
    a.set_title("单请求的解码速度", fontsize=9.5, pad=20)
    clean(a)

    c_max, tau_max = 2.7, 16.0
    cs = np.linspace(1.0, c_max, 50)
    for s_ in (2, 4, 8, 16):
        b.plot(cs, s_ * cs, color="#cfcfcf", lw=0.9, zorder=0)
        c_end = min(c_max, tau_max / s_)                     # 这条线从哪儿离开坐标区,标签就贴在哪儿
        b.text(c_end - (0.04 if c_end < c_max else 0.0), s_ * c_end - 0.25, f"{s_}×", fontsize=7.8, color=GRAY,
               ha="right" if c_end == c_max else "left", va="top")
    place = {  # (dx, dy, ha, va) in points, for the SpecPilot (final) and stock points of each platform
        "h100": {"final": (8, 0, "left", "center"), "stock": (-8, 0, "right", "center")},
        "pro6000": {"final": (-8, 0, "right", "center"), "stock": (0, 8, "center", "bottom")},
        "c500": {"final": (8, 0, "left", "center"), "stock": (7, -7, "left", "top")},
    }
    for n, short in (("h100", "H100"), ("pro6000", "PRO 6000"), ("c500", "C500")):
        c = cm[n]
        for kind, m, col in (("stock", "o", BLUE), ("final", "s", GREEN)):
            b.plot(c["c_" + kind], c["tau_" + kind], m, color=col, ms=7, mec="white", mew=0.8, zorder=3)
            dx, dy, ha, va = place[n][kind]
            b.annotate(short, (c["c_" + kind], c["tau_" + kind]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=col, ha=ha, va=va)
    b.plot([], [], "o", color=BLUE, label="公开实现")
    b.plot([], [], "s", color=GREEN, label="SpecPilot")
    b.legend(loc="lower right", fontsize=8)
    b.set_xlim(1.0, 2.7)
    b.set_ylim(0, 16)
    b.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    b.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    b.set_xlabel("c:一步的时间 ÷ 自回归一步的时间")
    b.set_ylabel("τ:一步接受的 token 数")
    b.set_title("加速比 = τ ÷ c", fontsize=9.5)
    clean(b)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig5_platforms.png")


# ---------------------------------------------------------------------------------------------- fig6, fig7 (RTX 4090 D)
SHAPES = [("q_proj", "q_proj\n4096×4096", BLUE), ("gate_proj", "gate_proj\n14336×4096", ORANGE),
          ("down_proj", "down_proj\n4096×14336", GREEN), ("lm_head", "lm_head\n128256×4096", PURP)]
METHODS = [("bf16_triton", "BF16,Triton (对照组)", GRAY), ("dequant", "先解码回 BF16,再 cuBLAS", GOLD),
           ("w12", "W12,边读边解码", BLUE), ("q8", "只读高 8 位 (Q pass)", GREEN)]


def fig6_rows():
    rows, bw = bench["rows"], bench["copy_bandwidth_gbps"]
    sizes = {tag: (d["info"]["N"], d["info"]["K"]) for tag, d in bench["verify"].items()}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for tag, label, c in SHAPES:
        ms = sorted(int(m) for m in rows[tag])
        us = [rows[tag][str(m)]["us_median"] for m in ms]
        ax.plot(ms, us, "o-", color=c, ms=3.6, lw=1.4, label=label)
        n, k = sizes[tag]
        ax.axhline(2 * n * k / (bw * 1e9) * 1e6, color=c, lw=0.8, linestyle=(0, (5, 4)))
    ms = sorted(int(m) for m in rows["q_proj"])
    r1, r16, r64 = (rows["q_proj"][str(m)]["us_median"] for m in (1, 16, 64))
    ax.annotate(f"q_proj:1 行 {r1:.1f} µs,16 行 {r16:.1f} µs,64 行 {r64:.1f} µs", (16, r16), xytext=(1.25, 62),
                fontsize=8, color=BLUE, va="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.annotate("lm_head:1 行 1,070 µs,64 行 1,211 µs", (64, bench["rows"]["lm_head"]["64"]["us_median"]), xytext=(1.25, 1750),
                fontsize=8, color=PURP, va="center", arrowprops=dict(arrowstyle="-", color=PURP, lw=0.7))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_yticks([30, 40, 60, 100, 200, 400, 1000, 2000, 4000, 8000], ["30", "40", "60", "100", "200", "400", "1000", "2000", "4000", "8000"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("行数 M (一趟验证的 token 数)")
    ax.set_ylabel("矩阵乘的耗时 (µs)")
    ax.set_title("RTX 4090 D,BF16,cuBLAS,真实权重", fontsize=9.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=GRAY, lw=0.8, linestyle=(0, (5, 4))))
    labels.append("虚线:权重字节数 ÷ 拷贝带宽")
    ax.legend(handles, labels, loc="upper left", fontsize=7.8, ncol=2, bbox_to_anchor=(0.0, 1.0), columnspacing=1.2, handlelength=1.8)
    ax.set_ylim(25, 16000)
    ax.set_xlim(0.8, 1100)
    clean(ax)
    save(fig, "fig6_rows.png")


def fig7_w12_speed():
    m = "16"
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(SHAPES))
    w = 0.2
    for i, (key, label, c) in enumerate(METHODS):
        vals = [bench["verify"][tag]["points"][m]["bf16"]["us_median"] / bench["verify"][tag]["points"][m][key]["us_median"] for tag, _, _ in SHAPES]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=c, label=label)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4, color=c)
    ax.axhline(1.0, color=DARK, lw=1.0)
    ax.set_xticks(x, [label for _, label, _ in SHAPES], fontsize=8.5)
    ax.set_ylabel("相对 BF16 cuBLAS 的速度")
    ax.set_title("RTX 4090 D:同一个 Triton 骨架,权重换成 Llama-3.1-8B 的真实权重,M = 16 行", fontsize=9.5)
    ax.set_ylim(0, 2.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                  ["0", "0.25", "0.50", "0.75", "1.00  BF16 直接读,cuBLAS", "1.25", "1.50", "1.75", "2.00"])
    ax.legend(loc="upper left", fontsize=8, ncol=2, columnspacing=1.2)
    clean(ax)
    save(fig, "fig7_w12_speed.png")


if __name__ == "__main__":
    fig1_timeline()
    fig2_tree()
    fig3_w12_window()
    fig4_promotions()
    fig5_platforms()
    if bench:
        fig6_rows()
        fig7_w12_speed()
