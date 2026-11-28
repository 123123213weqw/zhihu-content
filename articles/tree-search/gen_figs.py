#!/usr/bin/env python3
"""树搜索一篇的图(单线程 CPU 实测,Xeon 8352V)。

fig1_search.png       (a) 二分查找的决策树:15 个键里找 11,看了 4 个,扔掉 11 个  (b) 比较次数随表的大小怎么涨
fig2_trees.png        23 个键的三棵二叉搜索树:按从小到大插入,随机插入,AVL 树
fig3_bptree.png       (a) B+ 树长什么样,一次查找的路径  (b) 层数跟分叉数的关系  (c) SQLite:按主键,按索引,全表扫描
fig4_prefix.png       (a) 前缀树  (b) 压缩之后  (c) 存的条数涨 8 倍,五种找法各要多久
fig5_heap_range.png   (a) 最大的 k 个:排序,和只用一个堆  (b) 区间和:数组大 16 倍,五种操作各慢了多少
fig6_kd.png           (a) kd 树在平面上的切分和剪枝  (b) 维数越高,要看的点越多  (c) cKDTree 对直接算全部距离
fig7_hnsw.png         HNSW:时间和召回率(均匀随机的点,和真实数据)
fig8_game.png         (a) 井字棋:三种搜法各看了多少个局面  (b) UCT:模拟次数和胜率

每个数都来自 bench/calc_summary.json(由 calc.py 写出)。
运行:  python3 gen_figs.py   ->  ./images/      (TREE_BENCH 指定别的 bench 目录)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)
BENCH = os.environ.get("TREE_BENCH", os.path.join(HERE, "bench"))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, ORANGE, PURP, GOLD = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#995588", "#b8922e"
LIGHT = "#e9e9e9"
calc = json.load(open(os.path.join(BENCH, "calc_summary.json")))


def fmt(x):
    return "{:,}".format(int(round(x)))


def us_text(v):
    """把微秒写成人话。"""
    if v < 10:
        return "%.1f 微秒" % v
    if v < 1000:
        return "%.0f 微秒" % v
    if v < 1e4:
        return "%.1f 毫秒" % (v / 1e3)
    return "%.0f 毫秒" % (v / 1e3)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, mutation_scale=11,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


# ---------------------------------------------------------------------------------------------- fig1
def fig1_search():
    lv = calc["linear_vs_binary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
    a.set_xlim(-0.5, 14.5); a.set_ylim(-0.5, 4.9); a.axis("off")
    path = (8, 12, 10, 11)        # 找 11 经过的四个节点
    nodes, edges = [], []

    def build(lo, hi, depth, x0, x1, parent=None):
        """每一层取剩下那一段的中间那个当节点;左边是比它小的,右边是比它大的。"""
        if lo > hi:
            return
        mid = (lo + hi) // 2
        x, y = (x0 + x1) / 2, 4 - depth
        on = (parent is None or parent[2]) and (mid in path)
        if parent is not None:
            edges.append((parent[0], parent[1], x, y, on and parent[2]))
        nodes.append((x, y, mid, on))
        build(lo, mid - 1, depth + 1, x0, x, (x, y, on))
        build(mid + 1, hi, depth + 1, x, x1, (x, y, on))

    build(1, 15, 0, 0.0, 14.0)
    for x0, y0, x1, y1, on in edges:
        a.plot([x0, x1], [y0, y1], color=RED if on else "#b5b5b5", lw=2.4 if on else 1.0, zorder=1, solid_capstyle="round")
    for x, y, mid, on in nodes:
        a.scatter([x], [y], s=900, facecolor="#fdf3ec" if on else "white", edgecolor=RED if on else GRAY, linewidth=1.9 if on else 1.0, zorder=3)
        a.text(x, y, str(mid), ha="center", va="center", fontsize=9.5, color=RED if on else "#333333",
               fontweight="bold" if on else "normal", zorder=4)
    a.text(3.5, 0.15, "扔掉 7 个", ha="center", va="top", fontsize=8.8, color=GRAY)
    a.text(12.25, 0.15, "扔掉 3 个", ha="center", va="top", fontsize=8.8, color=GRAY)
    a.text(7.875, 0.15, "扔掉 1 个", ha="center", va="top", fontsize=8.8, color=GRAY)
    a.text(7.0, 4.55, "在 1 到 15 里找 11:比了 4 次(红色),扔掉 11 个", ha="center", va="bottom", fontsize=10)
    a.set_ylim(-0.9, 5.2)

    n = np.logspace(1, 9, 200)
    b.plot(n, n / 2, color=GRAY, lw=1.6, label="顺序查找(要找的在表里):平均比 $N/2$ 次")
    b.plot(n, np.log2(n), color=RED, lw=1.6, label="二分查找:最多比 $\\lceil \\log_2(N{+}1) \\rceil$ 次")
    m = lv["n"]
    b.scatter([m, m], [lv["linear_hit"]["comparisons"], lv["binary_hit"]["comparisons"]], color=[GRAY, RED], zorder=3, s=32)
    b.annotate("实测,$N$ = 1,048,576\n顺序 %s 次,二分平均 %.1f 次" % (fmt(lv["linear_hit"]["comparisons"]), lv["binary_hit"]["comparisons"]),
               xy=(m, lv["linear_hit"]["comparisons"]), xytext=(18, 6e5), fontsize=8.5, color="#333333", linespacing=1.4,
               arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(10, 1e9); b.set_ylim(1, 1e9)
    b.set_xlabel("$N$:表里有多少个数(对数)"); b.set_ylabel("比了几次(对数)")
    b.set_title("(b) 表每大 1,000 倍:顺序多比一千倍,二分多比 10 次", fontsize=10)
    b.legend(fontsize=8.5, loc="lower right")
    b.grid(alpha=0.25, lw=0.5)
    save(fig, "fig1_search.png")


# ---------------------------------------------------------------------------------------------- fig2
def draw_tree(ax, t, title, color):
    pos = {n[0]: (n[1], -n[2]) for n in t["points"]}       # 横坐标是中序遍历的次序,纵坐标是深度
    for k1, k2 in t["edges"]:
        (x1, y1), (x2, y2) = pos[k1], pos[k2]
        ax.plot([x1, x2], [y1, y2], color="#a0a0a0", lw=0.9, zorder=1)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=210, facecolor="#eef4fa", edgecolor=color, linewidth=1.2, zorder=3)
    for k, (x, y) in pos.items():
        ax.text(x, y, str(k), ha="center", va="center", fontsize=6.4, color=color, zorder=4)
    h = t["height"]
    ax.set_xlim(-1.0, max(p[0] for p in pos.values()) + 1.0); ax.set_ylim(-h + 0.2, 1.2); ax.axis("off")
    ax.set_title(title, fontsize=10)


def fig2_trees():
    t = calc["trees_small"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2))
    draw_tree(axes[0], t["sorted"], "(a) 按从小到大插入:高度 %d,平均深度 %.1f" % (t["sorted"]["height"], t["sorted"]["avg_depth"]), RED)
    draw_tree(axes[1], t["random"], "(b) 随机插入:高度 %d,平均深度 %.1f" % (t["random"]["height"], t["random"]["avg_depth"]), BLUE)
    draw_tree(axes[2], t["avl"], "(c) AVL 树:高度 %d,平均深度 %.1f" % (t["avl"]["height"], t["avl"]["avg_depth"]), GREEN)
    fig.suptitle("同样 %d 个键(圆圈里是键本身),插入顺序不同,树的高低差这么多。三张图的纵轴比例不一样" % t["n"], fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig2_trees.png")


# ---------------------------------------------------------------------------------------------- fig3
def fig3_bptree():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [2.1, 1.0, 1.0]})
    a.set_xlim(0, 100); a.set_ylim(-4, 64); a.axis("off")
    leaves = [[3, 5, 8], [11, 13], [17, 19, 21], [25, 27], [35, 38], [40, 45, 50]]
    xs = [8, 24.5, 41, 57.5, 74, 90.5]
    for i, leaf in enumerate(leaves):
        x, hit = xs[i], i == 2                 # 找 19:落在第 3 个叶子上
        a.add_patch(FancyBboxPatch((x - 7.2, 5), 14.4, 7, boxstyle="round,pad=0.02,rounding_size=0.6",
                                   fc="#fdf3ec" if hit else "#f5f5f5", ec=ORANGE if hit else GRAY, lw=1.9 if hit else 1.0))
        for j, k in enumerate(leaf):
            cx = x - 7.2 + 14.4 * (j + 0.5) / len(leaf)
            a.text(cx, 8.5, str(k), ha="center", va="center", fontsize=8.6, color="#333333")
            if j:
                a.plot([x - 7.2 + 14.4 * j / len(leaf)] * 2, [5, 12], color="#cccccc", lw=0.7)
        if i < len(leaves) - 1:
            arrow(a, x + 7.5, 8.5, xs[i + 1] - 7.6, 8.5, color="#aaaaaa", lw=0.9)
    internals = [(11, 16.25), (25, 49.25), (40, 82.25)]
    for i, (key, x) in enumerate(internals):
        hit = i == 1
        a.add_patch(FancyBboxPatch((x - 6, 27), 12, 7, boxstyle="round,pad=0.02,rounding_size=0.6",
                                   fc="#fdf3ec" if hit else "#eef4fa", ec=ORANGE if hit else BLUE, lw=1.9 if hit else 1.0))
        a.text(x, 30.5, str(key), ha="center", va="center", fontsize=9.5, color=ORANGE if hit else BLUE, fontweight="bold")
        arrow(a, x, 27, xs[2 * i], 12.5, color=ORANGE if hit else "#aaaaaa", lw=1.9 if hit else 0.9)
        arrow(a, x, 27, xs[2 * i + 1], 12.5, color="#aaaaaa", lw=0.9)
    a.add_patch(FancyBboxPatch((43, 49), 18, 7, boxstyle="round,pad=0.02,rounding_size=0.6", fc="#fdf3ec", ec=ORANGE, lw=1.9))
    a.text(47.5, 52.5, "17", ha="center", va="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    a.text(56.5, 52.5, "35", ha="center", va="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    a.plot([52, 52], [49, 56], color="#cccccc", lw=0.7)
    arrow(a, 47.5, 49, internals[0][1], 34.5, color="#aaaaaa", lw=0.9)
    arrow(a, 52, 49, internals[1][1], 34.5, color=ORANGE, lw=1.9)
    arrow(a, 56.5, 49, internals[2][1], 34.5, color="#aaaaaa", lw=0.9)
    a.text(50, 60.5, "找 19:17 ≤ 19 < 35,走中间;19 < 25,走左边;在叶子里找到。读 3 个节点", ha="center", va="center", fontsize=9.6)
    a.text(50, -0.5, "一个节点就是磁盘上的一页。根和内部节点只放路标,真正的键都在叶子上,叶子之间用箭头串起来",
           ha="center", va="center", fontsize=8.6, color=GRAY)

    names = {"1000000": "一百万", "1000000000": "十亿", "1000000000000": "一万亿"}
    for n, color in (("1000000", BLUE), ("1000000000", ORANGE), ("1000000000000", GREEN)):
        rows = calc["bptree"]["by_n"][n]
        b.plot([r["fan"] for r in rows], [r["levels"] for r in rows], "o-", color=color, lw=1.5, ms=4, label="%s个键" % names[n])
        for r, dx, ha in ((rows[0], 9, "left"), (rows[-1], 9, "left")):     # 只在两头标层数,免得挤在一起
            b.annotate("%d 层" % r["levels"], xy=(r["fan"], r["levels"]), xytext=(dx, -2), textcoords="offset points", ha=ha, va="center", fontsize=8, color=color)
    b.set_xscale("log", base=2); b.set_xticks([4, 16, 64, 256, 1024]); b.set_xticklabels(["4", "16", "64", "256", "1,024"])
    b.set_xlim(3.0, 4200); b.set_ylim(1, 22); b.set_yticks([2, 5, 10, 15, 20])
    b.set_xlabel("一个节点最多分几个叉"); b.set_ylabel("一次查找要读几个节点")
    b.set_title("(b) 叉分得越多,层数越少", fontsize=10)
    b.legend(fontsize=8.5, loc="upper right"); b.grid(alpha=0.25, lw=0.5)

    s = calc["sqlite"]
    vals = [s["by_pk_us"], s["by_index_us"], s["scan_s"] * 1e6]
    c.bar(range(3), vals, 0.55, color=[BLUE, PURP, GRAY])
    for i, v in enumerate(vals):
        c.text(i, v * 1.3, "%.1f 微秒" % v if v < 1e4 else "%.0f 毫秒" % (v / 1e3), ha="center", va="bottom", fontsize=8.8)
    c.set_yscale("log"); c.set_ylim(1, 1e7)
    c.set_xticks(range(3)); c.set_xticklabels(["按主键找", "按二级索引找", "没有索引,\n全表扫描"], fontsize=8.6)
    c.set_ylabel("一次查询的时间(微秒,对数)")
    c.set_title("(c) SQLite,%s 行" % fmt(s["rows"]), fontsize=10)
    c.text(0.5, 0.80, "两头差 %s 倍" % fmt(s["speedup"]), transform=c.transAxes, ha="center", fontsize=9.5, color=RED)
    c.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    save(fig, "fig3_bptree.png")


# ---------------------------------------------------------------------------------------------- fig4
def word_tree(ax, words, compress, title):
    """words 里的词建成前缀树;compress 为真时,只有一个孩子的节点并进边里。"""
    trie = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node["#"] = {}
    pos, edges = {}, []

    def place(node, label, depth, x0, x1, parent_xy, from_label):
        kids = [k for k in node if k != "#"]
        cx = (x0 + x1) / 2
        pos[id(node)] = (cx, -depth)
        if parent_xy is not None:
            edges.append((parent_xy, (cx, -depth), from_label))
        return kids, cx

    if not compress:
        def walk(node, label, depth, x0, x1, parent_xy):
            kids, cx = place(node, label, depth, x0, x1, parent_xy, label)
            for i, ch in enumerate(kids):
                walk(node[ch], ch, depth + 1, x0 + (x1 - x0) * i / len(kids), x0 + (x1 - x0) * (i + 1) / len(kids), (cx, -depth))
        walk(trie, "", 0, 0.0, 10.0, None)
    else:
        def walk(node, label, depth, x0, x1, parent_xy):
            while len([k for k in node if k != "#"]) == 1 and "#" not in node:
                (ch, child), = [(k, v) for k, v in node.items() if k != "#"]
                label += ch
                node = child
            kids, cx = place(node, label, depth, x0, x1, parent_xy, label)
            for i, ch in enumerate(kids):
                walk(node[ch], ch, depth + 1, x0 + (x1 - x0) * i / len(kids), x0 + (x1 - x0) * (i + 1) / len(kids), (cx, -depth))
        walk(trie, "", 0, 0.0, 10.0, None)
    ends = set()
    stack = [(trie, id(trie))]
    # 哪些节点是词尾:沿着坐标表反查不方便,这里再走一遍记下来
    pos_ids = list(pos)
    def mark(node, depth, x0, x1):
        pass
    # 直接按位置标词尾:每个词的最后一个节点
    def last_pos(node, w):
        for ch in w:
            node = node[ch]
        return pos[id(node)]
    # 压缩之后节点的位置表里,键是压缩前 trie 里"分叉点或词尾"的节点,直接查得到
    for w in words:
        ends.add(last_pos(trie, w))
    for (p0, p1, label) in edges:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#a0a0a0", lw=0.9, zorder=1)
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.09, label, ha="center", va="bottom", fontsize=8.6, color=BLUE)
    inner = [xy for xy in pos.values() if xy not in ends]
    ax.scatter([p[0] for p in inner], [p[1] for p in inner], s=230, facecolor="#eef4fa", edgecolor=BLUE, linewidth=1.2, zorder=3)
    ax.scatter([p[0] for p in ends], [p[1] for p in ends], s=230, facecolor="#fdf3ec", edgecolor=ORANGE, linewidth=1.5, zorder=3)
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-4.7, 0.7); ax.axis("off")
    ax.set_title("%s:%d 个节点" % (title, len(pos)), fontsize=10)


def fig4_prefix():
    p = calc["prefix"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})
    words = ["cat", "car", "card", "dog"]
    word_tree(a, words, False, "(a) 前缀树,一个字母一层")
    word_tree(b, words, True, "(b) 压缩之后,单链并成一条边")
    ns = [r["n"] for r in p["rows"]]
    styles = (("linear_token", GRAY, "-", "逐个 token 线性比"), ("linear_block", GOLD, "-", "按块线性比(16 个 token 一块)"),
              ("trie", BLUE, "-", "前缀树"), ("radix", GREEN, "--", "压缩的前缀树"), ("chained_hash", RED, "-", "链式哈希表"))
    for name, color, ls, label in styles:
        c.plot(ns, [r[name]["ms"] for r in p["rows"]], "o" + ls, color=color, lw=1.5, ms=5, label=label)
        for x, r in zip(ns, p["rows"]):
            c.annotate("%.3f" % r[name]["ms"] if r[name]["ms"] < 0.1 else "%.1f" % r[name]["ms"], xy=(x, r[name]["ms"]),
                       xytext=(0, 6 if name in ("trie", "linear_block") else -11), textcoords="offset points", fontsize=7.8, color=color, ha="center")
    c.set_xscale("log"); c.set_yscale("log")
    c.xaxis.set_minor_formatter(NullFormatter())
    c.set_xticks(ns); c.set_xticklabels([fmt(n) for n in ns]); c.set_xlim(330, 6000)
    c.set_ylim(0.002, 150)
    c.set_xlabel("存了多少条序列(每条 512 个 token,前 256 个相同)")
    c.set_ylabel("查一次的用时(毫秒,对数)")
    c.set_title("(c) 条数涨 8 倍:线性比跟着涨,树和哈希表不涨", fontsize=10)
    c.legend(fontsize=8.3, loc="upper left")
    c.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig4_prefix.png")


# ---------------------------------------------------------------------------------------------- fig5
def fig5_heap_range():
    t = calc["topk"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ks = [r["k"] for r in t["rows"]]
    xs = np.arange(len(ks))
    a.bar(xs - 0.19, [t["sort_s"]] * 3, 0.36, color=GRAY, label="全部排序,再取前 k 个")
    a.bar(xs + 0.19, [r["nlargest_s"] for r in t["rows"]], 0.36, color=ORANGE, label="只装 k 个数的堆(heapq.nlargest)")
    for x, r in zip(xs, t["rows"]):
        a.text(x - 0.19, t["sort_s"] * 1.04, "%.2f 秒" % t["sort_s"], ha="center", va="bottom", fontsize=8.8)
        a.text(x + 0.19, r["nlargest_s"] * 1.04, "%.3f 秒\n快 %.0f 倍" % (r["nlargest_s"], r["speedup"]), ha="center", va="bottom", fontsize=8.8, color=ORANGE, linespacing=1.3)
    a.set_xticks(xs); a.set_xticklabels(["k = %d" % k for k in ks])
    a.set_ylim(0, 2.5); a.set_yticks([0, 0.5, 1.0, 1.5, 2.0]); a.set_ylabel("时间(秒)")
    a.set_title("(a) %s 个数里挑最大的 k 个" % fmt(t["n"]), fontsize=10)
    a.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    sc = t["sort_counted"]
    a.text(0.03, 0.985, "排序:每个数摊到 %.2f 倍的 $\\log_2 N$ 次比较,约 %s 次\n堆:每个数只和堆顶比一次,约 %s 次\nk = 10 时进过堆的只有 %d 个数"
           % (sc["per_n_log2n"], fmt(t["sort_comparisons_4m"]), fmt(t["rows"][0]["heap_comparisons_est"]), t["rows"][0]["entered_heap"]),
           transform=a.transAxes, fontsize=8.5, color="#333333", va="top", linespacing=1.5)

    r = calc["range"]
    ns = [x["n"] for x in r["rows"]]
    series = (("naive_query_us", GRAY, "-", "直接加 "), ("fenwick_query_us", GREEN, "-", "树状数组求和 "),
              ("prefix_query_us", BLUE, "-", "前缀和求和 "), ("prefix_update_us", RED, "--", "前缀和改一个数 "),
              ("fenwick_update_us", ORANGE, "--", "树状数组改一个数 "))
    offsets = {"naive_query_us": (7, 0), "fenwick_query_us": (7, 4), "prefix_query_us": (7, -9),
               "prefix_update_us": (7, 0), "fenwick_update_us": (7, -11)}
    for key, color, ls, label in series:
        b.plot(ns, [x[key] for x in r["rows"]], "o" + ls, color=color, lw=1.5, ms=4)
        last = r["rows"][-1][key]
        unit = "微秒" if last < 1000 else "毫秒"
        value = last if last < 1000 else last / 1e3
        b.annotate("%s %.1f %s" % (label, value, unit) if value < 100 else "%s %.0f %s" % (label, value, unit),
                   xy=(ns[-1], last), xytext=offsets[key], textcoords="offset points", fontsize=8.2, color=color, va="center")
    b.set_xscale("log", base=2); b.set_yscale("log")
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xticks(ns); b.set_xticklabels([fmt(n) for n in ns])
    b.set_xlim(7e5, 1.3e9); b.set_ylim(0.3, 2e5)
    b.set_xlabel("数组有多少个数"); b.set_ylabel("一次操作的时间(微秒,对数)")
    b.set_title("(b) 区间和:数组大 16 倍,树状数组只多走 %d 步" % r["scale"]["extra_query_steps"], fontsize=10)
    b.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig5_heap_range.png")


# ---------------------------------------------------------------------------------------------- fig6
def kd_demo(pts, q, leaf=3):
    """给二维的点搭一棵 kd 树(叶子最多 leaf 个点),查离 q 最近的点。返回切分线,算过距离的叶子,被剪掉的叶子,最近距离。"""
    lines, kept, pruned, best = [], [], [], [float("inf")]

    def build(idx, depth, lo, hi):
        if len(idx) <= leaf:
            return {"leaf": True, "lo": lo, "hi": hi, "idx": idx}
        axis = depth % 2
        order = idx[np.argsort(pts[idx, axis], kind="stable")]
        m = len(idx) // 2
        split = float(pts[order[m], axis])
        cut = np.arange(2) == axis
        lines.append((axis, split, lo.copy(), hi.copy()))
        return {"leaf": False, "axis": axis, "split": split, "lo": lo, "hi": hi,
                "l": build(order[:m], depth + 1, lo, np.where(cut, split, hi)),
                "r": build(order[m:], depth + 1, np.where(cut, split, lo), hi)}

    def leaves(node, acc):
        (acc.append(node) if node["leaf"] else (leaves(node["l"], acc), leaves(node["r"], acc)))

    def visit(node):
        gap = 0.0                                  # q 到这块矩形最近可能的距离的平方
        for k in range(2):
            gap += max(0.0, node["lo"][k] - q[k]) ** 2 + max(0.0, q[k] - node["hi"][k]) ** 2
        if gap >= best[0]:                         # 这块里就算有最近的点,也比手里的答案远:整块剪掉
            leaves(node, pruned)
            return
        if node["leaf"]:
            best[0] = min(best[0], float(((pts[node["idx"]] - q) ** 2).sum(1).min()))
            kept.append(node)
            return
        near, far = (node["l"], node["r"]) if q[node["axis"]] < node["split"] else (node["r"], node["l"])
        visit(near)
        visit(far)

    visit(build(np.arange(len(pts)), 0, np.zeros(2), np.ones(2)))
    return lines, kept, pruned, math.sqrt(best[0])


def fig6_kd():
    k, ck = calc["kd_count"], calc["ckdtree"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.8, 4.9), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    pts = np.random.default_rng(12).random((28, 2))
    q = np.array([0.62, 0.30])
    lines, kept, pruned, radius = kd_demo(pts, q)
    for node in pruned:
        a.add_patch(Rectangle(node["lo"], node["hi"][0] - node["lo"][0], node["hi"][1] - node["lo"][1], fc=LIGHT, ec="#d0d0d0", lw=0.6, zorder=0))
    for node in kept:
        a.add_patch(Rectangle(node["lo"], node["hi"][0] - node["lo"][0], node["hi"][1] - node["lo"][1], fc="#e4f1e4", ec=GREEN, lw=1.1, zorder=0))
    for axis, split, lo, hi in lines:
        if axis == 0:
            a.plot([split, split], [lo[1], hi[1]], color=BLUE, lw=0.9)
        else:
            a.plot([lo[0], hi[0]], [split, split], color=BLUE, lw=0.9)
    a.scatter(pts[:, 0], pts[:, 1], s=24, color="#333333", zorder=3)
    a.add_patch(Circle(q, radius, fill=False, ec=RED, lw=1.4, zorder=4))
    a.scatter([q[0]], [q[1]], marker="*", s=170, color=RED, zorder=5)
    nearest = pts[int(((pts - q) ** 2).sum(1).argmin())]
    a.scatter([nearest[0]], [nearest[1]], s=80, facecolor="none", edgecolor=RED, lw=1.3, zorder=5)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) 2 维:每个格子管一块地盘。离红圈还远的\n整块剪掉(灰),只看绿格子里的点", fontsize=9.6)

    rows = k["rows"]
    colors = [BLUE if r["fraction"] < 0.001 else (BLUE if r["fraction"] < 0.01 else (ORANGE if r["fraction"] < 0.5 else RED)) for r in rows]
    b.bar(range(len(rows)), [r["fraction"] * 100 for r in rows], 0.6, color=colors)
    b.set_yscale("log"); b.set_ylim(0.03, 1000)
    for i, r in enumerate(rows):
        f = r["fraction"] * 100
        b.text(i, f * 1.3, "%.2f%%" % f if f < 0.1 else ("%.1f%%" % f if f < 10 else "%.0f%%" % f), ha="center", va="bottom", fontsize=8.4)
    b.set_xticks(range(len(rows))); b.set_xticklabels([str(r["d"]) for r in rows])
    b.set_xlabel("维数"); b.set_ylabel("一次查询算了百分之几的点的距离(对数)")
    b.set_title("(b) 维数越高,能剪掉的越少(%s 个点)" % fmt(k["n"]), fontsize=10)
    b.grid(alpha=0.25, lw=0.5, axis="y")

    c.plot([r["d"] for r in ck["rows"]], [r["speedup"] for r in ck["rows"]], "o-", color=PURP, lw=1.5, ms=5)
    c.set_xscale("log", base=2); c.set_yscale("log")
    c.xaxis.set_minor_formatter(NullFormatter())
    c.axhline(1.0, color=GRAY, lw=1.0, ls="--")
    c.text(1.95, 1.3, "虚线:和直接算一样快", fontsize=8.3, color=GRAY)
    for r in ck["rows"][1:]:
        c.annotate("%.1f 倍" % r["speedup"], xy=(r["d"], r["speedup"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.4, color=PURP)
    c.annotate("%.1f 倍" % ck["rows"][0]["speedup"], xy=(2, ck["rows"][0]["speedup"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.4, color=PURP)
    c.set_xticks([r["d"] for r in ck["rows"]]); c.set_xticklabels([str(r["d"]) for r in ck["rows"]])
    c.set_xlim(1.3, 3200); c.set_ylim(0.15, 60)
    c.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20]); c.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    c.set_xlabel("维数(对数)"); c.set_ylabel("直接算的时间 ÷ cKDTree 的时间(对数)")
    c.set_title("(c) 低维快 9 倍上下,32 维起反而慢(%s 个点)" % fmt(ck["n"]), fontsize=10)
    c.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig6_kd.png")


# ---------------------------------------------------------------------------------------------- fig7
def fig7_hnsw():
    h, m = calc["hnsw"], calc["hnsw_mnist"]
    GRAYS = [GRAY, ORANGE, GOLD, GREEN]          # 四组数据各用一种颜色(名字起早了),下面按颜色给"老实算"的标签错开位置
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    series = [
        ([(r["hnsw_us"], r["recall@10"], r["ef"], r["brute_us"]) for r in h["rows"] if r["d"] == 64], GRAY, "-", "均匀随机的点,100 万个,64 维"),
        ([(r["hnsw_us"], r["recall@10"], r["ef"], r["brute_us"]) for r in h["rows"] if r["d"] == 768], ORANGE, "--", "均匀随机的点,100 万个,768 维"),
        ([(r["hnsw_us"], r["recall@10"], r["ef"], r["brute_us"]) for r in m["uniform_same_shape"]], GOLD, "-", "均匀随机的点,6 万个,784 维"),
        ([(r["hnsw_us"], r["recall@10"], r["ef"], r["brute_us"]) for r in m["mnist"]], GREEN, "-", "手写数字图片,6 万个,784 维"),
    ]
    for pts, color, ls, label in series:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, lw=1.2, alpha=0.8)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=color, s=46, zorder=3)
        for x, y, ef, brute in pts:
            ax.annotate("ef = %d" % ef, xy=(x, y), xytext=(6, -4), textcoords="offset points", fontsize=7.6, color=color)
        star_xy = {GRAYS[0]: (0, 14, "center"), GRAYS[1]: (0, -17, "center"), GRAYS[2]: (0, -17, "center"), GRAYS[3]: (-12, 10, "right")}[color]
        ax.scatter([pts[0][3]], [1.0], marker="*", s=190, color=color, zorder=4)
        ax.annotate("老实算 %s" % us_text(pts[0][3]), xy=(pts[0][3], 1.0), xytext=star_xy[:2], textcoords="offset points",
                    ha=star_xy[2], va="bottom" if star_xy[1] > 0 else "top", fontsize=7.6, color=color)
    ax.set_xscale("log"); ax.set_xlim(60, 9e5); ax.set_ylim(-0.02, 1.16)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("一次查询的时间(微秒,对数)")
    ax.set_ylabel("召回率:真最近的 10 个,找回了几成")
    ax.set_title("HNSW 用时间换召回率(ef 是搜索时一路盯着看的候选数)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(handles=[Line2D([0], [0], color=v[1], ls=v[2], marker="o", label=v[3]) for v in series], fontsize=8.2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig7_hnsw.png")


# ---------------------------------------------------------------------------------------------- fig8
def fig8_game():
    g = calc["game"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15]})
    labels = ["极小极大\n每个局面都看到底", "Alpha-Beta 剪枝\n按格子编号试走法", "Alpha-Beta 剪枝\n先中间,再角,后边"]
    vals = [g["minimax"]["nodes"], g["alphabeta_natural"]["nodes"], g["alphabeta_center_first"]["nodes"]]
    a.bar(range(3), vals, 0.55, color=[GRAY, BLUE, GREEN])
    for i, v in enumerate(vals):
        a.text(i, v * 1.25, "%s 个" % fmt(v), ha="center", va="bottom", fontsize=9)
        if i:
            a.text(i, v * 0.42, "原来的\n%.1f%%" % (v / vals[0] * 100), ha="center", va="center", fontsize=8.6, color="white")
    a.set_yscale("log"); a.set_ylim(1e3, 1.2e7)
    a.set_xticks(range(3)); a.set_xticklabels(labels, fontsize=8.5)
    a.set_ylabel("看了多少个局面(对数)")
    a.set_title("(a) 井字棋从空盘开始找先手的最好走法,三种搜法的答案一样", fontsize=10)
    a.grid(alpha=0.25, lw=0.5, axis="y")

    sims = [r["sims_per_move"] for r in g["uct"]]
    xs = np.arange(len(sims))
    wins = np.array([r["win_rate"] * 100 for r in g["uct"]])
    draws = np.array([r["draw"] / r["games"] * 100 for r in g["uct"]])
    losses = np.array([r["x_loss"] / r["games"] * 100 for r in g["uct"]])
    b.bar(xs, wins, 0.55, color=GREEN, label="先手赢")
    b.bar(xs, draws, 0.55, bottom=wins, color=GOLD, label="平局")
    b.bar(xs, losses, 0.55, bottom=wins + draws, color=RED, label="先手输")
    for x, w, r in zip(xs, wins, g["uct"]):
        b.text(x, w / 2, "%.1f%%" % w, ha="center", va="center", fontsize=9, color="white")
        note = "两边都随机走" if not r["sims_per_move"] else "每局模拟 %.0f 次" % r["simulations_per_game"]
        b.text(x, 103, "%s\n%d 局" % (note, r["games"]), ha="center", va="bottom", fontsize=8, color=GRAY)
    b.set_xticks(xs); b.set_xticklabels(["随机走", "每步模拟 10 次", "每步模拟 100 次", "每步模拟 1,000 次"], fontsize=8.5)
    b.set_ylim(0, 132); b.set_yticks([0, 25, 50, 75, 100]); b.set_ylabel("占对局的百分比")
    b.set_title("(b) 先手用 UCT,对手随机走:模拟越多,赢得越多", fontsize=10)
    b.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig8_game.png")


for fn in (fig1_search, fig2_trees, fig3_bptree, fig4_prefix, fig5_heap_range, fig6_kd, fig7_hnsw, fig8_game):
    fn()
    print("ok", fn.__name__)
