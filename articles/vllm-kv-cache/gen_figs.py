#!/usr/bin/env python3
"""Figures for the vLLM KV cache article (vLLM v0.34.0).

fig1_blocks.png    -- logical view: tokens are cut into blocks, the block table maps logical -> physical blocks,
                      two requests sharing a prefix point at the same physical blocks
fig2_layout.png    -- physical view: one flat buffer, one strided view per layer, address = base + l*layer_stride + b*block_stride
fig3_hash.png      -- prefix caching: the hash chain (parent hash, block tokens, extra keys); shared prefix -> same hashes
fig4_pool.png      -- block pool: ref counts, the free queue (non-cached LIFO at the head, cached LRU at the tail), eviction
fig5_alloc.png     -- allocate_slots: the per-step allocation flow and what the scheduler does when it returns None
fig6_hybrid.png    -- hybrid models: KV cache groups with equal page size share one pool; one logical block = several pieces
fig7_startup.png   -- how the KV cache size is decided at startup: estimate-based vs. the extensible (measured) cache
fig8_naive.png     -- measured cost of writing one token's KV: torch.cat (grows with length) vs. preallocated vs. paged
fig9_calls.png     -- the call chain: scheduler -> KV cache manager -> coordinator -> block pool -> worker -> attention kernel

Run:  python3 gen_figs.py   -> ./images/
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
BENCH = os.environ.get("KV_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, ORANGE, PURP, GOLD = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#995588", "#b8922e"
SHARED, A_ONLY, B_ONLY, FREE, NULLC = "#9ecae1", "#fdd0a2", "#c7e9c0", "#f0f0f0", "#bdbdbd"
AR = "$\\rightarrow$"


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, fw="normal", color=None, ls="-", lw=1.2, pad=0.25, ls_text=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or ec, fontweight=fw,
            linespacing=ls_text)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw, mutation_scale=12,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


# ----------------------------------------------------------------------------------------------
def fig1_blocks():
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    ax.set_xlim(0, 138); ax.set_ylim(0, 72); ax.axis("off")
    ax.text(69, 69.4, "逻辑视图:token 切成块,块表把逻辑块映射到物理块(块大小 B = 4,示意)", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(32, 63.6, "① 请求的 token", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#333333")
    ax.text(71, 63.6, "② 块表", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#333333")
    ax.text(115, 63.6, "③ 物理块池", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#333333")

    cw, ch = 2.15, 3.8
    def tokens(y, kinds, label, ly):
        ax.text(2, y + ch / 2, label, ha="left", va="center", fontsize=9.6, color="#222222")
        for i, k in enumerate(kind_seq(kinds)):
            x = 24 + i * cw
            ax.add_patch(Rectangle((x, y), cw - 0.12, ch, fc=k, ec="white", lw=0.8))
            ax.text(x + (cw - 0.12) / 2, y + ch / 2, str(i), ha="center", va="center", fontsize=7.2, color="#444444")
            if i % 4 == 3:  # closing bracket of a logical block
                xs = x - 3 * cw
                ax.plot([xs, x + cw - 0.12], [y - 0.5, y - 0.5], color="#555555", lw=0.9)
                ax.plot([xs, xs], [y - 0.5, y - 1.2], color="#555555", lw=0.9)
                ax.plot([x + cw - 0.12, x + cw - 0.12], [y - 0.5, y - 1.2], color="#555555", lw=0.9)
                ax.text(xs + 2 * cw - 0.06, y - 1.5, f"逻辑块 {i // 4}", ha="center", va="top", fontsize=7.8, color="#555555")
    def kind_seq(kinds):
        out = []
        for k, n in kinds:
            out += [k] * n
        return out
    tokens(50, [(SHARED, 12), (A_ONLY, 4)], "请求 A\n16 个 token", 0)
    tokens(33, [(SHARED, 8), (B_ONLY, 8)], "请求 B\n前 8 个与 A 相同", 0)

    # block tables
    def table(x, y_top, rows, title, colors):
        ax.text(x + 7.4, y_top + 6.2, title, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#333333")
        ax.text(x + 3.6, y_top + 3.6, "逻辑块", ha="center", va="center", fontsize=8.2, color="#666666")
        ax.text(x + 11.2, y_top + 3.6, "物理块", ha="center", va="center", fontsize=8.2, color="#666666")
        for i, ((l, p), c) in enumerate(zip(rows, colors)):
            yy = y_top - i * 3.4
            box(ax, x, yy, 7.2, 2.7, str(l), "white", "#888888", fs=8.6, color="#222222", pad=0.15)
            box(ax, x + 7.6, yy, 7.2, 2.7, str(p), c, c, fs=8.6, color="#222222", pad=0.15)
    table(58, 55.0, [(0, 12), (1, 5), (2, 30), (3, 7), (4, 3)], "请求 A 的块表", [SHARED, SHARED, A_ONLY, A_ONLY, A_ONLY])
    table(58, 31.0, [(0, 12), (1, 5), (2, 18), (3, 26)], "请求 B 的块表", [SHARED, SHARED, B_ONLY, B_ONLY])

    # physical pool: 8 x 4
    both, aonly, bonly = {12, 5}, {30, 7, 3}, {18, 26}
    px, py, cw2, ch2 = 93, 50.5, 5.6, 6.0
    for r in range(4):
        for c in range(8):
            bid = r * 8 + c
            col = NULLC if bid == 0 else SHARED if bid in both else A_ONLY if bid in aonly else B_ONLY if bid in bonly else FREE
            x, y = px + c * cw2, py - r * ch2
            ax.add_patch(Rectangle((x, y - ch2 + 0.5), cw2 - 0.45, ch2 - 0.8, fc=col, ec="#999999", lw=0.8))
            ax.text(x + (cw2 - 0.45) / 2, y - ch2 + 0.5 + (ch2 - 0.8) / 2, str(bid), ha="center", va="center", fontsize=8.4,
                    color="#222222")
    ax.text(px, 25.2, "同一时间,一个块只属于一个请求,或属于共用它的几个请求", ha="left", va="center", fontsize=8.4, color="#555555")
    # legend
    lx, ly = 93, 21.5
    for k, (c, t) in enumerate([(SHARED, "A 和 B 共用(命中前缀缓存)"), (A_ONLY, "只有 A 在用"), (B_ONLY, "只有 B 在用"),
                                (FREE, "空闲"), (NULLC, "占位块")]):
        xx = lx + (k % 2) * 23.0; yy = ly - (k // 2) * 3.8
        ax.add_patch(Rectangle((xx, yy), 2.4, 2.2, fc=c, ec="#999999", lw=0.8))
        ax.text(xx + 3.2, yy + 1.1, t, ha="left", va="center", fontsize=8.2, color="#333333")
    # arrows from tables to the pool
    arrow(ax, 73.2, 56.35, 117.98, 50.8, BLUE, lw=1.4)
    arrow(ax, 73.2, 25.55, 105.0, 36.4, GREEN, lw=1.2)
    ax.text(2, 10.5, "token 落到哪里,由块表算出来:", ha="left", va="center", fontsize=9.8, color="#222222")
    ax.text(2, 5.6, r"slot = block_table[req][pos // B] $\times$ B + pos % B",
            ha="left", va="center", fontsize=12, color="#111111", fontweight="bold")
    ax.text(64, 5.6, "先由位置算出第几个逻辑块、块内第几格,再用块表换成物理块号", ha="left", va="center", fontsize=9.2,
            color="#555555")
    fig.savefig(f"{OUT}/fig1_blocks.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig2_layout():
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.set_xlim(0, 138); ax.set_ylim(0, 62); ax.axis("off")
    ax.text(69, 60.0, "物理视图:一整块显存,每层一个视图(层在外、块在内的默认摆法)", ha="center", va="center",
            fontsize=13, fontweight="bold")
    # left: whole buffer
    ax.text(31, 53.2, "整块 KV 缓存(L 层 × N 块)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#333333")
    L, N, x0, y0, bw, bh = 6, 8, 9.5, 10.5, 5.2, 5.4
    for l in range(L):
        y = y0 + (L - 1 - l) * (bh + 0.9)
        ax.text(x0 - 1.2, y + bh / 2, f"层 {l}" if l < L - 1 else "层 L-1", ha="right", va="center", fontsize=9, color="#333333")
        for b in range(N):
            x = x0 + b * bw
            fc = "#fdeee2" if (l + b) % 2 == 0 else "#fbdcc6"
            ax.add_patch(Rectangle((x, y), bw - 0.3, bh, fc=fc, ec=ORANGE, lw=0.8))
            if l == 0:
                ax.text(x + (bw - 0.3) / 2, y + bh + 0.5, f"块 {b}" if b < N - 1 else "块 N-1", ha="center", va="bottom",
                        fontsize=8.2, color="#555555")
            if l == 2 and b == 4:
                ax.add_patch(Rectangle((x, y), bw - 0.3, bh, fc=ORANGE, ec=ORANGE, lw=1.4, alpha=0.55))
    arrow(ax, x0 + 4 * bw + (bw - 0.3) / 2, y0 + 2 * (bh + 0.9) + bh / 2, 77.4, 36.0, ORANGE, lw=1.3)
    xr = x0 + N * bw - 0.3
    # strides
    ytop = y0 + (L - 1) * (bh + 0.9) + bh
    arrow(ax, xr + 2.2, ytop, xr + 2.2, ytop - bh - 0.9, ORANGE, lw=1.3, style="<|-|>")
    ax.text(xr + 3.2, ytop - (bh + 0.9) / 2, "层的步长\nlayer_stride", ha="left", va="center", fontsize=8.8, color=ORANGE)
    arrow(ax, x0, y0 - 1.6, x0 + bw, y0 - 1.6, ORANGE, lw=1.3, style="<|-|>")
    ax.text(x0 + bw / 2, y0 - 2.4, "块的步长 block_stride", ha="center", va="top", fontsize=8.8, color=ORANGE)
    ax.text(31, 4.2, r"地址 = 起始 + 层号 $\times$ layer_stride + 块号 $\times$ block_stride", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#111111")
    # right: one block zoomed
    ax.text(104, 53.2, "放大其中一块", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#333333")
    H, Bn, bx, by, cw, ch = 3, 4, 84.0, 25.0, 6.4, 7.2
    for h in range(H):
        for n in range(Bn):
            x, y = bx + n * cw, by + (H - 1 - h) * ch
            ax.add_patch(Rectangle((x, y), cw / 2 - 0.15, ch - 0.6, fc="#dbeaf7", ec=BLUE, lw=0.9))
            ax.add_patch(Rectangle((x + cw / 2, y), cw / 2 - 0.15, ch - 0.6, fc="#e2f0e2", ec=GREEN, lw=0.9))
            ax.text(x + cw / 4 - 0.08, y + (ch - 0.6) / 2, "K", ha="center", va="center", fontsize=8.6, color=BLUE)
            ax.text(x + 3 * cw / 4 - 0.08, y + (ch - 0.6) / 2, "V", ha="center", va="center", fontsize=8.6, color=GREEN)
            if h == 0:
                ax.text(x + (cw - 0.3) / 2, y + ch - 0.3, f"token {n}" if n < Bn - 1 else "…", ha="center", va="bottom",
                        fontsize=8.0, color="#555555")
    for h in range(H):
        ax.text(bx - 1.2, by + (H - 1 - h) * ch + (ch - 0.6) / 2, f"KV 头 {h}" if h < H - 1 else "…", ha="right", va="center",
                fontsize=8.6, color="#555555")
    ax.text(bx + Bn * cw / 2, 20.0, "一个块 = [KV 头数, 块内 token 数, K 的维度 + V 的维度]", ha="center", va="center",
            fontsize=9.6, color="#222222")
    ax.text(bx + Bn * cw / 2, 13.6, "块内偏移 = (头号 × B + 块内位置) × (K 维 + V 维) × 每数占的字节数", ha="center",
            va="center", fontsize=9.0, color="#555555")
    ax.text(bx + Bn * cw / 2, 8.4, "K 和 V 拼在最后一维;视图只是步长,不复制数据", ha="center", va="center", fontsize=9.0,
            color="#555555")
    fig.savefig(f"{OUT}/fig2_layout.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig3_hash():
    fig, ax = plt.subplots(figsize=(13.8, 5.4))
    ax.set_xlim(0, 138); ax.set_ylim(4.6, 58); ax.axis("off")
    ax.text(69, 55.6, "前缀缓存:每个块的哈希,由上一个块的哈希、本块的 token、额外键算出来(B = 4,示意)",
            ha="center", va="center", fontsize=13, fontweight="bold")
    cw, ch, tx = 3.4, 4.0, 18.0
    def row(y, label, toks, colors, hashes, hcolors, htags, seed_y=None):
        ax.text(2.5, y + ch / 2, label, ha="left", va="center", fontsize=10.0, color="#222222")
        for i, (t, c) in enumerate(zip(toks, colors)):
            x = tx + i * cw
            ax.add_patch(Rectangle((x, y), cw - 0.15, ch, fc=c, ec="white", lw=0.9))
            ax.text(x + (cw - 0.15) / 2, y + ch / 2, t, ha="center", va="center", fontsize=7.8, color="#333333")
        hy = y + ch + 5.0
        prev = None
        for i, (h, c, tag) in enumerate(zip(hashes, hcolors, htags)):
            cx = tx + i * 4 * cw + 2 * cw - 0.075
            box(ax, cx - 3.8, hy, 7.6, 3.2, h, c, c, fs=9.2, color="#111111", pad=0.3)
            ax.text(cx, hy + 3.7, tag, ha="center", va="bottom", fontsize=8.4, color=c, fontweight="bold")
            arrow(ax, cx, y + ch + 0.2, cx, hy - 0.3, GRAY, lw=1.0)
            if prev is not None:
                arrow(ax, prev + 3.9, hy + 1.6, cx - 3.9, hy + 1.6, GRAY, lw=1.2)
            prev = cx
        return hy
    ta = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    tb = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "x", "y", "z", "w"]
    hyA = row(36.0, "请求 A", ta, [SHARED] * 12 + [A_ONLY] * 4, ["h0", "h1", "h2", "h3"], [GREEN, GREEN, GOLD, GOLD],
              ["命中", "命中", "新写入", "新写入"])
    hyB = row(6.0, "请求 B", tb, [SHARED] * 12 + [B_ONLY] * 4, ["h0", "h1", "h2'", "h3'"], [GREEN, GREEN, RED, RED],
              ["命中", "命中", "不同:未命中", "不同:未命中"])
    sy0, sy1 = hyA + 1.6, hyB + 1.6
    box(ax, 2.5, (sy0 + sy1) / 2 - 1.7, 10.8, 3.4, "空哈希(种子)", "#f2f2f2", GRAY, fs=8.6, color="#333333", pad=0.3)
    arrow(ax, 13.5, (sy0 + sy1) / 2 + 0.4, tx - 3.9 - 0.2, sy0, GRAY, lw=1.1, rad=-0.25)
    arrow(ax, 13.5, (sy0 + sy1) / 2 - 0.4, tx - 3.9 - 0.2, sy1, GRAY, lw=1.1, rad=0.25)
    ax.text(76.0, hyA + 6.4, r"$h_i$ = H($h_{i-1}$, 第 i 块的 token, 额外键)", ha="left", va="center", fontsize=11.5,
            color="#111111", fontweight="bold")
    ax.text(76.0, hyA + 1.4, "额外键:LoRA 编号、图片哈希、\n请求的盐(只混进第一块)", ha="left", va="center", fontsize=9.0,
            color="#555555", linespacing=1.5)
    ax.text(76.0, hyA - 4.6, "只缓存填满的块;\n每个哈希都覆盖从头到这个块的整段前缀", ha="left", va="center", fontsize=9.0,
            color="#555555", linespacing=1.5)
    ax.text(76.0, hyB - 1.0, "链的首没有父块,用种子代替;\n最后一块没填满,不算哈希、不缓存", ha="left", va="center", fontsize=9.0,
            color="#555555", linespacing=1.5)
    fig.savefig(f"{OUT}/fig3_hash.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig4_pool():
    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    ax.set_xlim(0, 138); ax.set_ylim(0, 70); ax.axis("off")
    ax.text(69, 67.4, "块池:引用计数、空闲队列和淘汰(接着第一张图的例子,只画出一部分块)", ha="center", va="center",
            fontsize=13, fontweight="bold")
    # in-use blocks
    ax.text(3, 60.6, "正在使用(引用计数 > 0),不在空闲队列里", ha="left", va="center", fontsize=10.2, fontweight="bold",
            color="#333333")
    cards = [("块 12", "引用计数 2\n哈希 h0", SHARED, "请求 A 和 B 共用\n都是命中拿到的"),
             ("块 5", "引用计数 2\n哈希 h1", SHARED, "请求 A 和 B 共用"),
             ("块 30", "引用计数 1\n哈希 h2", A_ONLY, "只有 A 在用\n登记过,别的请求能来共用"),
             ("块 3", "引用计数 1\n哈希:无", A_ONLY, "只有 A 在用\n最后一块还没填满,不缓存")]
    for i, (t1, t2, c, note) in enumerate(cards):
        x = 4 + i * 33.0
        box(ax, x, 47.0, 11.5, 9.0, t1, c, c, fs=10.0, fw="bold", color="#111111", pad=0.4)
        box(ax, x + 12.3, 47.0, 15.5, 9.0, t2, "white", c, fs=8.6, color="#222222", pad=0.3)
        ax.text(x, 44.6, note, ha="left", va="top", fontsize=8.4, color="#555555", linespacing=1.5)
    # free queue
    ax.text(3, 34.2, "空闲队列(引用计数 = 0),用块自己身上的前后指针串成双向链表", ha="left", va="center", fontsize=10.2,
            fontweight="bold", color="#333333")
    ax.text(3, 29.2, "分配新块从这头取", ha="left", va="center", fontsize=9.2, color=RED, fontweight="bold")
    arrow(ax, 15.5, 22.4, 21.0, 22.4, RED, lw=1.8)
    q = [("块 9", "无哈希", FREE, GRAY), ("块 8", "无哈希", FREE, GRAY), ("块 21", "有哈希", "#e8d9f0", PURP),
         ("块 19", "有哈希", "#e8d9f0", PURP), ("块 20", "有哈希", "#e8d9f0", PURP), ("块 6", "有哈希", "#e8d9f0", PURP)]
    for i, (t1, t2, fc, ec) in enumerate(q):
        x = 22 + i * 16.0
        box(ax, x, 19.0, 11.0, 6.8, f"{t1}\n{t2}", fc, ec, fs=9.2, color="#111111", pad=0.35)
        if i < len(q) - 1:
            arrow(ax, x + 11.2, 22.4, x + 15.8, 22.4, GRAY, lw=1.3, style="<|-|>")
    ax.text(131.5, 29.2, "释放的缓存块接到这尾", ha="right", va="center", fontsize=9.2, color=PURP, fontweight="bold")
    arrow(ax, 131.5, 26.6, 126.0, 25.9, PURP, lw=1.5)
    ax.annotate("", xy=(22 + 2 * 16.0 - 0.8, 17.2), xytext=(21.6, 17.2), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
    ax.plot([22, 22 + 2 * 16.0 - 0.6], [17.9, 17.9], color=GRAY, lw=0.8)
    ax.text(22 + 16.0 - 0.3, 12.6, "没缓存的空闲块\n后释放的排前(后进先出)", ha="center", va="top", fontsize=8.8, color=GRAY,
            linespacing=1.5)
    ax.plot([22 + 2 * 16.0, 22 + 6 * 16.0 - 1.0], [17.9, 17.9], color=PURP, lw=0.8)
    ax.text(22 + 4 * 16.0 - 0.5, 12.6, "缓存着的空闲块\n先释放的排前(最久未用),下次分配先淘汰它", ha="center", va="top",
            fontsize=8.8, color=PURP, linespacing=1.5)
    ax.text(3, 3.4, "命中的块会被“摸”一下:引用计数加一,并从空闲队列里摘出来。取到一个带哈希的块,就等于把它从哈希表里删掉,别的请求再也命中不了。",
            ha="left", va="center", fontsize=9.2, color="#222222")
    fig.savefig(f"{OUT}/fig4_pool.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig5_alloc():
    fig, ax = plt.subplots(figsize=(13.8, 9.6))
    ax.set_xlim(0, 138); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(69, 97.4, "每一步的分配:allocate_slots 做了什么", ha="center", va="center", fontsize=13, fontweight="bold")
    steps = [
        ("① 算需要多少新块", "要的,减去滑出窗口的、已经有的、刚命中的", BLUE, "#dbeaf7"),
        ("② 先还债", "滑出窗口的块还回块池,块表里换成占位块", BLUE, "#dbeaf7"),
        ("③ 够不够", "需要的块数,和空闲块数比(水位打开时再多留一些)", GOLD, "#fbf1d6"),
        ("④ 接管命中的块", "摸一下:引用计数加一,从空闲队列里摘出来", GREEN, "#e2f0e2"),
        ("⑤ 取新块", "从空闲队列头取;取到带哈希的块,就是淘汰了它", GREEN, "#e2f0e2"),
        ("⑥ 写缓存", "把刚填满的块登记进哈希表,只登记已确定的 token", GREEN, "#e2f0e2"),
        ("⑦ 发给工作进程", "新块号随调度结果下发,追加到这个请求的块表上", PURP, "#efe4f0"),
    ]
    y = 84.0
    for i, (t, d, ec, fc) in enumerate(steps):
        box(ax, 4, y, 50, 8.6, "", fc, ec, pad=0.4)
        ax.text(29, y + 5.8, t, ha="center", va="center", fontsize=10.4, fontweight="bold", color=ec)
        ax.text(29, y + 2.6, d, ha="center", va="center", fontsize=8.6, color="#333333")
        if i < len(steps) - 1:
            arrow(ax, 29, y - 0.4, 29, y - 5.0, GRAY, lw=1.4)
        y -= 13.6
    # right column: segments strip
    ax.text(66, 92.4, "一个请求的 token,分成几段", ha="left", va="center", fontsize=9.8, fontweight="bold", color="#333333")
    segs = [("已算过\ncomp", "#f2f2f2", GRAY), ("刚命中\nnew_comp", SHARED, BLUE), ("外部拿来的\next_comp", "#e8d9f0", PURP),
            ("这一步要算\nnew", "#fdeee2", ORANGE), ("推测解码\nlookahead", "white", ORANGE)]
    x = 66.0
    for t, fc, ec in segs:
        box(ax, x, 82.0, 12.2, 7.0, t, fc, ec, fs=8.4, color="#222222", pad=0.2)
        x += 12.8
    ax.text(66, 79.2, "前缀缓存的段,引用计数已经加过;外部拿来的段,还没加", ha="left", va="center", fontsize=8.4, color="#555555")
    # the None branch
    y3 = 84.0 - 2 * 13.6 + 4.3
    arrow(ax, 54.3, y3, 65.8, y3, RED, lw=1.7)
    ax.text(60, y3 + 1.0, "不够", ha="center", va="bottom", fontsize=9.4, color=RED, fontweight="bold")
    box(ax, 66.0, y3 - 8.0, 68.0, 16.0, "", "#fbe9e9", RED, pad=0.4)
    ax.text(100, y3 + 4.4, "返回 None", ha="center", va="center", fontsize=10.4, fontweight="bold", color=RED)
    ax.text(100, y3 - 2.4, "调度器让这个请求等待,或者抢占别的请求:\n被抢的请求,块全部交回,进度清零,回头重算", ha="center", va="center",
            fontsize=8.8, color=RED, linespacing=1.6)
    # notes
    box(ax, 66.0, 6.0, 68.0, 34.0, "", "white", GRAY, pad=0.4, ls="--")
    ax.text(100, 36.0, "几个数怎么来的", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#333333")
    ax.text(70.0, 28.4, "水位", ha="left", va="center", fontsize=9.0, fontweight="bold", color="#333333")
    ax.text(70.0, 24.2, "按总块数的一个比例留余量,只对等待中、被抢的请求生效;\n默认是 0,也就是关着", ha="left", va="center", fontsize=8.6,
            color="#444444", linespacing=1.6)
    ax.text(70.0, 17.4, "已确定的 token", ha="left", va="center", fontsize=9.0, fontweight="bold", color="#333333")
    ax.text(70.0, 12.6, "写缓存时最多写到它:草稿模型猜出来的 token\n可能被拒绝,不算数,不登记", ha="left", va="center", fontsize=8.6,
            color="#444444", linespacing=1.6)
    fig.savefig(f"{OUT}/fig5_alloc.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig6_hybrid():
    fig, ax = plt.subplots(figsize=(13.8, 6.9))
    ax.set_xlim(0, 138); ax.set_ylim(7.5, 76); ax.axis("off")
    ax.text(69, 73.4, "混合模型:两种层共用一个块池,页大小必须相等(块 = 16 个 token,窗口 = 32,示意)", ha="center",
            va="center", fontsize=13, fontweight="bold")
    x0, bw = 5.0, 9.8
    ax.text(x0 + 2 * bw, 69.0, "更早的 32 个 token:滑动窗口层已经不要了", ha="center", va="center", fontsize=8.6, color="#777777")
    ax.text(x0 + 3 * bw, 65.4, "最近的 32 个 token:滑动窗口层还要看", ha="center", va="center", fontsize=8.6, color=ORANGE)
    ax.text(x0, 62.6, "一个 64 个 token 的请求,4 个逻辑块", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#333333")
    for b in range(4):
        x = x0 + b * bw
        for t in range(16):
            xx = x + t * (bw - 0.4) / 16
            ax.add_patch(Rectangle((xx, 57.0), (bw - 0.4) / 16 - 0.05, 3.8, fc="#fdeee2" if b >= 2 else "#f4f4f4", ec="none"))
        ax.add_patch(Rectangle((x, 57.0), bw - 0.4, 3.8, fc="none", ec=ORANGE if b >= 2 else "#bbbbbb", lw=1.0))
        ax.text(x + (bw - 0.4) / 2, 55.8, f"逻辑块 {b}", ha="center", va="top", fontsize=8.4, color="#555555")
    # the two block tables, aligned under the logical blocks
    def btable(y, title, cells):
        ax.text(x0, y + 8.6, title, ha="left", va="center", fontsize=9.6, fontweight="bold", color="#333333")
        for i, (t, fc, ec) in enumerate(cells):
            box(ax, x0 + i * bw, y, bw - 0.4, 6.6, t, fc, ec, fs=8.6, color="#111111", pad=0.3)
    btable(38.0, "组 0:全注意力层 F0、F1 的块表,每个 token 都要看,4 块都要",
           [(f"{AR} 块 {i}", "#fdeee2", ORANGE) for i in range(4)])
    btable(19.0, "组 1:滑动窗口层 S0、S1 的块表,只留窗口里的,前两块还给了块池",
           [("空块", FREE, GRAY), ("空块", FREE, GRAY), (f"{AR} 块 4", "#e2f0e2", GREEN), (f"{AR} 块 5", "#e2f0e2", GREEN)])
    ax.text(60.0, 62.6, "物理显存:分成 2 个缓冲,每个被 2 层共用", ha="left", va="center", fontsize=9.8, fontweight="bold",
            color="#333333")
    for r, name in enumerate(["缓冲 0:  F0 的页 和 S0 的页", "缓冲 1:  F1 的页 和 S1 的页"]):
        y = 48.0 - r * 12.0
        ax.text(60.0, y + 3.2, name, ha="left", va="center", fontsize=9.0, color="#333333")
        for b in range(6):
            x = 100.0 + b * 6.2
            fc = ORANGE if b <= 3 else GREEN
            ax.add_patch(Rectangle((x, y), 5.7, 6.4, fc=fc, alpha=0.3, ec=fc, lw=1.0))
            ax.text(x + 2.85, y + 3.2, str(b), ha="center", va="center", fontsize=8.6, color="#222222")
        if r == 0:
            ax.text(100.0 + 2 * 6.2, y + 7.6, "组 0 用的块", ha="center", va="bottom", fontsize=8.2, color=ORANGE)
            ax.text(100.0 + 4.5 * 6.2, y + 7.6, "组 1 用的块", ha="center", va="bottom", fontsize=8.2, color=GREEN)
    ax.text(60.0, 28.0, "同一个块号,同一时间只属于一个组:\n组 0 用了块 0 到 3,组 1 只能用块 4、5", ha="left", va="center",
            fontsize=8.8, color="#333333", linespacing=1.6)
    ax.text(60.0, 19.6, "一个逻辑块 = 2 片(每组有 m 层,就是 m 片)", ha="left", va="center", fontsize=9.4, color="#111111",
            fontweight="bold")
    ax.text(x0, 11.2, "页大小相等是硬约束:组 0 的一页 = 2 层 × 16 个 token × 每层每个 token 的字节数;组 1 的一页字节数相同。",
            ha="left", va="center", fontsize=9.2, color="#222222")
    fig.savefig(f"{OUT}/fig6_hybrid.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig7_startup():
    fig, ax = plt.subplots(figsize=(13.8, 4.9))
    ax.set_xlim(0, 138); ax.set_ylim(0, 46); ax.axis("off")
    ax.text(69, 43.6, "启动时:KV 缓存能有多大,是怎么定的(示意,长度不代表真实比例)", ha="center", va="center", fontsize=13,
            fontweight="bold")
    def bar(y, name, segs, note):
        ax.text(3, y + 8.0, name, ha="left", va="center", fontsize=10.0, fontweight="bold", color="#333333")
        x = 3.0
        for t, w, fc, ec in segs:
            ax.add_patch(Rectangle((x, y), w, 6.2, fc=fc, ec=ec, lw=1.0))
            ax.text(x + w / 2, y + 3.1, t, ha="center", va="center", fontsize=8.6, color="#222222", linespacing=1.4)
            x += w
        ax.text(3, y - 2.0, note, ha="left", va="top", fontsize=8.8, color="#555555")
    bar(27.0, "以前(或关掉它):先估算,再分配",
        [("模型权重", 40, "#e4e4e4", GRAY), ("空跑一次\n测到的峰值", 18, "#dbeaf7", BLUE), ("KV 缓存\n(预算减去估算)", 50, "#fdeee2", ORANGE),
         ("余量\n(没把握,只能多留)", 22, "white", "#aaaaaa")],
        "估算和实际总有出入:CUDA Graph、预热算子都要再吃显存。余量留小了会 OOM,留大了 KV 缓存就小了")
    bar(7.0, "0.34 默认:可扩展 KV 缓存,预热之后按实测定大小",
        [("模型权重", 40, "#e4e4e4", GRAY), ("预热、录图之后\n的真实占用", 18, "#dbeaf7", BLUE),
         ("KV 缓存:地址空间先占好,物理显存按实测提交", 62, "#fdeee2", ORANGE), ("余量\n(按实测留)", 11, "white", "#aaaaaa")],
        "地址从头到尾不变,所以已经建好的视图和录好的 CUDA Graph 都不用动")
    fig.savefig(f"{OUT}/fig7_startup.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig8_naive():
    data = json.load(open(os.path.join(BENCH, "naive_vs_paged_4090d.json")))
    pts = data["points"]
    ns = [p["n"] for p in pts]
    N = 4096
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    for k, c, t in (("cat", ORANGE, "拼接(torch.cat)"), ("static", BLUE, "预分配,原地写"), ("paged", GREEN, "分页,按槽位散写")):
        ys = [p[f"{k}_us_median"] for p in pts]
        ax.plot(ns, ys, "-o", color=c, lw=2.2 if k == "cat" else 1.4, ms=5, label=t)
        for x, y in zip(ns, ys):
            if k == "cat" or x in (ns[0], ns[-1]):
                if k == "cat":  # 拼接的线往右上走,标签放到左上;第一个点在轴边上,放右下
                    dx, dy, ha = ((7, -13, "left") if x == ns[0] else ((7, 9, "left") if x == ns[-1] else (-9, 7, "right")))
                elif x == ns[0]:  # 第一个点:标签放到左边,免得和别的挤在一起
                    dx, dy, ha = -7, 5 if k == "static" else -12, "right"
                else:
                    dx, dy, ha = 7, 5 if k == "static" else -12, "left"
                ax.annotate(f"{y:,.0f}", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.6, color=c,
                            fontweight="bold" if k == "cat" else "normal", ha=ha)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns]); ax.minorticks_off()
    ax.set_ylim(35, 16000)
    ax.set_xlabel("当前长度 n(token,对数轴)")
    ax.set_ylabel("写一个 token 的 KV,每步耗时(微秒,对数轴)")
    ax.set_title("实测:拼接写法随长度线性变慢,另两种是平的", fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9.4)

    ax = axes[1]
    slope, intercept = np.polyfit(ns, [p["cat_us_median"] for p in pts], 1)
    cum = [("拼接", (intercept * N + slope * N * (N - 1) / 2) / 1e6, ORANGE),
           ("预分配,原地写", pts[-1]["static_us_median"] * N / 1e6, BLUE),
           ("分页,按槽位散写", pts[-1]["paged_us_median"] * N / 1e6, GREEN)]
    ys = np.arange(len(cum))[::-1]
    ax.barh(ys, [c[1] for c in cum], color=[c[2] for c in cum], height=0.55, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in cum], fontsize=9.6)
    ax.set_xlim(0.1, 60)
    for y, c in zip(ys, cum):
        ax.text(c[1] * 1.25, y, f"{c[1]:.2f} 秒", va="center", ha="left", fontsize=9.6, color=c[2], fontweight="bold")
    ax.set_xlabel("累计耗时(秒,对数轴)")
    ax.set_title(f"生成 {N:,} 个 token,光写 KV 的累计耗时", fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.05, f"RTX 4090 D,32 层,batch = 1,CUDA Graph 计时\n右图按左图实测的每步耗时累加", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.6, color="#666666", linespacing=1.6)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(f"{OUT}/fig8_naive.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig9_calls():
    fig, ax = plt.subplots(figsize=(12.0, 8.9))  # 比别的图窄:这张字多,窄一点字才看得清
    ax.set_xlim(0, 138); ax.set_ylim(-2.5, 101.4); ax.axis("off")
    ax.text(69, 99.9, "调用链:一个块号从调度到写进显存,要过哪些函数(v0.34.0)", ha="center", va="center", fontsize=13, fontweight="bold")
    XA, XB, XC, W = 1.5, 50.5, 99.5, 36.0
    H, PITCH, Y0 = 9.4, 11.4, 80.0

    def yb(i):
        return Y0 - PITCH * i

    for x, title, c, fc in ((XA, "① 调度器(引擎核心进程,在 CPU 上)", ORANGE, "#fdf6f0"),
                            (XB, "② KV 缓存管理(和调度器同一个进程)", GREEN, "#f2f8f2"),
                            (XC, "③ GPU 工作进程", BLUE, "#f0f6fb")):
        ax.add_patch(FancyBboxPatch((x - 1.4, -1.6), W + 2.8, yb(0) + H + 2.6, boxstyle="round,pad=0.02,rounding_size=0.8",
                                    fc=fc, ec=c, lw=1.0))
        ax.text(x + W / 2, 96.5, title, ha="center", va="center", fontsize=10.8, fontweight="bold", color=c)

    def box(x, i, fn, where, what, c, fs=8.8):
        y = yb(i)
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.5", fc="white", ec=c, lw=1.2))
        if "\n" in fn:  # 名字太长的拆成两行:类名一行,方法一行
            top, bottom = fn.split("\n")
            ax.text(x + 1.3, y + H - 1.8, top, ha="left", va="center", fontsize=fs, fontweight="bold", color="#111111")
            ax.text(x + 1.3, y + H - 3.9, bottom, ha="left", va="center", fontsize=fs, fontweight="bold", color="#111111")
            ax.text(x + 1.3, y + H - 6.1, where, ha="left", va="center", fontsize=7.4, color=c)
            ax.text(x + 1.3, y + 1.5, what, ha="left", va="center", fontsize=8.2, color="#333333")
            return
        ax.text(x + 1.3, y + H - 2.2, fn, ha="left", va="center", fontsize=fs, fontweight="bold", color="#111111")
        ax.text(x + 1.3, y + H - 4.9, where, ha="left", va="center", fontsize=7.8, color=c)
        ax.text(x + 1.3, y + 2.0, what, ha="left", va="center", fontsize=8.2, color="#333333")

    def down(x, i):
        arrow(ax, x + 4.0, yb(i), x + 4.0, yb(i + 1) + H + 0.15, GRAY, lw=1.3)

    def note(x, i, lines, c):
        y = yb(i + 2) - 0.2
        ax.add_patch(FancyBboxPatch((x, y), W, 2 * PITCH + H - 1.6, boxstyle="round,pad=0.02,rounding_size=0.5", fc="white",
                                    ec=c, lw=1.0, ls="--"))
        for k, t in enumerate(lines):
            ax.text(x + 1.3, y + (2 * PITCH + H - 1.6) - 2.2 - 2.9 * k, t, ha="left", va="center", fontsize=8.0,
                    color="#444444", fontweight="bold" if k == 0 else "normal")

    # column 1: scheduler
    box(XA, 0, "EngineCore.step()", "vllm/v1/engine/core.py:713", "一步三段:调度,执行,更新", ORANGE)
    box(XA, 1, "Scheduler.schedule()", "vllm/v1/core/sched/scheduler.py:716", "定这一步每个请求算几个 token", ORANGE)
    box(XA, 2, "get_computed_blocks()", "scheduler.py:691", "等待中的请求,先查前缀命中", ORANGE)
    box(XA, 3, "allocate_slots()", "scheduler.py:930, 1497", "位置不够就等待,或者抢占", ORANGE)
    for i in range(3):
        down(XA, i)
    note(XA, 5, ["每步算完、请求结束的时候:", "cache_blocks(2468):登记填满的块", "free(3229):倒序还块"], ORANGE)
    # column 2: KV cache management
    box(XB, 3, "KVCacheManager.allocate_slots()", "vllm/v1/core/kv_cache_manager.py:401", "先还滑出窗口的,再比空闲块数", GREEN)
    box(XB, 4, "KVCacheCoordinator\n.allocate_new_blocks()", "kv_cache_coordinator.py:315", "转给每一种层自己的管理器", GREEN)
    box(XB, 5, "SingleTypeKVCacheManager\n.allocate_new_blocks()", "single_type_kv_cache_manager.py:375", "需要的块数,减去已有的块数", GREEN)
    box(XB, 6, "BlockPool.get_new_blocks()", "block_pool.py:681", "带哈希的块,取到就是淘汰", GREEN)
    box(XB, 7, "FreeKVCacheBlockQueue.popleft_n()", "kv_cache_utils.py:343", "摘下空闲队列头的 n 个块", GREEN)
    for i in range(3, 7):
        down(XB, i)
    # column 3: worker
    box(XC, 1, "update_requests()", "worker/gpu/model_runner.py:1340", "从调度结果里取出新块号", BLUE)
    box(XC, 2, "BlockTables.append_block_ids()", "vllm/v1/worker/gpu/block_table.py:125", "接到块表这一行的后面,只追加", BLUE)
    box(XC, 3, "prepare_attn()", "model_runner.py:1672", "收拢块表,给每个 token 算槽位", BLUE)
    box(XC, 4, "reshape_and_cache_flash()", "attention/backends/flash_attn.py:1490", "每层算完 K、V,按槽位散写进缓存", BLUE)
    for i in range(1, 4):
        down(XC, i)
    note(XC, 6, ["读的时候", "注意力 kernel 拿着块表和每个请求的长度,", "一块一块地取"], BLUE)

    # across the columns
    arrow(ax, XA + W, yb(3) + 6.6, XB, yb(3) + 6.6, GREEN, lw=1.6)
    ax.text((XA + W + XB) / 2, yb(3) + 7.3, "调用", ha="center", va="bottom", fontsize=8.0, color=GREEN, fontweight="bold")
    arrow(ax, XB, yb(3) + 2.8, XA + W, yb(3) + 2.8, RED, lw=1.4, ls="--")
    ax.text((XA + W + XB) / 2 + 0.8, yb(3) - 0.2, "新块号或 None", ha="center", va="top", fontsize=6.8, color=RED)
    arrow(ax, XA + W, yb(1) + 6.6, XC, yb(1) + 6.6, BLUE, lw=1.6)
    ax.text((XA + W + XC) / 2, yb(1) + 7.3, "调度结果:新块号是整数,经共享内存发过去", ha="center", va="bottom", fontsize=8.6, color=BLUE)
    # 返回:从工作进程一列的顶上绕回去,走各列边框之间的空当
    xm, ytop = (XA + W + XB) / 2 + 0.4, yb(0) + H + 2.2
    ax.plot([XC + W / 2, XC + W / 2, xm, xm], [yb(1) + H, ytop, ytop, yb(0) + H / 2], color=GRAY, lw=1.3, ls="--")
    arrow(ax, xm, yb(0) + H / 2, XA + W, yb(0) + H / 2, GRAY, lw=1.3, ls="--")
    ax.text((xm + XC + W / 2) / 2, ytop + 0.6, "新 token 回到引擎,进下一步", ha="center", va="bottom", fontsize=8.6, color=GRAY)
    fig.savefig(f"{OUT}/fig9_calls.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_blocks(); fig2_layout(); fig3_hash(); fig4_pool(); fig5_alloc(); fig6_hybrid(); fig7_startup()
    fig8_naive(); fig9_calls()
    print("done ->", OUT)
