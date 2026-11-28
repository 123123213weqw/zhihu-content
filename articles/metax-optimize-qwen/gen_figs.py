#!/usr/bin/env python3
"""Figures for the article on optimising Qwen3.8-27B on a MetaX C500.

fig1_chain.png   one GDN layer split into three kinds of work, and which unit of the AP does each
fig2_kernel.png  how MetaX's GDN chunk kernel lays out its state, registers and shared memory on one AP
fig3_graph.png   CUDA Graph on a C500 as measured by a third party: draft step and verify step
fig4_hooks.png   which of vLLM's extension points to use to replace an op on vLLM-metax
fig5_tlv.png     two-level verification on a C500: two drafts, a quantised first pass, one exact pass, and what a step costs
fig6_line.png    when a change pays off: tokens per verification against step time, with the iso-throughput line
fig7_bridge.png  bridging the author's engine to vLLM's model: where the three products of prefill go

Numbers come from bench/calc_summary.json (written by calc.py).
Run:  python3 gen_figs.py  ->  ./images/
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)
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
BLUE, RED, GREEN, GRAY, ORANGE, GOLD = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#b8922e"
PURP, DARK, PLUM = "#995588", "#333333", "#5b2a86"
calc = json.load(open(os.path.join(HERE, "bench", "calc_summary.json")))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rbox(ax, x, y, w, h, fc, ec="#8a8a8a", lw=0.9, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, p0, p1, color=GRAY, lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# ---------------------------------------------------------------------------------------------- fig1
def fig1_chain():
    rows = [
        ("投影:输入乘上 Q,K,V,Z,β 的权重", "矩阵乘 (mcBLAS)", "张量单元"),
        ("门控:sigmoid, softplus, exp,chunk 内累加", "逐元素运算", "矢量单元"),
        (r"状态更新:$S \leftarrow S \times$ 衰减 $+\ K^{\top}W$", "一条矩阵指令,一次乘加", "张量单元,矢量单元"),
        (r"输出:$O = Q\,S$", "块乘 (矩阵指令)", "张量单元"),
        ("MLP:两个矩阵乘,中间夹一个 SiLU", "矩阵乘,加一个融合 kernel", "张量单元,矢量单元"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    cols = [(2, 36), (41, 24), (69, 29)]
    heads = ["模型里的活 (一层 GDN,外加 MLP)", "落到哪个算子", "AP 里谁来做"]
    for (x, w), head in zip(cols, heads):
        ax.text(x + w / 2, 92, head, ha="center", va="center", fontsize=8.8, fontweight="bold", color=DARK)
    h, gap = 11.6, 3.0
    tints = ["#f7e3d3", "#ead9f3", "#f7e3d3", "#f7e3d3", "#dcebf7"]
    for i, (a, b, c) in enumerate(rows):
        y = 82 - (i + 1) * (h + gap)
        rbox(ax, cols[0][0], y, cols[0][1], h, tints[i], ec=DARK)
        ax.text(cols[0][0] + cols[0][1] / 2, y + h / 2, a, ha="center", va="center", fontsize=8.0, color=DARK)
        rbox(ax, cols[1][0], y, cols[1][1], h, "#f1f1f1", ec="#9a9a9a")
        ax.text(cols[1][0] + cols[1][1] / 2, y + h / 2, b, ha="center", va="center", fontsize=7.8, color=DARK)
        rbox(ax, cols[2][0], y, cols[2][1], h, "#d9efd9", ec=GREEN)
        ax.text(cols[2][0] + cols[2][1] / 2, y + h / 2, c, ha="center", va="center", fontsize=7.8, color=DARK)
        arrow(ax, (cols[0][0] + cols[0][1] + 0.6, y + h / 2), (cols[1][0] - 0.6, y + h / 2))
        arrow(ax, (cols[1][0] + cols[1][1] + 0.6, y + h / 2), (cols[2][0] - 0.6, y + h / 2))
    ax.text(50, 3.2, "模型结构再复杂,到了硬件上只剩这几种指令;结构决定的是每种指令有多少,数据怎么搬。",
            ha="center", va="center", fontsize=7.8, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "fig1_chain.png")

# ---------------------------------------------------------------------------------------------- fig2
def fig2_kernel():
    k = calc["kernel"]
    fig = plt.figure(figsize=(7.2, 3.6))
    a = fig.add_axes((0.0, 0.0, 0.56, 0.97))
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.axis("off")
    a.text(50, 97.5, "沐曦的 GDN chunk kernel:一个线程块", ha="center", va="center", fontsize=9.6, fontweight="bold", color=DARK)
    rbox(a, 4, 79, 92, 10.5, "#ead9f3", ec=PURP)
    a.text(50, 84.2, f"寄存器里的状态:64 × 128 个 FP32,每个线程 {k['state_regs_per_thread']} 个\n"
                     f"{k['threads_per_block']} 个线程,{k['waves_per_block']} 条线程束",
          ha="center", va="center", fontsize=7.8, color=DARK, linespacing=1.4)
    steps = [(4, "读一个 chunk\nK 和 W 进共享内存\n(64 个 token)"), (37.5, "矩阵指令\n" + r"更新 $= K^{\top}W$"),
             (68, "乘加\n" + r"状态 $\times$ 衰减" + "\n" + r"$+$ 更新")]
    ws = [30, 27, 28]
    for (x, text), w in zip(steps, ws):
        rbox(a, x, 52, w, 20, "#f7e3d3", ec=ORANGE)
        a.text(x + w / 2, 62, text, ha="center", va="center", fontsize=7.6, color=DARK, linespacing=1.4)
    arrow(a, (34.6, 62), (36.9, 62))
    arrow(a, (65.1, 62), (67.4, 62))
    arrow(a, (82, 51.4), (19, 51.4), rad=-0.22, style="-|>", ls=(0, (4, 3)))
    a.text(50, 29.5, "下一个 chunk,直到这段序列走完", ha="center", va="center", fontsize=7.4, color=GRAY)
    a.text(50, 17.5, f"共享内存 {k['smem_total_bytes']:,} 字节,占 64 KB 的 {k['smem_share']}%", ha="center", va="center",
           fontsize=8.0, color=DARK, fontweight="bold")
    a.text(50, 9.5, "状态快照 16,896 B,残差面板 8,704 B\nW 面板 16,896 B,衰减面板 512 B", ha="center", va="center",
           fontsize=7.2, color=GRAY, linespacing=1.4)

    # right: two budgets
    b = fig.add_axes((0.67, 0.62, 0.30, 0.24))
    b.barh(0, k["state_regs_per_thread"], height=0.42, color=BLUE, label="状态")
    b.barh(0, k["update_regs_per_thread"], left=k["state_regs_per_thread"], height=0.42, color=ORANGE, label="更新的累加器")
    b.axvline(k["mtreg_limit_in_kernel_comment"], color=RED, lw=1.0, linestyle=(0, (5, 4)))
    b.text(k["mtreg_limit_in_kernel_comment"] - 3, 0.34, "128:沐曦自己定的上限", ha="right", va="bottom", fontsize=6.6, color=RED)
    b.text(k["state_regs_per_thread"] + k["update_regs_per_thread"] + 3, -0.36, "48 个", ha="left", va="center", fontsize=7.4, color=DARK)
    b.set_xlim(0, 150)
    b.set_ylim(-0.6, 1.05)
    b.set_yticks([])
    b.set_title("寄存器,每个线程", fontsize=8.2, pad=3)
    b.tick_params(labelsize=7)
    b.legend(fontsize=6.6, loc="upper left", handlelength=1.0, borderaxespad=0.1)
    clean(b)
    sm = k["smem_bytes"]
    c = fig.add_axes((0.67, 0.08, 0.30, 0.24))
    left = 0
    for name, col in (("state_bf16", BLUE), ("residual_panel", ORANGE), ("w_panel", GREEN), ("decay_panel", GOLD)):
        c.barh(0, sm[name] / 1024, left=left / 1024, height=0.42, color=col)
        left += sm[name]
    c.axvline(64, color=RED, lw=1.0, linestyle=(0, (5, 4)))
    c.text(62, 0.34, "64 KB:工作组共享内存", ha="right", va="bottom", fontsize=6.6, color=RED)
    c.text(k["smem_total_kb"] + 1.5, -0.36, f"{k['smem_total_kb']:.1f} KB", ha="left", va="center", fontsize=7.4, color=DARK)
    c.set_xlim(0, 78)
    c.set_ylim(-0.6, 1.05)
    c.set_yticks([])
    c.set_title("共享内存,每个线程块", fontsize=8.2, pad=3)
    c.tick_params(labelsize=7)
    clean(c)
    save(fig, "fig2_kernel.png")


# ---------------------------------------------------------------------------------------------- fig3
def fig3_graph():
    g = calc["graph"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    lo, hi = g["draft_after_ms"]
    a.bar(0, g["draft_before_ms"], width=0.5, color="#bbbbbb")
    a.bar(1, lo, width=0.5, color=BLUE)
    a.bar(1, hi - lo, bottom=lo, width=0.5, color=BLUE, alpha=0.35, hatch="////", ec=BLUE, lw=0.0)
    a.text(0, g["draft_before_ms"] + 0.6, f"{g['draft_before_ms']:.0f} ms", ha="center", va="bottom", fontsize=8.2, color=DARK)
    a.text(1, hi + 0.6, f"{lo:.1f} 到 {hi:.1f} ms", ha="center", va="bottom", fontsize=8.2, color=DARK)
    a.set_xticks([0, 1], ["无图\neager 派发", "分桶 CUDA Graph"], fontsize=8.2)
    a.set_ylim(0, 31)
    a.set_ylabel("Draft 单步耗时 (ms)", fontsize=8.4)
    a.set_title(f"快 {g['draft_speedup'][0]} 到 {g['draft_speedup'][1]} 倍", fontsize=9.2)
    e, c = g["verify_eager_custom_op_ms"], g["verify_graph_ms"]
    b.bar(0, e, width=0.5, color="#bbbbbb")
    b.bar(1, c, width=0.5, color=GREEN)
    b.text(0, e + 0.9, f"{e:.2f} ms", ha="center", va="bottom", fontsize=8.2, color=DARK)
    b.text(1, c + 0.9, f"{c:.2f} ms", ha="center", va="bottom", fontsize=8.2, color=DARK)
    b.set_xticks([0, 1], ["eager\n走 custom_op", "CUDA Graph"], fontsize=8.2)
    b.set_ylim(0, 57)
    b.set_ylabel("Verify 一步的耗时 (ms),M = 1", fontsize=8.4)
    b.set_title(f"省 {g['verify_saved_ms']:.1f} ms, {g['verify_saved_share']}%", fontsize=9.2)
    for ax in (a, b):
        clean(ax)
    fig.suptitle("CUDA Graph 在 C500 上的收益 (第三方实测,64 层的 Qwen3.8-27B)", fontsize=9.6, y=1.02)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_graph.png")


# ---------------------------------------------------------------------------------------------- fig4
def fig4_hooks():
    h = calc["hooks"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96.5, "要在 vLLM 里换掉一个算子,先找口子,补丁放最后", ha="center", va="center", fontsize=9.8, fontweight="bold", color=DARK)
    rbox(ax, 29, 85.5, 42, 7.5, "#f1f1f1", ec=DARK)
    ax.text(50, 89.2, "要把 vLLM 里的一个算子换成自己的", ha="center", va="center", fontsize=8.4, color=DARK)
    rbox(ax, 25, 70.5, 50, 9.5, "#dcebf7", ec=BLUE)
    ax.text(50, 75.2, "vLLM 把它登记成 CustomOp\n或者 PluggableLayer 了吗?", ha="center", va="center", fontsize=8.4, color=DARK, linespacing=1.4)
    arrow(ax, (50, 85.0), (50, 80.6))
    # yes: registry
    rbox(ax, 2, 53.5, 36, 8.5, "#d9efd9", ec=GREEN)
    ax.text(20, 57.7, "是:用 register_oot 登记自己的类", ha="center", va="center", fontsize=8.2, color=DARK, fontweight="bold")
    arrow(ax, (40, 70.0), (20, 62.6))
    ax.text(29, 68.6, "是", ha="center", va="center", fontsize=8.0, color=GREEN, fontweight="bold")
    rbox(ax, 2, 38.5, 36, 11.5, "#fbfbfb", ec=GREEN)
    ax.text(20, 44.2, "这个名字还空着:直接登记\n例:GDN 整层 QwenGatedDeltaNetAttention", ha="center", va="center", fontsize=7.3, color=DARK, linespacing=1.45)
    rbox(ax, 2, 20.0, 36, 14.0, "#fbfbfb", ec=GREEN)
    ax.text(20, 27.0, "沐曦已经登记过:同名不能再登记\n继承它的类,只改 forward_oot,\n再换掉表里那一项\n例:GDN prefill 的 ChunkGatedDeltaRule",
            ha="center", va="center", fontsize=7.3, color=DARK, linespacing=1.4)
    arrow(ax, (20, 53.0), (20, 50.5))
    arrow(ax, (20, 38.0), (20, 34.5))
    # no: patch
    rbox(ax, 52, 53.5, 46, 8.5, "#f7e3d3", ec=PLUM)
    ax.text(75, 57.7, "否:用 vllm_metax 的 @patch 补那一处的名字", ha="center", va="center", fontsize=8.2, color=DARK, fontweight="bold")
    arrow(ax, (60, 70.0), (75, 62.6))
    ax.text(70.5, 68.6, "否", ha="center", va="center", fontsize=8.0, color=PLUM, fontweight="bold")
    rbox(ax, 52, 20.0, 46, 25.0, "#fbfbfb", ec=PLUM)
    ax.text(75, 32.0, "目标必须已经存在,改名或挪位置会在 import 时报错\n同一个目标补两次会报错\n每个补丁写清受影响的版本和移除条件\n例:GDN decode 的 fused_sigmoid_gating_delta_rule_update",
            ha="center", va="center", fontsize=7.3, color=DARK, linespacing=1.6)
    arrow(ax, (75, 53.0), (75, 45.6))
    ax.text(50, 12.5, f"vLLM-metax {h['vllm_metax_version']} 自己的用法:注册表里 {h['registered_oot_ops']} 个算子,"
                      f"补丁层写了 {h['patches_total']} 处 @patch,启用 {h['patches_active']} 处,碰了 {h['patched_modules']} 个上游模块", ha="center", va="center", fontsize=7.6, color=GRAY)
    ax.text(50, 5.5, "全注意力的层由平台选注意力后端,不在这张图里。", ha="center", va="center", fontsize=7.6, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "fig4_hooks.png")


# ---------------------------------------------------------------------------------------------- fig5
def fig5_tlv():
    t = calc["tlv"]
    st = t["stage_ms"]
    scan = {r["mode"]: r for r in t["width_scan"]}
    # 示意图: token 是编的。节点数照实测的配置: 树 1 连锚点 16 个, 精确趟 C 加树 2 共 16 个
    #         id: (token, 父节点, 在不在真实的路上)
    nodes = {
        "A": (" 42", None, True),                                                    # 锚点: 上一步给出的 token, 树的 0 号节点
        "B1": (" because", "A", True), "B2": (", which", "A", False), "B3": (" and", "A", False),
        "C1": (" 6", "B1", True), "C2": (" it", "B1", False),
        "C3": (" is", "B2", False), "C4": (" then", "B3", False),
        "D1": (" times", "C1", True), "D2": (" plus", "C1", False), "D3": (" correct", "C3", False),
        "E1": (" six", "D1", False), "E2": (" equals", "D1", False), "E3": (".", "D3", False),
        "F1": (" ==", "E1", False), "F2": (" 42", "E2", False),
        "G1": (" 7", "D1", True), "G2": (" so", "D1", False), "G3": (" via", "D1", False),
        "H1": (" which", "G1", True), "H2": (", giving", "G1", False), "H3": (" 42", "G2", False),
        "I1": (" is", "H1", True), "I2": (" equals", "H1", False), "I3": (" then", "H2", False),
        "J1": (" correct", "I1", True), "J2": (" right", "I1", False),
        "K1": (".", "J1", True),
        "X": ("\\n", "K1", True),                                                    # 目标模型白给的, 不是草稿写的
    }
    chain = ["A", "B1", "C1", "D1"]                                                 # 第一趟 (8 位) 认可的链 C
    accepted = chain + ["G1", "H1", "I1", "J1", "K1"]                               # 精确趟通过的
    kids = {}
    for k, (_, p, _) in nodes.items():
        if p is not None:
            kids.setdefault(p, []).append(k)
    for p in kids:                                                                  # 真的那条路排前面, 画在最上面
        kids[p].sort(key=lambda k: not nodes[k][2])
    depth = {}

    def set_depth(k, d):
        depth[k] = d
        for c in kids.get(k, []):
            set_depth(c, d + 1)
    set_depth("A", 0)
    order = []

    def dfs(k):
        if k in kids:
            for c in kids[k]:
                dfs(c)
        else:
            order.append(k)
    dfs("A")

    fig = plt.figure(figsize=(7.2, 4.2))
    a = fig.add_axes((0.0, 0.0, 0.70, 1.0))
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.axis("off")
    X0, DX, NW, NH, slot = 14.0, 8.6, 8.0, 4.8, 6.9
    xk = lambda k: X0 + depth[k] * DX
    y = {k: 91.0 - i * slot - slot / 2 for i, k in enumerate(order)}
    for d in range(8, -1, -1):
        for k in nodes:
            if k in kids and depth[k] == d:
                y[k] = (y[kids[k][0]] + y[kids[k][-1]]) / 2

    def line(p0, p1, color, lw, ls="-", z=1):
        a.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-", color=color, lw=lw, linestyle=ls,
                                    shrinkA=0, shrinkB=0, zorder=z))
    for p in nodes:
        for c in kids.get(p, []):
            if c == "X":
                line((xk(p) + NW, y[p]), (xk(c), y[c]), BLUE, 1.3, (0, (4, 2)), 1)
            elif p in accepted and c in accepted:
                if p in chain and c in chain:                                        # 橙色虚线露在绿线两边: 这一段是链 C
                    line((xk(p) + NW, y[p]), (xk(c), y[c]), ORANGE, 3.4, (0, (3.2, 2.2)), 1)
                line((xk(p) + NW, y[p]), (xk(c), y[c]), GREEN, 1.3, "-", 2)
            else:
                line((xk(p) + NW, y[p]), (xk(c), y[c]), "#d0d0d0", 0.8, "-", 1)
    for k, (tok, _, _) in nodes.items():
        x, yy = xk(k), y[k]
        if k == "X":
            fc, ec, tc, lw = BLUE, BLUE, "white", 0.9
        elif k == "A":
            fc, ec, tc, lw = "#ead9f3", PURP, DARK, 1.2
        elif k in accepted:
            fc, ec, tc, lw = GREEN, GREEN, "white", 0.9
        else:
            fc, ec, tc, lw = "#f3f3f3", "#d5d5d5", "#a3a3a3", 0.8
        if k in chain[1:]:                                                           # 链 C 的虚线框, 垫在节点下面
            a.add_patch(FancyBboxPatch((x - 0.8, yy - NH / 2 - 0.8), NW + 1.6, NH + 1.6,
                                       boxstyle="round,pad=0.02,rounding_size=1.2", fill=False,
                                       ec=ORANGE, lw=1.2, linestyle=(0, (3.2, 2.0)), zorder=1.5))
        rbox(a, x, yy - NH / 2, NW, NH, fc, ec=ec, lw=lw, r=1.0)
        a.text(x + NW / 2, yy, tok, ha="center", va="center", fontsize=6.0, color=tc,
               fontweight="bold" if k in accepted else "normal", zorder=3)
    rbox(a, 4.0, y["A"] - NH / 2, 7.0, NH, "#e6e6e6", ec="#8a8a8a", r=0.8)
    a.text(7.5, y["A"], "…", ha="center", va="center", fontsize=7.0, color=DARK)
    line((11.3, y["A"]), (X0 - 0.6, y["A"]), "#9a9a9a", 0.9, "-", 1)
    a.text(X0 - 1.2, y["A"] + NH / 2 + 1.0, "锚点,上一步给的", ha="right", va="bottom", fontsize=5.8, color=PURP)
    a.text(X0, 97.8, "树 1:锚点加 15 个草稿节点,共 16 个", ha="left", va="center", fontsize=7.4, color=DARK, fontweight="bold")
    a.text(99.5, 97.8, "树 2:从 C 的末尾接着写,12 个", ha="right", va="center", fontsize=7.4, color=DARK, fontweight="bold")
    a.text(99.5, 93.3, "精确趟:C 加树 2,共 16 个节点,一趟验完", ha="right", va="center", fontsize=6.6, color=GREEN)
    a.text(14.5, 80.0, "链 C:第一趟认可的 4 个,\n只接受,不提交", ha="left", va="center", fontsize=6.6,
           color="#b5561f", linespacing=1.5)
    arrow(a, (27.5, 77.4), (xk("D1") + 1.0, y["D1"] + NH / 2 + 1.0), color="#b5561f", lw=0.9)
    a.text(xk("X") + NW / 2, y["X"] - NH / 2 - 1.3, "目标模型白给的", ha="center", va="top", fontsize=5.6, color=BLUE)
    lx = X0
    for kind, text in (("chain", "第一趟认可的链 C"), ("ok", "精确趟通过,提交"), ("bonus", "目标模型白给的"), ("rej", "没被接受")):
        if kind == "chain":
            a.add_patch(FancyBboxPatch((lx, 2.6), 3.2, 2.6, boxstyle="round,pad=0.02,rounding_size=0.6", fill=False,
                                       ec=ORANGE, lw=1.2, linestyle=(0, (3.2, 2.0))))
        else:
            fc, ec = {"ok": (GREEN, GREEN), "bonus": (BLUE, BLUE), "rej": ("#f3f3f3", "#c9c9c9")}[kind]
            rbox(a, lx, 2.6, 3.2, 2.6, fc, ec=ec, lw=0.9, r=0.5)
        a.text(lx + 4.2, 3.9, text, ha="left", va="center", fontsize=5.9, color=DARK)
        lx += 20.0

    b = fig.add_axes((0.755, 0.115, 0.235, 0.70))
    names = ["草稿 1", "第一趟,8 位\n只接受不提交", "草稿 2", "精确验证\n说了算", "提交"]
    vals = [st["draft_block_n16"], st["verify_q_n16_q8view"], st["draft_block_n16"], st["verify_exact_n16"], st["commit_replay_l6"]]
    cols = ["#bbbbbb", ORANGE, "#bbbbbb", GREEN, BLUE]
    ypos = [4, 3, 2, 1, 0]
    b.barh(ypos, vals, height=0.62, color=cols)
    for yp, v in zip(ypos, vals):
        b.text(v + 1.0, yp, f"{v:.1f}", va="center", fontsize=7.2, color=DARK)
    b.set_yticks(ypos, names, fontsize=6.3)
    b.set_xlim(0, 64)
    b.set_ylim(-0.6, 4.6)
    b.set_xlabel("毫秒", fontsize=7)
    b.tick_params(axis="x", labelsize=6.5)
    clean(b)
    fig.text(0.885, 0.945, "一步的账", ha="center", va="center", fontsize=9.2, fontweight="bold", color=DARK)
    fig.text(0.885, 0.905, f"各阶段相加 {t['step_stage_sum_ms']['tlv_16_16']:.1f} ms,\n每次拿 {scan['tlv_16_16']['tau']:.2f} 个 token",
             ha="center", va="top", fontsize=7.0, color=DARK, linespacing=1.5)
    save(fig, "fig5_tlv.png")


# ---------------------------------------------------------------------------------------------- fig6
def fig6_line():
    t = calc["tlv"]
    S = {r["mode"]: r for r in t["width_scan"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    xs = [60, 160]
    s16 = S["tree_16"]["geomean_tok_s"]
    ax.plot(xs, [s16 * x / 1e3 for x in xs], ls=(0, (5, 4)), lw=1.1, color="#7a8fa6", zorder=1)
    tree = [m for m in ("tree_16", "tree_24", "tree_32")]
    tlv = [m for m in ("tlv_16_16", "tlv_16_24", "tlv_16_32")]
    ax.scatter([S[m]["step_ms"] for m in tree], [S[m]["tau"] for m in tree], s=46, color=BLUE, zorder=3, label="单棵树")
    ax.scatter([S[m]["step_ms"] for m in tlv], [S[m]["tau"] for m in tlv], s=46, color=GREEN, marker="s", zorder=3, label="两级验证")
    r = S["tree_16"]
    ax.annotate(f"树,16 个节点\n{r['tau']:.2f} 个,一步 {r['step_ms']:.1f} ms", (r["step_ms"], r["tau"]), (r["step_ms"] + 2.0, r["tau"] - 0.45),
                fontsize=7.2, color=DARK, ha="left", va="top", linespacing=1.5)
    ax.annotate("24 个节点", (S["tree_24"]["step_ms"], S["tree_24"]["tau"]), (S["tree_24"]["step_ms"] - 1.0, S["tree_24"]["tau"] + 0.18),
                fontsize=6.8, color=DARK, ha="right", va="bottom")
    ax.annotate("32 个节点", (S["tree_32"]["step_ms"], S["tree_32"]["tau"]), (S["tree_32"]["step_ms"] + 1.2, S["tree_32"]["tau"] + 0.02),
                fontsize=6.8, color=DARK, ha="left", va="bottom")
    r = S["tlv_16_16"]
    ax.annotate(f"两级验证,16 + 16\n{r['tau']:.2f} 个,一步 {r['step_ms']:.1f} ms", (r["step_ms"], r["tau"]), (r["step_ms"] - 1.5, r["tau"] + 0.22),
                fontsize=7.2, color=GREEN, ha="right", va="bottom", linespacing=1.5, fontweight="bold")
    r = S["tlv_16_24"]
    ax.annotate(f"16 + 24 和 16 + 32\n一步 {r['step_ms']:.0f} ms 上下", (r["step_ms"], r["tau"]), (r["step_ms"] + 1.5, r["tau"] - 0.3),
                fontsize=6.8, color=DARK, ha="left", va="top", linespacing=1.5)
    a0, a1 = S["tree_16"], S["tree_32"]
    ax.annotate("", (a1["step_ms"] - 1.6, a1["tau"] - 0.02), (a0["step_ms"] + 1.4, a0["tau"] + 0.04),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1, mutation_scale=10))
    ax.text((a0["step_ms"] + a1["step_ms"]) / 2, (a0["tau"] + a1["tau"]) / 2 - 0.2, "加宽:token 涨 7%,耗时涨 36%", ha="center", va="top", fontsize=6.8, color=GRAY)
    b0, b1 = S["tree_16"], S["tlv_16_16"]
    ax.annotate("", (b1["step_ms"] - 1.5, b1["tau"] - 0.22), (b0["step_ms"] + 1.4, b0["tau"] + 0.1),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4, mutation_scale=11))
    ax.text(101.0, 9.35, "两级验证:token 涨 85%,耗时涨 77%\n越过了线", ha="center", va="center", fontsize=6.9, color=GREEN, linespacing=1.5)
    ax.set_xlim(60, 160)
    ax.set_ylim(3.9, 13.1)
    ax.set_xlabel("一步耗时 (ms)", fontsize=8.2)
    ax.set_ylabel("每次验证拿到的 token", fontsize=8.2)
    ax.tick_params(labelsize=7.2)
    ax.set_title("划不划算看一条线:点要落到等吞吐线的上面才更快", fontsize=9.4, pad=8)
    ax.legend(fontsize=7.2, loc="upper left", handlelength=1.0, borderaxespad=0.2)
    clean(ax)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.17)
    fig.canvas.draw()                                                               # 等吞吐线的标注顺着线写, 角度按显示坐标算
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((100, s16 * 0.1))
    ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    ax.text(159.0, s16 * 0.159 + 0.1, f"等吞吐线:{s16:.0f} tok/s,和单棵树一样快", rotation=ang, rotation_mode="anchor",
            ha="right", va="bottom", fontsize=7.0, color="#5f7488")
    fig.text(0.5, 0.012, "虚线是等吞吐线,线上的点都和单棵树一样快。6 个点取自同一份日志,5 道题各 2048 个 token;一步耗时是 TPOT 乘每次拿到的 token。",
             ha="center", fontsize=6.9, color=GRAY)
    save(fig, "fig6_line.png")


# ---------------------------------------------------------------------------------------------- fig7
def fig7_bridge():
    f = calc["tlv"]["feature"]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96.5, "接到 vLLM 上:prefill 跑完,三样产物各去哪", ha="center", va="center", fontsize=9.6, fontweight="bold", color=DARK)
    rbox(ax, 1, 43.5, 13, 11, "#f1f1f1", ec=DARK)
    ax.text(7.5, 49, "prompt 的\ntoken", ha="center", va="center", fontsize=7.4, color=DARK, linespacing=1.4)
    arrow(ax, (14.6, 49), (18.4, 49))
    rbox(ax, 19, 12, 38, 76, "#fbfbfb", ec=DARK)
    ax.text(38, 83.5, "vLLM 的 64 层循环", ha="center", va="center", fontsize=8.4, color=DARK, fontweight="bold")
    ax.text(38, 78.8, "每个子模块的 forward 换成作者的 kernel", ha="center", va="center", fontsize=6.8, color=GRAY)
    y0, hh = 20.0, 0.84
    for i in range(64):
        ax.add_patch(Rectangle((21.5, y0 + (63 - i) * hh), 2.6, hh * 0.92,
                               fc="#ead9f3" if i % 4 != 3 else "#dcebf7", ec="none"))
    for i in f["layers_one_based"]:                                                  # 取特征的层, 从 1 数
        yy = y0 + (64 - i) * hh
        ax.add_patch(Rectangle((21.3, yy - 0.06), 3.0, hh * 0.92 + 0.12, fill=False, ec=ORANGE, lw=1.2))
        ax.text(25.2, yy + hh * 0.46, str(i), va="center", ha="left", fontsize=5.2, color="#b5561f")
    ax.text(22.8, y0 + 64 * hh + 1.0, "层", ha="center", va="bottom", fontsize=5.6, color=GRAY)
    rows = [("RMSNorm", "rmsnorm,fused_add_rmsnorm"), ("MLP 的激活", "silu_and_mul"),
            ("GDN,48 层", "prefill 走 gdn_prefill"), ("注意力,16 层", "prefill 走因果 SDPA")]
    for j, (u, v) in enumerate(rows):
        yy = 67.5 - j * 9.0
        ax.text(30.5, yy + 1.6, u, fontsize=6.7, color=DARK, fontweight="bold")
        ax.text(30.5, yy - 2.0, v, fontsize=6.7, color=GRAY)
    ax.text(30.5, 25.0, "线性层不换,照常走 vLLM\n加载的那份 W12 权重", fontsize=6.4, color=GRAY, linespacing=1.5)
    dests = [
        (61.5, 26.5, GREEN, "草稿的上下文",
         f"第 {', '.join(map(str, f['layers_one_based']))} 层跑完后的隐状态\n拼成 {f['concat_dim']:,} 维,write_context_kv\n写进 DFlash2 的 K,V"),
        (33.5, 23.0, BLUE, "目标模型的状态", "16 层的 K,V\n48 层 GDN 的卷积状态和递归状态\n落在引擎私有的连续缓冲里"),
        (9.5, 19.5, GRAY, "第 1 个输出 token", "最后一层过 LM head,取 argmax\n当下一轮的锚点"),
    ]
    for yy, h, col, head, body in dests:
        rbox(ax, 63, yy, 36.5, h, "white", ec=col, lw=1.1)
        ax.text(65, yy + h - 4.0, head, fontsize=7.4, color=DARK, fontweight="bold", ha="left", va="center")
        ax.text(65, yy + h - 8.4, body, fontsize=6.5, color=DARK, ha="left", va="top", linespacing=1.55)
        arrow(ax, (57.6, yy + h / 2), (62.4, yy + h / 2), color=col, lw=1.2)
    ax.text(50, 4.0, "两级验证里,链 C 的特征先取第一趟的,临时写进草稿的 K,V;精确趟之后再用精确的特征覆盖同一批位置",
            ha="center", va="center", fontsize=6.8, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save(fig, "fig7_bridge.png")


if __name__ == "__main__":
    fig1_chain()
    fig2_kernel()
    fig3_graph()
    fig4_hooks()
    fig5_tlv()
    fig6_line()
    fig7_bridge()
