#!/usr/bin/env python3
"""Figures for the article on optimising Qwen3.8-27B on a MetaX C500.

fig1_chain.png   one GDN layer split into three kinds of work, and which unit of the AP does each
fig2_kernel.png  how MetaX's GDN chunk kernel lays out its state, registers and shared memory on one AP
fig3_graph.png   CUDA Graph on a C500 as measured by a third party: draft step and verify step
fig4_hooks.png   which of vLLM's extension points to use to replace an op on vLLM-metax

Numbers come from bench/calc_summary.json (written by calc.py).
Run:  python3 gen_figs.py  ->  ./images/
"""
from __future__ import annotations

import json
import os

import matplotlib
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


if __name__ == "__main__":
    fig1_chain()
    fig2_kernel()
    fig3_graph()
    fig4_hooks()
