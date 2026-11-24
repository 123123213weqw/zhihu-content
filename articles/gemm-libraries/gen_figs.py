#!/usr/bin/env python3
"""Figures for the gemm-libraries article (teaching version: three figures).

fig1_layers.png    -- who decides each of the five layers of a GEMM, for each of the five approaches (author's synthesis)
fig2_selection.png -- what the "selection" layer is worth: FP16 GEMM on two RTX cards. cuBLAS default (the heuristic's #1) vs. the fastest of
                      cuBLASLt's 152 enumerated configs (all timed) vs. Triton's 13 working configs (best / median / range).
                      Reads the measurements saved in bench/ (steady_*.json, lt_search_*_acc32.txt); see the article's references
fig3_methods.png   -- the three ways to find the fastest configuration for a shape: measure everything, heuristic, learned cost model (schematic)

Run:  python3 gen_figs.py   -> ./images/      (GEMM_BENCH=<dir> overrides where the bench/ data is read from)
"""
from __future__ import annotations
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

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
BLUE, RED, GREEN, GRAY, PURP, GOLD, ORANGE = "#2288CC", "#BB3333", "#117733", "#666666", "#995588", "#b8922e", "#d97742"
BENCH = os.environ.get("GEMM_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
SIZES = [512, 1024, 2048, 4096, 8192]


def load_lt(path):
    """Parse lt_search output -> {N: [heuristic #1 TFLOPS, best of all TFLOPS, worst TFLOPS, configs timed]}"""
    out, n = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.search(r"N=(\d+) fp16 inputs", line)
        if m:
            n = int(m.group(1)); out[n] = [None, None, None, None]; continue
        if n is None:
            continue
        for i, pat in enumerate((r"heuristic #1 \(default\):\s+([\d.]+) TFLOPS", r"best of all:\s+([\d.]+) TFLOPS",
                                 r"worst timed config:\s+([\d.]+) TFLOPS")):
            m = re.search(pat, line)
            if m and out[n][i] is None:
                out[n][i] = float(m.group(1))
        m = re.search(r"(\d+) timed,", line)
        if m:
            out[n][3] = int(m.group(1))
    return out




def fig2_selection():
    panels = [("RTX 4080", "4080", 120), ("RTX 4090 D", "4090d", 178)]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    med = lambda v: float(np.median(v)) if isinstance(v, list) else float(v)
    for ax, (name, tag, ymax) in zip(axes, panels):
        d = json.load(open(os.path.join(BENCH, f"steady_{tag}.json")))["data"]
        lt = load_lt(os.path.join(BENCH, f"lt_search_{tag}_acc32.txt"))
        tri = lambda s_: sorted(med(d[k]) for k in d if k.startswith(f"{s_}:triton_fp16("))
        cub = [lt[s_][0] for s_ in SIZES]  # the heuristic's own #1, so the gap to the diamond is self-contained
        t_best, t_med, t_worst = [tri(s_)[-1] for s_ in SIZES], [float(np.median(tri(s_))) for s_ in SIZES], [tri(s_)[0] for s_ in SIZES]
        lt_best = [lt[s_][1] for s_ in SIZES]
        n_lt = lt[4096][3]
        ax.fill_between(SIZES, t_worst, t_best, color=GREEN, alpha=0.09, lw=0, label="Triton:13 组配置的范围(下边是最差的一组)")
        ax.plot(SIZES, cub, "o-", color=RED, lw=2.2, ms=7.5, label="cuBLAS 默认:启发式排第一的那一套")
        ax.plot(SIZES, t_best, "s-", color=GREEN, lw=2.0, ms=7, label="Triton:13 组里最好的一组")
        ax.plot(SIZES, t_med, "--", color=GREEN, lw=1.3, ms=5, label="Triton:13 组的中位数")
        ax.plot(SIZES, lt_best, "D", color="#111111", mfc="white", ms=9.5, mew=1.8, ls="none",
                label=f"cuBLAS:{n_lt} 套配置全部实测,最快的一套")
        # mark the shapes where the heuristic visibly missed: a bracket from the default to the best, and a label above it
        off = {512: (-8, 12), 1024: (-22, 16)} if tag == "4090d" else {512: (-8, 12)}
        for k_, (s_, c_, b_) in enumerate(zip(SIZES, cub, lt_best)):
            gap = b_ / c_ - 1
            if gap < 0.04:
                continue
            ax.annotate("", xy=(s_, b_), xytext=(s_, c_),
                        arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.1, shrinkA=6, shrinkB=9))
            dx, dy = off.get(s_, (0, 20))
            ax.annotate(f"默认慢 {gap:.0%}", (s_, b_), textcoords="offset points", xytext=(dx, dy), ha="right" if dx else "center",
                        va="bottom", fontsize=9.5, fontweight="bold", color="#111111", bbox=dict(fc="white", ec="none", pad=1.5))
        ax.set_xscale("log", base=2); ax.set_xticks(SIZES); ax.set_xticklabels([str(v) for v in SIZES])
        ax.minorticks_off()
        ax.set_xlabel("矩阵大小(N × N × N)"); ax.set_ylabel("TFLOPS")
        ax.set_ylim(0, ymax); ax.set_xlim(310, 9700)
        ax.set_title(name + " 上的半精度矩阵乘", fontsize=11.5, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=8.9, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_selection.png"); plt.close(fig)


def fig1_layers():
    rows = ["① 算法\n要不要切开、换成 GEMV", "② 分块\ntile、流水级数、遍历顺序", "③ 线程和寄存器\n一个线程算几个输出",
            "④ 指令\nTensor Core、异步拷贝", "⑤ 选择\n这个形状用哪个 kernel"]
    cols = ["手写 CUDA", "cuBLAS", "CUTLASS", "Triton", "TVM"]
    YOU, PICK, VENDOR, COMP, MACHINE = "you", "pick", "vendor", "compiler", "machine"
    cells = [
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("你", YOU), ("你写计算", YOU)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("你", YOU), ("规则或搜索", MACHINE)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("编译器", COMP), ("规则或搜索", MACHINE)],
        [("你", YOU), ("厂商", VENDOR), ("你挑", PICK), ("编译器", COMP), ("规则或搜索", MACHINE)],
        [("你", YOU), ("启发式·可穷举", MACHINE), ("自己解决", PICK), ("自己调优", PICK), ("搜索", MACHINE)],
    ]
    style = {
        YOU: ("#dbeaf7", BLUE), PICK: ("#eef5fb", "#4a8fc8"), VENDOR: ("#e4e4e4", "#666666"),
        COMP: ("#e2f0e2", GREEN), MACHINE: ("#f0e0ea", PURP),
    }
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0, 25.2); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.text(12.6, 12.0, "一个矩阵乘的五层决定,五种做法里各由谁来定", ha="center", fontsize=13, fontweight="bold")
    x0, w, h, ytop = 7.0, 3.56, 1.86, 10.1
    for j, c in enumerate(cols):
        ax.text(x0 + j * w + w / 2, ytop + 0.32, c, ha="center", va="center", fontsize=11.5, fontweight="bold")
    for i, (r, row) in enumerate(zip(rows, cells)):
        y = ytop - (i + 1) * h
        ax.text(0.4, y + h / 2, r, ha="left", va="center", fontsize=9.8, color="#222222", linespacing=1.5)
        for j, (txt, kind) in enumerate(row):
            fc, ec = style[kind]
            ax.add_patch(FancyBboxPatch((x0 + j * w + 0.08, y + 0.1), w - 0.16, h - 0.2, boxstyle="round,pad=0.02",
                                        fc=fc, ec=ec, lw=1.3))
            ax.text(x0 + j * w + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.8, color=ec, fontweight="bold")
    # 图例
    ly = 0.45
    legend = [("你", YOU, "你自己写"), ("你挑", PICK, "你从现成的里挑,或自己解决"), ("厂商", VENDOR, "厂商定好,改不了"),
              ("编译器", COMP, "编译器替你定"), ("搜索", MACHINE, "机器来选:启发式、规则、搜索")]
    lx = 0.4
    for txt, kind, desc in legend:
        fc, ec = style[kind]
        ax.add_patch(FancyBboxPatch((lx, ly - 0.22), 0.9, 0.46, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2))
        ax.text(lx + 1.08, ly, desc, ha="left", va="center", fontsize=8.8, color="#444444")
        lx += 1.08 + len(desc) * 0.25 + 0.65
    fig.savefig(f"{OUT}/fig1_layers.png"); plt.close(fig)



def fig3_methods():
    titles = ["① 全部实测", "② 启发式(经验法则)", "③ 代价模型"]
    colors = [RED, ORANGE, PURP]
    fills = ["#fbe9e9", "#fdeee2", "#efe4f0"]
    foots = ["花的时间:最多\n结果:最稳\n候选一多,就慢得受不了",
             "花的时间:几乎没有\n结果:规则覆盖到的形状准\n没覆盖到的会错,而且自己不知道",
             "花的时间:少,但要先攒数据训练\n结果:介于两者之间\n换一块 GPU,得重新学"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.5))
    for k, ax in enumerate(axes):
        ax.set_xlim(0, 10); ax.set_ylim(-1.1, 10.3); ax.axis("off")
        ax.set_title(titles[k], fontsize=12.5, fontweight="bold", color=colors[k], pad=6)
        ec, fc = colors[k], fills[k]

        def box(x, y, w, h, txt, fs=10, fw="normal"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.3))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color="#222222",
                    fontweight=fw, linespacing=1.6)

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4, shrinkA=0, shrinkB=0))

        box(2.6, 9.0, 4.8, 0.95, "输入:矩阵的形状 $x$", fs=10.5, fw="bold")
        names = ["$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", r"$c_{|C|}$"]
        if k == 0:
            for i, n in enumerate(names):
                box(2.4, 7.6 - i * 0.82, 5.2, 0.62, f"{n}:跑一遍", fs=9.8)
            arrow(5.0, 9.0, 5.0, 8.22); arrow(5.0, 3.5, 5.0, 2.07)
            box(1.6, 1.05, 6.8, 1.0, "看表:取最快的那一套", fs=10.5, fw="bold")
            ax.text(7.9, 5.75, "每一套\n都真的跑", ha="left", va="center", fontsize=9.6, color=ec, linespacing=1.6)
        elif k == 1:
            box(1.4, 3.5, 7.2, 4.35, "经验规则 $h$\n(人从大量实测里总结出来的)\nM 很小 $\\rightarrow$ 小块,切开 K\n"
                                      "$M \\times N$ 不大 $\\rightarrow$ 64×64\n否则 $\\rightarrow$ 128×128", fs=9.8)
            arrow(5.0, 9.0, 5.0, 7.85); arrow(5.0, 3.5, 5.0, 2.07)
            box(1.6, 1.05, 6.8, 1.0, "直接给出答案,没有跑过", fs=10.5, fw="bold")
        else:
            box(0.25, 6.6, 2.7, 1.0, "一批实测数据", fs=9.6)
            arrow(2.95, 7.1, 3.55, 7.1)
            box(3.55, 6.6, 6.1, 1.0, "训练出来的模型 $\\hat{T}$", fs=10.2)
            arrow(6.6, 9.0, 6.6, 7.6)
            for i, n in enumerate(names):
                box(2.4, 5.85 - i * 0.62, 5.2, 0.5, f"{n}:估一个耗时", fs=9.6)
            arrow(5.0, 6.6, 5.0, 6.35); arrow(5.0, 2.75, 5.0, 2.07)
            box(1.6, 1.05, 6.8, 1.0, "取估得最小的,没有跑过", fs=10.5, fw="bold")
            ax.text(7.9, 4.35, "每一套\n都只是估", ha="left", va="center", fontsize=9.6, color=ec, linespacing=1.6)
        ax.text(5.0, 0.55, foots[k], ha="center", va="top", fontsize=9.6, color="#333333", linespacing=1.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_methods.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_layers(); fig2_selection(); fig3_methods()
    print("done ->", OUT)
