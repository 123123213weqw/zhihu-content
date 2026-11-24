#!/usr/bin/env python3
"""Figures for the vLLM 0.34 overview article.

fig1_processes.png    -- process layout of `vllm serve -tp=4`: 1 API server + 1 engine core + 4 GPU workers = 6 processes,
                         and what crosses each boundary (ZMQ + MessagePack; shared-memory broadcast)
fig2_request_path.png -- sequence diagram: the 11 steps from an HTTP request to the first streamed token
fig3_loc.png          -- where the ~1.28M lines of Python in vllm/ go (left), and what sits inside vllm/v1 (right)

Every number in fig3 was counted on the v0.34.0 source tree (line counts of *.py files per package).
Run:  python3 gen_figs.py   -> ./images/
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Noto Serif CJK JP", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, ORANGE, PURP = "#2288CC", "#BB3333", "#117733", "#666666", "#d97742", "#995588"


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, fw="normal", color=None, ls="-", lw=1.2, pad=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or ec, fontweight=fw,
            linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.5, style="-|>", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


# ----------------------------------------------------------------------------------------------
def fig1_processes():
    fig, ax = plt.subplots(figsize=(14.2, 5.9))
    ax.set_xlim(0, 120); ax.set_ylim(0, 42); ax.axis("off")
    ax.text(60, 40.4, r"vllm serve -tp=4:1 个 API 服务进程 + 1 个引擎核心进程 + 4 个 GPU 工作进程 = 6 个进程",
            ha="center", va="center", fontsize=13, fontweight="bold")
    A, E, W = dict(fc="#dbeaf7", ec=BLUE), dict(fc="#fdeee2", ec=ORANGE), dict(fc="#e2f0e2", ec=GREEN)
    groups = [(12.5, 23.5, "API 服务进程 × 1", A), (51.0, 22.0, "引擎核心进程 × 1", E), (86.0, 33.0, "GPU 工作进程 × 4(每张卡 1 个)", W)]
    for x, w, title, st in groups:
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 37.0, boxstyle="round,pad=0.02,rounding_size=0.6", fc=st["fc"], ec=st["ec"], lw=1.6))
        ax.text(x + w / 2, 35.6, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=st["ec"])

    # client
    box(ax, 1.0, 17.5, 9.0, 5.0, "客户端", "#f2f2f2", GRAY, fs=10.5, fw="bold")
    arrow(ax, 10.2, 20.0, 12.3, 20.0, GRAY)

    # API server process: five boxes, 5.2 high, evenly spread
    api = [
        (28.2, "HTTP 入口(FastAPI + uvloop)"),
        (21.95, "渲染器:消息 $\\rightarrow$ 提示词 $\\rightarrow$ token"),
        (15.7, "输入处理器:校验参数,打包"),
        (9.45, "AsyncLLM:前端的发动机接口"),
        (3.2, "输出处理器:反分词,推进队列"),
    ]
    for y, t in api:
        box(ax, 13.8, y, 20.9, 5.2, t, "white", BLUE)

    # engine core process
    box(ax, 52.3, 28.2, 19.4, 5.2, "EngineCoreProc:收消息,发消息", "white", ORANGE, fs=8.8)
    box(ax, 52.3, 15.2, 19.4, 12.0, "", "white", ORANGE)
    ax.text(62.0, 25.3, "忙循环:收输入,推进一步", ha="center", va="center", fontsize=9.2, color=ORANGE)
    for x, w, t in [(53.2, 5.2, "调度器"), (58.7, 6.6, "KV 缓存\n管理器"), (65.6, 5.2, "结构化\n输出")]:
        box(ax, x, 16.4, w, 6.4, t, "white", ORANGE, fs=8.8)
    box(ax, 52.3, 9.45, 19.4, 5.2, "执行器\nexecute_model /\nsample_tokens", "white", ORANGE, fs=8.4)
    box(ax, 52.3, 3.2, 19.4, 5.2, "两个线程:一个收请求,一个发结果", "white", ORANGE, fs=8.6)

    # worker processes: four narrow columns
    for k in range(4):
        x = 87.3 + k * 7.8
        box(ax, x, 24.5, 7.3, 5.6, "GPU\nWorker", "white", GREEN, fs=8.6)
        box(ax, x, 17.6, 7.3, 5.6, "Model\nRunner V2", "white", GREEN, fs=8.6)
        box(ax, x, 10.7, 7.3, 5.6, "模型\n(PyTorch)", "white", GREEN, fs=8.6)
        box(ax, x, 3.5, 7.3, 5.0, f"GPU {k}", "#f7f7f7", GRAY, fs=9.0, fw="bold")
        arrow(ax, x + 3.65, 10.7, x + 3.65, 8.7, GREEN, lw=1.1)

    # what crosses each boundary
    arrow(ax, 36.4, 31.5, 50.6, 31.5, RED, lw=1.8)
    ax.text(43.5, 32.9, "请求", ha="center", va="bottom", fontsize=9.0, color=RED, fontweight="bold")
    ax.text(43.5, 30.1, "ZMQ +\nMessagePack", ha="center", va="top", fontsize=8.2, color=RED)
    arrow(ax, 50.6, 15.7, 36.4, 15.7, PURP, lw=1.8)
    ax.text(43.5, 17.1, "结果", ha="center", va="bottom", fontsize=9.0, color=PURP, fontweight="bold")
    ax.text(43.5, 14.3, "ZMQ +\nMessagePack", ha="center", va="top", fontsize=8.2, color=PURP)
    arrow(ax, 73.4, 21.5, 85.6, 21.5, GREEN, lw=1.8)
    ax.text(79.5, 22.9, "调度结果", ha="center", va="bottom", fontsize=9.0, color=GREEN, fontweight="bold")
    ax.text(79.5, 20.1, "共享内存\n写一次,全体可读", ha="center", va="top", fontsize=8.2, color=GREEN)
    arrow(ax, 85.6, 8.0, 73.4, 8.0, GRAY, lw=1.5)
    ax.text(79.5, 6.7, "新 token", ha="center", va="top", fontsize=8.6, color=GRAY)

    fig.savefig(f"{OUT}/fig1_processes.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig2_request_path():
    fig, ax = plt.subplots(figsize=(11.4, 13.2))
    ax.set_xlim(0, 105); ax.set_ylim(2.5, 102); ax.axis("off")
    ax.text(50, 100.8, "一个请求,从 HTTP 进来,到第一个 token 发回去", ha="center", va="center", fontsize=13, fontweight="bold")
    lanes = [(8, "客户端", GRAY, "#f2f2f2"), (32, "API 服务进程", BLUE, "#dbeaf7"),
             (60, "引擎核心进程", ORANGE, "#fdeee2"), (88, "GPU 工作进程", GREEN, "#e2f0e2")]
    top, bottom = 94.0, 5.0
    for x, name, ec, fc in lanes:
        box(ax, x - 7.5, top, 15, 4.6, name, fc, ec, fs=10.5, fw="bold", pad=0.5)
        ax.plot([x, x], [bottom, top], ls=(0, (4, 4)), color=ec, lw=1.2)
    xc, xa, xe, xw = [l[0] for l in lanes]

    def msg(y, x0, x1, text, color=GRAY, dy=0.9):
        arrow(ax, x0, y, x1, y, color, lw=1.5)
        ax.text((x0 + x1) / 2, y + dy, text, ha="center", va="bottom", fontsize=9.0, color="#222222")

    def selfmsg(y, x, text, color, side=1, w=5.0, h=3.6):
        ax.plot([x, x + side * w, x + side * w], [y, y, y - h], color=color, lw=1.3)
        arrow(ax, x + side * w, y - h, x, y - h, color, lw=1.3)
        ax.text(x + side * (w + 0.9), y - h / 2, text, ha="left" if side > 0 else "right", va="center", fontsize=9.0, color="#222222")

    ys = dict(s1=88.0, s2=81.5, s3=75.0, s4=68.5, s5=62.0, s6=55.5, s7=49.0, s8=42.5, s8b=36.0, s9=29.5, s10=23.0, s11=16.5)
    msg(ys["s1"], xc, xa, "① HTTP 请求")
    selfmsg(ys["s2"], xa, "② 渲染:消息 $\\rightarrow$ 提示词,再分词", BLUE)
    selfmsg(ys["s3"], xa, "③ 校验参数,打包", BLUE)
    msg(ys["s4"], xa, xe, "④ 发给引擎核心(ZMQ)", RED)
    selfmsg(ys["s5"], xe, "⑤ 入队;下一步:调度", ORANGE)
    msg(ys["s6"], xe, xw, "⑥ 调度结果(共享内存广播)", GREEN)
    selfmsg(ys["s7"], xw, "⑦ 准备输入,跑模型,采样", GREEN, side=-1)
    msg(ys["s8"], xw, xe, "⑧ 新 token", GRAY)
    selfmsg(ys["s8b"], xe, "⑧ 更新每个请求的状态", ORANGE)
    msg(ys["s9"], xe, xa, "⑨ 结果回去(ZMQ)", color=PURP)
    selfmsg(ys["s10"], xa, "⑩ 反分词,推进这个请求的队列", BLUE)
    msg(ys["s11"], xa, xc, "⑪ 流式发回:第一个 token")

    # first-token latency bracket on the left
    ax.annotate("", xy=(2.6, ys["s11"]), xytext=(2.6, ys["s1"]), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
    ax.text(1.6, (ys["s1"] + ys["s11"]) / 2, "首 token 延迟", rotation=90, ha="center", va="center", fontsize=9.5, color=RED, fontweight="bold")
    # the loop: from the streamed token back to the scheduling step
    yloop = ys["s5"] + 1.6
    arrow(ax, xw + 8.0, ys["s11"], xw + 8.0, yloop, GREEN, lw=1.3)
    arrow(ax, xw + 8.0, yloop, xe + 0.8, yloop, GREEN, lw=1.3)
    ax.plot([xa, xw + 8.0], [ys["s11"], ys["s11"]], ls=(0, (2, 3)), color=GREEN, lw=1.1)
    ax.text(xw + 8.0 + 2.6, (yloop + ys["s11"]) / 2 + 3.0, "之后每转一圈\n多一个 token", rotation=90, ha="center", va="center",
            fontsize=9.5, color=GREEN, fontweight="bold", linespacing=1.6)
    ax.text(50, 7.0, "异步调度:GPU 在跑第 N 步的时候,CPU 已经在调度第 N+1 步", ha="center", va="center", fontsize=9.6,
            color="#222222", bbox=dict(fc="white", ec="#cccccc", pad=5.0))
    fig.savefig(f"{OUT}/fig2_request_path.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig3_loc():
    cats = [("模型实现", 733742), ("引擎 v1", 193090), ("对外服务", 98220), ("分布式与 KV 迁移", 93166),
            ("其他", 88542), ("配置与编译", 40587), ("多模态与 HF 配置", 35173)]
    v1 = [("注意力后端", 72782), ("工作进程(含 Model Runner、推测解码、采样)", 48574), ("KV 卸载", 19906),
          ("调度器与 KV 缓存管理", 14518), ("前后端边界(engine)", 12718), ("其余", 26092)]
    total = sum(v for _, v in cats)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax, data, title, total_, unit in ((axes[0], cats, "vllm/ 下 128 万行 Python 的分布", total, "全部"),
                                          (axes[1], v1, "其中引擎 v1(19.3 万行)里面", 193090, "v1")):
        names = [n for n, _ in data][::-1]; vals = [v for _, v in data][::-1]
        colors = [BLUE if n == "引擎 v1" else ORANGE if n == "模型实现" else GREEN if "工作进程" in n else PURP if "注意力" in n else "#9aa7b1" for n in names]
        y = np.arange(len(names))
        ax.barh(y, [v / 1e4 for v in vals], color=colors, height=0.62, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.6)
        ax.set_xlim(0, max(vals) / 1e4 * 1.30); ax.set_xlabel("万行")
        for yi, v in zip(y, vals):
            ax.text(v / 1e4 + max(vals) / 1e4 * 0.012, yi, f"{v / 1e4:.1f} 万行({v / total_:.0%})", va="center", ha="left", fontsize=9.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_loc.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_processes(); fig2_request_path(); fig3_loc()
    print("done ->", OUT)
