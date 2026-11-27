#!/usr/bin/env python3
"""前端一篇的图(vLLM 0.34.0,Llama-3.1-8B-Instruct,一张 RTX 4090 D,前端在 Xeon 8352V 上)。

fig0_flow.png         一条请求在前端走一圈:两个进程,两条队列
fig1_detok.png        (a) 每解码一个 token 花多久:朴素做法线性增长,增量做法不变  (b) 解码 N 个 token 的总时间
fig2_ttft.png         (a) 首 token 时间按提示词长度拆成四段  (b) 其中前端自己占的两段
fig3_first_token.png  "算完到出来"这一段随提示词长度增长:流式反分词器第一次 step 把提示词重放了一遍
fig4_lag.png          (a) 并发路数和每个 token 的延迟  (b) 并发 32 路时前端每个 token 的 CPU 时间,核凉着和核热着
fig5_tokq.png         分词线程 1 个和 4 个:32 条短请求单独发,和排在一条 32k 请求后面
fig6_cold.png         空闲多久之后的第一条请求

每个数都来自 bench/calc_summary.json(calc.py 写的)。
运行:  python3 gen_figs.py   ->  ./images/      (FRONT_BENCH 指定别的 bench 目录)
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)
BENCH = os.environ.get("FRONT_BENCH", os.path.join(HERE, "bench"))
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
calc = json.load(open(os.path.join(BENCH, "calc_summary.json")))


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, fw="normal", color=None, ls="-", lw=1.2, pad=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={pad}", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or ec, fontweight=fw, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, rad=0.0, ls="-", text=None, tpos=0.5, toff=(0, 1.2)):
    ax.add_patch(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, mutation_scale=12,
                                                    connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text(x0 + (x1 - x0) * tpos + toff[0], y0 + (y1 - y0) * tpos + toff[1], text, ha="center", va="bottom", fontsize=8.5, color=color)


# ----------------------------------------------------------------------------------------------
def fig0_flow():
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_xlim(0, 138); ax.set_ylim(0, 74); ax.axis("off")
    ax.text(69, 72.8, "一条请求在 vLLM 0.34.0 的前端走一圈:API server 一个进程,引擎核心另一个进程,中间是 ZMQ", ha="center", va="center", fontsize=11)
    # 进程框
    box(ax, 1, 6, 8, 60, "", "#ffffff", GRAY, lw=1.0)
    ax.text(5, 67.8, "客户端", ha="center", va="center", fontsize=9.5, color=GRAY)
    box(ax, 11, 6, 75, 60, "", "#f7fafc", BLUE, lw=1.2)
    ax.text(48.5, 67.8, "API server 进程:uvicorn + FastAPI,一个事件循环(一个线程)", ha="center", va="center", fontsize=9.5, color=BLUE)
    box(ax, 104, 6, 33, 60, "", "#fbf7f7", RED, lw=1.2)
    ax.text(120.5, 67.8, "EngineCore 进程:调度 + 模型执行", ha="center", va="center", fontsize=9.5, color=RED)
    # 去程
    box(ax, 13, 53, 18, 10, "HTTP 请求\n解析 JSON、校验参数", "#eef4fa", BLUE, fs=9)
    box(ax, 35, 53, 18, 10, "渲染\n聊天模板、分词(线程池)", "#eef4fa", BLUE, fs=9)
    box(ax, 57, 53, 27, 10, "AsyncLLM.add_request()\n建请求状态、反分词器", "#eef4fa", BLUE, fs=9)
    box(ax, 106, 53, 29, 10, "输入线程 → 调度器\n(排进 WAITING)", "#f8eeee", RED, fs=9)
    arrow(ax, 9, 58, 13, 58, BLUE); arrow(ax, 31, 58, 35, 58, BLUE); arrow(ax, 53, 58, 57, 58, BLUE)
    arrow(ax, 84, 58, 106, 58, BLUE)
    ax.text(95, 59.2, "ZMQ\n请求(token 编号)", ha="center", va="bottom", fontsize=8.2, color=BLUE, linespacing=1.3)
    # 回程
    box(ax, 106, 34, 29, 11, "模型执行 → 输出线程\n每一步的 EngineCoreOutput", "#f8eeee", RED, fs=9)
    box(ax, 57, 34, 27, 11, "output_handler\n每批最多 128 条:反分词、停止词", "#fdf3ec", ORANGE, fs=9)
    box(ax, 35, 34, 18, 11, "每条请求一个队列\n前端快了就合并", "#fdf3ec", ORANGE, fs=9)
    box(ax, 13, 34, 18, 11, "SSE 生成器\ndata: {...}\\n\\n", "#fdf3ec", ORANGE, fs=9)
    arrow(ax, 106, 39.5, 84, 39.5, ORANGE)
    ax.text(95, 40.7, "ZMQ\n输出(新 token 编号)", ha="center", va="bottom", fontsize=8.2, color=ORANGE, linespacing=1.3)
    arrow(ax, 57, 39.5, 53, 39.5, ORANGE); arrow(ax, 35, 39.5, 31, 39.5, ORANGE); arrow(ax, 13, 39.5, 9, 39.5, ORANGE)
    # 断连
    box(ax, 13, 15, 18, 10, "客户端断开\n生成器被取消", "#f4f4f4", GRAY, fs=9, ls="--")
    box(ax, 35, 15, 49, 10, "generate() 捕获 CancelledError → abort()\n前端删掉请求状态,给引擎发中止消息", "#f4f4f4", GRAY, fs=9, ls="--")
    box(ax, 106, 15, 29, 10, "FINISHED_ABORTED\n之后不再排它,块释放", "#f4f4f4", GRAY, fs=9, ls="--")
    arrow(ax, 31, 20, 35, 20, GRAY, ls="--"); arrow(ax, 84, 20, 106, 20, GRAY, ls="--")
    ax.text(95, 21.2, "ZMQ\n中止", ha="center", va="bottom", fontsize=8.2, color=GRAY, linespacing=1.3)
    ax.text(69, 9.5, "蓝色:去程,每个请求走一遍。橙色:回程,每个 token 走一遍。灰色虚线:客户端断开之后。",
            ha="center", va="center", fontsize=9, color=GRAY)
    fig.savefig(os.path.join(OUT, "fig0_flow.png")); plt.close(fig)


def fig1_detok():
    d = calc["detok"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.4))
    for rows, color, label, ls in ((d["naive"], RED, "朴素:每来一个 token 就把整段重新解码", "-"),
                                   (d["slow"], BLUE, "增量(SlowIncrementalDetokenizer,Python)", "-"),
                                   (d["fast"], GREEN, "增量(FastIncrementalDetokenizer,tokens 库的 DecodeStream)", "-")):
        x = np.array([r["tokens"] for r in rows]); y = np.array([r["us_last_block"] for r in rows])
        a.plot(x, y, "o-", color=color, lw=1.6, ms=4, label=label)
        b.plot(x, np.array([r["total_ms"] for r in rows]) / 1e3, "o-", color=color, lw=1.6, ms=4, label=label)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks([256, 1024, 4096, 8192]); a.set_xticklabels(["256", "1,024", "4,096", "8,192"])
    a.set_ylim(0.4, 3000)
    a.set_yticks([1, 10, 100, 1000]); a.set_yticklabels(["1", "10", "100", "1,000"])
    a.set_xlabel("已经解码了多少个 token"); a.set_ylabel("第 t 个 token 的反分词耗时(微秒,对数)")
    a.set_title("(a) 每个 token 花多久", fontsize=10)
    a.annotate("朴素做法:耗时和上下文长度成正比\n4096 处每个 910 微秒", xy=(4096, 910), xytext=(258, 1500), fontsize=8.5, color=RED,
               va="top", arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    a.annotate("增量:每个 token 2 到 6 微秒,和长度无关", xy=(4096, 1.98), xytext=(258, 0.5), fontsize=8.5, color=GREEN,
               va="bottom", arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1)
    b.set_xscale("log", base=2); b.set_yscale("log")
    b.set_xticks([256, 1024, 4096]); b.set_xticklabels(["256", "1,024", "4,096"])
    b.set_xlabel("一共解码了多少个 token"); b.set_ylabel("总时间(秒,对数)")
    b.set_title("(b) 总时间:朴素是二次方,增量是线性", fontsize=10)
    n = np.array([256, 4096]); pred = d["c_us_per_context_token"] * n * (n - 1) / 2 / 1e6
    b.plot(n, pred, "--", color=RED, lw=1.0)
    b.set_ylim(3e-4, 40)
    b.text(256, 6.0, "朴素做法第 $t$ 个 token 约 $c\\,t$ 微秒,合计 $c\\,N(N{+}1)/2$\n$c$ 取 0.22 微秒:4096 个要 1.86 秒(实线是实测,虚线是公式)", fontsize=8.5, color=RED, va="top")
    fast4096 = next(r for r in d["fast"] if r["tokens"] == 4096)
    b.annotate("增量:4096 个一共 %.0f 毫秒,\n是朴素做法的 1/%.0f" % (fast4096["total_ms"], d["naive_measured_total_ms_4096"] / fast4096["total_ms"]),
               xy=(4096, fast4096["total_ms"] / 1e3), xytext=(1500, 4.2e-4), fontsize=8.5, color=GREEN,
               va="bottom", arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(OUT, "fig1_detok.png")); plt.close(fig)


def fig2_ttft():
    t = calc["ttft"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.5))
    Ls = [128, 1024, 4096, 16384, 32768]
    ids = [next(r for r in t["rows"] if r["L"] == L and r["kind"] == "ids") for L in Ls]
    txt = [next(r for r in t["rows"] if r["L"] == L and r["kind"] == "text") for L in Ls]
    x = np.arange(len(Ls))
    parts = (("to_engine_ms", "到引擎之前(解析、渲染、建状态)", BLUE), ("queue_ms", "在引擎里排队", GOLD),
             ("prefill_ms", "prefill", "#c9c9c9"), ("out_ms", "算完到客户端收到第一个 token", ORANGE))
    for rows, w, off, alpha, label in ((ids, 0.36, -0.19, 1.0, "提示词直接给 token 编号(/v1/completions)"),
                                       (txt, 0.36, 0.19, 0.55, "提示词给文字(/v1/chat/completions)")):
        bottom = np.zeros(len(Ls))
        for key, name, color in parts:
            v = np.array([r[key] for r in rows])
            a.bar(x + off, v, w, bottom=bottom, color=color, alpha=alpha, edgecolor="white", lw=0.5, label=name if alpha == 1.0 else None)
            bottom += v
        for xi, r in zip(x + off, rows):
            a.text(xi, r["ttft_ms"] * 1.06, "%.0f" % r["ttft_ms"], ha="center", va="bottom", fontsize=8, color=GRAY)
    a.set_yscale("log"); a.set_ylim(10, 2e4)
    a.set_xticks(x); a.set_xticklabels(["128", "1,024", "4,096", "16,384", "32,768"])
    a.set_xlabel("提示词长度(token)"); a.set_ylabel("首 token 时间(毫秒,对数)")
    a.set_title("(a) 拆成四段,柱顶是合计", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    a.bar([], [], color=BLUE, alpha=0.45, label="浅色柱:提示词给的是文字(多一步分词)")
    for rows, off, alpha, color, label in ((ids, -0.19, 1.0, BLUE, "token 编号"), (txt, 0.19, 0.55, PURP, "文字")):
        v_in = np.array([r["to_engine_ms"] for r in rows]); v_out = np.array([r["out_ms"] for r in rows])
        b.bar(x + off, v_in, 0.36, color=color, alpha=alpha, edgecolor="white", lw=0.5, label="到引擎之前(%s)" % label)
        b.bar(x + off, v_out, 0.36, bottom=v_in, color=ORANGE, alpha=alpha, edgecolor="white", lw=0.5, hatch="//" if alpha < 1 else None)
        for xi, r in zip(x + off, rows):
            tot = r["to_engine_ms"] + r["out_ms"]
            b.text(xi, tot + 2.5, "%.0f\n%.1f%%" % (tot, tot / r["ttft_ms"] * 100), ha="center", va="bottom", fontsize=8, color=GRAY, linespacing=1.1)
    b.set_xticks(x); b.set_xticklabels(["128", "1,024", "4,096", "16,384", "32,768"])
    b.set_xlabel("提示词长度(token)"); b.set_ylabel("毫秒")
    b.set_ylim(0, 200)
    b.set_title("(b) 前端自己占的两段:柱顶是合计和它在首 token 时间里的占比", fontsize=10)
    b.legend(handles=[matplotlib.patches.Patch(facecolor=BLUE, label="到引擎之前(token 编号)"),
                      matplotlib.patches.Patch(facecolor=PURP, alpha=0.55, label="到引擎之前(文字)"),
                      matplotlib.patches.Patch(facecolor=ORANGE, label="算完到客户端收到第一个 token(橙色)")], fontsize=8, loc="upper left")
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.savefig(os.path.join(OUT, "fig2_ttft.png")); plt.close(fig)


def fig3_first_token():
    t = calc["ttft"]; f = calc["first_token"]; p = calc["profile_32k"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.5))
    rows = [r for r in t["rows"] if r["kind"] == "ids"]
    L = np.array([r["prompt_tokens"] for r in rows]) / 1e3
    a.plot(L, [r["out_ms"] for r in rows], "o-", color=ORANGE, lw=1.8, ms=5, label="实测:最后一块 prefill 算完,到客户端收到第一个 token")
    g = f["rows"]
    a.plot(np.array([r["tokens"] for r in g]) / 1e3, [r["first_ms"] for r in g], "s--", color=GREEN, lw=1.4, ms=4.5,
           label="单独跑这一步:DecodeStream 第一次 step(进程一直忙着,核是热的)")
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks([0.128, 1.024, 4.096, 16.384, 32.768]); a.set_xticklabels(["128", "1,024", "4,096", "16,384", "32,768"])
    a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"])
    a.set_xlabel("提示词长度(token)"); a.set_ylabel("毫秒(对数)")
    a.set_title("(a) “算完到出来”这一段,和提示词长度成正比", fontsize=10)
    a.set_ylim(0.02, 300)
    a.annotate("32k:实测 49 毫秒\n剖析里 49.9 毫秒在 DecodeStream.step", xy=(32.768, 49.1), xytext=(2.6, 0.06), fontsize=8.5, color=ORANGE,
               va="bottom", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9, connectionstyle="arc3,rad=-0.25"))
    a.legend(fontsize=8, loc="upper left")
    r = f["inproc"]
    xs = np.arange(len(r))
    b.bar(xs - 0.2, [x["after_idle_3s_ms"] for x in r], 0.38, color=RED, label="先睡 3 秒再跑(核是凉的)")
    b.bar(xs + 0.2, [x["immediately_again_ms"] for x in r], 0.38, color=BLUE, label="紧接着再跑一遍(核热了)")
    for xi, x in zip(xs, r):
        b.text(xi - 0.2, x["after_idle_3s_ms"] + 0.8, "%.0f ms\n%s MHz" % (x["after_idle_3s_ms"], round(x["freq_khz"] / 1e3)), ha="center", va="bottom", fontsize=8, color=RED)
        b.text(xi + 0.2, x["immediately_again_ms"] + 0.8, "%.0f" % x["immediately_again_ms"], ha="center", va="bottom", fontsize=8, color=BLUE)
    b.axhline(f["first_ms_32k"], color=GREEN, lw=1.0, ls="--", xmax=0.90)
    b.text(3.45, f["first_ms_32k"] + 1.2, "单独起一个进程\n一直跑:%.0f 毫秒" % f["first_ms_32k"], ha="left", va="bottom", fontsize=8.5, color=GREEN)
    b.set_xticks(xs); b.set_xticklabels(["第 %d 次" % (i + 1) for i in xs])
    b.set_xlim(-0.6, 5.0)
    b.set_ylim(0, 60)
    b.set_ylabel("同一个 32k 提示词,DecodeStream 第一次 step(毫秒)")
    b.set_title("(b) 在 API server 自己的线程里跑:刚睡醒那一次最慢", fontsize=10)
    b.legend(fontsize=8, loc="upper right")
    b.text(0.26, 0.80, "第 1 次紧接着一条真实的 32k 请求:\n线程在整个 prefill 期间几乎都在睡,\n醒来时核的频率读数是 %.0f MHz" % f["inproc_first_freq_mhz"],
           transform=b.transAxes, fontsize=8.5, color=RED, ha="left", va="top")
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(OUT, "fig3_first_token.png")); plt.close(fig)


def fig4_lag():
    lag = calc["lag"]; ll = calc["laglong"]; hog = calc["hog"]; tail = calc["tail"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 4.7))
    runs = {r["b"]: r for r in lag["runs"]}
    bs = [1, 16, 32, 256]
    rs = [runs[1], ll, runs[32], runs[256]]
    x = np.arange(len(bs))
    a.bar(x, [r["delay_p50"] for r in rs], 0.55, color=BLUE, label="中位数")
    a.scatter(x, [min(r["delay_p99"], 57.5) for r in rs], color=ORANGE, zorder=3, label="99 分位", marker="D", s=22)
    for xi, r in zip(x, rs):
        a.text(xi + 0.30, r["delay_p50"] + 1.0, "%.1f" % r["delay_p50"], ha="left", va="bottom", fontsize=8.5, color=BLUE)
        if r["delay_p99"] <= 57.5:
            a.text(xi + 0.30, r["delay_p99"] - 1.0, "%.0f" % r["delay_p99"], ha="left", va="top", fontsize=8.5, color=ORANGE)
    a.annotate("256 路的 99 分位是 279 毫秒,超出坐标", xy=(3, 57.5), xytext=(1.45, 52), fontsize=8.5, color=ORANGE, va="center",
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
    a.set_xticks(x); a.set_xticklabels(["%d 路\n%s token/秒\n一步 %.0f 毫秒" % (v, "%.0f" % r["tokens_per_s"], r["step_ms"]) for v, r in zip(bs, rs)], fontsize=8.5)
    a.set_ylim(0, 64)
    a.set_ylabel("一个 token 从算完到客户端收到(毫秒)")
    a.set_title("(a) 并发路数越多,前端越忙,每个 token 等得越久", fontsize=10)
    a.legend(fontsize=8.5, loc="upper left")
    a.text(0.27, 0.70, "中位数看蓝色柱:并发 32 路一共跑了 %d 轮,都在 4 到 8 毫秒。\n99 分位看橙色点:这 %d 轮里 %d 轮在 10 到 35 毫秒之间,\n另外 %d 轮更高(%s 毫秒)。"
           % (lag["b32_runs_total"], lag["b32_runs_total"], lag["b32_p99_le_35"], lag["b32_runs_total"] - lag["b32_p99_le_35"], "、".join("%.0f" % v for v in lag["b32_p99_over_35"])),
           transform=a.transAxes, fontsize=8.5, color=GRAY, va="top")
    labels = {"hog_0": "绑 8 个核\n周围空闲", "hog_1": "同样的 8 个核上\n16 个死循环", "hog_2": "同一节点的\n别的核上 16 个", "hog_3": "另一节点的\n核上 16 个"}
    rows = [h for h in hog["rows"] if h["tag"] in labels]
    xs = np.arange(len(rows))
    b.bar(xs, [h["api_us_per_token"] for h in rows], 0.55, color=[GRAY, RED, GRAY, GRAY])
    for xi, h in zip(xs, rows):
        b.text(xi, h["api_us_per_token"] + 3, "%.0f" % h["api_us_per_token"], ha="center", va="bottom", fontsize=9)
        b.text(xi, 6, "中位延迟\n%.1f 毫秒" % h["delay_p50"], ha="center", va="bottom", fontsize=8, color="white")
    b.axhline(hog["no_hog_us_per_token"], color=GRAY, lw=0.8, ls=":")
    b.set_xticks(xs); b.set_xticklabels([labels[h["tag"]] for h in rows], fontsize=8.5)
    b.set_ylabel("API server 进程每个 token 的 CPU 时间(微秒)")
    b.set_ylim(0, 225)
    b.set_title("(b) 并发 32 路:核热着的时候,前端每个 token 更省 CPU", fontsize=10)
    b.text(0.98, 0.97, "线程每 22 毫秒醒一次。周围全空时核是凉的:\n醒来那一下要跑 140 微秒的 CPU 时间;\n同样的核上有别的进程把频率占住,只要 40 微秒。",
           transform=b.transAxes, fontsize=8.5, color=GRAY, ha="right", va="top")
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.savefig(os.path.join(OUT, "fig4_lag.png")); plt.close(fig)


def fig5_tokq():
    q = calc["tokq"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    groups = [("1 个分词线程(默认)", q["w1"]), ("4 个分词线程(--renderer-num-workers 4)", q["w4"])]
    xs = np.arange(len(groups))
    for i, (mode, color) in enumerate((("short_only", BLUE), ("long_then_short", ORANGE))):
        vals = [g[1][mode]["ttft_median_ms"] for g in groups]
        ax.bar(xs + (i - 0.5) * 0.36, vals, 0.34, color=color, edgecolor="white", lw=0.5,
               label="32 条短请求,单独发" if mode == "short_only" else "32 条短请求,排在一条 32k 请求后面 2 毫秒发")
        for xi, v, g in zip(xs + (i - 0.5) * 0.36, vals, groups):
            ax.text(xi, v * 1.10, "%.0f 毫秒\n排队 %.0f 毫秒" % (v, g[1][mode]["queue_median_ms"]), ha="center", va="bottom", fontsize=8.3, color=color, linespacing=1.3)
    ax.set_yscale("log"); ax.set_ylim(60, 3e4)
    ax.set_yticks([100, 1000, 10000]); ax.set_yticklabels(["100", "1,000", "10,000"])
    ax.set_xticks(xs); ax.set_xticklabels([g[0].replace("(", "\n(") for g in groups])
    ax.set_ylabel("32 条短请求的首 token 时间,三轮中位数(毫秒,对数)")
    ax.set_title("一条 32k 的请求会不会挡住后面的短请求", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.text(0.98, 0.66, "线程够多时,短请求先分完词、先到引擎,\n比那条长请求早 40 到 171 毫秒,调度器先让它们算完。",
            transform=ax.transAxes, fontsize=8.5, color=GRAY, ha="right", va="top", multialignment="right")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.savefig(os.path.join(OUT, "fig5_tokq.png")); plt.close(fig)


def fig6_cold():
    c = calc["cold"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    idles = ["3", "10", "30", "60"]
    xs = np.arange(len(idles) + 1)
    fin = c["first_ever"]
    v_in = [fin["to_engine_ms"]] + [c["by_idle"][k]["to_engine_ms"] for k in idles]
    v_pre = [fin["prefill_ms"]] + [c["by_idle"][k]["prefill_ms"] for k in idles]
    v_out = [fin["out_ms"]] + [c["by_idle"][k]["out_ms"] for k in idles]
    ax.bar(xs, v_in, 0.55, color=BLUE, label="到引擎之前(前端)")
    ax.bar(xs, v_pre, 0.55, bottom=v_in, color="#c9c9c9", label="prefill(GPU)")
    ax.bar(xs, v_out, 0.55, bottom=np.array(v_in) + np.array(v_pre), color=ORANGE, label="算完到客户端收到")
    for xi, a, p, o in zip(xs, v_in, v_pre, v_out):
        ax.text(xi + 0.31, a + p + o, "%.0f 毫秒" % (a + p + o), ha="left", va="center", fontsize=9)
    clocks = ["210"] + ["/".join(str(v) for v in c["by_idle"][k]["clocks"]) for k in idles]
    ax.set_xticks(xs); ax.set_xticklabels(["服务启动后\n第一条请求", "空闲\n3 秒后", "空闲\n10 秒后", "空闲\n30 秒后", "空闲\n60 秒后"], fontsize=8.8)
    ax.set_ylim(0, 300)
    ax.set_ylabel("首 token 时间(毫秒),提示词 128 个 token")
    ax.set_title("空闲多久之后的第一条请求(柱下的红字是发请求之前显卡的时钟)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.text(0.98, 0.97, "第一条请求前端自己先花了 104 毫秒:很多模块和对象是第一次才建。\n之后 %d 次里有 %d 次到引擎之前只要 8 到 11 毫秒,多出来的全在 prefill:\n显卡空闲 60 秒后时钟掉到 210 MHz,128 个 token 要 %.0f 毫秒。"
            % (c["later_runs"], c["later_to_engine_8_to_11"], c["by_idle"]["60"]["prefill_ms"]),
            transform=ax.transAxes, fontsize=8.5, color=GRAY, ha="right", va="top")
    for xi, cl in zip(xs, clocks):
        ax.text(xi, -46, "%s MHz" % cl, ha="center", va="top", fontsize=8.5, color=RED)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.savefig(os.path.join(OUT, "fig6_cold.png")); plt.close(fig)


for fn in (fig0_flow, fig1_detok, fig2_ttft, fig3_first_token, fig4_lag, fig5_tokq, fig6_cold):
    fn()
    print("ok", fn.__name__)
