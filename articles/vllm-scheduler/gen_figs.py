#!/usr/bin/env python3
"""Figures for the scheduler article (vLLM v0.34.0, Llama-3.1-8B-Instruct, one RTX 4090 D).

fig1_batching.png   -- requests running at each moment: "finish the whole batch, then the next" vs "refill as soon as one finishes"
fig2_step_time.png  -- decode step time vs batch size, aggregate throughput, and the scheduler's own CPU time per step
fig3_budget.png     -- what each step carries around one 8192-token prompt, for token budgets 512 / 2048 / 8192
fig4_tradeoff.png   -- time to first token and the worst inter-token gap of the decoding requests, vs token budget
fig5_preempt.png    -- (a) KV capacity caps concurrency  (b) one batch of 64 requests on a small KV cache: 32 get preempted
fig6_watermark.png  -- sustained pressure on a small KV cache, watermark off vs 10%
fig0_flow.png       -- what schedule() does each step

Every number comes from bench/calc_summary.json (written by calc.py) and bench/results/*.json.
Run:  python3 gen_figs.py   ->  ./images/      (SCHED_BENCH overrides the bench directory)
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
BENCH = os.environ.get("SCHED_BENCH", os.path.join(HERE, "bench"))
RES = os.path.join(BENCH, "results")
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
batching = json.load(open(os.path.join(RES, "batching_summary.json")))
mixed = {b: json.load(open(os.path.join(RES, f"mixed_{b}_summary.json"))) for b in (512, 2048, 8192)}
capacity = json.load(open(os.path.join(RES, "capacity_summary.json")))
preempt = json.load(open(os.path.join(RES, "preempt_summary.json")))
sustain = {"small": json.load(open(os.path.join(RES, "sustain_summary.json"))),
           "small_watermark": json.load(open(os.path.join(RES, "sustain_wm_summary.json")))}


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, fw="normal", color=None, ls="-", lw=1.2, pad=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or ec, fontweight=fw,
            linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw,
                                                    mutation_scale=12, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                                                    shrinkA=0, shrinkB=0))


# ----------------------------------------------------------------------------------------------
def fig0_flow():
    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    ax.set_xlim(0, 138); ax.set_ylim(0, 64); ax.axis("off")
    ax.text(69, 62.5, "一步里 schedule() 做的事(vLLM 0.34.0,vllm/v1/core/sched/scheduler.py)", ha="center", va="center",
            fontsize=11)
    box(ax, 3, 44, 24, 10, "引擎主循环\nstep_with_batch_queue()", "#eef4fa", BLUE, fs=9.5, fw="bold")
    box(ax, 36, 44, 30, 10, "schedule()\n两本账:token 预算和条数上限", "#eef4fa", BLUE, fs=9.5, fw="bold")
    box(ax, 75, 50, 28, 9, "模型执行 execute_model()", "#f6f6f6", GRAY, fs=9.5)
    box(ax, 75, 37, 28, 9, "update_from_output()", "#f6f6f6", GRAY, fs=9.5)
    box(ax, 110, 37, 26, 22, "下一步的 schedule()\n在 GPU 还在算上一步时\n就已经排好了\n(异步调度)", "#fdf3ec", ORANGE, fs=9.0)
    arrow(ax, 27, 49, 36, 49, BLUE); arrow(ax, 66, 49, 75, 54.5, GRAY)
    arrow(ax, 89, 50, 89, 46, GRAY); arrow(ax, 103, 41.5, 110, 41.5, GRAY)
    arrow(ax, 123, 59, 51, 54, ORANGE, ls="--", rad=0.12)
    # 两个循环
    box(ax, 3, 20, 66, 18, "", "#ffffff", BLUE, lw=1.2, ls="--")
    ax.text(36, 35.6, "循环一:RUNNING 队列,每条请求都过一遍", ha="center", va="center", fontsize=9.5, color=BLUE)
    ax.text(36, 29.5, "看它已经算到哪、一共要算多少,差几个这一步就排几个",
            ha="center", va="center", fontsize=9.0)
    ax.text(36, 25.5, "向块池要块。要不到:抢掉 RUNNING 的最后一条,再试", ha="center", va="center", fontsize=9.0)
    ax.text(36, 21.8, "每排进一个 token,token 预算减 1", ha="center", va="center", fontsize=9.0)
    box(ax, 3, 2, 66, 15.5, "", "#ffffff", GREEN, lw=1.2, ls="--")
    ax.text(36, 14.9, "循环二:WAITING 队列,先到先服务(这一步没发生过抢占才进)", ha="center", va="center",
            fontsize=9.5, color=GREEN)
    ax.text(36, 10.3, "查前缀缓存,命中多少就少算多少;剩下的砍到预算以内", ha="center", va="center", fontsize=9.0)
    ax.text(36, 6.6, "向块池要块。要不到:停下,它留在队头等下一步", ha="center", va="center", fontsize=9.0)
    ax.text(36, 3.3, "排进去的请求挪进 RUNNING", ha="center", va="center", fontsize=9.0)
    arrow(ax, 51, 44, 51, 38.2, BLUE)
    arrow(ax, 51, 20, 51, 17.5, GREEN)
    ax.text(72, 24, "两个循环都在 schedule() 里面。\n排完把结果交给模型执行:\n每条请求这一步算哪几个 token,\n用的是哪几块 KV。",
            fontsize=9.0, color=GRAY, va="center", linespacing=1.6)
    fig.savefig(os.path.join(OUT, "fig0_flow.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig1_batching():
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 5.6), sharex=True)
    runs = {r["mode"]: r for r in batching["runs"] if r["round"] == 0}
    agg = batching["agg"]
    titles = {
        "static": "整批走完再放下一批:每波 32 条,批里在跑的条数一路往下掉",
        "continuous": "完成一个补一个:空出一个位置立刻补上,批里一直接近满的",
    }
    for ax, mode, color in ((axes[0], "static", RED), (axes[1], "continuous", BLUE)):
        s = runs[mode]["series"]
        t = np.array([x[0] for x in s]) / 1e3
        run = np.array([x[1] for x in s])
        ax.fill_between(t, 0, run, step="post", color=color, alpha=0.18)
        ax.step(t, run, where="post", color=color, lw=1.3)
        ax.set_ylim(0, 40); ax.set_yticks([0, 8, 16, 24, 32])
        a = agg[mode]
        ax.text(0.99, 0.93,
                f"{a['wall_s']:.1f} 秒跑完,{a['tokens_per_s']:.0f} token/秒;每步平均 {calc['batching']['mean_running'][mode]:.1f} 条在跑",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
        ax.set_title(titles[mode], loc="left", fontsize=10)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
    axes[1].set_xlabel("时间(秒)。同一批 256 条请求,提示词 128 token,输出 35 到 768 个 token 不等,总共 57,248 个")
    axes[1].set_xlim(0, 100.5)
    for ax in axes:
        ax.set_ylabel("同时在跑的请求")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig1_batching.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig2_step_time():
    d = calc["decode"]
    bs = sorted(int(b) for b in d["t_short_ms"])
    t_short = [d["t_short_ms"][str(b)] for b in bs]
    bl = sorted(int(b) for b in d["t_long_ms"])
    t_long = [d["t_long_ms"][str(b)] for b in bl]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    ax = axes[0]
    ax.plot(bs, t_short, "o-", color=BLUE, lw=1.6, ms=4.5, label="上下文 64 到 120 个 token")
    ax.plot(bl, t_long, "s-", color=ORANGE, lw=1.4, ms=4, label="上下文 256 到 576 个 token")
    ax.axhline(d["t1_ms"], color=GRAY, ls="--", lw=1.0)
    ax.text(0.9, d["t1_ms"] - 0.35, f"batch 1:{d['t1_ms']:.2f} 毫秒,正好是读一遍权重 {calc['model']['weights_read_per_decode_step_gb']:.2f} GB 的时间",
            ha="left", va="top", fontsize=8.8, color=GRAY)
    ax.annotate(f"batch 128:{d['t_short_ms']['128']:.1f} 毫秒", xy=(128, d["t_short_ms"]["128"]),
                xytext=(40, 25.5), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.set_xscale("log", base=2); ax.set_xlim(0.85, 300)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_ylim(13.6, 38); ax.set_xlabel("batch 大小(同时在解码的请求条数)"); ax.set_ylabel("一步的时间(毫秒)")
    ax.set_title("(a) 一步 decode 的时间", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.8); ax.grid(color="#e6e6e6", lw=0.6)

    ax = axes[1]
    thr = [d["throughput_tok_s"][str(b)] for b in bs]
    per = [d["per_request_tok_s"][str(b)] for b in bs]
    ax.plot(bs, thr, "o-", color=GREEN, lw=1.6, ms=4.5, label="整卡每秒出多少 token")
    ax.set_xscale("log", base=2); ax.set_xlim(0.85, 300)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("batch 大小"); ax.set_ylabel("整卡吞吐(token/秒)", color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    ax2 = ax.twinx()
    ax2.plot(bs, per, "s--", color=PURP, lw=1.4, ms=4, label="每条请求每秒出多少 token")
    ax2.set_ylabel("单条请求的速度(token/秒)", color=PURP); ax2.tick_params(axis="y", colors=PURP)
    ax.set_ylim(0, 8200); ax2.set_ylim(0, 75)
    ax.text(0.985, 0.90, f"batch 256:每秒 {thr[-1]:,.0f} 个 token", transform=ax.transAxes, ha="right", va="center",
            fontsize=9, color=GREEN)
    ax.text(0.985, 0.20, f"每条请求每秒 {per[-1]:.0f} 个", transform=ax.transAxes, ha="right", va="center", fontsize=9, color=PURP)
    ax.text(0.04, 0.13, f"batch 1:每秒 {thr[0]:.0f} 个 token", transform=ax.transAxes, ha="left", va="center", fontsize=9, color=GREEN)
    ax.text(0.04, 0.90, f"每条请求每秒 {per[0]:.0f} 个", transform=ax.transAxes, ha="left", va="center", fontsize=9, color=PURP)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.8)
    ax.set_title("(b) 同一张卡,换来的吞吐", loc="left", fontsize=10); ax.grid(color="#e6e6e6", lw=0.6)

    ax = axes[2]
    sched = [d["sched_ms"][str(b)] for b in bs]
    upd = [d["update_ms"][str(b)] for b in bs]
    x = np.arange(len(bs))
    ax.bar(x, sched, color=PURP, alpha=0.85, width=0.7, label="schedule()")
    ax.bar(x, upd, bottom=sched, color=GOLD, alpha=0.85, width=0.7, label="update_from_output()")
    ax.plot(x, t_short, "o-", color=BLUE, lw=1.5, ms=4, label="这一步 GPU 的时间")
    ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=8.3)
    ax.set_xlabel("batch 大小(CUDA Graph 抓取过的大小)"); ax.set_ylabel("毫秒")
    i = bs.index(256)
    ax.annotate(f"占一步的 {d['cpu_share_of_step']['256'] * 100:.0f}%", xy=(i, sched[i] + upd[i]), xytext=(i - 5.4, 30.5),
                fontsize=9, color=PURP, arrowprops=dict(arrowstyle="-", color=PURP, lw=0.8))
    ax.set_ylim(0, 42)
    ax.legend(loc="upper left", fontsize=8.6); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_title("(c) 调度器自己的 CPU 时间", loc="left", fontsize=10)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(OUT, "fig2_step_time.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig3_budget():
    fig, axes = plt.subplots(3, 1, figsize=(13.8, 8.6))
    for ax, b, ymax_t, ymax_ms in ((axes[0], 512, 600, 100), (axes[1], 2048, 2400, 320), (axes[2], 8192, 9200, 1050)):
        s = mixed[b]
        tt = {x[0]: x[1] for x in s["series"]}
        pre = {x[0]: x[2] for x in s["series"]}     # 每一项:[步号, 时刻, prefill token, decode token, 在跑的, 等待的]
        dec = {x[0]: x[3] for x in s["series"]}
        r = s["fg"][1]
        k0 = r["admit_step"]
        n = r["n_prefill_steps"]
        ks = list(range(k0 - 6, k0 + n + 7))
        xs = np.array([k - k0 for k in ks])
        p = np.array([pre.get(k, 0) for k in ks]); dd = np.array([dec.get(k, 0) for k in ks])
        dt = np.array([tt[k] - tt[k - 1] for k in ks])
        ax.bar(xs, dd, color=BLUE, width=0.72, label="decode token:12 条后台请求各 1 个")
        ax.bar(xs, p, bottom=dd, color=ORANGE, width=0.72, label="prefill token:这一步算的长提示词")
        ax.axhline(b, color=GRAY, ls="--", lw=1.0)
        ax.text(xs[0] + 0.3, b * 1.02, f"预算 {b:,}:一步最多排这么多 token", ha="left", va="bottom", fontsize=8.8, color=GRAY)
        ax.set_ylim(0, ymax_t); ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        ax.set_ylabel("这一步排了几个 token")
        ax2 = ax.twinx()
        ax2.plot(xs, dt, "o-", color=RED, lw=1.5, ms=3.8, label="这一步的时间(正在解码的请求相邻两个 token 的间隔)")
        ax2.set_ylim(0, ymax_ms); ax2.set_ylabel("这一步的时间(毫秒)", color=RED); ax2.tick_params(axis="y", colors=RED)
        ax.axvline(0, color=GREEN, lw=1.2)
        ax.text(0.3, ymax_t * 0.985, "长提示词被接进来", color=GREEN, fontsize=9, va="top")
        e = calc["mixed"]["by_budget"][str(b)]
        ax.set_title(f"max_num_batched_tokens = {b:,}:切了 {n} 块,首 token {e['engine_ttft_ms']['mean'] / 1e3:.2f} 秒,"
                     f"最慢的一步 {e['itl_max_ms']:.0f} 毫秒", loc="left", fontsize=10)
        lines = [matplotlib.patches.Patch(color=BLUE), matplotlib.patches.Patch(color=ORANGE), ax2.get_lines()[0]]
        ax.legend(lines, ["decode token:12 条后台请求各 1 个", "prefill token:这一步算的长提示词", "这一步的时间"],
                  loc="center left", bbox_to_anchor=(0.005, 0.40), fontsize=8.6)
        ax.set_xticks([k for k in xs if k % (5 if b == 512 else 2) == 0])
    axes[2].set_xlabel("步(以长提示词被接进来的那一步为 0)。一条 8192 个 token 的提示词进来,同时有 12 条请求在解码")
    fig.tight_layout(h_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig3_budget.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig4_tradeoff():
    by = calc["mixed"]["by_budget"]
    bs = (512, 2048, 8192)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.3))
    ax = axes[0]
    x = np.arange(3)
    eng = [by[str(b)]["engine_ttft_ms"]["mean"] for b in bs]
    cli = [by[str(b)]["client_ttft_ms"]["mean"] for b in bs]
    ax.bar(x - 0.18, eng, width=0.36, color=BLUE, label="引擎内部:从调度器接进来,到第一个 token 算出来")
    ax.bar(x + 0.18, cli, width=0.36, color=PURP, alpha=0.8, label="客户端看到的(多了 HTTP 和前端)")
    for i, v in enumerate(eng):
        ax.text(i - 0.18, v + 12, f"{v / 1e3:.2f} 秒", ha="center", fontsize=9)
        ax.text(i + 0.18, cli[i] + 12, f"{cli[i] / 1e3:.2f}", ha="center", fontsize=9, color=PURP)
    ax.set_xticks(x); ax.set_xticklabels([f"{b:,}" for b in bs])
    ax.set_ylim(0, 1500); ax.set_xlabel("max_num_batched_tokens"); ax.set_ylabel("首 token 时间(毫秒)")
    ax.set_title(f"(a) 一条 {calc['mixed']['fg_prompt']:,} token 的提示词,多久出第一个 token", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.6); ax.grid(axis="y", color="#e6e6e6", lw=0.6)

    ax = axes[1]
    mx = [by[str(b)]["itl_max_ms"] for b in bs]
    p50 = [by[str(b)]["itl_p50_ms"] for b in bs]
    ax.bar(x, mx, width=0.5, color=RED, label="最慢的一步(正在解码的请求,相邻两个 token 最大的间隔)")
    ax.plot(x, p50, "o--", color=GRAY, lw=1.3, ms=5, label="中位数(基本就是纯解码那一步)")
    for i, v in enumerate(mx):
        ax.text(i, v + 18, f"{v:.0f} 毫秒\n≈ {v / calc['mixed']['decode_only_ms']:.0f} 步 decode", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{b:,}" for b in bs])
    ax.set_ylim(0, 1150); ax.set_xlabel("max_num_batched_tokens"); ax.set_ylabel("毫秒")
    ax.set_title("(b) 同一时刻正在解码的 12 条请求,最长卡了多久", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.6); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "fig4_tradeoff.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig5_preempt():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6))
    ax = axes[0]
    s = capacity["runs"]["128"][0]["series"]
    t = np.array([x[0] for x in s]) / 1e3
    run = np.array([x[1] for x in s]); wait = np.array([x[2] for x in s])
    ax.step(t, run, where="post", color=BLUE, lw=1.5, label="在跑的")
    ax.step(t, wait, where="post", color=ORANGE, lw=1.4, label="等待的")
    cap_tok = calc["model"]["capacity_tokens"]
    ax.axhline(cap_tok // 576, color=RED, ls="--", lw=1.0)
    ax.text(0.6, cap_tok / 576 + 2.5, f"KV 容量 {cap_tok:,} token ÷ 每条 576 token = {cap_tok / 576:.1f} 条", color=RED, fontsize=9)
    c = calc["capacity"]["128"]
    ax.text(0.99, 0.93, f"一次发 128 条,每条 256 + 320 个 token\n抢占 {c['preemptions_both_rounds'][0]} 次,白算 {c['lost_tokens']:,} 个 token",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.2)
    ax.set_xlim(0, 20); ax.set_ylim(0, 150)
    ax.set_xlabel("时间(秒)"); ax.set_ylabel("请求条数")
    ax.set_title("(a) KV 装不下时,同时在跑的条数跟着容量走", loc="left", fontsize=10)
    ax.text(12.9, 75.5, "在跑的", color=BLUE, fontsize=9, ha="center")
    ax.text(9.0, 27, "等待的(被踢出去的)", color=ORANGE, fontsize=9, ha="center")
    ax.grid(color="#e6e6e6", lw=0.6)

    ax = axes[1]
    s = preempt["series"]
    t = np.array([x[0] for x in s]) / 1e3
    run = np.array([x[1] for x in s]); wait = np.array([x[2] for x in s])
    ax.step(t, run, where="post", color=BLUE, lw=1.5, label="在跑的")
    ax.step(t, wait, where="post", color=ORANGE, lw=1.4, label="等待的(被踢出去的)")
    ax.set_xlim(0, 19.5); ax.set_ylim(0, 110)
    ax.set_xlabel("时间(秒)"); ax.set_ylabel("请求条数")
    ax2 = ax.twinx()
    ax2.step(t, [x[5] * 100 for x in s], where="post", color=GRAY, lw=1.1, ls=":")
    ax2.set_ylim(0, 125); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.set_ylabel("KV 缓存使用率(%)", color=GRAY)
    ax2.tick_params(axis="y", colors=GRAY)
    P = calc["preempt"]
    for bt, n in P["bursts"]:
        ax.axvline(bt, color=RED, lw=1.0, alpha=0.6)
    for i, (bt, n) in enumerate(P["bursts"]):
        ax.text(bt + 0.12, 78 if i % 2 == 0 else 66, f"踢 {n} 条", color=RED, fontsize=8.8, ha="left", va="center")
    ax.text(0.3, 44, f"{P['bursts'][0][0]:.1f} 秒 KV 写满,开始抢占\n之后每隔约 {P['burst_interval_s']:.2f} 秒一轮,共 {len(P['bursts'])} 轮",
            color=RED, fontsize=9, va="top")
    ax.text(0.99, 0.97, f"KV 缓存调小到 {P['small_tokens']:,} token\n64 条请求各要 {P['per_request_tokens']} token,只放得下 {P['fit_requests']} 条\n"
                        f"抢占 {P['preemptions']} 次,白算 {P['lost_tokens']:,} 个 token",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.2)
    ax.text(14.6, 20, "在跑的", color=BLUE, fontsize=9, ha="center")
    ax.text(12.0, 4, "等待的", color=ORANGE, fontsize=9, ha="left")
    ax.text(12.0, 14, "KV 使用率(灰色点线,右轴)", color=GRAY, fontsize=8.8, ha="left", va="center")
    ax.set_title("(b) 64 条一起进来:后进来的 32 条被踢,等前面的跑完再重来", loc="left", fontsize=10)
    ax.grid(color="#e6e6e6", lw=0.6)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "fig5_preempt.png")); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig6_watermark():
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 6.0), sharex=True, sharey=True)
    su = calc["sustain"]
    for ax, key, color, title in (
        (axes[0], "small", RED, "不开水位(默认 0)"),
        (axes[1], "small_watermark", GREEN, "水位 10%(--watermark 0.1)"),
    ):
        s = sustain[key]["series"]
        t = np.array([x[0] for x in s]) / 1e3
        run = np.array([x[1] for x in s]); wait = np.array([x[2] for x in s])
        ax.fill_between(t, 0, run, step="post", color=BLUE, alpha=0.16)
        ax.step(t, run, where="post", color=BLUE, lw=1.3, label="在跑的")
        ax.step(t, wait, where="post", color=ORANGE, lw=1.4, label="等待的")
        ax.set_ylim(0, 150); ax.set_yticks([0, 32, 64, 96, 128])
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, [x[5] * 100 for x in s], step="post", color=GRAY, alpha=0.16)
        ax2.set_ylim(0, 195); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.set_ylabel("KV 使用率(%)")
        a = su[key]
        times = sorted(p[0] / 1e3 for v in sustain[key]["requests"].values() for p in v["preempts"])
        ax.plot(times, [3] * len(times), "|", color=RED, ms=9, mew=1.3, label=f"一次抢占(共 {len(times)} 次)")
        ax.text(0.985, 0.80, f"{a['requests']} 条请求,抢占 {a['preemptions']} 次,{a['requests_preempted_2plus']} 条被踢两次以上,"
                             f"白算 {a['lost_tokens']:,} 个 token;每秒 {a['tokens_per_s']:.0f} 个 token",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.2)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(color="#e6e6e6", lw=0.6)
    lines = axes[0].get_lines()
    axes[0].legend(lines, ["在跑的", "等待的", "一次抢占(底部的红色刻度)"], loc="upper left", fontsize=8.8)
    axes[1].set_xlabel("时间(秒)。64 个客户端各自做完一条立刻发下一条,压 30 秒;KV 缓存 20,992 token")
    axes[1].set_xlim(0, 40)
    for ax in axes:
        ax.set_ylabel("请求条数")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig6_watermark.png")); plt.close(fig)


if __name__ == "__main__":
    for name in ("fig0_flow", "fig1_batching", "fig2_step_time", "fig3_budget", "fig4_tradeoff", "fig5_preempt", "fig6_watermark"):
        globals()[name]()
        print("wrote", name)
