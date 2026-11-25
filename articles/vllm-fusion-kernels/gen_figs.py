#!/usr/bin/env python3
"""Figures for the fused-kernel article (vLLM v0.34.0, RTX 4090 D).

fig1_traffic.png   -- bytes moved per token: PyTorch-native vs fused, for three ops (float32 intermediates dominate the native version)
fig2_speed.png     -- measured time per call: eager / Inductor-compiled / vLLM's CUDA kernel / a Triton kernel written from scratch
fig3_roofline.png  -- roofline of the card, with the three ops and decode-time GEMMs placed on it
fig4_flow.png      -- how a call reaches a kernel at run time, and what the compile-time fusion passes do

Measured numbers come from bench/fusion_ops_4090d.json (written by bench/fusion_ops.py).
Run:  python3 gen_figs.py   ->  ./images/
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
BENCH = os.environ.get("FUSION_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
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
bench = json.load(open(os.path.join(BENCH, "fusion_ops_4090d.json")))
calc = json.load(open(os.path.join(BENCH, "calc_summary.json")))
OPS = ["rms_norm", "fused_add_rms_norm", "silu_and_mul"]
OPS_CN = {"rms_norm": "RMSNorm", "fused_add_rms_norm": "加残差 + RMSNorm", "silu_and_mul": "SwiGLU 激活"}
VAR = {"eager": ("朴素写法(PyTorch 逐个算子)", GRAY), "inductor": ("交给 Inductor 自动融合", BLUE),
       "vllm_c": ("vLLM 手写的 CUDA kernel", ORANGE), "triton": ("本文从零写的 Triton kernel", GREEN)}
VAR_CN = {"eager": "朴素", "inductor": "Inductor", "vllm_c": "vLLM CUDA", "triton": "Triton"}


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, fw="normal", color=None, ls="-", lw=1.2, pad=0.25, ls_text=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or ec, fontweight=fw,
            linespacing=ls_text)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw,
                                                    mutation_scale=12, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                                                    shrinkA=0, shrinkB=0))


def med(n, op, k):
    return next(p["ops"][op][k]["us_median"] for p in bench["points"] if p["tokens"] == n)


# ----------------------------------------------------------------------------------------------
def fig1_traffic():
    """One group per op. Native bar = necessary I/O + bf16 intermediates + float32 intermediates.

    Units are "d": d elements x 1 byte. The weight vector (d elements) is left out of both bars.
    """
    d = {"rms_norm": 4096, "fused_add_rms_norm": 4096, "silu_and_mul": 14336}
    parts = {  # (necessary I/O, bf16 intermediates, float32 intermediates, fused total)
        "rms_norm": (4, 4, 28, 4),
        "fused_add_rms_norm": (8, 4, 48, 8),
        "silu_and_mul": (6, 4, 0, 6),
    }
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.set_xlim(0, 138); ax.set_ylim(0, 58); ax.axis("off")
    ax.text(69, 55.6, "每个 token 要搬多少字节:朴素写法的大头是 float32 的中间结果", ha="center", va="center", fontsize=13,
            fontweight="bold")
    bh, x0, scale = 4.4, 32.0, 84.0 / 250.0  # the widest bar is 240 KiB
    for i, op in enumerate(OPS):
        need, b16, f32, fused = parts[op]
        u = d[op] / 1024  # one "d" in KiB
        base = 43.0 - i * 15.5
        for j, (tag, segs, color_tag) in enumerate([
            ("朴素写法", [("need", need, BLUE), ("b16", b16, GOLD), ("f32", f32, "#b8b8b8")], GRAY),
            ("融合之后", [("need", fused, BLUE)], BLUE),
        ]):
            y = base + (4.9 if j == 0 else 0.0)
            x = x0
            for name, units, c in segs:
                w = units * u * scale
                if w > 0:
                    ax.add_patch(Rectangle((x, y), w, bh, fc=c, ec="white", lw=0.8, alpha=0.85))
                    if w > 13:
                        ax.text(x + w / 2, y + bh / 2, f"{units * u:.0f}", ha="center", va="center", fontsize=8.6,
                                color="white" if c != "#b8b8b8" else "#222222", fontweight="bold")
                x += w
            total_kib = (need + b16 + f32) * u if j == 0 else fused * u
            k = next(p["ops"][op]["eager" if j == 0 else "vllm_c"]["kernels"] for p in bench["points"] if p["tokens"] == 1)
            ax.text(2.0, y + bh / 2, tag, ha="left", va="center", fontsize=9.6, color=color_tag, fontweight="bold")
            ax.text(x + 1.6, y + bh / 2, f"共 {total_kib:.0f} KiB,{k} 个 kernel", ha="left", va="center", fontsize=9.4,
                    color="#333333")
        ratio = (need + b16 + f32) / fused
        ax.text(10.0, base + 6.4, OPS_CN[op], ha="left", va="center", fontsize=10.8, fontweight="bold", color="#222222")
        ax.text(10.0, base + 2.3, f"搬运量 {ratio:.1f} 倍", ha="left", va="center", fontsize=9.6, color=RED,
                fontweight="bold")
    lx, ly = 32.0, 3.4
    for k, (c, t) in enumerate([(BLUE, "必要的读写:读输入,写输出"), (GOLD, "bf16 的中间结果:写一遍,再读回来"),
                                ("#b8b8b8", "float32 的中间结果:先升精度再算")]):
        xx = lx + k * 33.0
        ax.add_patch(Rectangle((xx, ly), 2.4, 2.2, fc=c, ec="white", lw=0.8, alpha=0.85))
        ax.text(xx + 3.2, ly + 1.1, t, ha="left", va="center", fontsize=8.8, color="#333333")
    ax.text(136.0, 0.9, "形状按 Llama-3-8B,bfloat16;权重两边都没算。条上的数字是 KiB。", ha="right", va="center", fontsize=8.4,
            color="#777777")
    fig.savefig(f"{OUT}/fig1_traffic.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig2_speed():
    ns = [p["tokens"] for p in bench["points"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0))
    for ax, op in zip(axes, OPS):
        for k in ("eager", "inductor", "vllm_c", "triton"):
            ys = [med(n, op, k) for n in ns]
            lw = 2.6 if k == "eager" else 1.6
            ax.plot(ns, ys, "-o", color=VAR[k][1], lw=lw, ms=4.5, label=VAR[k][0])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns]); ax.minorticks_off()
        ax.set_xlabel("这一步的 token 数 N")
        ax.set_title(OPS_CN[op], fontsize=11.5, fontweight="bold", loc="left")
        sp = med(ns[-1], op, "eager") / med(ns[-1], op, "vllm_c")
        sp1 = med(ns[0], op, "eager") / med(ns[0], op, "vllm_c")
        ax.text(0.97, 0.93, f"N = 8192:快 {sp:.1f} 倍\nN = 1:快 {sp1:.1f} 倍", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.4, color="#333333", linespacing=1.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="major", color="#e6e6e6", lw=0.7)
    axes[0].set_ylabel("一次调用的耗时(微秒,对数轴)")
    axes[0].legend(loc="lower right", fontsize=8.8)
    fig.suptitle("实测:融合之后快多少(RTX 4090D,CUDA Graph 里连着调,每次的微秒数)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{OUT}/fig2_speed.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig3_roofline():
    bw = bench["copy_bandwidth_gbps"] * 1e9
    peak = bench["gemm_bf16_peak_tflops"] * 1e12
    ridge = peak / bw
    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    xs = np.logspace(-0.3, 4.0, 400)
    ys = np.minimum(xs * bw, peak) / 1e12
    ax.plot(xs, ys, color="#333333", lw=2.2, label="这张卡的上限")
    ax.fill_between(xs, ys, 1e-3, where=xs < ridge, color=BLUE, alpha=0.07)
    ax.fill_between(xs, ys, 1e-3, where=xs >= ridge, color=ORANGE, alpha=0.07)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 1e4); ax.set_ylim(0.25, 500)
    ax.axvline(ridge, color=GRAY, lw=1.0, ls="--")
    ax.text(ridge * 1.12, 0.42, f"拐点:每个字节 {ridge:.0f} 次运算", ha="left", va="bottom", fontsize=9.4, color="#444444")
    ax.text(2.2, 220, "左边:带宽受限\n时间看搬多少字节", ha="left", va="top", fontsize=10.2, color=BLUE, fontweight="bold",
            linespacing=1.6)
    ax.text(9000, 430, "右边:算力受限\n时间看多少次运算", ha="right", va="top", fontsize=10.2, color=ORANGE, fontweight="bold",
            linespacing=1.6)
    flops = {"rms_norm": 4, "fused_add_rms_norm": 5, "silu_and_mul": 8}
    pts = []
    for op in OPS:
        d = 4096 if op != "silu_and_mul" else 14336
        n = 8192
        ai = flops[op] / calc["traffic"][op][1]
        achieved = flops[op] * d * n / (med(n, op, "vllm_c") * 1e-6) / 1e12
        pts.append((OPS_CN[op], ai, achieved, "o", GREEN))
    for n in (1, 32, 256):
        g = bench["gemm_decode"][str(n)]
        pts.append((f"解码的矩阵乘,N = {n}", g["flop_per_byte"], g["tflops"], "s", ORANGE))
    pts.append(("预填充的矩阵乘,8192 阶", 2 * 8192 ** 3 / (3 * 8192 ** 2 * 2), bench["gemm_bf16_peak_tflops"], "D", ORANGE))
    # (dx, dy, horizontal alignment): labels go below the sloped line or into empty space
    offs = {"RMSNorm": (9, -13, "left"), "加残差 + RMSNorm": (7, -22, "left"), "SwiGLU 激活": (-9, 13, "right"),
            "解码的矩阵乘,N = 1": (13, -13, "left"), "解码的矩阵乘,N = 32": (-10, 11, "right"),
            "解码的矩阵乘,N = 256": (10, -16, "left"), "预填充的矩阵乘,8192 阶": (-10, -19, "right")}
    for name, ai, achieved, mk, c in pts:
        ax.plot([ai], [achieved], mk, color=c, ms=8, mec="white", mew=1.0)
        dx, dy, ha = offs[name]
        ax.annotate(name, (ai, achieved), xytext=(dx, dy), textcoords="offset points", fontsize=9.2, color=c,
                    ha=ha, va="center", fontweight="bold")
    ax.set_xlabel("算术强度:每从显存搬一个字节,做多少次浮点运算(对数轴)")
    ax.set_ylabel("达到的算力(TFLOPS,对数轴)")
    ax.set_title(f"roofline:三个小算子全落在带宽受限的一侧(拷贝带宽 {bw / 1e9:.0f} GB/s,bf16 峰值 {peak / 1e12:.0f} TFLOPS)",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#e9e9e9", lw=0.7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_roofline.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig4_flow():
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_xlim(0, 138); ax.set_ylim(20, 92); ax.axis("off")
    ax.text(69, 89.6, "一个融合算子是怎么被调到的:vLLM 0.34", ha="center", va="center", fontsize=13, fontweight="bold")

    # ---- lane 1: run time
    ax.text(3, 84.2, "运行时:一次调用怎么走到 kernel", ha="left", va="center", fontsize=11.0, fontweight="bold", color=ORANGE)
    W, H, Y = 24.0, 14.0, 64.0
    xs = [3.0, 30.2, 57.4, 84.6, 111.8]
    labels = [
        ("模型代码", "self.input_layernorm(\nhidden_states, residual)\nmodels/llama.py:324", "#fdf6f0", ORANGE),
        ("层的 forward", "RMSNorm.forward_native\n调 ir.ops.fused_add_rms_norm\nlayers/layernorm.py:79", "#fdf6f0", ORANGE),
        ("IR 算子", "签名 + PyTorch 写的参考实现\nir/ops/layernorm.py:43", "#f2f8f2", GREEN),
        ("按优先级挑实现", "IrOp.dispatch\n取优先级里第一个\n支持这组参数的\nvllm/ir/op.py:321", "#f2f8f2", GREEN),
        ("实现", "native / vllm_c /\noink / aiter …\nregister_impl 登记", "#f2f8f2", GREEN),
    ]
    for x, (t, body, fc, ec) in zip(xs, labels):
        box(ax, x, Y, W, H, "", fc, ec, pad=0.5)
        ax.text(x + W / 2, Y + H - 3.0, t, ha="center", va="center", fontsize=10.2, fontweight="bold", color=ec)
        ax.text(x + W / 2, Y + 4.9, body, ha="center", va="center", fontsize=8.2, color="#333333", linespacing=1.55)
    for i in range(4):
        arrow(ax, xs[i] + W + 0.3, Y + H / 2, xs[i + 1] - 0.3, Y + H / 2, ORANGE if i < 2 else GREEN, lw=1.6)
    ax.text(3, 60.6, "优先级在启动时定好(platforms/cuda.py:753):用 Inductor 编译时是 [native],不编译时是 [vllm_c, native]。",
            ha="left", va="center", fontsize=8.8, color="#444444")

    # ---- lane 2: compile time
    ax.text(3, 51.6, "编译期:torch.compile 之后,图上发生了什么", ha="left", va="center", fontsize=11.0, fontweight="bold", color=BLUE)
    Y2 = 32.0
    labels2 = [
        ("Inductor 拿到的图", "torch.compile 追踪出来的 fx 图,\n里面还留着 IR 算子", "#f0f6fb", BLUE),
        ("vLLM 的融合规则", "pattern:要找的子图\nreplacement:换成什么\ncompilation/passes/fusion/", "#f0f6fb", BLUE),
        ("下降", "每个 IR 算子按优先级\n换成选中的实现\npasses/ir/lowering_pass.py", "#f0f6fb", BLUE),
        ("Inductor 生成代码", "native:摊开成 PyTorch 算子,和前后的\n小算子一起生成 Triton kernel;\nvllm_c:留成自定义算子,Inductor 不碰",
         "#f0f6fb", BLUE),
        ("CUDA Graph", "录成一张图\n每步回放", "#f0f6fb", BLUE),
    ]
    for x, (t, body, fc, ec) in zip(xs, labels2):
        box(ax, x, Y2, W, H, "", fc, ec, pad=0.5)
        ax.text(x + W / 2, Y2 + H - 3.0, t, ha="center", va="center", fontsize=10.2, fontweight="bold", color=ec)
        fs = 7.6 if t == "Inductor 生成代码" else 8.2
        ax.text(x + W / 2, Y2 + 4.9, body, ha="center", va="center", fontsize=fs, color="#333333", linespacing=1.55)
    for i in range(4):
        arrow(ax, xs[i] + W + 0.3, Y2 + H / 2, xs[i + 1] - 0.3, Y2 + H / 2, BLUE, lw=1.6)
    ax.text(3, 28.2, "默认配置下,归一化、激活这些规则是关着的(注释:否则交给 Inductor 融合);手写算子或量化算子打开时才开。",
            ha="left", va="center", fontsize=8.8, color="#444444")
    ax.text(3, 23.6, "所以同一个 RMSNorm 有三副面孔:PyTorch 写的原生实现,Inductor 生成的融合 kernel,手写的 CUDA kernel。",
            ha="left", va="center", fontsize=9.6, color="#222222", fontweight="bold")
    fig.savefig(f"{OUT}/fig4_flow.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_traffic(); fig2_speed(); fig3_roofline(); fig4_flow()
    print("done ->", OUT)
