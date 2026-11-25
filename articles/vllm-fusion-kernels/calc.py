#!/usr/bin/env python3
"""Every number quoted in the fused-kernel article, computed in one place.

Shapes follow Llama-3-8B: hidden 4096, MLP intermediate 14336, bfloat16 (2 bytes a number).
A fused kernel keeps its intermediates in registers, so it only pays for reading the inputs
and writing the outputs. The PyTorch-native version writes every intermediate to memory
and reads it back; on top of that it upcasts to float32 (4 bytes a number) before the math.

Measured numbers come from bench/fusion_ops_4090d.json (RTX 4090 D, vLLM 0.34.0's kernels).
Run:  python3 calc.py
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN, INTER, LAYERS = 4096, 14336, 32
BF16, FP32 = 2, 4
KiB = 1024
bench = json.load(open(os.path.join(HERE, "bench", "fusion_ops_4090d.json")))


def traffic(steps):
    """steps: list of (name, read_bytes, write_bytes) per row, in units of d elements."""
    r = sum(s[1] for s in steps)
    w = sum(s[2] for s in steps)
    return r, w, r + w


# ---- bytes per row, in units of d elements x 1 byte. Each step is (name, read, write, read dtype, write dtype);
# the dtype is "-" when nothing is transferred. The weight vector (d elements) is left out of every count.
B, F = "bf16", "float32"
OPS = {
    "rms_norm": {
        "native": [
            ("x.to(float32)", BF16, FP32, B, F),
            ("x.pow(2)", FP32, FP32, F, F),
            ("mean(-1)", FP32, 0, F, "-"),
            ("x * rsqrt(var + eps)", FP32, FP32, F, F),
            ("x.to(weight.dtype)", FP32, BF16, F, B),
            ("x * weight", BF16, BF16, B, B),
        ],
        "fused": [("read x", BF16, 0, B, "-"), ("write y", 0, BF16, "-", B)],
        "d": HIDDEN,
        "flops": 4,  # squares, sum, scale, weight: four passes of d multiply-adds
    },
    "fused_add_rms_norm": {
        "native": [
            ("x.to(float32)", BF16, FP32, B, F),
            ("residual.to(float32)", BF16, FP32, B, F),
            ("x + residual", FP32 + FP32, FP32, F, F),
            ("new residual .to(bf16)", FP32, BF16, F, B),
            ("x.pow(2)", FP32, FP32, F, F),
            ("mean(-1)", FP32, 0, F, "-"),
            ("x * rsqrt(var + eps)", FP32, FP32, F, F),
            ("x.to(weight.dtype)", FP32, BF16, F, B),
            ("x * weight", BF16, BF16, B, B),
        ],
        "fused": [("read x", BF16, 0, B, "-"), ("read residual", BF16, 0, B, "-"),
                  ("write norm", 0, BF16, "-", B), ("write residual", 0, BF16, "-", B)],
        "d": HIDDEN,
        "flops": 5,  # add, squares, sum, scale, weight
    },
    "silu_and_mul": {
        "native": [
            ("F.silu(x[..., :d])", BF16, BF16, B, B),
            ("* x[..., d:]", BF16 + BF16, BF16, B, B),
        ],
        "fused": [("read gate", BF16, 0, B, "-"), ("read up", BF16, 0, B, "-"), ("write y", 0, BF16, "-", B)],
        "d": INTER,
        "flops": 8,  # sigmoid (exp, add, div) and the multiply, per output element
    },
}
PARTS = {}  # op -> (necessary I/O, bf16 intermediates, float32 intermediates, fused total), for the traffic figure

BW = bench["copy_bandwidth_gbps"] * 1e9
PEAK = bench["gemm_bf16_peak_tflops"] * 1e12
print(f"== 0. 这张卡:{bench['gpu']},拷贝带宽 {BW / 1e9:.1f} GB/s,bf16 矩阵乘峰值 {PEAK / 1e12:.1f} TFLOPS")
ridge = PEAK / BW
print(f"   拐点:每个字节 {ridge:.1f} 次浮点运算。低于它,时间由搬字节决定;高于它,由算力决定。")

print("\n== 1. 每个算子,每个 token 要搬多少字节(朴素写法 vs 融合)")
TRAFFIC = {}
for name, op in OPS.items():
    r1 = sum(s[1] for s in op["native"]); w1 = sum(s[2] for s in op["native"]); t1 = r1 + w1
    f32 = sum(s[1] for s in op["native"] if s[3] == F) + sum(s[2] for s in op["native"] if s[4] == F)
    t2 = sum(s[1] + s[2] for s in op["fused"])
    d = op["d"]
    TRAFFIC[name] = (t1, t2)
    PARTS[name] = (t2, t1 - t2 - f32, f32, t2)
    print(f"   {name:18s} 朴素:读 {r1}d 写 {w1}d,共 {t1}d = {t1 * d / KiB:.0f} KiB;融合:共 {t2}d = {t2 * d / KiB:.0f} KiB;比值 {t1 / t2:.2f}")
    for s_ in op["native"]:
        print(f"        {s_[0]:26s} 读 {s_[1]:2d}d  写 {s_[2]:2d}d  ({s_[3]} 进,{s_[4]} 出)")
    b16_mid = t1 - t2 - f32
    print(f"        其中 float32 中间结果 {f32}d({f32 / t1:.0%}),bf16 中间结果 {b16_mid}d,必要的读写 {t2}d(权重两边都不算)")

print("\n== 2. 实测(4090D):融合版比朴素版快多少,和搬运量的比值对不对得上")
SPEED = {}
for p in bench["points"]:
    n = p["tokens"]
    row = []
    for name, info in p["ops"].items():
        e = info["eager"]["us_median"]
        sp = {k: e / info[k]["us_median"] for k in ("inductor", "vllm_c", "triton")}
        SPEED[(name, n)] = sp
        row.append(f"{name}: 朴素 {e:.1f}us({info['eager']['kernels']}k) → inductor {info['inductor']['us_median']:.1f} vllm_c {info['vllm_c']['us_median']:.1f} triton {info['triton']['us_median']:.1f}(各 {info['vllm_c']['kernels']}k);加速 inductor {sp['inductor']:.1f}x vllm_c {sp['vllm_c']:.1f}x triton {sp['triton']:.1f}x")
    print(f"   N={n:5d}")
    for r in row:
        print("      " + r)
print("   N=8192 时的加速比 vs 搬运量比值:")
for name in OPS:
    t1, t2 = TRAFFIC[name]
    print(f"      {name:18s} 实测 vllm_c {SPEED[(name, 8192)]['vllm_c']:.2f}x,搬运量 {t1 / t2:.2f}x")

print("\n== 3. 融合版贴着带宽下限吗(N=8192,数据超出 L2)")
for name in OPS:
    d = OPS[name]["d"]
    byts = TRAFFIC[name][1] * d * 8192
    t = next(p["ops"][name]["vllm_c"]["us_median"] for p in bench["points"] if p["tokens"] == 8192)
    print(f"      {name:18s} 搬 {byts / 2**20:.1f} MiB;按拷贝带宽算要 {byts / BW * 1e6:.1f}us;实测 {t:.1f}us;折合 {byts / (t * 1e-6) / 1e9:.0f} GB/s")

print("\n== 4. roofline:这些算子落在哪")
print(f"      {'算子':18s} {'每字节浮点运算':>10s} {'实测 TFLOPS':>12s}  受限于")
ROOF = []
for name, op in OPS.items():
    t1, t2 = TRAFFIC[name]
    ai = op["flops"] / t2
    n = 8192
    t = next(p["ops"][name]["vllm_c"]["us_median"] for p in bench["points"] if p["tokens"] == n)
    achieved = op["flops"] * op["d"] * n / (t * 1e-6) / 1e12
    ROOF.append((name, ai, achieved))
    print(f"      {name:18s} {ai:10.2f} {achieved:12.2f}  {'带宽' if ai < ridge else '算力'}")
for n, g in bench["gemm_decode"].items():
    ROOF.append((f"矩阵乘 N={n}", g["flop_per_byte"], g["tflops"]))
    print(f"      {'矩阵乘 N=' + str(n):18s} {g['flop_per_byte']:10.2f} {g['tflops']:12.2f}  {'带宽' if g['flop_per_byte'] < ridge else '算力'}")
big_ai = 2 * 8192 ** 3 / (3 * 8192 ** 2 * BF16)
ROOF.append(("矩阵乘 8192 阶", big_ai, bench["gemm_bf16_peak_tflops"]))
print(f"      {'矩阵乘 8192 阶':18s} {big_ai:10.2f} {bench['gemm_bf16_peak_tflops']:12.2f}  算力")
g = bench["gemm_decode"]
print(f"   矩阵乘 [N, 4096] x [4096, 4096]:N=1 {g['1']['us']:.1f}us,N=32 {g['32']['us']:.1f}us,N=256 {g['256']['us']:.1f}us")
wbytes = 4096 * 4096 * BF16
print(f"   权重 {wbytes / 2**20:.0f} MiB,按拷贝带宽读一遍要 {wbytes / BW * 1e6:.1f}us")

print("\n== 5. 放到一层、一个 token、一步解码里")
naive_layer = 2 * TRAFFIC["fused_add_rms_norm"][0] * HIDDEN + TRAFFIC["rms_norm"][0] * HIDDEN * 0 + TRAFFIC["silu_and_mul"][0] * INTER
fused_layer = 2 * TRAFFIC["fused_add_rms_norm"][1] * HIDDEN + TRAFFIC["silu_and_mul"][1] * INTER
print(f"   每层(两次加残差归一化,一次 SwiGLU):朴素 {naive_layer / KiB:.0f} KiB,融合 {fused_layer / KiB:.0f} KiB")
print(f"   {LAYERS} 层,每个 token:朴素 {naive_layer * LAYERS / 2**20:.1f} MiB,融合 {fused_layer * LAYERS / 2**20:.1f} MiB")
print(f"   其中融合之后,SwiGLU 占 {TRAFFIC['silu_and_mul'][1] * INTER / fused_layer:.0%},因为它的中间维是隐层的 {INTER / HIDDEN:.1f} 倍")

def step_overhead(n):
    """一步解码里这三样小算子花的时间(微秒),按实测的每个调用的耗时乘调用次数。"""
    p = next(p for p in bench["points"] if p["tokens"] == n)
    out = {}
    for k in ("eager", "inductor", "vllm_c", "triton"):
        out[k] = LAYERS * (2 * p["ops"]["fused_add_rms_norm"][k]["us_median"] + p["ops"]["silu_and_mul"][k]["us_median"])
    return out

# 一步解码多久:取上一篇实测的解码吞吐(并发 32,一步出 32 个 token)
prev = os.path.join(HERE, "..", "vllm-kv-cache", "bench", "prefix_overhead_4090d.json")
if os.path.exists(prev):
    pr = json.load(open(prev))
    import statistics
    v = [x["output_tps"] for rd in pr["rounds"] if rd["config"] == "off" for x in rd["runs"] if x["name"] == "解码为主"]
    step_ms = 32 / statistics.median(v) * 1000
    ov = step_overhead(32)
    print(f"   并发 32 解码,一步 {step_ms:.1f} 毫秒(32 个 token)。这一步里这三样小算子:")
    for k in ("eager", "inductor", "vllm_c", "triton"):
        print(f"      {k:9s} {ov[k] / 1000:.2f} 毫秒,占一步的 {ov[k] / 1000 / step_ms:.1%}")
    STEP = {"step_ms": step_ms, "overhead_us": ov}
else:
    print("   (没找到上一篇的实测结果,不算一步解码里的占比)")
    STEP = None

print("\n== 6. kernel 数和小形状:N=1 时每个 kernel 的工夫")
p1 = next(p for p in bench["points"] if p["tokens"] == 1)
for name, info in p1["ops"].items():
    e = info["eager"]
    print(f"      {name:18s} 朴素 {e['us_median']:.2f}us / {e['kernels']} 个 kernel = 每个 {e['us_median'] / e['kernels']:.2f}us;融合(vllm_c) {info['vllm_c']['us_median']:.2f}us / {info['vllm_c']['kernels']} 个")
sil = p1["ops"]["silu_and_mul"]
print(f"   SwiGLU 在 N=1:vLLM 的 CUDA 版 {sil['vllm_c']['us_median']:.2f}us,比朴素的 {sil['eager']['us_median']:.2f}us 还慢;Triton 版 {sil['triton']['us_median']:.2f}us")

print("\n== 7. 误差(和 float64 参考比,最大绝对误差)")
p8 = next(p for p in bench["points"] if p["tokens"] == 8192)
for name, info in p8["ops"].items():
    print(f"      {name:18s} " + "  ".join(f"{k} {v['max_abs_err']:.4f}" for k, v in info.items()))

json.dump({"traffic": TRAFFIC, "parts": PARTS, "roofline": ROOF, "ridge": ridge, "step": STEP, "speed": {f"{k[0]}@{k[1]}": v for k, v in SPEED.items()}},
          open(os.path.join(HERE, "bench", "calc_summary.json"), "w"), indent=1)
print("\n-> bench/calc_summary.json")
