#!/usr/bin/env python3
"""融合算子实测:同一个算子的几种写法,各要多久(RTX 4090D)。

形状按 Llama-3-8B:隐层 4096,MLP 的中间维 14336,bfloat16。N 是这一步的 token 数。
  rms_norm            y = w * x / sqrt(mean(x^2) + eps)
  fused_add_rms_norm  s = x + residual;新残差 = s;y = w * s / sqrt(mean(s^2) + eps)
  silu_and_mul        y = silu(x[..., :d]) * x[..., d:]
每个算子量四种写法:
  eager     PyTorch 逐个算子写出来(就是 vLLM 里的原生实现),一行一个 kernel
  inductor  把同一个函数交给 torch.compile,Inductor 自动融合
  vllm_c    vLLM 手写的 CUDA kernel(torch.ops._C)
  triton    本文从零写的 Triton kernel
计时用 CUDA Graph:一张图里连着调 20 次(大形状 4 次),回放 50 次取平均,折成每次调用的微秒数;
几种写法在每个 token 数上交错测 5 轮取中位数。
另用 profiler 数每种写法一次调用发了几个 kernel,并和 float64 的参考实现对误差。
顺带量这张卡的 bf16 矩阵乘峰值和拷贝带宽,给文章里的 roofline 用。

Run:  python fusion_ops.py   ->  fusion_ops_4090d.json
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import vllm.kernels  # noqa: F401  导入即登记 vLLM 自带的算子实现,并加载 torch.ops._C

dev = torch.device("cuda")
DT = torch.bfloat16
HIDDEN, INTER, EPS = 4096, 14336, 1e-5
TOKENS = [1, 32, 256, 2048, 8192]
ROUNDS, REPLAYS, WARMUP_SECONDS = 5, 50, 7.0
CALLS_PER_GRAPH = 20  # 一张图里连着调几次;大形状在 main 里改成 4
OUT = "fusion_ops_4090d.json"


# ---------------------------------------------------------------- 原生实现:和 vLLM 里的一字不差
def rms_norm_native(x: torch.Tensor, weight: torch.Tensor, epsilon: float,
                    variance_size: int | None = None) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def fused_add_rms_norm_native(x: torch.Tensor, x_residual: torch.Tensor, weight: torch.Tensor,
                              epsilon: float, variance_size: int | None = None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)

    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


def silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ---------------------------------------------------------------- 从零写的 Triton kernel
@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)          # 一个 program 管一行
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    ptr = row * D + cols
    x = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D              # 整行一次读完,方差留在寄存器里
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + ptr, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(x_ptr, r_ptr, w_ptr, y_ptr, res_ptr, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    ptr = row * D + cols
    x = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(res_ptr + ptr, s.to(res_ptr.dtype.element_ty), mask=mask)   # 新的残差
    var = tl.sum(s * s, axis=0) / D
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + ptr, (s * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _silu_and_mul_kernel(x_ptr, y_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    gate = tl.load(x_ptr + row * 2 * D + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * 2 * D + D + cols, mask=mask, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up      # silu(gate) * up
    tl.store(y_ptr + row * D + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rms_norm_triton(x, w, eps):
    y = torch.empty_like(x)
    _rms_norm_kernel[(x.shape[0],)](x, w, y, eps, D=x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=8)
    return y


def fused_add_rms_norm_triton(x, r, w, eps):
    y, res = torch.empty_like(x), torch.empty_like(r)
    _fused_add_rms_norm_kernel[(x.shape[0],)](x, r, w, y, res, eps, D=x.shape[1],
                                              BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=8)
    return y, res


def silu_and_mul_triton(x):
    n, two_d = x.shape
    d = two_d // 2
    y = torch.empty((n, d), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    _silu_and_mul_kernel[(n, triton.cdiv(d, BLOCK))](x, y, D=d, BLOCK=BLOCK, num_warps=4)
    return y


# ---------------------------------------------------------------- vLLM 手写的 CUDA kernel
def rms_norm_vllm_c(x, w, eps):
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(out, x, w, eps)
    return out


def silu_and_mul_vllm_c(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


# fused_add_rms_norm 的 CUDA 版本是原地写的:新的残差写进 r,归一化的结果写进 x。
# 所以它一直在输入的副本 xc、rc 上跑。
def fused_add_rms_norm_vllm_c(xc, rc, w, eps):
    torch.ops._C.fused_add_rms_norm(xc, rc, w, eps)
    return xc


# ---------------------------------------------------------------- 计时、数 kernel、对误差
def time_graph(fn, k: int = CALLS_PER_GRAPH) -> float:
    """一张 CUDA Graph 里连着调 k 次,回放 50 次取平均,返回每次调用的微秒数。
    连着调,是为了摊薄启动一张图的开销:真实推理时,整个一步是一张大图,
    每个 kernel 付的是图里 kernel 之间的间隙,不是启动一张图的钱。"""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(k):
            fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (REPLAYS * k)


def kernel_names(fn) -> list[str]:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == DeviceType.CUDA and getattr(e, "self_device_time_total", 0) > 0]


def gpu_state() -> dict:
    q = "clocks.sm,power.draw,temperature.gpu"
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", phys],
                         capture_output=True, text=True).stdout.strip()
    sm, power, temp = [x.strip() for x in out.split(",")]
    return dict(sm_clock_mhz=int(sm), power_w=float(power), temp_c=int(temp))


def make_inputs(n: int):
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn((n, HIDDEN), generator=g, device=dev, dtype=torch.float32).to(DT)
    r = torch.randn((n, HIDDEN), generator=g, device=dev, dtype=torch.float32).to(DT)
    w = (1 + 0.1 * torch.randn((HIDDEN,), generator=g, device=dev, dtype=torch.float32)).to(DT)
    g2 = torch.randn((n, 2 * INTER), generator=g, device=dev, dtype=torch.float32).to(DT)
    return x, r, w, g2


def max_err(a: torch.Tensor, ref64: torch.Tensor) -> float:
    return float((a.double() - ref64).abs().max())


def main():
    name = torch.cuda.get_device_name()
    result = {"gpu": name, "torch": torch.__version__, "triton": triton.__version__,
              "shape": dict(hidden=HIDDEN, intermediate=INTER, dtype=str(DT)), "tokens": TOKENS,
              "bytes_moved_fused": {"rms_norm": 4 * HIDDEN, "fused_add_rms_norm": 8 * HIDDEN, "silu_and_mul": 6 * INTER},
              "calls_per_graph": {n: (20 if n <= 2048 else 4) for n in TOKENS}, "points": []}

    # ---- 峰值:拷贝带宽,bf16 矩阵乘
    a = torch.empty(1 << 27, device=dev, dtype=torch.float32)  # 512 MiB
    b = torch.empty_like(a)
    copy_us = time_graph(lambda: b.copy_(a), k=1)
    result["copy_bandwidth_gbps"] = round(2 * a.numel() * 4 / (copy_us * 1e-6) / 1e9, 1)
    del a, b
    m = torch.randn(8192, 8192, device=dev, dtype=DT)
    n_ = torch.randn(8192, 8192, device=dev, dtype=DT)
    gemm_us = time_graph(lambda: m @ n_, k=1)
    result["gemm_bf16_peak_tflops"] = round(2 * 8192 ** 3 / (gemm_us * 1e-6) / 1e12, 1)
    # 解码时的矩阵乘:[n, 4096] 乘 [4096, 4096]。
    # 图里轮流用 4 组不同的权重(共 128 MiB,超过 72 MB 的 L2),
    # 免得同一组权重一直留在缓存里,量出来的不是真实推理时的速度。
    wqs = [torch.randn(4096, 4096, device=dev, dtype=DT) for _ in range(4)]
    result["gemm_decode"] = {}
    for n in (1, 32, 256):
        xq = torch.randn(n, 4096, device=dev, dtype=DT)
        t = time_graph(lambda: [xq @ wq for wq in wqs], k=1) / len(wqs)
        flops, byts = 2 * n * 4096 * 4096, 2 * (4096 * 4096 + n * 4096 + n * 4096)
        result["gemm_decode"][n] = {"us": round(t, 2), "tflops": round(flops / (t * 1e-6) / 1e12, 2), "flop_per_byte": round(flops / byts, 2)}
    del m, n_, wqs
    torch.cuda.empty_cache()
    print("copy", result["copy_bandwidth_gbps"], "GB/s; gemm", result["gemm_bf16_peak_tflops"], "TFLOPS; decode", result["gemm_decode"], flush=True)

    # ---- 稳态
    x, r, w, g2 = make_inputs(2048)
    compiled = {
        "rms_norm": torch.compile(rms_norm_native, dynamic=True),
        "fused_add_rms_norm": torch.compile(fused_add_rms_norm_native, dynamic=True),
        "silu_and_mul": torch.compile(silu_and_mul_native, dynamic=True),
    }
    t0 = time.time()
    while time.time() - t0 < WARMUP_SECONDS:
        rms_norm_native(x, w, EPS); compiled["rms_norm"](x, w, EPS); rms_norm_triton(x, w, EPS); rms_norm_vllm_c(x, w, EPS)
    torch.cuda.synchronize()
    result["gpu_after_warmup"] = gpu_state()

    for n in TOKENS:
        k_per_graph = 20 if n <= 2048 else 4
        x, r, w, g2 = make_inputs(n)
        xc, rc = x.clone(), r.clone()                       # 给原地写的那个版本用
        # float64 参考
        x64, r64, w64, g64 = x.double(), r.double(), w.double(), g2.double()
        ref_rms = (x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * w64).to(DT).double()
        s64 = x64 + r64
        res_ref = s64.to(DT).double()
        ref_far = ((s64 * torch.rsqrt(s64.pow(2).mean(-1, keepdim=True) + EPS) * w64).to(DT)).double()
        ref_silu = (F.silu(g64[:, :INTER]) * g64[:, INTER:]).to(DT).double()

        variants = {
            "rms_norm": {
                "eager": (lambda: rms_norm_native(x, w, EPS), ref_rms),
                "inductor": (lambda: compiled["rms_norm"](x, w, EPS), ref_rms),
                "vllm_c": (lambda: rms_norm_vllm_c(x, w, EPS), ref_rms),
                "triton": (lambda: rms_norm_triton(x, w, EPS), ref_rms),
            },
            "fused_add_rms_norm": {
                "eager": (lambda: fused_add_rms_norm_native(x, r, w, EPS)[0], ref_far),
                "inductor": (lambda: compiled["fused_add_rms_norm"](x, r, w, EPS)[0], ref_far),
                "vllm_c": (lambda: fused_add_rms_norm_vllm_c(xc, rc, w, EPS), None),       # 对误差单独做
                "triton": (lambda: fused_add_rms_norm_triton(x, r, w, EPS)[0], ref_far),
            },
            "silu_and_mul": {
                "eager": (lambda: silu_and_mul_native(g2), ref_silu),
                "inductor": (lambda: compiled["silu_and_mul"](g2), ref_silu),
                "vllm_c": (lambda: silu_and_mul_vllm_c(g2), ref_silu),
                "triton": (lambda: silu_and_mul_triton(g2), ref_silu),
            },
        }
        point = {"tokens": n, "ops": {}}
        for op, vs in variants.items():
            # 预热每种写法(编译、自动调优都在计时之外)
            xc.copy_(x); rc.copy_(r)
            for fn, _ in vs.values():
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()
            times = {k: [] for k in vs}
            for rnd in range(ROUNDS):  # 交错:每轮换一个先后顺序
                order = list(vs)
                order = order[rnd % len(order):] + order[:rnd % len(order)]
                for k_ in order:
                    times[k_].append(time_graph(vs[k_][0], k=k_per_graph))
            info = {}
            for k, (fn, ref) in vs.items():
                if op == "fused_add_rms_norm" and k == "vllm_c":
                    xc.copy_(x); rc.copy_(r)
                    out = fn()
                    err = max(max_err(out, ref_far), max_err(rc, res_ref))   # 归一化的结果和新残差都要对
                else:
                    out = fn()
                    err = max_err(out, ref)
                kernels = kernel_names(fn)
                info[k] = {"us": [round(t, 3) for t in times[k]], "us_median": round(statistics.median(times[k]), 3),
                           "max_abs_err": round(err, 6), "kernels": len(kernels), "kernel_names": kernels[:8]}
            point["ops"][op] = info
            print(f"N={n:5d} {op:18s} " + "  ".join(f"{k} {info[k]['us_median']:9.2f}us/{info[k]['kernels']}k" for k in vs), flush=True)
        point["gpu"] = gpu_state()
        result["points"].append(point)
        del x, r, w, g2, xc, rc, x64, r64, w64, g64
        torch.cuda.empty_cache()

    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
