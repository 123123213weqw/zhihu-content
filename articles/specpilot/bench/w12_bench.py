#!/usr/bin/env python3
"""W12 在 RTX 4090 D 上的实测:多读几行要不要钱,权重少读四分之一能快多少。

权重用 Llama-3.1-8B-Instruct 的真实权重,四个形状:
  q_proj     [4096, 4096]       注意力
  gate_proj  [14336, 4096]      MLP
  down_proj  [4096, 14336]      MLP
  lm_head    [128256, 4096]     输出层

一、BF16 矩阵乘 (cuBLAS) 的耗时随行数 M 的变化,M 从 1 到 1024。
    推测解码里,M 就是一趟验证的行数(草稿的节点数)。
二、M 不超过 64 时,同一个矩阵乘 y = x @ W^T 的五种读法:
  bf16         读 BF16 权重,cuBLAS                                  2.0 字节/权重
  bf16_triton  同一个 Triton kernel,读 BF16 权重 (对照组)              2.0
  dequant      读 W12,先解码成 BF16 落到显存,再交给 cuBLAS (两个 kernel)  1.5 + 2 + 2 = 5.5
  w12          Triton kernel:读 W12,在寄存器里解码,直接进矩阵乘         1.5
  q8           只读 W12 的高 8 bit (两级验证里 Q pass 读的就是它)       1.0

计时用 CUDA Graph。为了让每次读权重都真的来自显存而不是 L2 缓存,每种格式准备几份同样的权重轮流用,
合起来不少于 256 MiB (这张卡的 L2 是 72 MiB)。一张图里把几份权重各用一遍(小形状重复几遍),
回放 50 次取平均,折成一次调用的微秒数。五种读法在每个 M 上交错测 5 轮取中位数。
Triton kernel 的块大小、warps、流水级数、K 方向切分从 19 组里挑:先和 PyTorch 的参考比误差 (对不上的淘汰),再按 M = 16 的耗时挑最快的一组,
每个形状每个 kernel 各挑一次。挑配置时也轮换权重副本。
escape 的稀疏校正不计时:它只碰十万分之几的权重。

Run:  CUDA_VISIBLE_DEVICES=<空闲的那张卡> python w12_bench.py  ->  w12_bench_4090d.json
      (--shapes q_proj 只跑一个形状, --quick 每种 kernel 只试 4 组配置; 调试用)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from w12_ref import NEG15, w12_encode

dev = torch.device("cuda")
DT = torch.bfloat16
SHAPES = [("q_proj", "model.layers.0.self_attn.q_proj.weight"),
          ("gate_proj", "model.layers.0.mlp.gate_proj.weight"),
          ("down_proj", "model.layers.0.mlp.down_proj.weight"),
          ("lm_head", "lm_head.weight")]
M_ROWS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024]
M_VERIFY = [1, 4, 8, 16, 32, 64]
ROUNDS, REPLAYS, WARMUP_SECONDS = 5, 50, 7.0
MIN_ROTATE_BYTES = 256 << 20
BYTES_PER_WEIGHT = {"bf16": 2.0, "bf16_triton": 2.0, "dequant": 5.5, "w12": 1.5, "q8": 1.0}
# (BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K)
CONFIGS = [(16, 64, 4, 3, 1), (16, 128, 4, 3, 1), (16, 128, 4, 4, 1), (32, 64, 4, 3, 1), (32, 64, 4, 4, 1),
           (32, 128, 4, 3, 1), (32, 128, 8, 3, 1), (64, 64, 4, 3, 1), (64, 64, 8, 3, 1), (64, 128, 8, 2, 1),
           (16, 64, 4, 3, 2), (32, 64, 4, 3, 2), (64, 64, 4, 2, 2),
           (16, 128, 4, 2, 4), (32, 64, 4, 3, 4), (32, 128, 4, 2, 4), (64, 64, 4, 2, 4),
           (16, 128, 4, 2, 8), (32, 128, 4, 2, 8)]
REDUCE_BLOCK = 1024


# ------------------------------------------------------------------ Triton kernel
@triton.jit
def _unpack(hi, low, Q_ONLY: tl.constexpr):
    """hi: [BN, BK] int32,每个是 W12 记录的高 8 bit;low: [BN, BK] int32,低 4 bit。

    拼回 BF16 的 16 个 bit:[符号 | 0111 c3c2c1c0 | m6..m0],也就是 w~ = (-1)^s * 2^(c-15) * (1 + m/128)。
    """
    if Q_ONLY:
        low = 8                                   # 扔掉的低 4 位取中点
    bits = ((hi << 8) & 0x8000) | ((hi & 0x7F) << 4) | 0x3800 | low
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _w12_gemm_kernel(x_ptr, hi_ptr, lo_ptr, scale_ptr, y_ptr, ws_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, Q_ONLY: tl.constexpr):
    """y[M, N] = (x[M, K] @ W~[N, K]^T) * scale[N]。一个 program 管 BLOCK_N 行权重。

    W~ 是约分后的权重(指数域在 112 到 127 之间),scale[n] = 2^(E_n - 127) 在最后乘回 FP32 的累加结果。
    SPLIT_K 大于 1 时,每个 program 只算 K 的一段,部分和写进 ws[SPLIT_K, M, N],由 _split_k_reduce 加起来。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_per = K // SPLIT_K
    for k0 in range(pid_k * k_per, (pid_k + 1) * k_per, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        hi = tl.load(hi_ptr + rn[:, None] * K + rk[None, :]).to(tl.int32)              # [BN, BK]
        rj = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        lb = tl.load(lo_ptr + rn[:, None] * (K // 2) + rj[None, :]).to(tl.int32)       # [BN, BK/2],一字节两个
        low = tl.interleave(lb & 0xF, lb >> 4)                                         # [BN, BK]
        w = _unpack(hi, low, Q_ONLY)                                                  # [BN, BK] bf16
        xk = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=rm[None, :] < M, other=0.0)  # [BK, BM]
        acc = tl.dot(w, xk, acc)                                                      # [BN, BM]
    out = tl.trans(acc)                                                               # [BM, BN]
    scale = tl.load(scale_ptr + rn)[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], (out * scale).to(tl.bfloat16), mask=rm[:, None] < M)
    else:
        tl.store(ws_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :], out * scale, mask=rm[:, None] < M)


@triton.jit
def _bf16_gemm_kernel(x_ptr, w_ptr, y_ptr, ws_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      SPLIT_K: tl.constexpr):
    """对照组:骨架和 _w12_gemm_kernel 一样,只是权重直接是 BF16,没有解码,也不乘公因子。"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k_per = K // SPLIT_K
    for k0 in range(pid_k * k_per, (pid_k + 1) * k_per, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])                             # [BN, BK] bf16
        xk = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=rm[None, :] < M, other=0.0)
        acc = tl.dot(w, xk, acc)
    out = tl.trans(acc)
    if SPLIT_K == 1:
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], out.to(tl.bfloat16), mask=rm[:, None] < M)
    else:
        tl.store(ws_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :], out, mask=rm[:, None] < M)


@triton.jit
def _split_k_reduce(ws_ptr, y_ptr, total, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(ws_ptr + s * total + cols)
    tl.store(y_ptr + cols, acc.to(tl.bfloat16))


@triton.jit
def _w12_dequant_kernel(hi_ptr, lo_ptr, e_ptr, out_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """把一份 W12 权重整份解码回 BF16,写进 out[N, K]。"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    hi = tl.load(hi_ptr + rn[:, None] * K + rk[None, :]).to(tl.int32)
    rj = (pid_k * BLOCK_K) // 2 + tl.arange(0, BLOCK_K // 2)
    lb = tl.load(lo_ptr + rn[:, None] * (K // 2) + rj[None, :]).to(tl.int32)
    low = tl.interleave(lb & 0xF, lb >> 4)
    e = tl.load(e_ptr + rn)[:, None]
    bits = (((hi << 8) & 0x8000) | ((hi & 0x7F) << 4) | 0x3800 | low) + ((e - 127) << 7)   # 指数域加回 E - 127
    tl.store(out_ptr + rn[:, None] * K + rk[None, :], bits.to(tl.uint16).to(tl.bfloat16, bitcast=True))


# ------------------------------------------------------------------ Python 包装
def launch_w12(x, hi, lo, scale, y, ws, c, q_only=False):
    M, N, K = x.shape[0], hi.shape[0], hi.shape[1]
    bn, bk, nw, ns, sk = c
    _w12_gemm_kernel[(N // bn, sk)](x, hi, lo, scale, y, ws, M, N, K,
                                    BLOCK_M=64, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, Q_ONLY=q_only,
                                    num_warps=nw, num_stages=ns)
    if sk > 1:
        _split_k_reduce[(M * N // REDUCE_BLOCK,)](ws, y, M * N, SPLIT_K=sk, BLOCK=REDUCE_BLOCK, num_warps=4)
    return y


def launch_bf16(x, w, y, ws, c):
    M, N, K = x.shape[0], w.shape[0], w.shape[1]
    bn, bk, nw, ns, sk = c
    _bf16_gemm_kernel[(N // bn, sk)](x, w, y, ws, M, N, K, BLOCK_M=64, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk,
                                     num_warps=nw, num_stages=ns)
    if sk > 1:
        _split_k_reduce[(M * N // REDUCE_BLOCK,)](ws, y, M * N, SPLIT_K=sk, BLOCK=REDUCE_BLOCK, num_warps=4)
    return y


def launch_dequant(hi, lo, E, out, block=(64, 128)):
    N, K = hi.shape
    _w12_dequant_kernel[(N // block[0], K // block[1])](hi, lo, E, out, N, K, BLOCK_N=block[0], BLOCK_K=block[1], num_warps=4)
    return out


# ------------------------------------------------------------------ 计时
def time_graph(fn, k: int = 1) -> float:
    """一张 CUDA Graph 里把 fn 连着调 k 次,回放 50 次取平均,返回调用一次 fn 的微秒数。"""
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


def gpu_state() -> dict:
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits",
                          "-i", phys], capture_output=True, text=True).stdout.strip()
    sm, power, temp = [x.strip() for x in out.split(",")]
    return dict(sm_clock_mhz=int(sm), power_w=float(power), temp_c=int(temp))


def others_on_gpu(phys: str) -> list[str]:
    """这张卡上除自己以外还有哪些进程 (按卡的 UUID 过滤,别的卡上有没有人不关这张卡的事)。"""
    def q(args_):
        return subprocess.run(["nvidia-smi", *args_, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    uuid = q([f"--id={phys}", "--query-gpu=uuid"])
    me = str(os.getpid())
    return [l for l in q(["--query-compute-apps=pid,used_memory,gpu_uuid"]).splitlines()
            if l.split(",")[0].strip() != me and l.strip().endswith(uuid)]


def encode_big(w: torch.Tensor, chunk: int = 8192):
    """按行分块做 w12_encode,免得 128256 行的 lm_head 一次生成过大的中间量。"""
    parts = [w12_encode(w[r0:r0 + chunk].contiguous()) for r0 in range(0, w.shape[0], chunk)]
    return [torch.cat([p[i] for p in parts]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/wangyue/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", default="w12_bench_4090d.json")
    ap.add_argument("--shapes", default="", help="逗号分开的形状名,留空是全部")
    ap.add_argument("--quick", action="store_true", help="每种 kernel 只试 4 组配置")
    args = ap.parse_args()
    shapes = [s_ for s_ in SHAPES if not args.shapes or s_[0] in args.shapes.split(",")]
    configs = CONFIGS[:4] if args.quick else CONFIGS
    busy = others_on_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    if busy:
        raise SystemExit("这张卡上还有别的进程,先不测:" + "; ".join(busy))

    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
              "model": os.path.basename(args.model), "bytes_per_weight": BYTES_PER_WEIGHT,
              "rows": {}, "verify": {}, "configs": {}}
    a = torch.empty(1 << 27, device=dev, dtype=torch.float32)
    b = torch.empty_like(a)
    copy_us = time_graph(lambda: b.copy_(a), k=1)
    result["copy_bandwidth_gbps"] = round(2 * a.numel() * 4 / (copy_us * 1e-6) / 1e9, 1)
    del a, b
    print("copy bandwidth", result["copy_bandwidth_gbps"], "GB/s", flush=True)

    wmap = json.load(open(glob.glob(os.path.join(args.model, "*.index.json"))[0]))["weight_map"]
    for tag, tname in shapes:
        with safe_open(os.path.join(args.model, wmap[tname]), "pt") as f:
            w = f.get_tensor(tname).to(dev)
        N, K = w.shape
        hi, lo, E, esc = encode_big(w)
        scale = torch.exp2((E - 127).float())
        r_bf16 = max(1, -(-MIN_ROTATE_BYTES // (2 * N * K)))
        r_w12 = max(1, -(-MIN_ROTATE_BYTES // (3 * N * K // 2)))
        ws_bf16, ws_hi, ws_lo = [w.clone() for _ in range(r_bf16)], [hi.clone() for _ in range(r_w12)], [lo.clone() for _ in range(r_w12)]
        scratch = torch.empty_like(w)
        y = torch.empty(64, N, device=dev, dtype=DT)
        ws = torch.empty(8, 64, N, device=dev, dtype=torch.float32)
        k_b, k_w = max(1, 16 // r_bf16), max(1, 16 // r_w12)
        info = {"tensor": tname, "N": N, "K": K, "escapes": int(esc.sum()),
                "rotations": {"bf16": r_bf16, "w12": r_w12}, "calls_per_graph": {"bf16": k_b, "w12": k_w}}
        print(f"== {tag} [{N}, {K}] escapes {int(esc.sum())} rotations bf16 {r_bf16} / w12 {r_w12}", flush=True)

        # ---- 一、BF16 cuBLAS 的耗时随行数的变化
        rows = {}
        for m in M_ROWS:
            x = torch.randn(m, K, device=dev, dtype=torch.float32).to(DT)
            out = torch.empty(m, N, device=dev, dtype=DT)
            fn = lambda: [torch.mm(x, wt.t(), out=out) for wt in ws_bf16]
            ts = []
            for _ in range(ROUNDS):
                ts.append(time_graph(fn, k=k_b) / r_bf16)
            med = statistics.median(ts)
            rows[m] = {"us_median": round(med, 2), "us": [round(t, 2) for t in ts], "gbps": round(2 * N * K / (med * 1e-6) / 1e9, 1)}
        result["rows"][tag] = rows
        print("   rows:", " ".join(f"M={m}:{rows[m]['us_median']}us" for m in M_ROWS), flush=True)
        del x, out

        # ---- 二、M 不超过 64:给 Triton kernel 挑配置 (按 M = 16 的耗时;和正式计时一样轮换权重副本,不留在缓存里)
        x16 = torch.randn(16, K, device=dev, dtype=torch.float32).to(DT)
        ref16 = torch.mm(x16, w.t()).float()
        w_placeholder = w.clone()                              # kernel 不做 escape 校正,参考值里 escape 处也用占位值
        esc_rows, esc_cols = esc.nonzero(as_tuple=True)
        w_placeholder[esc_rows, esc_cols] = (NEG15 * scale[esc_rows]).to(DT)
        ref_w12 = torch.mm(x16, w_placeholder.t()).float()
        best = {}
        for kind in ("bf16_triton", "w12", "q8"):
            cand = []
            for c in configs:
                if N % c[0] or K % (c[4] * c[1]):
                    continue
                if kind == "bf16_triton":
                    fn = lambda c=c: [launch_bf16(x16, wt, y, ws, c) for wt in ws_bf16]
                    per_call = r_bf16
                else:
                    fn = lambda c=c: [launch_w12(x16, h_, l_, scale, y, ws, c, q_only=(kind == "q8")) for h_, l_ in zip(ws_hi, ws_lo)]
                    per_call = r_w12
                limit = 0.02 if kind != "q8" else 0.6          # Q pass 本来就是近似的,只要求不离谱
                try:
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    err = float((y[:16].float() - (ref16 if kind == "bf16_triton" else ref_w12)).abs().max() / ref16.abs().max())
                    cand.append((time_graph(fn, k=1) / per_call if err < limit else float("inf"), c, round(err, 4)))
                except Exception as ex:  # 编译不过或者跑不了的配置跳过
                    cand.append((float("inf"), c, str(ex)[:80]))
            cand.sort(key=lambda t: t[0])
            best[kind] = cand[0][1]
            result["configs"][f"{tag}:{kind}"] = {"best": list(cand[0][1]), "best_us": round(cand[0][0], 1), "err": cand[0][2],
                                                  "all": [(list(c), round(t, 1), e) for t, c, e in cand]}
            print(f"   config {kind}: {cand[0][1]} {cand[0][0]:.1f}us (err {cand[0][2]}); next {[(list(c), round(t, 1)) for t, c, _ in cand[1:4]]}", flush=True)

        # ---- 稳态:先持续跑 7 秒以上
        t0 = time.time()
        while time.time() - t0 < WARMUP_SECONDS:
            for wt in ws_bf16:
                torch.mm(x16, wt.t(), out=y[:16])
            launch_w12(x16, ws_hi[0], ws_lo[0], scale, y, ws, best["w12"])
            launch_w12(x16, ws_hi[0], ws_lo[0], scale, y, ws, best["q8"], q_only=True)
        torch.cuda.synchronize()
        info["gpu_after_warmup"] = gpu_state()

        # ---- 对误差 (M = 64):和 float64 的参考比
        x64 = torch.randn(64, K, device=dev, dtype=torch.float32).to(DT)
        ref = x64.double() @ w.double().T
        rows_i, cols_i = esc.nonzero(as_tuple=True)
        outs = {
            "bf16": torch.mm(x64, w.t()),
            "bf16_triton": launch_bf16(x64, w, y, ws, best["bf16_triton"]).clone(),
            "dequant": torch.mm(x64, launch_dequant(hi, lo, E, scratch).t()),
            "w12": launch_w12(x64, hi, lo, scale, y, ws, best["w12"]).clone(),
            "q8": launch_w12(x64, hi, lo, scale, y, ws, best["q8"], q_only=True).clone(),
        }
        scratch.view(-1)[rows_i * K + cols_i] = w.view(-1)[rows_i * K + cols_i]      # 整份解码的那一路:escape 处盖回原值
        outs["dequant"] = torch.mm(x64, scratch.t())
        if rows_i.numel():                                                            # W12 那一路:补上 escape 的校正
            delta = w[rows_i, cols_i].float() - NEG15 * scale[rows_i]                 # 精确值 - 占位值 * 公因子
            outs["w12"] = outs["w12"].float().index_add(1, rows_i, x64.float()[:, cols_i] * delta[None, :])
        info["error"] = {k: float((v.double() - ref).abs().max() / ref.abs().max()) for k, v in outs.items()}
        info["bitwise_equal_to_cublas"] = {k: float((v.view(torch.int16) == outs["bf16"].view(torch.int16)).float().mean())
                                           for k, v in outs.items() if v.dtype == DT}
        print("   max abs error / max |ref|:", {k: f"{v:.2e}" for k, v in info["error"].items()}, flush=True)
        del ref, outs

        # ---- 二、计时
        ver = {}
        for m in M_VERIFY:
            x = x64[:m].contiguous()
            fns = {
                "bf16": (lambda: [torch.mm(x, wt.t(), out=y[:m]) for wt in ws_bf16], k_b),
                "bf16_triton": (lambda: [launch_bf16(x, wt, y, ws, best["bf16_triton"]) for wt in ws_bf16], k_b),
                "dequant": (lambda: [torch.mm(x, launch_dequant(h_, l_, E, scratch).t(), out=y[:m]) for h_, l_ in zip(ws_hi, ws_lo)], k_w),
                "w12": (lambda: [launch_w12(x, h_, l_, scale, y, ws, best["w12"]) for h_, l_ in zip(ws_hi, ws_lo)], k_w),
                "q8": (lambda: [launch_w12(x, h_, l_, scale, y, ws, best["q8"], q_only=True) for h_, l_ in zip(ws_hi, ws_lo)], k_w),
            }
            for fn, _ in fns.values():
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()
            times = {kk: [] for kk in fns}
            for rnd in range(ROUNDS):
                order = list(fns)
                order = order[rnd % len(order):] + order[:rnd % len(order)]        # 每轮换一个先后顺序
                for kk in order:
                    fn, k = fns[kk]
                    times[kk].append(time_graph(fn, k=k) / (r_bf16 if BYTES_PER_WEIGHT[kk] == 2.0 else r_w12))
            ver[m] = {}
            for kk, v in times.items():
                med = statistics.median(v)
                ver[m][kk] = {"us_median": round(med, 2), "us": [round(t, 2) for t in v],
                              "gbps": round(BYTES_PER_WEIGHT[kk] * N * K / (med * 1e-6) / 1e9, 1)}
            info["gpu"] = gpu_state()
            print(f"   M={m:2d} " + "  ".join(f"{kk} {ver[m][kk]['us_median']:8.2f}us ({ver[m][kk]['gbps']:6.1f} GB/s)" for kk in fns), flush=True)
        info["speedup_vs_bf16"] = {str(m): {kk: round(ver[m]["bf16"]["us_median"] / ver[m][kk]["us_median"], 3)
                                            for kk in fns if kk != "bf16"} for m in M_VERIFY}
        result["verify"][tag] = {"info": info, "points": ver}
        del ws_bf16, ws_hi, ws_lo, scratch, y, ws, w, hi, lo, esc, x64, x16, x
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
