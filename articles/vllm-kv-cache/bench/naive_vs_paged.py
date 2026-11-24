#!/usr/bin/env python3
"""写一个 token 的 KV,三种写法每一步的耗时(RTX 4090D 实测)。

形状按 Llama-2-7B:32 层,每层 32 个 KV 头,每头 128 维,半精度,batch = 1。当前长度 n 取 128 到 8192。
  cat     每层每步把新的 K、V 拼到旧张量后面(torch.cat)。
          HuggingFace 的 DynamicCache 就是这么写的:代价随当前长度线性增长。
  static  启动时按最大长度预分配一段连续显存,每步原地写第 n 个位置。
  paged   一整块显存切成块,每步按槽位散写(vLLM 的做法)。

测法:每种写法在当前长度 n 上录一张 CUDA Graph(一步,32 层),回放 20 次取平均;
三种写法在每个长度上交错测 5 轮取中位数;计时前先持续空跑至少 7 秒让显卡进入稳态,
并记录测量期间的时钟和功耗。
Run:  python naive_vs_paged.py   ->  naive_vs_paged_4090d.json
"""
from __future__ import annotations
import json
import os
import statistics
import subprocess
import time

import torch

H, D, L = 32, 128, 32  # KV 头数,每头维度,层数
PAGE = 16  # 分页写法里每个物理块的 token 数(FlashAttention 的分页粒度)
LENGTHS = [128, 512, 2048, 8192]
ROUNDS, REPLAYS, WARMUP_SECONDS = 5, 20, 7.0
OUT = "naive_vs_paged_4090d.json"
dev = torch.device("cuda")
fp16 = dict(dtype=torch.float16, device=dev)


def gpu_state() -> dict:
    q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]  # nvidia-smi 认物理卡号
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
                          "-i", phys], capture_output=True, text=True).stdout
    sm, mem, power, temp, util = [x.strip() for x in out.strip().split(",")]
    return dict(sm_clock_mhz=int(sm), mem_clock_mhz=int(mem), power_w=float(power), temp_c=int(temp), util_pct=int(util))


def build(n: int):
    """三种写法各建一份状态。返回 (step 函数, 参数表)。"""
    kn = [torch.randn(1, H, 1, D, **fp16) for _ in range(L)]
    vn = [torch.randn(1, H, 1, D, **fp16) for _ in range(L)]
    cat_k = [torch.randn(1, H, n, D, **fp16) for _ in range(L)]
    cat_v = [torch.randn(1, H, n, D, **fp16) for _ in range(L)]
    max_len = n + 1 + PAGE  # 只开到当前长度够用
    sta_k = [torch.zeros(1, H, max_len, D, **fp16) for _ in range(L)]
    sta_v = [torch.zeros(1, H, max_len, D, **fp16) for _ in range(L)]
    num_pages = max_len // PAGE
    cache = [torch.zeros(num_pages, H, PAGE, 2 * D, **fp16) for _ in range(L)]
    new_kv = [torch.cat([kn[i][:, :, 0, :], vn[i][:, :, 0, :]], dim=-1) for i in range(L)]
    # 位置 n 的槽位:逻辑块 n // PAGE 落在物理块 (n // PAGE) 上,块内偏移 n % PAGE
    slot = torch.tensor([(n // PAGE) * PAGE + n % PAGE], device=dev)
    flat = [c.view(-1, H, 2 * D) for c in cache]
    out: list = [None] * (2 * L)

    def step_cat():
        for i in range(L):
            out[i] = torch.cat([cat_k[i], kn[i]], dim=2)
            out[L + i] = torch.cat([cat_v[i], vn[i]], dim=2)

    def step_static():
        for i in range(L):
            sta_k[i][:, :, n, :] = kn[i][:, :, 0, :]
            sta_v[i][:, :, n, :] = vn[i][:, :, 0, :]

    def step_paged():
        for i in range(L):
            flat[i].index_copy_(0, slot, new_kv[i])

    return {"cat": step_cat, "static": step_static, "paged": step_paged}, (sta_k, sta_v, cache, cat_k, cat_v)


def time_graph(step) -> float:
    """录一步,回放 REPLAYS 次,返回每步微秒数。"""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / REPLAYS


def main():
    name = torch.cuda.get_device_name()
    steps, keep = build(128)
    t0 = time.time()  # 稳态:持续负载至少 WARMUP_SECONDS 秒,让时钟升上来
    while time.time() - t0 < WARMUP_SECONDS:
        steps["cat"](); steps["static"](); steps["paged"]()
    torch.cuda.synchronize()
    warm_seconds = time.time() - t0
    result = {"gpu": name, "torch": torch.__version__, "shape": dict(layers=L, kv_heads=H, head_dim=D, batch=1),
              "bytes_per_token_all_layers": 2 * L * H * D * 2, "warmup_seconds": round(warm_seconds, 1),
              "gpu_after_warmup": gpu_state(), "points": []}
    for n in LENGTHS:
        steps, _ = build(n)
        for _ in range(30):
            steps["cat"](); steps["static"](); steps["paged"]()
        torch.cuda.synchronize()
        times = {"cat": [], "static": [], "paged": []}
        for r in range(ROUNDS):  # 交错:每轮换一个先后顺序
            order = ["cat", "static", "paged"][r % 3:] + ["cat", "static", "paged"][:r % 3]
            for k in order:
                times[k].append(time_graph(steps[k]))
        per_step_bytes = {"cat": 2 * result["bytes_per_token_all_layers"] * (n + 1),  # 读一遍再写一遍
                          "static": result["bytes_per_token_all_layers"], "paged": result["bytes_per_token_all_layers"]}
        point = {"n": n, "bytes_moved_per_step": per_step_bytes,
                 **{f"{k}_us": [round(x, 2) for x in v] for k, v in times.items()},
                 **{f"{k}_us_median": round(statistics.median(v), 2) for k, v in times.items()},
                 "gpu": gpu_state()}
        result["points"].append(point)
        print(f"n={n:6d}  cat {point['cat_us_median']:9.1f} us   static {point['static_us_median']:7.1f} us   "
              f"paged {point['paged_us_median']:7.1f} us   {point['gpu']}", flush=True)
        del steps, _
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
