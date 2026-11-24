#!/usr/bin/env python3
"""Every number quoted in the KV cache article, computed in one place.

Model configs (read from config.json on Hugging Face, 2026-11-24):
  Llama-2-7B    : 32 layers, 32 attention heads, 32 KV heads, hidden 4096, float16
  Llama-3-8B    : 32 layers, 32 attention heads,  8 KV heads, hidden 4096, bfloat16
  DeepSeek-V3   : 61 layers, kv_lora_rank 512, qk_rope_head_dim 64, 128 heads, qk_nope_head_dim 128, v_head_dim 128
Block size: 64 (vLLM 0.34 default), 16 before.
Run:  python3 calc.py
"""
KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
B_NEW, B_OLD = 64, 16


def kv_per_token(layers, kv_heads, head_dim, bytes_per_num):
    return 2 * layers * kv_heads * head_dim * bytes_per_num


def cdiv(a, b):
    return -(-a // b)


llama2 = dict(layers=32, kv_heads=32, head_dim=4096 // 32, bytes_per_num=2)
llama3 = dict(layers=32, kv_heads=8, head_dim=4096 // 32, bytes_per_num=2)
dsv3_latent = (512 + 64) * 61 * 2
dsv3_mha = kv_per_token(61, 128, 128, 2)  # if every head stored a 128-dim K and a 128-dim V

print("== 1. KV bytes per token")
for name, m in (("Llama-2-7B", llama2), ("Llama-3-8B", llama3)):
    v = kv_per_token(**m)
    print(f"{name}: {v} B = {v / KiB:.0f} KiB; block of {B_NEW} tokens = {v * B_NEW / MiB:.0f} MiB, "
          f"block of {B_OLD} tokens = {v * B_OLD / MiB:.1f} MiB")
print(f"DeepSeek-V3 MLA: {dsv3_latent} B = {dsv3_latent / KiB:.3f} KiB; block of {B_NEW} tokens = "
      f"{dsv3_latent * B_NEW / MiB:.3f} MiB")
print(f"DeepSeek-V3 if it stored full K and V for 128 heads: {dsv3_mha} B = {dsv3_mha / MiB:.3f} MiB; "
      f"ratio {dsv3_mha / dsv3_latent:.1f}x")

print("\n== 2. One request")
v2 = kv_per_token(**llama2)
print(f"Llama-2-7B, 4096 tokens: {v2 * 4096 / GiB:.0f} GiB; 10 such requests: {v2 * 4096 * 10 / GiB:.0f} GiB")
print(f"7B params in half precision: {7e9 * 2 / 1e9:.0f} GB")
v3 = kv_per_token(**llama3)
print(f"Llama-3-8B, 8192 tokens: {v3 * 8192 / GiB:.2f} GiB")
print(f"DeepSeek-V3, 131072 tokens: {dsv3_latent * 131072 / GiB:.2f} GiB")

print("\n== 3. Startup: 20 GiB for the KV cache")
free = 20 * GiB
for name, v, ctx in (("Llama-2-7B", v2, 4096), ("Llama-3-8B", v3, 8192)):
    for bs in (B_NEW, B_OLD):
        page = v * bs
        n = free // page
        print(f"{name}, block {bs}: page {page / MiB:.0f} MiB -> {n} blocks -> {n * bs} tokens -> "
              f"concurrency at {ctx}: {n * bs / ctx:.1f}x ({cdiv(ctx, bs)} blocks per request)")
left = free % (v2 * B_NEW)
print(f"leftover after floor division: {left} B")

print("\n== 4. Tail waste and bookkeeping vs block size")
for bs in (B_NEW, B_OLD):
    tail = bs - 1
    print(f"block {bs}: worst tail {tail} tokens = {tail * v2 / MiB:.2f} MiB; of a 4096-token request "
          f"{tail / 4096:.2%}; 131072-token table {131072 // bs} entries x 4 B = {131072 // bs * 4 / KiB:.0f} KiB; "
          f"hashes per 131072 tokens: {131072 // bs}")

print("\n== 5. Prefix caching")
shared = 2000
full_blocks = shared // B_NEW
print(f"shared prompt of {shared} tokens: {full_blocks} full blocks = {full_blocks * B_NEW} tokens hit; "
      f"99 later requests save {99 * full_blocks * B_NEW} prefill tokens; "
      f"shortest shared prefix that can hit: {B_NEW} (block 16: {B_OLD})")
print(f"prompt of 128 tokens, all seen before: hit length capped at 127 -> {127 // B_NEW} block = "
      f"{127 // B_NEW * B_NEW} tokens hit, {128 - 127 // B_NEW * B_NEW} recomputed")

print("\n== 6. One request's allocation trace (prompt 100 tokens, block 64)")
trace = []
held = 0
for step, need in [("prefill", 100)] + [(f"decode {k}", 100 + k) for k in range(1, 200)]:
    req = cdiv(need, B_NEW)
    if req > held:
        trace.append((step, req - held))
        held = req
print(trace[:4])

print("\n== 7. Sliding window vs full attention, context 131072, block 64")
for w in (1024, 4096):
    print(f"window {w}: at most {cdiv(w - 1, B_NEW) + 1} blocks per request (in-flight tokens not counted); "
          f"full attention: {cdiv(131072, B_NEW)} blocks")
print(f"example from the source: window 6 tokens, block 4 -> {cdiv(6, 4) + 1} blocks")

print("\n== 8. Hybrid grouping (logic of _get_kv_cache_groups_uniform_page_size)")
for name, full, sw in (("10 full + 20 sliding", 10, 20), ("gpt-oss-20b + eagle: 12 sliding + 13 full", 13, 12)):
    mn, mx = min(full, sw), max(full, sw)
    gs = mx if mx < mn * 1.5 else mn
    pad = (gs - full % gs) % gs + (gs - sw % gs) % gs
    print(f"{name}: group size {gs}, {cdiv(full, gs) + cdiv(sw, gs)} groups, {pad} padding layers")

print("\n== 9. Extensible cache")
print(f"MiniMax-M3 on H200: 433k -> 672k tokens = {(672 / 433 - 1):.1%} more")
print(f"margin floor {(1 << 30) / GiB:.0f} GiB; commit granule {(2 << 20) / MiB:.0f} MiB; "
      f"utilization default 0.92 -> 1.0")

print("\n== 10. 朴素写法的算量(b = 一个 token 的 KV,全部层相加,Llama-2-7B)")
b, N, Lmax = v2, 4096, 4096
written_cat = b * (N * (N + 1) // 2 + N)      # 每步拼接:第 n 步要写 n + 1 个 token
written_once = b * N                           # 其余写法:每个 token 只写一次
print(f"拼接 {N} 步:累计写入 {written_cat / 2**40:.2f} TiB,是 KV 缓存本身({written_once / 2**30:.0f} GiB)的 "
      f"{written_cat / written_once:.0f} 倍")
for n in (128, 512, 2048, 8192):
    print(f"  当前长度 {n:5d}:一步要搬(读一遍再写一遍) {2 * b * (n + 1) / 2**30:.2f} GiB")
dbl = sum(2 ** k for k in range(13))           # 倍增扩容:1, 2, 4, ..., 4096 各写一次
print(f"倍增扩容到 {N}:累计写入 {dbl * b / 2**30:.2f} GiB,不到最终大小的 2 倍;最后一刻占用在 50% 到 100% 之间")
for actual in (256, 512, 1024):
    print(f"  按最大长度 {Lmax} 预留,实际 {actual:4d}:利用率 {actual / Lmax:.1%},浪费 {(Lmax - actual) * b / 2**30:.2f} GiB")
print(f"分页,块 64:每个请求最多浪费 63 个 token = {63 * b / 2**20:.1f} MiB")

print("\n== 11. 实测(bench/naive_vs_paged_4090d.json,RTX 4090 D)")
import json
import os
import statistics
import time
pts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "naive_vs_paged_4090d.json")))["points"]
xs = [p["n"] for p in pts]
cat = [p["cat_us_median"] for p in pts]
xbar, ybar = statistics.mean(xs), statistics.mean(cat)
slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, cat)) / sum((x - xbar) ** 2 for x in xs)
intercept = ybar - slope * xbar
cum = {"拼接": intercept * N + slope * N * (N - 1) / 2, "预分配": pts[-1]["static_us_median"] * N, "分页": pts[-1]["paged_us_median"] * N}
for k, v in cum.items():
    print(f"  生成 {N} 个 token,{k}:每步 {pts[-1][{'拼接': 'cat', '预分配': 'static', '分页': 'paged'}[k] + '_us_median']} 微秒(长度 8192),"
          f"累计 {v / 1e6:.2f} 秒")
print(f"  拼接的拟合:t = {intercept:.1f} + {slope:.3f} x n(微秒);长度 8192 时一步搬 {2 * b * 8193 / 2**30:.2f} GiB,"
      f"折合 {2 * b * 8193 / (pts[-1]['cat_us_median'] * 1e-6) / 1e9:.0f} GB/s")

print("\n== 12. 算一个块的哈希要多久(vLLM 的 sha256:先 pickle,再哈希;本机 CPU,7 次取中位数)")
import hashlib
import pickle
import random
block = (random.randbytes(32), tuple(random.randrange(16, 128_000) for _ in range(64)), None)
runs = []
for _ in range(7):
    t0 = time.perf_counter()
    for _ in range(20000):
        hashlib.sha256(pickle.dumps(block, protocol=pickle.HIGHEST_PROTOCOL)).digest()
    runs.append((time.perf_counter() - t0) / 20000)
per_block = statistics.median(runs) * 1e6
print(f"  每块 {per_block:.2f} 微秒,摊到每个 token {per_block / 64:.3f} 微秒;4096 个 token 的提示词 64 块,共 {per_block * 64 / 1000:.2f} 毫秒")
