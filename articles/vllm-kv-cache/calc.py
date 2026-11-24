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
