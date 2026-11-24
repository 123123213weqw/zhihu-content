"""Skinny GEMMs (decode-like shapes) with the same steady-state, CUDA-graph timing as steady.py."""
import statistics, time, torch
from steady import graph_timer

def kname(fn):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ks = sorted({e.key for e in p.key_averages() if "gemm" in e.key.lower() or "gemv" in e.key.lower() or "xmma" in e.key.lower()
                 or "cutlass" in e.key.lower() or "reduce" in e.key.lower()})
    return [k[k.find("<") + 1:][:120] if "<" in k else k[:120] for k in ks]

print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16); b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
t_end = time.time() + 8.0
while time.time() < t_end:
    for _ in range(20): torch.matmul(a, b)
    torch.cuda.synchronize()
del a, b
shapes = [(1, 4096, 4096), (8, 4096, 4096), (16, 4096, 4096), (64, 4096, 4096), (512, 4096, 4096)]
items = []
for (M, N, K) in shapes:
    for dt in (torch.float32, torch.float16):
        a = torch.randn(M, K, device="cuda", dtype=dt); b = torch.randn(K, N, device="cuda", dtype=dt)
        fn = lambda a=a, b=b: torch.matmul(a, b)
        run, inner, outer = graph_timer(fn)
        items.append((M, N, K, dt, fn, run, inner, outer))
res = {}
for r in range(7):  # interleaved rounds, so clock drift hits every shape alike
    for it in items:
        res.setdefault(id(it), []).append(it[5]())
out = {}
for it in items:
    M, N, K, dt, fn, run, inner, outer = it
    ms = statistics.median(res[id(it)])
    nbytes = (K * N + M * K + M * N) * (2 if dt == torch.float16 else 4)
    tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    bw = nbytes / (ms * 1e-3) / 1e9
    print(f"M={M:4d} N={N} K={K} {'fp16' if dt == torch.float16 else 'fp32'}: {ms*1e3:8.1f} us  {tf:7.3f} TFLOPS  {bw:7.1f} GB/s  (graph {inner}x{outer})  -> {kname(fn)}", flush=True)
    out[f"{M}:{'fp16' if dt == torch.float16 else 'fp32'}"] = {"us": round(ms * 1e3, 2), "tflops": round(tf, 4), "gbps": round(bw, 1)}
import json; json.dump(out, open("skinny_" + ("4080" if "4080" in torch.cuda.get_device_name(0) else "4090d") + ".json", "w"), indent=1)
