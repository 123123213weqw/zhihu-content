"""Steady-state GEMM comparison: cuBLAS default vs Triton (16 configs) vs three hand-written CUDA kernels.

Protocol
  1. keep the GPU busy for WARM_S seconds first, so clocks have reached their steady state
  2. time every method with CUDA graphs (no Python launch overhead on the GPU timeline)
  3. round-robin ROUNDS times over all (size, method) pairs, so drift hits all methods alike
  4. sample SM clock / power with nvidia-smi in a background thread during the measurement phase
Env: NO_TRITON=1 skips Triton; SMI_INDEX selects the GPU for nvidia-smi.
"""
import ctypes, json, os, statistics, subprocess, threading, time
import torch

SIZES = [512, 1024, 2048, 4096, 8192]
ROUNDS, WARM_S, MIN_INNER_MS, MIN_TOTAL_MS = 5, 8.0, 4.0, 40.0
DO_TRITON = os.environ.get("NO_TRITON", "0") != "1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "steady.json"))
dev = "cuda"


if DO_TRITON:
    import triton, triton.language as tl

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bm, stride_bn, stride_cm, stride_cn,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      GROUP_SIZE_M: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M); num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bm + offs_bn[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
            a_ptrs += BLOCK_SIZE_K * stride_ak; b_ptrs += BLOCK_SIZE_K * stride_bm
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    CFGS = [(128, 128, 64, 8, 8, 3), (128, 256, 64, 8, 8, 3), (256, 128, 64, 8, 8, 3), (128, 128, 32, 8, 4, 3),
            (128, 64, 64, 8, 4, 4), (64, 128, 64, 8, 4, 4), (64, 64, 64, 8, 4, 4), (128, 128, 64, 8, 4, 3),
            (128, 128, 64, 8, 8, 4), (128, 256, 32, 8, 8, 4), (256, 256, 64, 8, 8, 2), (128, 128, 32, 8, 4, 4),
            (64, 256, 64, 8, 4, 4), (256, 64, 64, 8, 4, 4), (128, 256, 64, 8, 8, 4), (64, 64, 32, 4, 4, 4)]

lib = ctypes.CDLL(os.path.join(HERE, "libgemm.so"))
for _f in (lib.run_naive, lib.run_tiled, lib.run_regtile):
    _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def tflops(s, ms):
    return 2.0 * s * s * s / (ms * 1e-3) / 1e12


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.samples = []; self.on = False; self.stop = False

    def run(self):
        while not self.stop:
            if self.on:
                r = subprocess.run(["nvidia-smi", "-i", os.environ.get("SMI_INDEX", "0"),
                                    "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks.mem",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip():
                    self.samples.append([float(v) for v in r.stdout.strip().split(",")])
            time.sleep(0.2)


def est_ms(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(3): fn()
    t1.record(); t1.synchronize()
    return t0.elapsed_time(t1) / 3


def graph_timer(fn):
    """Capture `inner` back-to-back launches in a CUDA graph; replay `outer` times between two events."""
    e = max(est_ms(fn), 1e-3)
    inner = max(1, min(64, int(MIN_INNER_MS / e) + 1))
    outer = max(1, min(40, int(MIN_TOTAL_MS / (inner * e)) + 1))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)

    def run():
        g.replay(); torch.cuda.synchronize()
        t0.record()
        for _ in range(outer): g.replay()
        t1.record(); t1.synchronize()
        return t0.elapsed_time(t1) / (inner * outer)
    return run, inner, outer


def main():
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | torch {torch.__version__}" + (f" | triton {triton.__version__}" if DO_TRITON else ""), flush=True)
    # 1. warm-up: continuous load
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16); b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    t_end = time.time() + WARM_S
    while time.time() < t_end:
        for _ in range(20): torch.matmul(a, b)
        torch.cuda.synchronize()
    del a, b
    # 2. build methods
    methods = []  # (size, label, fn)
    for s in SIZES:
        a16 = torch.randn(s, s, device=dev, dtype=torch.float16); b16 = torch.randn(s, s, device=dev, dtype=torch.float16)
        a32 = a16.float(); b32 = b16.float()
        methods.append((s, "cublas_fp16", (lambda a=a16, b=b16: torch.matmul(a, b))))
        methods.append((s, "cublas_fp32", (lambda a=a32, b=b32: torch.matmul(a, b))))
        if DO_TRITON:
            for tag, (A, B, ref) in (("fp16", (a16, b16, torch.matmul(a16, b16).float())),
                                     ("fp32ieee", (a32, b32, torch.matmul(a32, b32)))):
                c = torch.empty((s, s), device=dev, dtype=A.dtype)
                for cfg in CFGS:
                    BM, BN, BK, G, w, st = cfg

                    def run(A=A, B=B, c=c, s=s, BM=BM, BN=BN, BK=BK, G=G, w=w, st=st):
                        grid = lambda META: (triton.cdiv(s, META["BLOCK_SIZE_M"]) * triton.cdiv(s, META["BLOCK_SIZE_N"]),)
                        matmul_kernel[grid](A, B, c, s, s, s, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                            c.stride(0), c.stride(1), BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                                            GROUP_SIZE_M=G, num_warps=w, num_stages=st)
                    try:
                        run(); torch.cuda.synchronize()
                        err = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
                        if err > 5e-2:
                            continue
                    except Exception:
                        continue
                    methods.append((s, f"triton_{tag}{cfg}", run))
        A32 = a32.contiguous(); B32 = b32.contiguous(); C32 = torch.empty((s, s), device=dev, dtype=torch.float32)
        ref32 = A32 @ B32
        for k in ("naive", "tiled", "regtile"):
            f = getattr(lib, "run_" + k)
            # the stream must be looked up at call time: during capture it is the capture stream
            fn = lambda f=f, A=A32, B=B32, C=C32, s=s: f(A.data_ptr(), B.data_ptr(), C.data_ptr(), s, torch.cuda.current_stream().cuda_stream)
            C32.zero_(); fn(); torch.cuda.synchronize()
            err = (C32 - ref32).abs().max().item() / (ref32.abs().max().item() + 1e-9)
            if err > 2e-4:
                print(f"  !! cuda_{k} at {s} looks wrong (rel err {err:.2e}), skipped", flush=True)
                continue
            methods.append((s, f"cuda_{k}", fn))
    timers = {}
    for i, (s, label, fn) in enumerate(methods):
        try:
            timers[(s, label)] = graph_timer(fn)
        except Exception as ex:
            print(f"  (skip {s} {label}: {str(ex)[:80]})", flush=True)
        if (i + 1) % 20 == 0:
            torch.cuda.synchronize()
    sampler = Sampler(); sampler.start(); sampler.on = True
    results = {}
    keys = list(timers.keys())
    import random
    for r in range(ROUNDS):
        order = keys[:]; random.Random(r).shuffle(order)  # a different order every round
        for key in order:
            run, inner, outer = timers[key]
            results.setdefault(key, []).append(run())
    sampler.on = False; time.sleep(0.4); sampler.stop = True
    if sampler.samples:
        clk = [x[0] for x in sampler.samples]; pw = [x[1] for x in sampler.samples]; tp = [x[2] for x in sampler.samples]
        print(f"clock during measurement: SM {min(clk):.0f}/{statistics.median(clk):.0f}/{max(clk):.0f} MHz (min/median/max), "
              f"power {min(pw):.0f}/{statistics.median(pw):.0f}/{max(pw):.0f} W, temp {min(tp):.0f}-{max(tp):.0f} C, {len(clk)} samples", flush=True)
    out = {"gpu": name, "torch": torch.__version__, "rounds": ROUNDS, "data": {}}
    for s in SIZES:
        print(f"--- {s} ---", flush=True)
        rows = [(label, v) for (ss, label), v in results.items() if ss == s]
        for label, v in sorted(rows, key=lambda kv: -statistics.median([tflops(s, m) for m in kv[1]])):
            t = [tflops(s, m) for m in v]
            out["data"][f"{s}:{label}"] = [round(x, 3) for x in t]
            spread = (max(t) / min(t) - 1) * 100
            print(f"  {label:28s} median {statistics.median(t):8.2f}  min {min(t):8.2f}  max {max(t):8.2f} TFLOPS  (rounds spread {spread:.1f}%)", flush=True)
        tr = [statistics.median([tflops(s, m) for m in v]) for (ss, label), v in results.items() if ss == s and label.startswith("triton_fp16")]
        if tr:
            out["data"][f"{s}:triton_fp16_best"] = max(tr); out["data"][f"{s}:triton_fp16_median"] = statistics.median(tr); out["data"][f"{s}:triton_fp16_worst"] = min(tr)
            print(f"  Triton fp16 over {len(tr)} configs: best {max(tr):.2f} / median {statistics.median(tr):.2f} / worst {min(tr):.2f} TFLOPS", flush=True)
        for base in ("cublas_fp16", "cublas_fp32", "cuda_naive", "cuda_tiled", "cuda_regtile"):
            v = results.get((s, base))
            if v:
                out["data"][f"{s}:{base}"] = round(statistics.median([tflops(s, m) for m in v]), 3)
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
