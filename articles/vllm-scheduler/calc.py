#!/usr/bin/env python3
"""调度器一篇里引用的每个数,都在这里算一遍。输出 bench/calc_summary.json,图和正文都从它读。

模型:Llama-3.1-8B-Instruct,一张 RTX 4090 D,vLLM 0.34.0。
实测数据在 bench/results/ 里,由 bench/analyze.py 从调度记录和客户端结果汇总而来。
运行:python3 calc.py
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import statistics
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "bench", "results")


def load(name):
    return json.load(open(os.path.join(RES, name)))


# ---- 模型形状(config.json):隐层 4096,MLP 中间维 14336,32 层,32 个注意力头,8 个 KV 头,每头 128 维,词表 128,256
HIDDEN, INTER, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 4096, 14336, 32, 32, 8, 128, 128256
BF16 = 2
PARAMS_LAYER = HIDDEN * HIDDEN + 2 * HIDDEN * KV_HEADS * HEAD_DIM + HIDDEN * HIDDEN + 3 * HIDDEN * INTER
PARAMS_EMBED = VOCAB * HIDDEN
PARAMS_TOTAL = LAYERS * PARAMS_LAYER + 2 * PARAMS_EMBED       # 词嵌入和输出层各一份,不共享
KV_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BF16        # K 和 V 各一份,单位字节
BLOCK = 64                                                     # 0.34 的默认块大小
BLOCKS = 744                                                   # --gpu-memory-utilization 0.95 时启动日志给的块数
CAPACITY_TOKENS = BLOCKS * BLOCK
COPY_BANDWIDTH = 943.9                                         # 融合算子一篇在 4090D 上量的拷贝带宽,GB/s
GEMM_PEAK_TFLOPS = 143.0                                       # 同上,bf16 矩阵乘的峰值

short = load("sweep_short_summary.json")
long_ = load("sweep_long_summary.json")
capacity = load("capacity_summary.json")
batching = load("batching_summary.json")
mixed = {b: load(f"mixed_{b}_summary.json") for b in (512, 2048, 8192)}
preempt = load("preempt_summary.json")
sustain = {"small": load("sustain_summary.json"), "small_watermark": load("sustain_wm_summary.json")}

out = {"model": {}}
m = out["model"]
m["params_layer"] = PARAMS_LAYER
m["params_embed"] = PARAMS_EMBED
m["params_total"] = PARAMS_TOTAL
m["checkpoint_gb"] = PARAMS_TOTAL * BF16 / 1e9
# 解码一步要读的权重:各层加输出层。嵌入表只取用到的几行,不整张读。
m["weights_read_per_decode_step_gb"] = (LAYERS * PARAMS_LAYER + PARAMS_EMBED) * BF16 / 1e9
m["flops_per_prefill_token"] = 2 * LAYERS * PARAMS_LAYER
m["kv_per_token_bytes"] = KV_PER_TOKEN
m["block_tokens"] = BLOCK
m["block_bytes"] = BLOCK * KV_PER_TOKEN
m["blocks"] = BLOCKS
m["capacity_tokens"] = CAPACITY_TOKENS
m["concurrency_at"] = {str(n): CAPACITY_TOKENS / n for n in (512, 576, 1024, 2048, 4096, 8192, 16384)}


# ---- 1. 一步 decode 的时间随 batch 怎么变
def pick(sweep, b):
    """两轮的稳态中位数。两轮差 2% 以上,说明大的那轮被别的进程打断过,取小的那轮。"""
    vals = [r["step_ms_steady_median"] for r in sweep["runs"] if r["b"] == b and r["step_ms_steady_median"]]
    return (min(vals) if max(vals) / min(vals) > 1.02 else statistics.fmean(vals)), vals


def cpu_at(sweep, b, key):
    return statistics.fmean(r[key] for r in sweep["runs"] if r["b"] == b and r.get(key) is not None)


B_SHORT = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256)
B_LONG = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64)
t_short = {b: pick(short, b)[0] for b in B_SHORT}
t_long = {b: pick(long_, b)[0] for b in B_LONG}
d = out["decode"] = {}
# 稳态窗口取中间 5/7 的步:提示词长度,加上那一段已生成的平均数
d["context_short_mean"] = 64 + (8 + 48) / 2
d["context_long_mean"] = 256 + (46 + 274) / 2
d["t_short_ms"] = {str(b): round(v, 3) for b, v in t_short.items()}
d["t_long_ms"] = {str(b): round(v, 3) for b, v in t_long.items()}
d["t_short_rounds"] = {str(b): [round(x, 3) for x in pick(short, b)[1]] for b in B_SHORT}
d["t_long_rounds"] = {str(b): [round(x, 3) for x in pick(long_, b)[1]] for b in B_LONG}
T1 = t_short[1]
d["t1_ms"] = T1
d["bandwidth_effective_gbps"] = m["weights_read_per_decode_step_gb"] / (T1 / 1e3)
d["bandwidth_ratio_vs_copy"] = d["bandwidth_effective_gbps"] / COPY_BANDWIDTH
d["throughput_tok_s"] = {str(b): round(b / v * 1e3, 1) for b, v in t_short.items()}
d["throughput_gain_vs_1"] = {str(b): round((b / v) / (1 / T1), 1) for b, v in t_short.items()}
d["per_request_tok_s"] = {str(b): round(1e3 / v, 1) for b, v in t_short.items()}
d["step_time_growth"] = {str(b): round(v / T1, 3) for b, v in t_short.items()}
# 同样 64 条,上下文从平均 92 涨到 416:多出来的时间就是多读的 KV
extra_tokens = 64 * (d["context_long_mean"] - d["context_short_mean"])
d["kv_extra_gb_at_64"] = extra_tokens * KV_PER_TOKEN / 1e9
d["kv_extra_ms_at_64"] = t_long[64] - t_short[64]
d["kv_bandwidth_gbps"] = d["kv_extra_gb_at_64"] / (d["kv_extra_ms_at_64"] / 1e3)
d["sched_ms"] = {str(b): round(cpu_at(short, b, "sched_ms_median"), 3) for b in B_SHORT}
d["update_ms"] = {str(b): round(cpu_at(short, b, "update_ms_median"), 3) for b in B_SHORT}
d["cpu_total_ms"] = {str(b): round(d["sched_ms"][str(b)] + d["update_ms"][str(b)], 2) for b in B_SHORT}
d["cpu_share_of_step"] = {str(b): round(d["cpu_total_ms"][str(b)] / t_short[b], 3) for b in B_SHORT}

# 时钟:两次扫描期间 nvidia-smi 每 0.5 秒采一次
cl = []
with open(os.path.join(RES, "clocks_sweeps.csv")) as f:
    for row in csv.reader(f):
        if len(row) < 6:
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            cl.append((t, float(row[1]), float(row[3])))
        except ValueError:
            pass
clock = out["clocks"] = {}
for name, fn in (("short", "sweep_short.json"), ("long", "sweep_long.json")):
    res = {}
    for run in load(fn)["runs"]:
        sel = [x for x in cl if run["t0"] + 1.0 <= x[0] <= run["t1"]]
        if sel and run["b"] not in res:
            res[run["b"]] = [statistics.median(x[1] for x in sel), statistics.median(x[2] for x in sel), max(x[2] for x in sel)]
    clock[name] = {str(b): [round(v[0]), round(v[1]), round(v[2])] for b, v in res.items()}


# ---- 2. KV 容量决定并发上限(第一次扫描里 batch 128 和 256 两组,每条 256+320=576 token)
def track(series):
    """第一条请求结束之前,在跑的条数往下掉的几个点。"""
    pts = []
    last = None
    for x in series:
        if x[2] > 0 and x[1] != last:       # 等待队列不空,并且在跑的条数变了
            pts.append([round(x[0] / 1e3, 2), x[1], x[2]])
            last = x[1]
    return pts


cap = out["capacity"] = {}
for b, runs in capacity["runs"].items():
    r = runs[0]
    cap[b] = {
        "preemptions_both_rounds": sorted({x["preemptions"] for x in runs}),
        "lost_tokens": r["lost_tokens"],
        "max_run": r["max_run"],
        "waiting_peak": r["waiting_peak"],
        "client_ttft_median_s": r["client_ttft_median_s"],
        "client_ttft_max_s": r["client_ttft_max_s"],
        "client_e2e_max_s": r["client_e2e_max_s"],
        "track": track(r["series"])[:12],
    }
cap["expected_fit_576"] = CAPACITY_TOKENS // 576


# ---- 3. 整批走完再放下一批 vs 完成一个补一个
bt = out["batching"] = {"agg": batching["agg"], "n": len(batching["lengths"]), "total_output_tokens": sum(batching["lengths"])}
L = batching["lengths"]
bt["mean_len"] = statistics.fmean(L)
bt["max_len"] = max(L)
s_, c_ = batching["agg"]["static"], batching["agg"]["continuous"]
bt["speedup"] = c_["tokens_per_s"] / s_["tokens_per_s"]
waves = [L[i:i + 32] for i in range(0, len(L), 32)]
bt["wave_utilization"] = [round(statistics.fmean(w) / max(w), 3) for w in waves]
bt["wave_utilization_mean"] = statistics.fmean(bt["wave_utilization"])
# 输出长度服从指数分布时,b 条里最大值的期望是 平均长度 × 调和数,利用率是调和数的倒数
H32 = sum(1 / j for j in range(1, 33))
bt["harmonic_32"] = H32
bt["util_exponential_32"] = 1 / H32
bt["step_ms"] = {mode: statistics.fmean(r["engine_step_ms_median"] for r in batching["runs"] if r["mode"] == mode)
                 for mode in ("static", "continuous")}
# 每一轮实测的在跑条数的平均值(记录里每一步的 run)
bt["mean_running"] = {mode: round(statistics.fmean(x[1] for r in batching["runs"] if r["mode"] == mode for x in r["series"]), 2)
                      for mode in ("static", "continuous")}


GRAPH_SIZES = (1, 2, 4, 8) + tuple(range(16, 257, 8))     # 启动日志里的 decode CUDA Graph 抓取大小


def pad(n):
    """CUDA Graph 只抓了 1,2,4,8 和 8 的倍数这些 batch 大小,别的向上补齐到最近的一个。"""
    return min(x for x in GRAPH_SIZES if x >= n)


def t_decode(n, ctx_mean):
    """n 条在解码、平均上下文 ctx_mean 时的一步(毫秒):实测的短上下文步时间,加上多读的 KV。"""
    extra = n * (ctx_mean - d["context_short_mean"]) * KV_PER_TOKEN / (d["kv_bandwidth_gbps"] * 1e9) * 1e3
    return t_short[pad(n)] + extra


PROMPT = 128
PREFILL_US_PER_TOKEN = None      # 下面用 mixed 的实测填


def sim_static(lengths, b):
    """整批走完再放下一批:每一波的时间是其中最长那条的步数,批里在跑的条数一路往下掉。"""
    t = 0.0
    for i in range(0, len(lengths), b):
        wave = lengths[i:i + b]
        t += PREFILL_US_PER_TOKEN * 1e-6 * PROMPT * len(wave)
        for k in range(max(wave)):
            n = sum(1 for x in wave if x > k)
            t += t_decode(n, PROMPT + k) / 1e3
    return t


def sim_continuous(lengths, b, budget=2048):
    """完成一个补一个:每一步给在跑的请求各算一个 token,剩下的 token 预算给等待的请求做 prefill。"""
    t = 0.0
    progress = []                # 每条在跑的请求:(已生成几个 token, 总长)
    queue = deque(lengths)
    while progress or queue:
        n = len(progress)
        if n:
            t += t_decode(n, PROMPT + statistics.fmean(p for p, _ in progress)) / 1e3
            progress = [(p + 1, total) for p, total in progress if p + 1 < total]
        room = budget - len(progress)
        admitted = []
        while queue and room >= PROMPT and len(progress) + len(admitted) < b:
            admitted.append(queue.popleft())
            room -= PROMPT
        t += PREFILL_US_PER_TOKEN * 1e-6 * PROMPT * len(admitted)
        progress += [(0, total) for total in admitted]
    return t


# ---- 4. token 预算:首 token 时间和解码间隔的取舍
mx = out["mixed"] = {}
DECODE_ONLY = statistics.fmean(v["bg_engine_itl_decode_only_ms"]["mean"] for v in mixed.values())
mx["decode_only_ms"] = DECODE_ONLY
mx["bg_streams"] = mixed[512]["args"]["bg"]
mx["bg_prompt"] = mixed[512]["args"]["bg_prompt"]
mx["fg_prompt"] = mixed[512]["args"]["fg_prompt"]
mx["by_budget"] = {}
for b, s in mixed.items():
    tt = {x[0]: x[1] for x in s["series"]}
    chunk_dt, gpu_ms, first_chunk = [], [], []
    for r in s["fg"]:
        k0 = r["admit_step"]
        dt = [tt[k] - tt[k - 1] for k in range(k0, k0 + r["n_prefill_steps"]) if k in tt and k - 1 in tt]
        chunk_dt += dt
        gpu_ms.append(sum(dt))
        if dt:
            first_chunk.append(dt[0])
    e = mx["by_budget"][str(b)] = {
        "chunks_per_prompt": sorted(set(r["n_prefill_steps"] for r in s["fg"])),
        "prefill_tokens_per_step": sorted(set(n for r in s["fg"] for n in r["prefill_tokens_per_step"] if n > 100)),
        "engine_ttft_ms": s["fg_engine_ttft_ms"],
        "client_ttft_ms": s["fg_client_ttft_ms"],
        "itl_decode_only_ms": s["bg_engine_itl_decode_only_ms"]["mean"],
        "itl_p50_ms": s["bg_engine_itl_ms"]["p50"],
        "itl_p99_ms": s["bg_engine_itl_ms"]["p99"],
        "itl_max_ms": s["bg_engine_itl_ms"]["max"],
        "itl_mean_ms": s["bg_engine_itl_ms"]["mean"],
        "steps_with_prefill": s["steps_with_prefill"],
        "steps_total": s["window_steps"],
        "prefill_gpu_ms_mean": statistics.fmean(gpu_ms),
        "prefill_gpu_ms_min": min(gpu_ms),
        "prefill_gpu_ms_max": max(gpu_ms),
        "chunk_step_ms_min": min(chunk_dt),
        "chunk_step_ms_median": statistics.median(chunk_dt),
        "chunk_step_ms_max": max(chunk_dt),
        "first_chunk_step_ms_median": statistics.median(first_chunk),
        "preemptions": s["preemptions"],
        "prefix_hits": s["prefix_hits_in_window"],
        "bg_client_itl_mean_ms": s["bg_client_itl_mean_ms"],
    }
    e["prefill_us_per_token"] = e["prefill_gpu_ms_mean"] * 1e3 / 8192
    e["prefill_gemm_tflops"] = m["flops_per_prefill_token"] * 8192 / (e["prefill_gpu_ms_mean"] / 1e3) / 1e12
    e["prefill_vs_gemm_peak"] = e["prefill_gemm_tflops"] / GEMM_PEAK_TFLOPS
    e["itl_max_in_decode_steps"] = e["itl_max_ms"] / DECODE_ONLY
PREFILL_US_PER_TOKEN = mx["by_budget"]["2048"]["first_chunk_step_ms_median"] * 1e3 / 2036     # 2048 的块,位置靠前
mx["prefill_us_per_token_2048_chunk"] = PREFILL_US_PER_TOKEN
e512, e2048, e8192 = (mx["by_budget"][str(b)] for b in (512, 2048, 8192))
mx["ttft_ratio_512_vs_8192"] = e512["engine_ttft_ms"]["mean"] / e8192["engine_ttft_ms"]["mean"]
mx["itl_max_ratio_8192_vs_512"] = e8192["itl_max_ms"] / e512["itl_max_ms"]

# 上面 mixed 的数出来了,再回头做 batching 的模拟
bt["sim_static_s"] = sim_static(L, 32)
bt["sim_continuous_s"] = sim_continuous(L, 32)
bt["sim_static_vs_measured"] = bt["sim_static_s"] / s_["wall_s"]
bt["sim_continuous_vs_measured"] = bt["sim_continuous_s"] / c_["wall_s"]
bt["sim_speedup"] = bt["sim_static_s"] / bt["sim_continuous_s"]


# ---- 5. 抢占
P = out["preempt"] = {}
P["small_blocks"] = preempt["header"]["num_blocks"]
P["small_tokens"] = P["small_blocks"] * BLOCK
P["n"] = preempt["n_requests"]
P["per_request_tokens"] = preempt["args"]["prompt"] + preempt["args"]["max_tokens"]
P["per_request_blocks"] = -(-P["per_request_tokens"] // BLOCK)
P["fit_requests"] = P["small_blocks"] // P["per_request_blocks"]
P["preemptions"] = preempt["n_preemptions"]
P["lost_tokens"] = preempt["lost_tokens_sum"]
P["lost_tokens_per_preemption_min"] = min(p["lost"] for p in preempt["preemptions"])
P["lost_tokens_per_preemption_max"] = max(p["lost"] for p in preempt["preemptions"])
P["wall_s"] = preempt["wall_s"]
P["output_tokens"] = preempt["output_tokens"]
reqs = preempt["requests"]
clean = [v for v in reqs.values() if not v["preempts"]]
hit = [v for v in reqs.values() if v["preempts"]]
P["requests_never_preempted"] = len(clean)
P["requests_preempted"] = len(hit)
P["first_preemption_s"] = preempt["preemptions"][0]["t_ms"] / 1e3
# 抢占不是匀速发生的:剩下的请求每写满一块(64 个 token)就要再要一块,要不到就踢几条,腾出的块够它们再写一块
_pe = sorted(preempt["preemptions"], key=lambda x: x["t_ms"])
_bursts, _cur = [], [_pe[0]]
for _a, _b in zip(_pe, _pe[1:]):
    if _b["t_ms"] - _a["t_ms"] < 200:      # 0.2 秒内的算一批
        _cur.append(_b)
    else:
        _bursts.append(_cur)
        _cur = [_b]
_bursts.append(_cur)
P["bursts"] = [[round(c[0]["t_ms"] / 1e3, 2), len(c)] for c in _bursts]
P["burst_interval_s"] = statistics.fmean(b[0] - a[0] for a, b in zip(P["bursts"], P["bursts"][1:]))
# 每一轮之后在跑的条数。每条请求写满一块(64 个 token)就要再要一块,装得下的条数 = 总块数 ÷ 每条当时的块数
P["run_after_bursts"] = [next(x[1] for x in preempt["series"] if x[0] / 1e3 >= bt + 0.05) for bt, _ in P["bursts"]]
P["fit_by_blocks"] = {str(k): P["small_blocks"] // k for k in range(6, 6 + len(P["bursts"]))}
P["steps_per_block_ms"] = P["burst_interval_s"] / 64 * 1e3
P["clean_finish_s"] = [min(v["finish"] for v in clean) / 1e3, max(v["finish"] for v in clean) / 1e3]
P["hit_finish_s"] = [min(v["finish"] for v in hit) / 1e3, max(v["finish"] for v in hit) / 1e3]
P["hit_resume_s"] = [min(v["admits"][1] for v in hit) / 1e3, max(v["admits"][1] for v in hit) / 1e3]
P["hit_first_admit_s"] = [min(v["admits"][0] for v in hit) / 1e3, max(v["admits"][0] for v in hit) / 1e3]
# 同样的请求 KV 够用时要多久:64 条一起解码 512 步,按实测的步时间算,是推算不是实测
P["estimated_no_preemption_step_ms"] = t_decode(64, preempt["args"]["prompt"] + preempt["args"]["max_tokens"] / 2)
P["estimated_no_preemption_s"] = (preempt["args"]["max_tokens"] * P["estimated_no_preemption_step_ms"] / 1e3
                                  + PREFILL_US_PER_TOKEN * 1e-6 * preempt["args"]["prompt"] * P["n"])
P["wall_vs_estimate"] = P["wall_s"] / P["estimated_no_preemption_s"]

SU = out["sustain"] = {}
for key, s in sustain.items():
    SU[key] = {
        "watermark": s["header"]["watermark"],
        "blocks": s["header"]["num_blocks"],
        "requests": s["n_requests"],
        "preemptions": s["n_preemptions"],
        "requests_preempted": s["requests_preempted"],
        "requests_preempted_2plus": s["requests_preempted_2plus"],
        "max_preemptions_one_request": s["max_preemptions_one_request"],
        "lost_tokens": s["lost_tokens_sum"],
        "output_tokens": s["output_tokens"],
        "wall_s": s["wall_s"],
        "tokens_per_s": s["tokens_per_s"],
        "e2e_mean_s": s["e2e_mean_s"],
        "e2e_p50_s": s["e2e_p50_s"],
        "e2e_p99_s": s["e2e_p99_s"],
        "e2e_max_s": s["e2e_max_s"],
        "e2e_mean_s_never_preempted": s["e2e_mean_s_never_preempted"],
        "e2e_mean_s_preempted": s["e2e_mean_s_preempted"],
        "kv_mean_usage": s["kv_mean_usage"],
        "run_mean": s["run_mean"],
        "wait_mean": s["wait_mean"],
    }
a, w = SU["small"], SU["small_watermark"]
SU["preemption_ratio"] = a["preemptions"] / w["preemptions"]
SU["lost_ratio"] = a["lost_tokens"] / w["lost_tokens"]
SU["throughput_change"] = w["tokens_per_s"] / a["tokens_per_s"]
SU["prompt_tokens_all"] = a["requests"] * PROMPT
SU["lost_share_of_prompt_tokens"] = a["lost_tokens"] / SU["prompt_tokens_all"]

# ---- 汇总里的 series 太长,图用得上,calc 结果里不重复存
for v in list(mx["by_budget"].values()):
    v.pop("prefill_tokens_per_step", None)

with open(os.path.join(HERE, "bench", "calc_summary.json"), "w") as f:
    json.dump(out, f, indent=1)


def show():
    print(f"参数:每层 {PARAMS_LAYER/1e6:.1f} M,词嵌入 {PARAMS_EMBED/1e6:.1f} M,合计 {PARAMS_TOTAL/1e9:.3f} B,"
          f"检查点 {m['checkpoint_gb']:.2f} GB;解码一步读 {m['weights_read_per_decode_step_gb']:.2f} GB")
    print(f"KV 每个 token {KV_PER_TOKEN:,} 字节;{BLOCKS} 块 × {BLOCK} = {CAPACITY_TOKENS:,} token;"
          f"576 token 一条时能装 {CAPACITY_TOKENS/576:.1f} 条")
    print(f"t(1) = {T1:.2f} ms -> {d['bandwidth_effective_gbps']:.0f} GB/s,是拷贝带宽的 {d['bandwidth_ratio_vs_copy']:.3f}")
    print("t(b) 短上下文:", {b: round(v, 2) for b, v in t_short.items()})
    print("t(b) 长上下文:", {b: round(v, 2) for b, v in t_long.items()})
    print("吞吐 tok/s:", d["throughput_tok_s"])
    print(f"64 条时上下文从 {d['context_short_mean']:.0f} 到 {d['context_long_mean']:.0f}:多 {d['kv_extra_ms_at_64']:.2f} ms,"
          f"多读 {d['kv_extra_gb_at_64']:.2f} GB -> {d['kv_bandwidth_gbps']:.0f} GB/s")
    print("调度器 CPU(排 + 更新, ms):", d["cpu_total_ms"], "占一步的比例:", d["cpu_share_of_step"])
    print("时钟:", clock)
    print("容量限制:", json.dumps(cap, ensure_ascii=False)[:900])
    print(f"静态 {s_['wall_s']:.1f} s {s_['tokens_per_s']:.0f} tok/s;连续 {c_['wall_s']:.1f} s {c_['tokens_per_s']:.0f} tok/s;"
          f"快 {bt['speedup']:.2f} 倍;每波利用率 {bt['wave_utilization_mean']:.3f};平均在跑 {bt['mean_running']}")
    print(f"模拟:静态 {bt['sim_static_s']:.1f} s(实测的 {bt['sim_static_vs_measured']:.2f} 倍),"
          f"连续 {bt['sim_continuous_s']:.1f} s(实测的 {bt['sim_continuous_vs_measured']:.2f} 倍)")
    for b, e in mx["by_budget"].items():
        print(f"预算 {b}: 块数 {e['chunks_per_prompt']},纯解码一步 {e['itl_decode_only_ms']:.1f} ms,最大间隔 {e['itl_max_ms']:.0f} ms"
              f"({e['itl_max_in_decode_steps']:.1f} 步),块这一步 {e['chunk_step_ms_min']:.0f}-{e['chunk_step_ms_max']:.0f} ms,"
              f"首块中位 {e['first_chunk_step_ms_median']:.0f},首 token {e['engine_ttft_ms']['mean']:.0f} ms(客户端 {e['client_ttft_ms']['mean']:.0f}),"
              f"prefill 共 {e['prefill_gpu_ms_mean']:.0f} ms,每 token {e['prefill_us_per_token']:.0f} us,{e['prefill_gemm_tflops']:.0f} TFLOPS")
    print(f"首 token 512/8192 = {mx['ttft_ratio_512_vs_8192']:.3f};最大间隔 8192/512 = {mx['itl_max_ratio_8192_vs_512']:.1f};"
          f"prefill 每 token {PREFILL_US_PER_TOKEN:.1f} us")
    print(json.dumps(P, ensure_ascii=False, indent=1))
    print(json.dumps(SU, ensure_ascii=False, indent=1))


show()
