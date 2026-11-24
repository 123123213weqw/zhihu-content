#!/usr/bin/env python3
"""前缀缓存"开着但零命中"和"关着"相比,吞吐差多少(RTX 4090D 实测)。

同一个模型、同一张卡,交替启动两种配置的 vLLM 服务,各跑 ROUNDS 轮:
  off  --no-enable-prefix-caching
  on   默认(前缀缓存打开)
提示词全是随机 token 序列,互不相同,所以 on 的命中率是 0,每轮从 /metrics 核对。

每轮:启动服务 → 一小段预热 → 依次跑三种负载(每个提示词都要跑完 max_tokens 个):
  短进短出   输入 64,   输出 32,  并发 256, 2000 个请求
  解码为主   输入 512,  输出 512, 并发 32,  128 个请求
  预填充为主 输入 4096, 输出 32,  并发 8,   128 个请求
并发数按 KV 缓存的容量压着定(Llama-3.1-8B 在 24 GB 的卡上约 5 万个 token),全程不触发抢占。
每种负载的提示词由固定种子生成,两轮之间只有配置不同;测量期间每 2 秒记一次时钟和功耗。

Run:  python prefix_overhead.py --model /data/wangyue/models/Llama-3.1-8B-Instruct --gpu 0
      ->  prefix_overhead_4090d.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import re
import signal
import statistics
import subprocess
import threading
import time

import aiohttp

VLLM = "/data/wangyue/venvs/vllm/bin/vllm"
PORT = 18991
BASE = f"http://127.0.0.1:{PORT}"
ROUNDS = 5
VOCAB = 128_000  # 只用 16 到 128000 之间的普通 token,避开 Llama-3 的特殊 token
WARMUP = dict(name="预热", input_len=256, output_len=16, concurrency=64, num_prompts=256)
WORKLOADS = [
    dict(name="短进短出", input_len=64, output_len=32, concurrency=256, num_prompts=2000),
    dict(name="解码为主", input_len=512, output_len=512, concurrency=32, num_prompts=128),
    dict(name="预填充为主", input_len=4096, output_len=32, concurrency=8, num_prompts=128),
]
CONFIGS = {"off": ["--no-enable-prefix-caching"], "on": []}
COUNTERS = ("prompt_tokens", "generation_tokens", "prefix_cache_queries", "prefix_cache_hits", "num_preemptions")


def prompt_of(idx: int, length: int) -> list[int]:
    rng = random.Random(1_000_003 * idx + length)  # 每个请求的提示词都不一样
    return [rng.randrange(16, VOCAB) for _ in range(length)]


async def read_counters(session) -> dict:
    async with session.get(f"{BASE}/metrics") as r:
        text = await r.text()
    out = {k: 0 for k in COUNTERS}
    for k in COUNTERS:  # 计数器可能带标签,同一名字的全部加起来
        out[k] = sum(float(x) for x in re.findall(rf"^vllm:{k}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.e+]+)", text, re.M))
    return out


async def run_workload(session, w: dict) -> tuple[dict, dict]:
    prompts = [prompt_of(i, w["input_len"]) for i in range(w["num_prompts"])]
    sem = asyncio.Semaphore(w["concurrency"])
    done = [0]

    async def one(ids):
        async with sem:
            body = {"model": "m", "prompt": ids, "max_tokens": w["output_len"], "ignore_eos": True,
                    "temperature": 0.0, "seed": 0}
            for attempt in range(3):
                try:
                    async with session.post(f"{BASE}/v1/completions", json=body) as r:
                        r.raise_for_status()
                        data = await r.json()
                    break
                except aiohttp.ClientConnectionError:  # 连接被对端关掉就重发;下面的 token 数核对会发现重复计算
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1.0)
            assert data["usage"]["completion_tokens"] == w["output_len"], data["usage"]
            done[0] += 1

    before = await read_counters(session)
    with ClockSampler.hold() as clock:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in prompts))
        seconds = time.perf_counter() - t0
    after = await read_counters(session)
    out_tok, in_tok = w["num_prompts"] * w["output_len"], w["num_prompts"] * w["input_len"]
    assert after["generation_tokens"] - before["generation_tokens"] == out_tok
    assert after["prompt_tokens"] - before["prompt_tokens"] == in_tok
    res = dict(seconds=round(seconds, 3), requests=done[0], output_tokens=out_tok, prompt_tokens=in_tok,
               output_tps=round(out_tok / seconds, 1), total_tps=round((out_tok + in_tok) / seconds, 1),
               prefix_cache_queries=int(after["prefix_cache_queries"] - before["prefix_cache_queries"]),
               prefix_cache_hits=int(after["prefix_cache_hits"] - before["prefix_cache_hits"]),
               preemptions=int(after["num_preemptions"] - before["num_preemptions"]), clocks=clock.result)
    return res, after


class ClockSampler(threading.Thread):
    """每 2 秒记一次 GPU 的时钟、功耗、温度;hold() 期间的那一段单独汇总。"""
    current = None

    def __init__(self, gpu: int):
        super().__init__(daemon=True)
        self.q = "clocks.sm,power.draw,temperature.gpu,utilization.gpu"
        self.gpu = gpu
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            out = subprocess.run(["nvidia-smi", f"--query-gpu={self.q}", "--format=csv,noheader,nounits",
                                  "-i", str(self.gpu)], capture_output=True, text=True).stdout.strip()
            try:
                sm, power, temp, util = [float(x) for x in out.split(",")]
            except ValueError:
                sm = power = temp = util = 0.0
            if self.current is not None:
                self.current.append((sm, power, temp, util))
            time.sleep(2.0)

    class _Hold:
        def __enter__(self):
            ClockSampler.current = []
            return self

        def __exit__(self, *exc):
            samples = ClockSampler.current
            ClockSampler.current = None
            if samples:
                self.result = dict(samples=len(samples), sm_clock_mhz_min=int(min(s[0] for s in samples)),
                                   sm_clock_mhz_max=int(max(s[0] for s in samples)),
                                   power_w_max=round(max(s[1] for s in samples), 1), temp_c_max=int(max(s[2] for s in samples)))
            else:
                self.result = {}

    @classmethod
    def hold(cls):
        return cls._Hold()


def start_server(tag: str, extra: list[str], gpu: int, model: str, log_path: str):
    # VLLM_HTTP_TIMEOUT_KEEP_ALIVE:默认 5 秒就断开空闲连接,压测客户端会撞上
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", VLLM_HTTP_TIMEOUT_KEEP_ALIVE="600")
    cmd = [VLLM, "serve", model, "--port", str(PORT), "--max-model-len", "8192", "--max-num-seqs", "256",
           "--served-model-name", "m", "--disable-log-requests", *extra]
    log = open(log_path, "a")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, time.time() - t0


def stop_server(proc, gpu: int) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    for _ in range(120):  # 等显存真的还回去
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                               "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
        if used and int(used) < 1000:
            return
        time.sleep(2.0)


async def wait_healthy(session) -> bool:
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            async with session.get(f"{BASE}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(3.0)
    return False


async def one_round(tag: str, gpu: int, model: str, idx: int, out_path: str) -> dict:
    extra = CONFIGS[tag]
    log_path = f"server-{idx}-{tag}.log"
    proc, _ = start_server(tag, extra, gpu, model, log_path)
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=3600)) as s:
            if not await wait_healthy(s):
                raise RuntimeError(f"server {tag} did not become healthy; see {log_path}")
            w, _ = await run_workload(s, WARMUP)
            runs = []
            for wl in WORKLOADS:
                res, _ = await run_workload(s, wl)
                res["name"] = wl["name"]
                runs.append(res)
                print(f"  [{tag}] {wl['name']}: {res['output_tps']} tok/s 输出, {res['total_tps']} tok/s 总, "
                      f"命中 {res['prefix_cache_hits']}/{res['prefix_cache_queries']}, {res['seconds']} s", flush=True)
            kv_tokens = re.search(r"GPU KV cache size: ([0-9,]+) tokens", open(log_path).read())
            return {"round": idx, "config": tag, "warmup_seconds": w["seconds"], "runs": runs,
                    "kv_cache_tokens": int(kv_tokens.group(1).replace(",", "")) if kv_tokens else None}
    finally:
        stop_server(proc, gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args()
    sampler = ClockSampler(args.gpu)
    sampler.start()
    results = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(args.gpu)],
                                     capture_output=True, text=True).stdout.strip(),
               "vllm": subprocess.run([VLLM, "--version"], capture_output=True, text=True).stdout.strip(),
               "model": args.model, "workloads": WORKLOADS, "configs": CONFIGS, "rounds": []}
    order = ["off", "on"]
    for idx in range(args.rounds):  # 交错:这一轮先 off 后 on,下一轮反过来
        for tag in order if idx % 2 == 0 else order[::-1]:
            print(f"round {idx} config {tag} start", flush=True)
            results["rounds"].append(asyncio.run(one_round(tag, args.gpu, args.model, idx, f"prefix_overhead_4090d.json")))
            with open("prefix_overhead_4090d.json", "w") as f:
                json.dump(results, f, indent=1, ensure_ascii=False)
    sampler.stop_event.set()

    print("\n汇总(5 轮中位数):")
    for wl in WORKLOADS:
        row = {}
        for tag in order:
            vals = [r["output_tps"] for rd in results["rounds"] if rd["config"] == tag
                    for r in rd["runs"] if r["name"] == wl["name"]]
            tot = [r["total_tps"] for rd in results["rounds"] if rd["config"] == tag
                   for r in rd["runs"] if r["name"] == wl["name"]]
            row[tag] = (statistics.median(vals), statistics.median(tot), min(vals), max(vals))
        diff = row["on"][0] / row["off"][0] - 1
        print(f"  {wl['name']}: off {row['off'][0]:.0f} [{row['off'][2]:.0f},{row['off'][3]:.0f}]  "
              f"on {row['on'][0]:.0f} [{row['on'][2]:.0f},{row['on'][3]:.0f}]  差 {diff:+.2%}")


if __name__ == "__main__":
    main()
