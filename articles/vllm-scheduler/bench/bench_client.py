#!/usr/bin/env python3
"""调度器实测的客户端。在服务器本机跑,默认请求 http://127.0.0.1:8000(可用 BENCH_URL 改)。

提示词直接给 token id:从一份英文原文(牛顿《光学》)切出来的连续片段,起点随机,
所以彼此没有公共前缀,长度精确。每条请求记下发出时刻、每个 token 到达的时刻、
服务端给的请求 id。结果写 JSON,路径相对本目录。

别用随机 token id 当提示词。随机 id 解码出来大量是残缺的 UTF-8,前端的 detokenizer
走慢路径,事件循环被卡住,token 成坨到达,客户端量到的间隔全是假的。引擎内部的
步时间不受影响(调度记录器里是稳的),但首 token 时间和客户端一侧的间隔不能看。

  bench_client.py decode_sweep --out results/sweep_short.json --batches 1,2,4,8,16,24,32,40,48,64,96,128,192,256 --prompt 64 --max-tokens 56
  bench_client.py decode_sweep --out results/sweep_long.json --batches 1,2,4,8,16,24,32,40,48,64 --prompt 256 --max-tokens 320
  bench_client.py mixed      --out results/mixed_512.json --fg-prompt 8192 --fg-n 16 --gap 2
  bench_client.py batching   --out results/batching.json --n 256 --b 32
  bench_client.py preempt    --out results/preempt.json --n 64 --prompt 128 --max-tokens 512
  bench_client.py sustain    --out results/sustain.json --n 64 --prompt 128 --max-len 512 --seconds 30
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time

import aiohttp

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8000")
MODEL = "/data/wangyue/models/Llama-3.1-8B-Instruct"
CORPUS_FILE = os.environ.get(
    "BENCH_CORPUS",
    "/data/wangyue/venvs/vllm/lib/python3.12/site-packages/tvm_ffi/3rdparty/libbacktrace/Isaac.Newton-Opticks.txt",
)
CORPUS: list[int] = []


def load_corpus() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(CORPUS_FILE, encoding="utf-8", errors="ignore").read()
    CORPUS.extend(tok(text, add_special_tokens=False)["input_ids"])
METRICS = [
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:request_success_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
]


def token_ids(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    start = rng.randrange(0, len(CORPUS) - n - 1)
    return CORPUS[start : start + n]


async def stream(session: aiohttp.ClientSession, ids: list[int], max_tokens: int, tag: str) -> dict:
    body = {
        "model": "m",
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.time()
    ts: list[float] = []
    rid, usage = None, None
    async with session.post(URL + "/v1/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            rid = obj.get("id", rid)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices")
            if choices and choices[0].get("text"):
                ts.append(time.time())
    # 客户端或前端来不及取时,服务端会把几步的 token 并进一个流式事件:记下并了几次。
    assert usage is not None and len(ts) <= usage["completion_tokens"], (len(ts), usage)
    return {
        "merged": usage["completion_tokens"] - len(ts),
        "tag": tag,
        "rid": rid[-6:],
        "t_send": t_send,
        "ts": ts,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(r: dict) -> dict:
    itl = [b - a for a, b in zip(r["ts"], r["ts"][1:])]
    out = {k: v for k, v in r.items() if k != "ts"}
    out["ttft"] = r["ts"][0] - r["t_send"]
    out["e2e"] = r["ts"][-1] - r["t_send"]
    # 平均值按 token 数算,不受合并事件影响;分位数只在没有合并时才准。
    if itl:
        out.update(
            itl_mean=(r["ts"][-1] - r["ts"][0]) / (r["completion_tokens"] - 1),
            itl_p50=pct(itl, 0.5),
            itl_p99=pct(itl, 0.99),
            itl_max=max(itl),
        )
    return out


async def scrape(session) -> dict[str, float]:
    out = {}
    async with session.get(URL + "/metrics") as resp:
        for line in (await resp.text()).splitlines():
            if not line.startswith("vllm:"):
                continue
            name, _, value = line.rpartition(" ")
            name = name.split("{")[0]
            out.setdefault(name, float(value))
    return out


async def drain(session) -> None:
    for _ in range(150):
        m = await scrape(session)
        if not m.get("vllm:num_requests_running") and not m.get("vllm:num_requests_waiting"):
            return
        await asyncio.sleep(0.2)
    raise RuntimeError("engine did not drain within 30 s")


_warm_round = 0


async def warm(session) -> None:
    """先压 3 秒左右,让 GPU 时钟和前端热起来。

    GeForce 卡空闲时只有 210 MHz,前端的线程也是冷的。实测空闲 30 秒后的第一个请求:
    从发出到进调度器 317 毫秒,512 个 token 的 prefill 算了 426 毫秒;热的时候是 19 和 77。
    """
    global _warm_round
    for _ in range(2):
        await asyncio.gather(
            *(stream(session, token_ids(9000 + 50 * _warm_round + i, 512), 96, "warm") for i in range(24))
        )
    _warm_round += 1
    await drain(session)
    await asyncio.sleep(0.5)


async def run_batch(session, b: int, prompt: int, max_tokens: int, seed: int) -> list[dict]:
    return await asyncio.gather(
        *(stream(session, token_ids(seed * 1009 + i, prompt), max_tokens, "d") for i in range(b))
    )


async def decode_sweep(session, a) -> dict:
    out = {"kind": "decode_sweep", "prompt": a.prompt, "max_tokens": a.max_tokens, "runs": []}
    # 预热:让 GPU 时钟升到稳态
    await run_batch(session, 16, 256, 256, 10**6)
    # 一轮正着测(小 batch 到大 batch),一轮倒着测,两轮对不上说明有漂移
    for rnd, order in enumerate([a.batches, a.batches[::-1]][: a.rounds]):
        for b in order:
            await drain(session)
            await asyncio.sleep(2)
            await warm(session)
            t0 = time.time()
            rs = await run_batch(session, b, a.prompt, a.max_tokens, rnd * 1000 + b)
            out["runs"].append(
                {"round": rnd, "b": b, "t0": t0, "t1": time.time(), "reqs": [summarize(r) for r in rs]}
            )
            print(f"round {rnd} b={b}: {time.time() - t0:.2f} s", flush=True)
    return out


async def mixed(session, a) -> dict:
    await warm(session)
    t_start = time.time()
    before = await scrape(session)
    bg = [
        asyncio.create_task(stream(session, token_ids(1000 + i, a.bg_prompt), a.bg_tokens, "bg"))
        for i in range(a.bg)
    ]
    await asyncio.sleep(a.fg_delay)
    fg = []
    for i in range(a.fg_n):
        fg.append(asyncio.create_task(stream(session, token_ids(2000 + i, a.fg_prompt), a.fg_tokens, "fg")))
        await asyncio.sleep(a.gap)
    fg_rs = [summarize(r) for r in await asyncio.gather(*fg)]
    bg_rs = [summarize(r) for r in await asyncio.gather(*bg)]
    after = await scrape(session)
    return {
        "kind": "mixed",
        "args": vars(a),
        "t0": t_start,
        "t1": time.time(),
        "bg": bg_rs,
        "fg": fg_rs,
        "metrics_delta": {k: after.get(k, 0) - before.get(k, 0) for k in METRICS if k in after},
    }


async def batching(session, a) -> dict:
    rng = random.Random(a.seed)
    lengths = [
        min(a.max_len, max(32, round(rng.lognormvariate(math.log(a.median_len), 0.55))))
        for _ in range(a.n)
    ]
    prompts = [token_ids(5000 + i, a.prompt) for i in range(a.n)]
    out = {"kind": "batching", "args": vars(a), "lengths": lengths, "runs": []}
    semaphore = asyncio.Semaphore(a.b)

    async def one(i: int) -> dict:
        async with semaphore:
            return await stream(session, prompts[i], lengths[i], f"{a.b}")

    for rnd in range(a.rounds):
        for mode in ("static", "continuous"):
            await drain(session)
            await asyncio.sleep(2)
            await warm(session)
            t0 = time.time()
            if mode == "static":
                rs = []
                for s in range(0, a.n, a.b):
                    rs += await asyncio.gather(
                        *(stream(session, prompts[i], lengths[i], f"{a.b}") for i in range(s, s + a.b))
                    )
            else:
                rs = await asyncio.gather(*(one(i) for i in range(a.n)))
            wall = time.time() - t0
            toks = sum(r["completion_tokens"] for r in rs)
            out["runs"].append(
                {"round": rnd, "mode": mode, "t0": t0, "t1": time.time(), "wall": wall,
                 "output_tokens": toks, "tokens_per_s": toks / wall, "reqs": [summarize(r) for r in rs]}
            )
            print(f"round {rnd} {mode}: {wall:.2f} s, {toks} tokens, {toks / wall:.1f} tok/s", flush=True)
    return out


async def preempt(session, a) -> dict:
    await warm(session)
    before = await scrape(session)
    t0 = time.time()
    rs = await asyncio.gather(
        *(stream(session, token_ids(7000 + i, a.prompt), a.max_tokens, "p") for i in range(a.n))
    )
    after = await scrape(session)
    return {
        "kind": "preempt",
        "args": vars(a),
        "t0": t0,
        "t1": time.time(),
        "reqs": [summarize(r) for r in rs],
        "metrics_delta": {k: after.get(k, 0) - before.get(k, 0) for k in METRICS if k in after},
    }


async def sustain(session, a) -> dict:
    """持续压力:--n 个工人,每个做完一条立刻发下一条,压满 --seconds。

    输出长度在 [--min-len, --max-len] 里随机(同一个种子,各配置下请求完全一样)。
    一次性发一批只能看到一轮抢占;这样压才能看到同一条请求被反复抢占。
    """
    await warm(session)
    before = await scrape(session)
    deadline = time.time() + a.seconds
    rng = random.Random(a.seed)

    async def worker(w: int) -> list[dict]:
        out = []
        k = 0
        while time.time() < deadline:
            n = rng.randint(a.min_len, a.max_len)
            r = await stream(session, token_ids(8000 + 100 * w + k, a.prompt), n, f"w{w}")
            r["max_tokens"] = n
            out.append(summarize(r))
            k += 1
        return out

    t0 = time.time()
    rs = [r for part in await asyncio.gather(*(worker(w) for w in range(a.n))) for r in part]
    after = await scrape(session)
    toks = sum(r["completion_tokens"] for r in rs)
    return {
        "kind": "sustain",
        "args": vars(a),
        "t0": t0,
        "t1": time.time(),
        "wall": time.time() - t0,
        "output_tokens": toks,
        "tokens_per_s": toks / (time.time() - t0),
        "reqs": rs,
        "metrics_delta": {k: after.get(k, 0) - before.get(k, 0) for k in METRICS if k in after},
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("workload", choices=["decode_sweep", "mixed", "batching", "preempt", "sustain"])
    p.add_argument("--out", required=True)
    # decode_sweep
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--prompt", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--rounds", type=int, default=2)
    # mixed
    p.add_argument("--bg", type=int, default=12)
    p.add_argument("--bg-prompt", type=int, default=256)
    p.add_argument("--bg-tokens", type=int, default=2400)
    p.add_argument("--fg-delay", type=float, default=3.0)
    p.add_argument("--fg-prompt", type=int, default=8192)
    p.add_argument("--fg-n", type=int, default=16)
    p.add_argument("--fg-tokens", type=int, default=4)
    p.add_argument("--gap", type=float, default=2.0)
    # batching
    p.add_argument("--n", type=int, default=256)
    p.add_argument("-b", type=int, default=32)
    p.add_argument("--median-len", type=float, default=200)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--seed", type=int, default=7)
    # preempt / sustain(提示词长度、条数、最大输出长度复用上面的 --prompt --n --max-tokens)
    p.add_argument("--min-len", type=int, default=128)
    p.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    if isinstance(getattr(a, "batches", None), str):
        a.batches = [int(x) for x in a.batches.split(",")]
    load_corpus()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None), connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        res = await globals()[a.workload](session, a)
    res["corpus_tokens"] = len(CORPUS)
    res["corpus_file"] = CORPUS_FILE
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
