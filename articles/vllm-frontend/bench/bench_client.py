#!/usr/bin/env python3
"""前端一篇的服务端实测客户端。在服务器本机跑,默认请求 http://127.0.0.1:8765
(8000 被机器上别的服务占着,serve.sh 也默认用 8765;要用别的端口改 BENCH_URL)。

每一步的排班时刻来自调度记录器(trace_sched.py,通过 SCHED_TRACE_FILE 指定文件);
API server 进程的 CPU 时间读 /proc/<pid>/stat,进程号通过 API_PID 给。
提示词取自一份英文原文(牛顿《光学》)切出来的 token,是自然文本。

  bench_client.py cold   --out results/cold.json            闲一会儿之后,第一个请求慢在哪
  bench_client.py ttft   --out results/ttft.json            首 token 时间拆成三段,提示词 128 到 32,768 个 token
  bench_client.py lag    --out results/lag.json             并发 1 / 32 / 256 路流式输出时,前端每一步的延迟
  bench_client.py abort  --out results/abort.json           客户端断开之后,引擎多久停算,多算了几个 token
  bench_client.py tokq   --out results/tokq.json            一条长提示词的分词,会不会挡住后面的短请求
"""
import argparse
import asyncio
import gc
import json
import os
import random
import statistics
import subprocess
import time

import aiohttp
from transformers import AutoTokenizer

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8765")
MODEL = "/data/wangyue/models/Llama-3.1-8B-Instruct"
CORPUS_FILE = os.environ.get("BENCH_CORPUS", "/data/wangyue/corpora/Isaac.Newton-Opticks.txt")
TRACE = os.environ["SCHED_TRACE_FILE"]
API_PID = int(os.environ["API_PID"])
CLK_TCK = os.sysconf("SC_CLK_TCK")
CORPUS: list[int] = []


def load_corpus() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(CORPUS_FILE, encoding="utf-8", errors="ignore").read()
    CORPUS.extend(tok(text, add_special_tokens=False)["input_ids"])


def start_of(seed: int, n: int) -> int:
    return random.Random(seed).randrange(0, len(CORPUS) - n - 1)


def token_ids(seed: int, n: int) -> list[int]:
    s = start_of(seed, n)
    return CORPUS[s:s + n]


def text_of(seed: int, n: int) -> str:
    s = start_of(seed, n)
    return AutoTokenizer.from_pretrained(MODEL).decode(CORPUS[s:s + n])


def cpu_s() -> float:
    with open(f"/proc/{API_PID}/stat") as f:
        p = f.read().rsplit(")", 1)[1].split()
    return (int(p[11]) + int(p[12])) / CLK_TCK      # utime + stime,不含子进程


def gpu_clock() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True).stdout.strip().split("\n")[0]
    return int(out)


_OFFSET = 0


def load_trace() -> list[dict]:
    """只读上次之后新写进记录文件的行,免得每次实验都把整个文件重新解析一遍。

    步的记录是 {"k": ...};请求到达的记录是 {"add": ...},这里并进一个字典 add["<id>"] = {"t": 时刻, "ms": 耗时}。
    """
    global _OFFSET
    rows = [{"add": {}}]
    with open(TRACE) as f:
        f.seek(_OFFSET)
        for line in f:
            if line.startswith("{\"k\""):
                rows.append(json.loads(line))
            elif line.startswith("{\"add\""):
                a = json.loads(line)
                rows[0]["add"][a["add"]] = a
        _OFFSET = f.tell()
    return rows


async def stream(session, prompt, max_tokens: int, tag: str = "") -> dict:
    """发一条流式请求。prompt 是 token 编号的列表时走 /v1/completions,是字符串时走 /v1/chat/completions。"""
    if isinstance(prompt, list):
        url, body = URL + "/v1/completions", {"model": "m", "prompt": prompt}
    else:
        url, body = URL + "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": prompt}]}
    body.update({"max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
                 "stream_options": {"include_usage": True}})
    t_send = time.time()
    ts, rid, usage = [], None, None
    async with session.post(url, json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            o = json.loads(payload)
            rid = o.get("id", rid)
            if o.get("usage"):
                usage = o["usage"]
            ch = o.get("choices")
            if ch:
                piece = ch[0].get("text") if "text" in ch[0] else (ch[0].get("delta") or {}).get("content")
                if piece:
                    ts.append(time.time())
    return {"tag": tag, "rid": rid.split("-")[1][-6:], "t_send": t_send, "ts": ts,
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]}


_GC: list[list] = []


def _gc_callback(phase: str, info: dict) -> None:
    # 客户端自己的一次垃圾回收:开始时刻、停了多久(毫秒)、第几代、回收了多少对象
    if phase == "start":
        _GC.append([time.time(), None, info.get("generation")])
    elif _GC and _GC[-1][1] is None:
        _GC[-1][1] = round((time.time() - _GC[-1][0]) * 1e3, 3)
        _GC[-1].append(info.get("collected"))


def client_gc_pauses() -> list[list]:
    """BENCH_GC_LOG 设了的话,列出客户端自己停了 0.5 毫秒以上的垃圾回收(用来把慢的那几下和它对上)。"""
    if not os.environ.get("BENCH_GC_LOG"):
        return []
    if _gc_callback not in gc.callbacks:
        gc.callbacks.append(_gc_callback)
    return [p for p in _GC if p[1] is not None and p[1] >= 0.5]


async def pinger(session, stop: asyncio.Event, out: list) -> None:
    while not stop.is_set():
        t = time.time()
        try:
            async with session.get(URL + "/ping") as r:
                await r.text()
            out.append((t, time.time() - t))
        except Exception:
            out.append((t, -1.0))
        await asyncio.sleep(0.05)


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def span(rows: list[dict], rid: str) -> dict:
    """这条请求在记录里的两个时刻:被接进 RUNNING 的那一步,和它最后一块 prefill 所在的那一步。

    提示词比 token 预算长时分好几步算,第一个 token 在最后一块的那一步算出来。
    """
    admit_row = next((x for x in rows[1:] if any(n[0] == rid for n in x["new"])), None)
    pre_rows = [x for x in rows[1:] if any(q[0] == rid and q[1] > 0 for q in x.get("reqs", []))]
    ready_row = pre_rows[-1] if pre_rows else admit_row
    return {"admit": admit_row, "ready": ready_row, "prefill_steps": len(pre_rows), "add": rows[0]["add"].get(rid)}


def parts(r: dict, x: dict) -> dict:
    """一条请求的首 token 时间,拆成五段(毫秒)。"""
    add = x["add"] or {}
    return {"front_ms": (x["admit"]["ts"] - r["t_send"]) * 1e3,          # 客户端发出到被排进 RUNNING
            "to_engine_ms": ((add.get("t") or x["admit"]["ts"]) - r["t_send"]) * 1e3,   # 其中,到达引擎核心之前
            "engine_add_ms": add.get("ms", 0.0),                            # 引擎核心收下它
            "queue_ms": (x["admit"]["ts"] - (add.get("t") or x["admit"]["ts"])) * 1e3,   # 在引擎里排队
            "prefill_ms": (x["ready"]["tu"] - x["admit"]["ts"]) * 1e3,
            "out_ms": (r["ts"][0] - x["ready"]["tu"]) * 1e3,
            "ttft_ms": (r["ts"][0] - r["t_send"]) * 1e3}


def snap_load() -> dict:
    """机器空不空:负载均值,和不是这次测量的人在吃 CPU 的进程。"""
    load = open("/proc/loadavg").read().split()[:3]
    mine = ("bench_client.py", "vllm serve", "VLLM::", "sshd", "ps -eo", "snap_load")
    out = subprocess.run(["ps", "-eo", "pcpu,etime,args", "--sort=-pcpu"], capture_output=True, text=True).stdout
    top = [l.strip()[:110] for l in out.splitlines()[1:] if not any(m in l for m in mine)][:3]
    return {"loadavg": [float(x) for x in load], "other_top": top}


async def warm(session, rounds=3) -> None:
    """先把 GPU 和前端压热:空闲之后第一个请求慢。"""
    for _ in range(rounds):
        await asyncio.gather(*(stream(session, token_ids(9000 + i, 512), 96) for i in range(16)))
    await asyncio.sleep(0.5)


def admit(rows: list[dict], rid: str):
    """这条请求第一次被排进 RUNNING 的那一步。"""
    return next((x for x in rows[1:] if any(n[0] == rid for n in x["new"])), None)


async def cold(session, a) -> dict:
    out = {"runs": []}
    for idle in (1, 3, 10, 30, 60, 60, 30, 10, 3, 1):
        await asyncio.sleep(idle)
        clock = gpu_clock()
        load = snap_load()
        r = await stream(session, token_ids(int(idle * 100 + len(out["runs"])), 128), 4)
        await asyncio.sleep(0.3)
        rows = load_trace()
        x = span(rows, r["rid"])
        out["runs"].append({"idle_s": idle, "clock_mhz": clock, **parts(r, x), "load": load["loadavg"]})
        c = out["runs"][-1]
        print("闲 %2d 秒:时钟 %4d MHz,进引擎前 %6.1f 毫秒,prefill %6.1f 毫秒,出来 %5.1f 毫秒,首 token %7.1f 毫秒" % (
            idle, c["clock_mhz"], c["front_ms"], c["prefill_ms"], c["out_ms"], c["ttft_ms"]), flush=True)
    return out


async def ttft(session, a) -> dict:
    await warm(session)
    out = {"runs": [], "load_start": snap_load()}
    for L in a.lengths:
        for kind in ("ids", "text"):
            runs = []
            for i in range(a.n):
                prompt = token_ids(100 * L + i, L) if kind == "ids" else text_of(100 * L + i, L)
                r = await stream(session, prompt, 4, f"{L}-{kind}")
                await asyncio.sleep(0.3)
                x = span(load_trace(), r["rid"])
                runs.append({"kind": kind, "prompt_tokens": r["prompt_tokens"], **parts(r, x),
                             "prefill_steps": x["prefill_steps"], "sched_ms_admit": x["admit"]["sm"]})
            out["runs"] += runs
            m = lambda k: statistics.median(r[k] for r in runs)
            print("%6d %4s(%6d 个 token):到引擎 %5.1f(引擎收下 %4.1f),排队 %5.1f,prefill %7.1f,出来 %5.1f,首 token %7.1f 毫秒" % (
                L, kind, runs[0]["prompt_tokens"], m("to_engine_ms"), m("engine_add_ms"), m("queue_ms"), m("prefill_ms"),
                m("out_ms"), m("ttft_ms")), flush=True)
    out["load_end"] = snap_load()
    return out


def lag_analysis(rows, reqs, pings, cpu0, cpu1, wall):
    """每一行记录是一步。b 条请求一起解码的第 j 行,产出每条请求的第 j + 2 个 token(第 1 个来自 prefill 那一步)。"""
    by_rid = {r["rid"]: r for r in reqs}
    first = min(r["t_send"] for r in reqs)
    win = [x for x in rows[1:] if x["ts"] >= first - 1 and x["dec"] > 0]
    delays, gaps, merged, slow = [], [], 0, []
    for r in reqs:
        mine = [x for x in win if any(q[0] == r["rid"] for q in x.get("reqs", []))]
        adm = admit(rows, r["rid"])
        steps = [adm] + [x for x in mine if x["k"] > adm["k"]]
        for i, t in enumerate(r["ts"]):
            if i < len(steps):
                d = (t - steps[i]["tu"]) * 1e3
                delays.append(d)
                if d > 10.0:
                    slow.append([round(steps[i]["tu"], 6), round(d, 2)])   # 那一步算完的绝对时刻,和这一下慢了多少
        gaps += [(b - a) * 1e3 for a, b in zip(r["ts"], r["ts"][1:])]
        merged += r["completion_tokens"] - len(r["ts"])
    rtts = [d * 1e3 for _, d in pings if d >= 0]
    ntok = sum(r["completion_tokens"] for r in reqs)
    step_ms = [b["tu"] - a["tu"] for a, b in zip(win, win[1:])] if len(win) > 2 else [0]
    return {
        "b": len(reqs), "wall_s": wall, "output_tokens": ntok, "tokens_per_s": ntok / wall,
        "api_cpu_s": cpu1 - cpu0, "api_cpu_pct": (cpu1 - cpu0) / wall * 100, "api_us_per_token": (cpu1 - cpu0) / ntok * 1e6,
        "engine_step_ms_median": statistics.median(step_ms) * 1e3,
        "delay_ms_p50": pct(delays, 0.5), "delay_ms_p99": pct(delays, 0.99), "delay_ms_max": max(delays),
        "gap_ms_p50": pct(gaps, 0.5), "gap_ms_p99": pct(gaps, 0.99), "gap_ms_max": max(gaps),
        "merged_events": merged, "events": sum(len(r["ts"]) for r in reqs),
        "ping_ms_p50": pct(rtts, 0.5), "ping_ms_p99": pct(rtts, 0.99), "ping_ms_max": max(rtts), "pings": len(rtts),
        "ping_over_20ms": sum(x > 20 for x in rtts),
        "slow_delays": slow,
        "req_ts": [[r["rid"], [round(t, 6) for t in r["ts"]]] for r in reqs],   # 每条请求每个 token 到达的绝对时刻
        "client_gc": client_gc_pauses(),
        "ping_series": [[round(t - first, 3), round(d * 1e3, 2)] for t, d in pings],
        "gap_series": [[round((t - r0["ts"][0]), 3), round((t - a) * 1e3, 2)] for r0 in reqs[:4] for a, t in zip(r0["ts"], r0["ts"][1:])],
    }


async def lag_once(session, b, max_tokens, label=""):
    await warm(session)
    load = snap_load()
    stop = asyncio.Event()
    pings = []
    pt = asyncio.create_task(pinger(session, stop, pings))
    cpu0, t0 = cpu_s(), time.time()
    prompt = 128 if b <= 32 else 64
    reqs = await asyncio.gather(*(stream(session, token_ids(20000 + 1000 * b + i, prompt), max_tokens, str(b)) for i in range(b)))
    wall, cpu1 = time.time() - t0, cpu_s()
    await asyncio.sleep(0.3)
    stop.set()
    await pt
    res = lag_analysis(load_trace(), reqs, pings, cpu0, cpu1, wall)
    res["label"] = label
    res["load"] = load
    print("%-14s b=%3d:%5.0f token/秒,引擎一步 %5.1f 毫秒;token 算完到客户端收到 p50 %5.1f / p99 %6.1f / 最大 %7.1f 毫秒;"
          "客户端间隔最大 %6.1f;合并事件 %d / %d;ping p50 %4.1f / p99 %5.1f / 最大 %6.1f 毫秒,超过 20 毫秒 %d 次;API 进程 CPU %5.1f%%,每个 token %.1f 微秒" % (
              label, b, res["tokens_per_s"], res["engine_step_ms_median"], res["delay_ms_p50"], res["delay_ms_p99"], res["delay_ms_max"],
              res["gap_ms_max"], res["merged_events"], res["events"], res["ping_ms_p50"], res["ping_ms_p99"], res["ping_ms_max"],
              res["ping_over_20ms"], res["api_cpu_pct"], res["api_us_per_token"]), flush=True)
    return res


async def lag(session, a) -> dict:
    out = {"runs": []}
    for b in a.batches:      # 256 条时每条 64 + 96 个 token,一共 40,960 个,装得进 KV 缓存
        out["runs"].append(await lag_once(session, b, a.tokens if b <= 32 else 96, "short"))
    return out


async def laglong(session, a) -> dict:
    """压 60 秒,看延迟有没有周期性的尖峰;HOG 环境变量指定要不要同时起占 CPU 的进程。"""
    out = {"runs": []}
    hog = os.environ.get("HOG", "")
    proc = None
    if hog:
        cpus, n, secs = hog.split(",")
        proc = await asyncio.create_subprocess_exec("bash", "hog.sh", cpus, n, secs, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(1.0)
    out["runs"].append(await lag_once(session, a.b, a.tokens, os.environ.get("LABEL", "long")))
    if proc:
        await proc.wait()
    return out


async def abort(session, a) -> dict:
    await warm(session)
    out = {"runs": []}
    for i in range(a.abort_n):
        body = {"model": "m", "prompt": token_ids(30000 + i, 128), "max_tokens": a.max_tokens, "temperature": 0.0,
                "ignore_eos": True, "stream": True}
        t_send = time.time()
        resp = await session.post(URL + "/v1/completions", json=body)
        rid, ntok = None, 0
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            o = json.loads(payload)
            rid = o.get("id", rid)
            if o.get("choices") and o["choices"][0].get("text"):
                ntok += 1
                if ntok >= a.after:
                    break
        t_close = time.time()
        resp.close()
        await asyncio.sleep(1.5)
        rows = load_trace()
        rid = rid.split("-")[1][-6:]
        mine = [x for x in rows[1:] if any(q[0] == rid for q in x.get("reqs", []))]
        after = [x for x in mine if x["ts"] > t_close]
        fin = next((x for x in rows[1:] if any(f[0] == rid for f in x["fin"])), None)
        r = {"tokens_seen_by_client": ntok, "steps_after_close": len(after),
             "last_scheduled_after_close_ms": (after[-1]["ts"] - t_close) * 1e3 if after else 0.0,
             "last_computed_after_close_ms": (after[-1]["tu"] - t_close) * 1e3 if after else 0.0,
             "freed_after_close_ms": (fin["ts"] - t_close) * 1e3 if fin else None}
        out["runs"].append(r)
        print("第 %2d 次:断开之后又排了 %d 步,最后一次排班在断开后 %5.1f 毫秒,块在断开后 %5.1f 毫秒释放" % (
            i + 1, r["steps_after_close"], r["last_scheduled_after_close_ms"], r["freed_after_close_ms"] or -1), flush=True)
    return out


async def tokq(session, a) -> dict:
    await warm(session)
    out = {"runs": []}
    for mode in ("short_only", "long_then_short"):
        for rep in range(a.reps):
            long_text = text_of(777 + rep, a.long_tokens)
            short_texts = [text_of(5000 + 10 * rep + i, a.short_tokens) for i in range(32)]
            t0 = time.time()
            if mode == "long_then_short":
                long_task = asyncio.create_task(stream(session, long_text, 4, "long"))
                await asyncio.sleep(0.002)
                shorts = await asyncio.gather(*(stream(session, s, 4) for s in short_texts), return_exceptions=True)
                long_task.cancel()
            else:
                shorts = await asyncio.gather(*(stream(session, s, 4) for s in short_texts), return_exceptions=True)
            await asyncio.sleep(0.5)
            rows = load_trace()
            to_engine, queue, ttft = [], [], []
            for s in shorts:
                if isinstance(s, Exception):
                    continue
                x = span(rows, s["rid"])
                to_engine.append((x["add"]["t"] - s["t_send"]) * 1e3 if x["add"] else float("nan"))
                queue.append((x["admit"]["ts"] - (x["add"] or {"t": x["admit"]["ts"]})["t"]) * 1e3)
                ttft.append((s["ts"][0] - s["t_send"]) * 1e3)
            med = statistics.median
            out["runs"].append({"mode": mode, "rep": rep, "to_engine_ms": to_engine, "queue_ms": queue, "ttft_ms": ttft,
                                "to_engine_ms_median": med(to_engine), "to_engine_ms_max": max(to_engine),
                                "queue_ms_median": med(queue), "ttft_ms_median": med(ttft), "ttft_ms_max": max(ttft)})
            print("%-15s 第 %d 轮:32 条短请求,到引擎 中位 %6.1f 毫秒(最大 %6.1f),在引擎里排队 中位 %7.1f 毫秒,首 token 中位 %7.1f 毫秒" % (
                mode, rep + 1, med(to_engine), max(to_engine), med(queue), med(ttft)), flush=True)
            await asyncio.sleep(1.0)
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("workload", choices=["cold", "ttft", "lag", "laglong", "abort", "tokq"])
    p.add_argument("--out", required=True)
    p.add_argument("--lengths", default="128,1024,4096,16384,32768")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--batches", default="1,32,256")
    p.add_argument("-b", type=int, default=32)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--after", type=int, default=1)
    p.add_argument("--abort-n", dest="abort_n", type=int, default=20)
    p.add_argument("--long-tokens", type=int, default=32768)
    p.add_argument("--short-tokens", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.batches = [int(x) for x in a.batches.split(",")]
    load_corpus()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=aiohttp.TCPConnector(limit=0)) as session:
        res = await globals()[a.workload](session, a)
    res["api_pid"] = API_PID
    res["load_end"] = snap_load()
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
