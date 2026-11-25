#!/usr/bin/env python3
"""把调度记录(trace_*.jsonl)和客户端结果(*.json)合成一份 summary.json。

在服务器上跑,路径相对本目录:
  python analyze.py decode_sweep results/trace_b2048.jsonl results/decode_sweep.json
  python analyze.py mixed       results/trace_b512.jsonl results/mixed_512.json --budget 512
  python analyze.py batching    results/trace_b2048.jsonl results/batching.json
  python analyze.py preempt     results/trace_preempt.jsonl results/preempt.json
  python analyze.py sustain     results/trace_preempt.jsonl results/sustain.json

一步的时间怎么量:异步调度下 update_from_output() 每步调一次,第 k 步的结果一出来就调,
所以相邻两步这个时刻之差就是一步的时间(GPU 上一次前向加调度和启动的开销)。
客户端那边量到的间隔受 HTTP 流式传输和前端事件循环影响,这台机器上偶有几十到几百毫秒的停顿,
所以正文用引擎内部的数,客户端的数只做对照。
"""
import json
import statistics
import sys


def load_trace(path: str) -> tuple[dict, list[dict]]:
    import gzip

    hdr, rows = None, []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            o = json.loads(line)
            if "hdr" in o and hdr is None:
                hdr = o["hdr"]
            elif "k" in o:
                rows.append(o)
    return hdr, rows


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def in_window(rows, t0, t1):
    return [x for x in rows if t0 - 0.5 <= x["ts"] <= t1 + 0.5]


def step_times(sel: list[dict]) -> list[float]:
    """相邻两步 update 时刻之差,只取记录里连着的两步。"""
    out = []
    for a, b in zip(sel, sel[1:]):
        if b["k"] == a["k"] + 1:
            out.append((b["tu"] - a["tu"]) * 1e3)
    return out


def decode_sweep(trace, client) -> dict:
    hdr, rows = load_trace(trace)
    d = json.load(open(client))
    out = {"header": hdr, "prompt": d["prompt"], "max_tokens": d["max_tokens"], "runs": []}
    for run in d["runs"]:
        b = run["b"]
        win = in_window(rows, run["t0"], run["t1"])
        sel = [x for x in win if x["dec"] == b and x["pre"] == 0 and x["run"] == b]
        dt = step_times(sel)
        reqs = run["reqs"]
        r = {
            "round": run["round"],
            "b": b,
            "max_run": max(x["run"] for x in win),
            "waiting_peak": max(x["wait"] for x in win),
            "preemptions": sum(len(x["preem"]) for x in win),
            "kv_max": max(x["kv"] for x in win),
            "wall_s": run["t1"] - run["t0"],
            "client_ttft_ms_median": statistics.median(q["ttft"] for q in reqs) * 1e3,
            "client_ttft_ms_max": max(q["ttft"] for q in reqs) * 1e3,
            "client_e2e_ms_median": statistics.median(q["e2e"] for q in reqs) * 1e3,
            "client_e2e_ms_max": max(q["e2e"] for q in reqs) * 1e3,
            "client_merged_tokens": sum(q["merged"] for q in reqs),
            "step_ms_all_median": statistics.median(step_times(win)) if len(win) > 2 else None,
        }
        # "全部 b 条都在解码"的步不够多,说明这组装不下,发生过抢占:不进 t(b) 曲线
        if len(sel) > 30:
            r.update(
                steps=len(dt),
                step_ms_median=statistics.median(dt),
                step_ms_p1=pct(dt, 0.01),
                step_ms_p99=pct(dt, 0.99),
                step_ms_min=min(dt),
                sched_ms_median=statistics.median(x["sm"] for x in sel),
                sched_ms_p99=pct([x["sm"] for x in sel], 0.99),
                update_ms_median=statistics.median(x["um"] for x in sel),
                client_itl_ms_median=statistics.median(q["itl_p50"] for q in reqs) * 1e3,
            )
            # 稳态窗口:丢掉前后各 15% 的步(前面在升频,后面有请求陆续结束)
            core = sel[len(sel) // 7 : len(sel) - len(sel) // 7]
            r["step_ms_steady_median"] = statistics.median(step_times(core))
            r["core_steps"] = len(core)
        else:
            r.update(steps=len(dt), step_ms_steady_median=None, core_steps=0)
        out["runs"].append(r)
    by_b = {}
    for r in out["runs"]:
        if r["step_ms_steady_median"] is not None:
            by_b.setdefault(r["b"], []).append(r["step_ms_steady_median"])
    out["by_b"] = {str(b): statistics.median(v) for b, v in sorted(by_b.items())}
    out["by_b_rounds"] = {str(b): v for b, v in sorted(by_b.items())}
    return out


def mixed(trace, client, budget: int) -> dict:
    hdr, rows = load_trace(trace)
    d = json.load(open(client))
    a = d["args"]
    win = in_window(rows, d["t0"], d["t1"])
    bg_ids = {r["rid"] for r in d["bg"]}
    fg_ids = {r["rid"] for r in d["fg"]}
    # 背景流全部在跑、还没有一条结束的那一段
    first_bg = next(i for i, x in enumerate(win) if any(n[0] in bg_ids for n in x["new"]))
    end = next(
        (i for i, x in enumerate(win) if any(f[0] in bg_ids for f in x["fin"])), len(win)
    )
    win = win[first_bg:end]
    # 前台请求的准入步和 prefill 段
    fg_rows = []
    for r in d["fg"]:
        rid = r["rid"]
        i0 = next(i for i, x in enumerate(win) if any(n[0] == rid for n in x["new"]))
        j = i0
        while j < len(win) and win[j]["pre"] > 0:
            j += 1
        chunk_steps = win[i0:j]
        pre_steps = [x for x in chunk_steps if x["pre"] > 0]
        dts = step_times(win[max(i0 - 1, 0) : j + 1])
        fg_rows.append(
            {
                "rid": rid,
                "admit_step": win[i0]["k"],
                "n_prefill_steps": len(pre_steps),
                "prefill_tokens_per_step": [x["pre"] for x in pre_steps],
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": sum(n[1] for n in win[i0]["new"] if n[0] == rid),
                "engine_ttft_ms": (win[j - 1]["tu"] - win[i0]["ts"]) * 1e3 if j - 1 >= i0 else None,
                "prefill_ms_sum": sum(
                    (b["tu"] - a_["tu"]) * 1e3 for a_, b in zip(win[i0:j], win[i0 + 1 : j + 1])
                ),
                "client_ttft_ms": r["ttft"] * 1e3,
                "client_e2e_ms": r["e2e"] * 1e3,
            }
        )
    # 背景流的 token 间隔:这段里每一步的时间
    all_dt = step_times(win)
    dec_only = step_times([x for x in win if x["pre"] == 0])
    with_pre = [x for x in win if x["pre"] > 0]
    pre_dt = step_times(with_pre)
    itl_client = [v for r in d["bg"] for v in ()]  # 占位,客户端的分位数不可靠
    out = {
        "header": hdr,
        "budget": budget,
        "args": a,
        "window_steps": len(win),
        "window_s": win[-1]["tu"] - win[0]["tu"] if len(win) > 1 else 0,
        "steps_with_prefill": len(with_pre),
        "prefill_tokens_per_step_median": statistics.median(x["pre"] for x in with_pre) if with_pre else 0,
        "decode_tokens_per_step_median": statistics.median(x["dec"] for x in win),
        "bg_engine_itl_ms": {
            "mean": statistics.fmean(all_dt),
            "p50": pct(all_dt, 0.5),
            "p99": pct(all_dt, 0.99),
            "max": max(all_dt),
        },
        "bg_engine_itl_decode_only_ms": {
            "mean": statistics.fmean(dec_only),
            "p99": pct(dec_only, 0.99),
            "max": max(dec_only),
        },
        "bg_engine_itl_steps_with_prefill_ms": {
            "mean": statistics.fmean(pre_dt) if pre_dt else None,
            "max": max(pre_dt) if pre_dt else None,
        },
        "bg_client_itl_mean_ms": statistics.fmean(r["itl_mean"] for r in d["bg"]) * 1e3,
        "bg_client_merged_tokens": sum(r["merged"] for r in d["bg"]),
        "bg_tokens": sum(r["completion_tokens"] for r in d["bg"]),
        "fg": fg_rows,
        "fg_client_ttft_ms": {
            "mean": statistics.fmean(r["client_ttft_ms"] for r in fg_rows),
            "median": statistics.median(r["client_ttft_ms"] for r in fg_rows),
            "max": max(r["client_ttft_ms"] for r in fg_rows),
        },
        "fg_engine_ttft_ms": {
            "mean": statistics.fmean(r["engine_ttft_ms"] for r in fg_rows),
            "median": statistics.median(r["engine_ttft_ms"] for r in fg_rows),
            "max": max(r["engine_ttft_ms"] for r in fg_rows),
        },
        "fg_prefill_steps_total": sum(r["n_prefill_steps"] for r in fg_rows),
        "metrics_delta": d["metrics_delta"],
        "preemptions": sum(len(x["preem"]) for x in win),
        "prefix_hits_in_window": sum(n[1] for x in win for n in x["new"]),
        "series": [
            [x["k"], round((x["tu"] - win[0]["tu"]) * 1e3, 3), x["pre"], x["dec"], x["run"], x["wait"]]
            for x in win
        ],
    }
    return out


def batching(trace, client) -> dict:
    hdr, rows = load_trace(trace)
    d = json.load(open(client))
    out = {"header": hdr, "args": d["args"], "lengths": d["lengths"], "runs": []}
    for run in d["runs"]:
        rs = run["reqs"]
        e2e = sorted(r["e2e"] for r in rs)
        win = in_window(rows, run["t0"], run["t1"])
        dt = step_times([x for x in win if x["dec"] + x["pre"] > 0])
        out["runs"].append(
            {
                "round": run["round"],
                "mode": run["mode"],
                "wall_s": run["wall"],
                "output_tokens": run["output_tokens"],
                "tokens_per_s": run["tokens_per_s"],
                "e2e_mean_s": statistics.fmean(e2e),
                "e2e_median_s": pct(e2e, 0.5),
                "e2e_p99_s": pct(e2e, 0.99),
                "e2e_max_s": e2e[-1],
                "engine_step_ms_median": statistics.median(dt),
                "client_ttft_ms_median": statistics.median(r["ttft"] for r in rs) * 1e3,
                "client_ttft_ms_max": max(r["ttft"] for r in rs) * 1e3,
                "client_merged_tokens": sum(r["merged"] for r in rs),
                # 每一步的 batch 大小和步时间,画"批里有几条在跑"的曲线用
                "series": [
                    [round((x["tu"] - run["t0"]) * 1e3, 1), x["run"], x["dec"] + x["pre"]]
                    for x in win
                    if x["dec"] + x["pre"] > 0
                ],
            }
        )
    agg = {}
    for mode in ("static", "continuous"):
        rs = [r for r in out["runs"] if r["mode"] == mode]
        agg[mode] = {
            "tokens_per_s": statistics.fmean(r["tokens_per_s"] for r in rs),
            "wall_s": statistics.fmean(r["wall_s"] for r in rs),
            "e2e_mean_s": statistics.fmean(r["e2e_mean_s"] for r in rs),
            "e2e_median_s": statistics.fmean(r["e2e_median_s"] for r in rs),
            "e2e_p99_s": statistics.fmean(r["e2e_p99_s"] for r in rs),
            "e2e_max_s": max(r["e2e_max_s"] for r in rs),
        }
    out["agg"] = agg
    return out


def request_events(win: list[dict], t0: float) -> dict[str, dict]:
    """每条请求在记录里留下的事:每次进 RUNNING 的时刻,每次被抢占的时刻和当时已算好的 token 数,结束的时刻。"""
    per: dict[str, dict] = {}
    for x in win:
        for n in x["new"]:
            per.setdefault(n[0], {"admits": [], "preempts": [], "finish": None, "first_chunk": n[2]})
            per[n[0]]["admits"].append(round((x["ts"] - t0) * 1e3, 1))
        for p_ in x["preem"]:
            per.setdefault(p_[0], {"admits": [], "preempts": [], "finish": None, "first_chunk": 0})
            per[p_[0]]["preempts"].append([round((x["ts"] - t0) * 1e3, 1), p_[1], p_[2]])
        for f in x["fin"]:
            if f[0] in per:
                per[f[0]]["finish"] = round((x["ts"] - t0) * 1e3, 1)
    return per


def preempt(trace, client) -> dict:
    hdr, rows = load_trace(trace)
    d = json.load(open(client))
    a = d["args"]
    t0 = d["t0"]
    win = in_window(rows, d["t0"], d["t1"])
    preem = [{"step": x["k"], "t_ms": (x["ts"] - t0) * 1e3, "rid": p[0], "lost": p[1], "total": p[2]}
             for x in win for p in x["preem"]]
    per = request_events(win, t0)
    out = {
        "header": hdr,
        "args": a,
        "n_requests": len(per),
        "n_preemptions": len(preem),
        "preemptions": preem,
        "lost_tokens_sum": sum(p["lost"] for p in preem),
        "metrics_num_preemptions": d["metrics_delta"].get("vllm:num_preemptions_total"),
        "wall_s": d["t1"] - d["t0"],
        "output_tokens": sum(r["completion_tokens"] for r in d["reqs"]),
        "tokens_per_s": sum(r["completion_tokens"] for r in d["reqs"]) / (d["t1"] - d["t0"]),
        "e2e_mean_s": statistics.fmean(r["e2e"] for r in d["reqs"]),
        "e2e_max_s": max(r["e2e"] for r in d["reqs"]),
        "requests": per,
        "kv_max_usage": max(x["kv"] for x in win),
        "series": [[round((x["ts"] - t0) * 1e3, 1), x["run"], x["wait"], x["dec"], x["pre"], x["kv"]]
                   for x in win],
    }
    return out


def sustain(trace, client) -> dict:
    hdr, rows = load_trace(trace)
    d = json.load(open(client))
    a = d["args"]
    t0 = d["t0"]
    win = in_window(rows, d["t0"], d["t1"])
    per = request_events(win, t0)
    n_pre = {rid: len(v["preempts"]) for rid, v in per.items()}
    e2e = [r["e2e"] for r in d["reqs"]]
    e2e_clean = [r["e2e"] for r in d["reqs"] if n_pre.get(r["rid"], 0) == 0]
    e2e_hit = [r["e2e"] for r in d["reqs"] if n_pre.get(r["rid"], 0) > 0]
    preem = [p for v in per.values() for p in v["preempts"]]
    out = {
        "header": hdr,
        "args": a,
        "wall_s": d["wall"],
        "output_tokens": d["output_tokens"],
        "tokens_per_s": d["tokens_per_s"],
        "n_requests": len(d["reqs"]),
        "n_preemptions": len(preem),
        "metrics_num_preemptions": d["metrics_delta"].get("vllm:num_preemptions_total"),
        "requests_preempted": sum(1 for v in n_pre.values() if v > 0),
        "requests_preempted_2plus": sum(1 for v in n_pre.values() if v >= 2),
        "max_preemptions_one_request": max(n_pre.values(), default=0),
        "lost_tokens_sum": sum(p[1] for p in preem),
        "e2e_mean_s": statistics.fmean(e2e),
        "e2e_p50_s": pct(e2e, 0.5),
        "e2e_p99_s": pct(e2e, 0.99),
        "e2e_max_s": max(e2e),
        "e2e_mean_s_never_preempted": statistics.fmean(e2e_clean) if e2e_clean else None,
        "e2e_mean_s_preempted": statistics.fmean(e2e_hit) if e2e_hit else None,
        "kv_mean_usage": statistics.fmean(x["kv"] for x in win),
        "kv_max_usage": max(x["kv"] for x in win),
        "run_mean": statistics.fmean(x["run"] for x in win if x["dec"] + x["pre"] > 0),
        "wait_mean": statistics.fmean(x["wait"] for x in win),
        "requests": per,
        "series": [[round((x["ts"] - t0) * 1e3, 1), x["run"], x["wait"], x["dec"], x["pre"], x["kv"]]
                   for x in win],
    }
    return out


if __name__ == "__main__":
    kind, trace, client = sys.argv[1:4]
    budget = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0
    res = {
        "decode_sweep": lambda: decode_sweep(trace, client),
        "mixed": lambda: mixed(trace, client, budget),
        "batching": lambda: batching(trace, client),
        "preempt": lambda: preempt(trace, client),
        "sustain": lambda: sustain(trace, client),
    }[kind]()
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f)
        print("wrote", out_path)
    else:
        print(json.dumps(res, indent=1)[:6000])
