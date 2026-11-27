#!/usr/bin/env python3
"""前端一篇里引用的每个数,都在这里算一遍。输出 bench/calc_summary.json,图和正文都从它读。

模型:Llama-3.1-8B-Instruct,一张 RTX 4090 D,vLLM 0.34.0;CPU 是 Xeon Platinum 8352V。
实测数据在 bench/results/ 里:
  cpu_xeon8352v.json  纯 CPU 的几组测量(bench_cpu.py):分词、聊天模板、反分词、建请求状态
  input_cost.json     token 编号的请求进引擎之前的几样固定开销(input_cost.py)
  inproc.json         在 API server 自己的线程里跑同一个反分词步骤,核凉着和核热着各多久(sitecustomize.py 的 INPROC_BENCH)
  ttft.json cold.json lag.json laglong.json abort.json   服务端实测(bench_client.py)
  tokq_w1.json tokq_w4.json                             分词线程 1 个和 4 个时的排队
  hog_*.json                                           占 CPU 的对照
  send_*_attr.json                                     逐 token 对上客户端收到的时刻和 API server 写出的时刻(send_attr.py)
  prof_32k.json                                        一条 32k 请求的剖析(prof_delta.py)
  sse_probe.json raw32k.json                           极简流式服务的对照(sse_probe.py)、裸 socket 客户端的首 token(raw32k.py)
  gc_server.json gc_1.json gc_2.json                   API server 进程每次垃圾回收停了多久(sitecustomize.py 的 GC_LOG)
运行:python3 calc.py
"""
from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "bench", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def med(v):
    return statistics.median(v)


out = {}

# ---- 1. 分词、聊天模板 ------------------------------------------------------------------------
cpu = load("cpu_xeon8352v.json")
tok = {r["tokens"]: r for r in cpu["tokenize"]}
out["env"] = {"cpu": cpu["cpu"], "corpus_tokens": cpu["corpus_tokens"], "corpus_chars": cpu["corpus_chars"],
              "chars_per_token": cpu["corpus_chars"] / cpu["corpus_tokens"], "tokenizer": cpu["tokenizer"]}
out["tokenize"] = {
    "rows": [{"tokens": r["tokens"], "ms": r["ms"], "us_per_token": r["us_per_token"]} for r in cpu["tokenize"]],
    "us_per_token_4k": tok[4097]["us_per_token"], "us_per_token_32k": tok[32769]["us_per_token"],
    "ms_32k": tok[32769]["ms"], "ms_128k": tok[131073]["ms"],
    "one_by_one_ms": cpu["tokenize_32x128"]["one_by_one_ms"], "batch_ms": cpu["tokenize_32x128"]["batch_ms"],
}
out["tokenize"]["batch_speedup"] = out["tokenize"]["one_by_one_ms"] / out["tokenize"]["batch_ms"]
out["chat_template"] = [{"turns": r["turns"], "prompt_tokens": r["prompt_tokens"], "ms": r["ms"]} for r in cpu["chat_template"]]

# ---- 2. 反分词:朴素做法是 O(n^2),增量做法每个 token O(1) ------------------------------------
det = cpu["detokenize"]
naive = {r["tokens"]: r for r in det["naive"]}
slow = {r["tokens"]: r for r in det["slow"]}
fast = {r["tokens"]: r for r in det["fast"]}
# 朴素做法:解码第 t 个 token 要把前面 t 个重新解码一遍,每个约 c 微秒。用 4096 那一组拟合 c,再拿它预测总时间。
c_us = naive[4096]["us_per_token_last_block"] / 4096
pred_4096_ms = c_us * 4096 * 4095 / 2 / 1e3
out["detok"] = {
    "naive": [{"tokens": n, "total_ms": r["total_ms"], "us_last_block": r["us_per_token_last_block"]} for n, r in naive.items()],
    "slow": [{"tokens": n, "total_ms": r["total_ms"], "us_last_block": r["us_per_token_last_block"]} for n, r in slow.items()],
    "fast": [{"tokens": n, "total_ms": r["total_ms"], "us_last_block": r["us_per_token_last_block"]} for n, r in fast.items()],
    "all_correct": all(r["ok"] for r in det["naive"] + det["slow"] + det["fast"]),
    "c_us_per_context_token": c_us, "naive_pred_total_ms_4096": pred_4096_ms, "naive_measured_total_ms_4096": naive[4096]["total_ms"],
    "naive_total_us_8192_tokens_pred": c_us * 8192 * 8191 / 2,
    "naive_us_4096": naive[4096]["us_per_token_last_block"], "slow_us_4096": slow[4096]["us_per_token_last_block"],
    "fast_us_4096": fast[4096]["us_per_token_last_block"],
    "speedup_at_4096": naive[4096]["us_per_token_last_block"] / fast[4096]["us_per_token_last_block"],
    "speedup_at_8192": c_us * 8192 / fast[8192]["us_per_token_last_block"],
    "single_ok_english": det["single"][0]["ok_english"], "single_ok_chinese": det["single"][1]["ok_chinese"],
    "utf8": {"text": cpu["utf8_demo"]["text"], "token_pieces": cpu["utf8_demo"]["token_pieces"],
             "pieces": cpu["utf8_demo"]["rows"][1]["pieces"], "n_tokens": len(cpu["utf8_demo"]["token_pieces"]) - 1},
}

# ---- 3. 第一个生成 token:流式反分词器要把整段提示词重放一遍 -----------------------------------
g = {r["tokens"]: r for r in cpu["frontend_state"]}
inproc = load("inproc.json")
inp = load("input_cost.json")["rows"]
inp32 = inp[-1]
first_rows = [{"tokens": n, "init_ms": r["detok_init_ms"], "first_ms": r["first_token_ms"], "second_ms": r["second_token_ms"]} for n, r in g.items()]
out["first_token"] = {
    "rows": first_rows,
    "us_per_prompt_token": g[32768]["first_token_ms"] * 1e3 / 32768,
    "first_ms_32k": g[32768]["first_token_ms"], "second_ms": g[32768]["second_token_ms"],
    "make_output_ms": g[32768]["make_output_ms"],
    "inproc": inproc["runs"],
    "inproc_first_after_idle_ms": inproc["runs"][0]["after_idle_3s_ms"], "inproc_first_freq_mhz": inproc["runs"][0]["freq_khz"] / 1e3,
    "inproc_later_after_idle_median_ms": med([r["after_idle_3s_ms"] for r in inproc["runs"][1:]]),
    "inproc_later_again_median_ms": med([r["immediately_again_ms"] for r in inproc["runs"][1:]]),
    "inproc_slowdown": inproc["runs"][0]["after_idle_3s_ms"] / g[32768]["first_token_ms"],
}

# ---- 4. 进引擎之前:首 token 时间拆成四段 ----------------------------------------------------------
ttft = load("ttft.json")
rows = []
for L in (128, 1024, 4096, 16384, 32768):
    for kind in ("ids", "text"):
        rs = [r for r in ttft["runs"] if r["kind"] == kind and L <= r["prompt_tokens"] < 2 * L]
        rows.append({"L": L, "kind": kind, "prompt_tokens": rs[0]["prompt_tokens"], "n": len(rs),
                     "to_engine_ms": med([r["to_engine_ms"] for r in rs]), "to_engine_max_ms": max(r["to_engine_ms"] for r in rs),
                     "engine_add_ms": med([r["engine_add_ms"] for r in rs]), "queue_ms": med([r["queue_ms"] for r in rs]),
                     "prefill_ms": med([r["prefill_ms"] for r in rs]), "out_ms": med([r["out_ms"] for r in rs]),
                     "out_max_ms": max(r["out_ms"] for r in rs), "ttft_ms": med([r["ttft_ms"] for r in rs]),
                     "prefill_steps": med([r["prefill_steps"] for r in rs]), "sched_ms_admit": med([r["sched_ms_admit"] for r in rs])})
        rows[-1]["front_ms"] = rows[-1]["to_engine_ms"] + rows[-1]["out_ms"]
        rows[-1]["front_share"] = rows[-1]["front_ms"] / rows[-1]["ttft_ms"] * 100
        rows[-1]["to_engine_min_ms"] = min(r["to_engine_ms"] for r in rs)
by = {(r["L"], r["kind"]): r for r in rows}
ids32, ids128, txt32, txt128 = by[(32768, "ids")], by[(128, "ids")], by[(32768, "text")], by[(128, "text")]
out["ttft"] = {
    "rows": rows,
    "ids_us_per_token": (ids32["to_engine_ms"] - ids128["to_engine_ms"]) * 1e3 / (ids32["prompt_tokens"] - ids128["prompt_tokens"]),
    "text_us_per_token": (txt32["to_engine_ms"] - txt128["to_engine_ms"]) * 1e3 / (txt32["prompt_tokens"] - txt128["prompt_tokens"]),
    "out_us_per_token": (ids32["out_ms"] - ids128["out_ms"]) * 1e3 / (ids32["prompt_tokens"] - ids128["prompt_tokens"]),
    "tokens_per_step": ids32["prompt_tokens"] / ids32["prefill_steps"],
    "prefill_ms_per_step_32k": ids32["prefill_ms"] / ids32["prefill_steps"],
    "frontend_share_32k_ids": (ids32["to_engine_ms"] + ids32["out_ms"]) / ids32["ttft_ms"],
    "frontend_ms_32k_ids": ids32["to_engine_ms"] + ids32["out_ms"],
    "frontend_ms_32k_text": txt32["to_engine_ms"] + txt32["out_ms"],
    "frontend_ms_128_ids": ids128["to_engine_ms"] + ids128["out_ms"],
    "text_extra_ms_32k": txt32["to_engine_ms"] - ids32["to_engine_ms"],
    "prefill_ms_per_step_4k": by[(4096, "ids")]["prefill_ms"] / by[(4096, "ids")]["prefill_steps"],
    "frontend_share_128_ids": (ids128["to_engine_ms"] + ids128["out_ms"]) / ids128["ttft_ms"],
}
out["input_cost"] = {"rows": inp, "sum_ms_32k": inp32["json_ms"] + inp32["validate_ms"] + inp32["pack_ms"], "body_bytes_32k": inp32["body_bytes"]}

prof = load("prof_32k.json")
step = prof["decode_stream_step"]
out["profile_32k"] = {"request_out_ms": prof["request"]["out_ms"], "request_to_engine_ms": prof["request"]["to_engine_ms"],
                      "decode_stream_step_ms": step["cum_ms"], "decode_stream_step_calls": step["calls"],
                      "decode_stream_share": step["cum_ms"] / prof["request"]["out_ms"],
                      "json_ms": prof["json"]["cum_ms"], "validate_ms": prof["validate"]["cum_ms"], "packed_prompt_ms": prof["packed_prompt"]["self_ms"]}

# ---- 5. 冷启动 ------------------------------------------------------------------------------------
cold = load("cold.json")["runs"]
first_ever = cold[0]
idle = {}
for r in cold[1:]:
    idle.setdefault(r["idle_s"], []).append(r)
out["cold"] = {
    "first_ever": {k: first_ever[k] for k in ("idle_s", "clock_mhz", "to_engine_ms", "prefill_ms", "out_ms", "ttft_ms")},
    "runs": [{k: r[k] for k in ("idle_s", "clock_mhz", "to_engine_ms", "prefill_ms", "out_ms", "ttft_ms")} for r in cold],
    "by_idle": {str(s): {"prefill_ms": med([r["prefill_ms"] for r in v]), "to_engine_ms": med([r["to_engine_ms"] for r in v]),
                         "out_ms": med([r["out_ms"] for r in v]),
                         "ttft_ms": med([r["ttft_ms"] for r in v]), "clocks": sorted({r["clock_mhz"] for r in v})} for s, v in idle.items()},
}
out["cold"]["prefill_ratio_60s_vs_3s"] = (out["cold"]["by_idle"]["60"]["prefill_ms"] / out["cold"]["by_idle"]["3"]["prefill_ms"])
later = [r["to_engine_ms"] for r in cold[1:]]
out["cold"]["later_runs"] = len(later)
out["cold"]["later_to_engine_8_to_11"] = sum(8 <= x <= 11.5 for x in later)
out["cold"]["later_to_engine_range"] = [min(later), max(later)]

# ---- 6. 并发流式输出:前端每个 token 花多少 ----------------------------------------------------------
def lag_row(r, tag):
    return {"tag": tag, "b": r["b"], "tokens_per_s": r["tokens_per_s"], "step_ms": r["engine_step_ms_median"],
            "delay_p50": r["delay_ms_p50"], "delay_p99": r["delay_ms_p99"], "delay_max": r["delay_ms_max"],
            "gap_p50": r["gap_ms_p50"], "gap_p99": r["gap_ms_p99"], "gap_max": r["gap_ms_max"],
            "merged": r["merged_events"], "events": r["events"], "api_cpu_pct": r["api_cpu_pct"], "api_us_per_token": r["api_us_per_token"],
            "ping_p50": r["ping_ms_p50"], "ping_p99": r["ping_ms_p99"], "ping_max": r["ping_ms_max"]}
lag = [lag_row(r, "lag") for r in load("lag.json")["runs"]]
out["lag"] = {"runs": lag}
b256, b32, b1 = lag[2], lag[1], lag[0]
out["lag"]["merged_pct_256"] = b256["merged"] / b256["events"] * 100
out["lag"]["busy_ms_per_step_256"] = b256["api_us_per_token"] * 256 / 1e3
out["lag"]["busy_ms_per_step_32"] = b32["api_us_per_token"] * 32 / 1e3
out["lag"]["max_tokens_per_s_frontend"] = 1e6 / b256["api_us_per_token"]
out["lag"]["busy_share_of_step_256"] = out["lag"]["busy_ms_per_step_256"] / b256["step_ms"]
out["lag"]["rho_256"] = b256["tokens_per_s"] * b256["api_us_per_token"] / 1e6      # 到达率乘每个 token 的服务时间
out["lag"]["us_per_token"] = {str(r["b"]): r["api_us_per_token"] for r in lag}
out["laglong"] = lag_row(load("laglong.json")["runs"][0], "laglong")

hog = [lag_row(load("hog_%d.json" % i)["runs"][0], "hog_%d" % i) for i in range(4)]
hog += [lag_row(load("hog_ab_2.json")["runs"][0], "hog_ab_2"), lag_row(load("hog_ab_4.json")["runs"][0], "hog_ab_4")]
labels = {"hog_0": "绑 8 个核,不占", "hog_1": "同样的 8 个核上 16 个死循环", "hog_2": "同一 NUMA 节点的别的核", "hog_3": "另一个 NUMA 节点的核",
          "hog_ab_2": "同样的核(第二轮)", "hog_ab_4": "同样的核(第三轮)"}
for h in hog:
    h["label"] = labels[h["tag"]]
same = [h for h in hog if h["tag"] in ("hog_1", "hog_ab_2", "hog_ab_4")]
out["hog"] = {"rows": hog, "same_cores_us_per_token": med([h["api_us_per_token"] for h in same]),
              "same_cores_p50": med([h["delay_p50"] for h in same]), "no_hog_us_per_token": hog[0]["api_us_per_token"],
              "same_cores_tokens_per_s": med([h["tokens_per_s"] for h in same]), "no_hog_tokens_per_s": hog[0]["tokens_per_s"]}

# 并发 32 路的每一轮(各次实验里的 b=32,提示词 128 个 token):中位数很稳,99 分位不稳
b32_runs = [h for h in hog if h["b"] == 32]
out["lag"]["b32_runs"] = [{"tag": h["tag"], "delay_p50": h["delay_p50"], "delay_p99": h["delay_p99"]} for h in b32_runs]
out["lag"]["b32_p50_range"] = [min(h["delay_p50"] for h in b32_runs), max(h["delay_p50"] for h in b32_runs)]
out["lag"]["b32_p99_range"] = [min(h["delay_p99"] for h in b32_runs), max(h["delay_p99"] for h in b32_runs)]
# 逐 token 对上服务端写出时刻的那 8 轮也是并发 32 路:一起算 99 分位的范围
tail_all = load("send_srv_off_attr.json")["runs"] + load("send_srv_nd_attr.json")["runs"]
p99_all = sorted([h["delay_p99"] for h in b32_runs] + [next(r for r in lag if r["b"] == 32)["delay_p99"]] + [r["client_p99"] for r in tail_all])
out["lag"]["b32_runs_total"] = len(p99_all)
out["lag"]["b32_p99_le_35"] = sum(x <= 35 for x in p99_all)
out["lag"]["b32_p99_over_35"] = [round(x, 1) for x in p99_all if x > 35]
# 尾部:逐 token 对上服务端写出的时刻
tail_off = load("send_srv_off_attr.json")["runs"]
tail_nd = load("send_srv_nd_attr.json")["runs"]
out["tail"] = {
    "nodelay_off": tail_off, "nodelay_on": tail_nd,
    "off_server_p50": med([r["server_p50"] for r in tail_off[:3]]), "off_server_p99": med([r["server_p99"] for r in tail_off[:3]]),
    "off_client_p50": med([r["client_p50"] for r in tail_off[:3]]), "off_client_p99": [r["client_p99"] for r in tail_off],
    "off_write_to_recv_max": max(r["write_to_recv_max"] for r in tail_off), "off_write_to_recv_p99": max(r["write_to_recv_p99"] for r in tail_off),
    "on_server_p50": med([r["server_p50"] for r in tail_nd]), "on_server_p99": med([r["server_p99"] for r in tail_nd]),
    "on_client_p99": [r["client_p99"] for r in tail_nd],
    "on_write_to_recv_max": max(r["write_to_recv_max"] for r in tail_nd), "on_write_to_recv_p99": max(r["write_to_recv_p99"] for r in tail_nd),
    "off_slow_total": sum(r["slow_tokens"] for r in tail_off), "off_server_late": sum(r["server_late"] for r in tail_off),
    "off_after_write": sum(r["after_write"] for r in tail_off),
}

# 垃圾回收不是尾部的原因:API server 进程里每一次回收停了多久;客户端慢的 token 里有几个正好落在回收里
gc = load("gc_server.json")
gens = {}
for _, dur, gen, _ in gc:
    gens.setdefault(gen, []).append(dur)


def hit(tu, delay, pauses):
    return [p for p in pauses if p[0] < tu + delay / 1e3 + 0.002 and p[0] + p[1] / 1e3 > tu - 0.002]


gcr = [load("gc_%d.json" % i)["runs"][0] for i in (1, 2)]
out["gc"] = {"pauses": len(gc), "max_ms": max(d for _, d, _, _ in gc),
             "by_gen": {str(g): {"n": len(v), "median_ms": med(v), "max_ms": max(v)} for g, v in sorted(gens.items())},
             "slow_tokens": sum(len(r["slow_delays"]) for r in gcr),
             "slow_on_server_gc": sum(1 for r in gcr for tu, d in r["slow_delays"] if hit(tu, d, gc)),
             "slow_on_client_gc": sum(1 for r in gcr for tu, d in r["slow_delays"] if hit(tu, d, r["client_gc"])),
             "client_pauses_over_0_5ms": sum(len(r["client_gc"]) for r in gcr)}

# ---- 7. 断连之后 ----------------------------------------------------------------------------------
ab = load("abort.json")["runs"]
freed = sorted(r["freed_after_close_ms"] for r in ab)
out["abort"] = {"trials": len(ab), "max_steps_after_close": max(r["steps_after_close"] for r in ab),
                "freed_min_ms": freed[0], "freed_median_ms": med(freed), "freed_max_ms": freed[-1],
                "step_ms_128": by[(128, "ids")]["prefill_ms"], "max_tokens_requested": 1500, "decode_step_ms_b1": b1["step_ms"],
                "slowest_run_share": max(r["freed_after_close_ms"] for r in ab) / b1["step_ms"]}

# ---- 8. 分词排队 ----------------------------------------------------------------------------------
def tokq(name):
    o = load(name)
    res = {}
    for mode in ("short_only", "long_then_short"):
        rs = [r for r in o["runs"] if r["mode"] == mode]
        res[mode] = {"to_engine_median_ms": med([r["to_engine_ms_median"] for r in rs]), "to_engine_max_ms": max(r["to_engine_ms_max"] for r in rs),
                     "queue_median_ms": med([r["queue_ms_median"] for r in rs]), "ttft_median_ms": med([r["ttft_ms_median"] for r in rs]),
                     "per_rep": [{k: r[k] for k in ("to_engine_ms_median", "queue_ms_median", "ttft_ms_median")} for r in rs]}
    return res
out["tokq"] = {"w1": tokq("tokq_w1.json"), "w4": tokq("tokq_w4.json")}
o1, o4 = out["tokq"]["w1"], out["tokq"]["w4"]
out["tokq"]["ttft_ratio_long_first"] = o1["long_then_short"]["ttft_median_ms"] / o1["short_only"]["ttft_median_ms"]
out["tokq"]["per_request_ms"] = [o1["short_only"]["to_engine_median_ms"] / 32, o4["short_only"]["to_engine_median_ms"] / 32]
out["tokq"]["ranges"] = {w: {mode: {k: [min(r[k] for r in load("%s.json" % w)["runs"] if r["mode"] == mode),
                                       max(r[k] for r in load("%s.json" % w)["runs"] if r["mode"] == mode)]
                                   for k in ("to_engine_ms_median", "queue_ms_median", "ttft_ms_median")} for mode in ("short_only", "long_then_short")}
                         for w in ("tokq_w1", "tokq_w4")}
out["tokq"]["overtake"] = load("tokq_overtake.json")

# ---- 9. 极简流式服务的对照、裸 socket 客户端 ------------------------------------------------------
out["sse_probe"] = load("sse_probe.json")
raw = load("raw32k.json")
out["raw32k"] = {"plain_ms": [r["first_token_after_tu_ms"] for r in raw if not r["quickack"]],
                 "quickack_ms": [r["first_token_after_tu_ms"] for r in raw if r["quickack"]],
                 "empty_events_during_prefill": [r["empty_events"] for r in raw]}

with open(os.path.join(HERE, "bench", "calc_summary.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)


def show():
    def p(k, v):
        print("%-44s %s" % (k, json.dumps(v, ensure_ascii=False)[:150]))

    t = out["tokenize"]
    p("分词 4k / 32k 每 token(微秒)", [round(t["us_per_token_4k"], 2), round(t["us_per_token_32k"], 2)])
    p("32 条逐条 / 批量(毫秒), 倍数", [round(t["one_by_one_ms"], 2), round(t["batch_ms"], 2), round(t["batch_speedup"], 1)])
    p("聊天模板(轮, token, 毫秒)", [(r["turns"], r["prompt_tokens"], round(r["ms"], 2)) for r in out["chat_template"]])
    d = out["detok"]
    p("朴素 c(微秒/上下文 token), 预测 / 实测总时间(毫秒)", [round(d["c_us_per_context_token"], 4), round(d["naive_pred_total_ms_4096_ms"] if False else d["naive_pred_total_ms_4096"], 1), round(d["naive_measured_total_ms_4096"], 1)])
    p("4096 处 朴素 / slow / fast(微秒每 token), 倍数(4096, 8192)", [round(d["naive_us_4096"], 1), round(d["slow_us_4096"], 2), round(d["fast_us_4096"], 2), round(d["speedup_at_4096"], 0), round(d["speedup_at_8192"], 0)])
    p("朴素 8192 个 token 的总时间(秒, 预测)", round(d["naive_total_us_8192_tokens_pred"] / 1e6, 1))
    p("只解码新 token: 英文 / 中文", [d["single_ok_english"], d["single_ok_chinese"]])
    f1 = out["first_token"]
    p("第一个生成 token 的反分词(毫秒)", [(r["tokens"], round(r["first_ms"], 2)) for r in f1["rows"]])
    p("每提示词 token(微秒), 第二个 token(毫秒)", [round(f1["us_per_prompt_token"], 3), f1["second_ms"]])
    p("进程内: 各次", [(r["freq_khz"] // 1000, r["after_idle_3s_ms"], r["immediately_again_ms"]) for r in f1["inproc"]])
    p("进程内凉核 / 离线", round(f1["inproc_slowdown"], 1))
    t = out["ttft"]
    for r in t["rows"]:
        p("%6d %s" % (r["prompt_tokens"], r["kind"]), {k: round(r[k], 1) for k in ("to_engine_ms", "queue_ms", "prefill_ms", "out_ms", "ttft_ms", "prefill_steps")})
    p("ids / text / out 每 token(微秒)", [round(t["ids_us_per_token"], 2), round(t["text_us_per_token"], 2), round(t["out_us_per_token"], 2)])
    p("前端占比 32k ids / 128 ids", [round(t["frontend_share_32k_ids"] * 100, 1), round(t["frontend_share_128_ids"] * 100, 1)])
    p("32k text 的前端(毫秒)", round(t["frontend_ms_32k_text"], 1))
    p("input_cost", [(r["tokens"], round(r["json_ms"], 2), round(r["validate_ms"], 2), round(r["pack_ms"], 2)) for r in out["input_cost"]["rows"]])
    p("profile_32k", out["profile_32k"])
    c = out["cold"]
    p("cold 首条", {k: round(v, 1) if isinstance(v, float) else v for k, v in c["first_ever"].items()})
    p("cold 按闲多久", {k: {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in c["by_idle"].items()})
    p("60 秒 / 3 秒 prefill 倍数", round(c["prefill_ratio_60s_vs_3s"], 2))
    for r in out["lag"]["runs"] + [out["laglong"]]:
        p("并发 %d" % r["b"], {k: round(r[k], 1) if isinstance(r[k], float) else r[k] for k in ("tokens_per_s", "step_ms", "delay_p50", "delay_p99", "delay_max", "merged", "events", "api_cpu_pct", "api_us_per_token", "ping_p99", "ping_max")})
    l = out["lag"]
    p("256 路合并比例 / 每步忙多少毫秒(256, 32) / 忙碌占一步的比例 / 前端上限 token/s", [round(l["merged_pct_256"], 1), round(l["busy_ms_per_step_256"], 1), round(l["busy_ms_per_step_32"], 1), round(l["busy_share_of_step_256"], 2), round(l["max_tokens_per_s_frontend"])])
    p("并发 32 路各轮 p50 / p99 范围", [l["b32_p50_range"], [round(x, 1) for x in l["b32_p99_range"]]])
    p("并发 32 路一共几轮,99 分位不超过 35 的几轮,超出的几轮", [l["b32_runs_total"], l["b32_p99_le_35"], l["b32_p99_over_35"]])
    for h in out["hog"]["rows"]:
        p("hog " + h["label"], {k: round(h[k], 1) if isinstance(h[k], float) else h[k] for k in ("tokens_per_s", "delay_p50", "delay_p99", "delay_max", "api_cpu_pct", "api_us_per_token")})
    p("hog 同核 中位(微秒/token, p50)", [round(out["hog"]["same_cores_us_per_token"], 1), out["hog"]["same_cores_p50"]])
    tl = out["tail"]
    for k, v in tl.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            p("tail " + k, [{kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in r.items()} for r in v])
        else:
            p("tail " + k, [round(x, 1) if isinstance(x, float) else x for x in v] if isinstance(v, list) else (round(v, 1) if isinstance(v, float) else v))
    p("gc", out["gc"])
    p("abort", {k: round(v, 1) if isinstance(v, float) else v for k, v in out["abort"].items()})
    for w in ("w1", "w4"):
        for mode in ("short_only", "long_then_short"):
            p("tokq %s %s" % (w, mode), {k: round(v, 1) for k, v in out["tokq"][w][mode].items() if k != "per_rep"})
    p("tokq 超车", out["tokq"]["overtake"])
    p("tokq 各轮范围", {w: {m: {k: [round(x, 1) for x in v] for k, v in d.items()} for m, d in r.items()} for w, r in out["tokq"]["ranges"].items()})
    p("tokq 每条请求(毫秒)", [round(x, 2) for x in out["tokq"]["per_request_ms"]])
    p("rho_256 / 文字 32k 多出 / 4k 每步 prefill / 各行前端占比", [round(l["rho_256"], 3), round(t["text_extra_ms_32k"], 1), round(t["prefill_ms_per_step_4k"], 1),
                                                      [(r["prompt_tokens"], r["kind"], round(r["front_ms"], 1), round(r["front_share"], 1)) for r in t["rows"]]])
    p("sse_probe", out["sse_probe"]["clients"])
    p("raw32k", out["raw32k"])


show()
