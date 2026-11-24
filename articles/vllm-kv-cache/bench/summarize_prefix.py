#!/usr/bin/env python3
"""把 prefix_overhead_4090d.json 汇成文章里引用的表和数。

每个负载、每种配置取各轮输出的中位数,并给出最小到最大的范围;
同时核对:开着的那些轮命中数是 0,全程没有抢占。
Run:  python summarize_prefix.py [prefix_overhead_4090d.json]
"""
from __future__ import annotations
import json
import statistics
import sys

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "prefix_overhead_4090d.json", encoding="utf-8"))
rounds = r["rounds"]
tags = ("off", "on")
print(r["gpu"], "|", r["vllm"].strip(), "| 轮数:", {t: sum(rd["config"] == t for rd in rounds) for t in tags})


def spec(w):
    return f"{w['name']}:输入 {w['input_len']},输出 {w['output_len']},并发 {w['concurrency']}"


def runs(name, tag):
    return [x for rd in rounds if rd["config"] == tag for x in rd["runs"] if x["name"] == name]


print("\n| 负载 | 关着 | 开着,零命中 | 差 |")
print("|---|---|---|---|")
for w in r["workloads"]:
    row = {}
    for t in tags:
        v = [x["output_tps"] for x in runs(w["name"], t)]
        row[t] = (statistics.median(v), min(v), max(v))
    print(f"| {spec(w)} | {row['off'][0]:,.0f} | {row['on'][0]:,.0f} | {row['on'][0] / row['off'][0] - 1:+.2%} |")

print("\n范围(输出 token/秒)和总吞吐:")
for w in r["workloads"]:
    for t in tags:
        xs = runs(w["name"], t)
        v = [x["output_tps"] for x in xs]
        tot = [x["total_tps"] for x in xs]
        clk = [x["clocks"] for x in xs if x["clocks"]]
        print(f"  {w['name']:5s} {t:3s}: 输出中位数 {statistics.median(v):,.0f},范围 {min(v):,.0f} 到 {max(v):,.0f};"
              f"总吞吐中位数 {statistics.median(tot):,.0f};核心时钟 {min(c['sm_clock_mhz_min'] for c in clk)} 到 "
              f"{max(c['sm_clock_mhz_max'] for c in clk)} MHz;最高温度 {max(c['temp_c_max'] for c in clk)} C;功耗峰值 "
              f"{max(c['power_w_max'] for c in clk):.0f} W")

hits = sum(x["prefix_cache_hits"] for rd in rounds if rd["config"] == "on" for x in rd["runs"])
queries = sum(x["prefix_cache_queries"] for rd in rounds if rd["config"] == "on" for x in rd["runs"])
preempt = sum(x["preemptions"] for rd in rounds for x in rd["runs"])
print(f"\n核对:开着的各轮命中 {hits} 个 token,查询 {queries:,} 个 token;抢占 {preempt} 次")

print("\n正文引用的推导数:")
w = r["workloads"][2]
for t in tags:
    s = statistics.median([x["seconds"] for rd in rounds if rd["config"] == t for x in rd["runs"] if x["name"] == w["name"]])
    per = s / (w["num_prompts"] * w["input_len"])
    print(f"  预填充为主 {t}:4096 个 token {per * 4096 * 1000:.0f} 毫秒,每个 token {per * 1e6:.0f} 微秒")
spread = max((max(v) - min(v)) / statistics.median(v)
             for w0 in r["workloads"] for t in tags for v in [[x["output_tps"] for rd in rounds if rd["config"] == t for x in rd["runs"] if x["name"] == w0["name"]]])
print(f"  同一种配置 5 轮之间的最大起伏:{spread:.1%}")
print(f"  哈希(0.12 毫秒)占 4096 个 token 预填充时间的:{0.12 / 542:.5f},约五千分之一")
print("  每次启动的 KV 缓存容量(token):", sorted({rd["kv_cache_tokens"] for rd in rounds}))
