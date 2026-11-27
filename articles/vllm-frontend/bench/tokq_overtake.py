#!/usr/bin/env python3
"""分词线程够多的时候,排在那条 32k 长请求后面的短请求,是不是反而先到了引擎、先被排进去。

从调度记录(bench_client.py tokq 那一轮的 SCHED_TRACE_FILE)里看:每一条 32k 的长请求附近 0.6 秒内到的短请求,
有几条比它先到引擎,有几条在它之前被排进 RUNNING。
运行: python tokq_overtake.py <调度记录 jsonl>   ->  打印,并把结果追加到 results/tokq_overtake.json
"""
import json
import sys

adds, rows = {}, []
for line in open(sys.argv[1]):
    o = json.loads(line)
    if "add" in o:
        adds.setdefault(o["add"], []).append(o)
    elif "k" in o and "ts" in o:
        rows.append(o)
admit_step = {}
for r in rows:
    for n in r["new"]:
        admit_step.setdefault(n[0], r["k"])
out = []
for rid, L in adds.items():
    for a in L:
        if a["n"] < 32000:
            continue
        t_long = a["t"]
        near = [x for xs in adds.values() for x in xs if abs(x["t"] - t_long) < 0.6 and x["n"] < 1000]
        k = admit_step.get(rid)
        if k is None or not near:
            continue
        out.append({"long_prompt_tokens": a["n"], "short_requests": len(near), "arrived_before_long": sum(x["t"] < t_long for x in near),
                    "admitted_before_long": sum(admit_step.get(x["add"], 10 ** 9) < k for x in near),
                    "first_short_minus_long_ms": round((min(x["t"] for x in near) - t_long) * 1e3),
                    "last_short_minus_long_ms": round((max(x["t"] for x in near) - t_long) * 1e3)})
for o in out:
    print("长请求 %d 个 token:附近 %d 条短请求,比它先到引擎 %d 条,在它之前进 RUNNING %d 条;短请求到引擎比它早 %d 到 %d 毫秒" % (
        o["long_prompt_tokens"], o["short_requests"], o["arrived_before_long"], o["admitted_before_long"],
        -o["last_short_minus_long_ms"], -o["first_short_minus_long_ms"]))
path = sys.argv[1].replace(".jsonl", "").replace("trace_", "tokq_overtake_") + ".json"
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print("wrote", path)
