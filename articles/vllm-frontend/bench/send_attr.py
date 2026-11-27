#!/usr/bin/env python3
"""客户端晚收到的那些 token,是前端内部慢,还是前端写出去之后才慢的。

两边各有一份逐 token 的时刻,按请求、按第几个 token 对上:
  服务端:sitecustomize.py 的 SEND_LOG 记下 API server 每往外写一个带文字的 SSE 事件的时刻和请求 id
  客户端:bench_client.py lag / laglong 的结果里,req_ts 是每条请求每个 token 到达的绝对时刻
再从调度记录里取出每一步算完的时刻 tu,两边各自减去它,得到"算完到服务端写出去"和"算完到客户端收到"。

  服务端也晚了差不多一样多  -> 慢在前端里(引擎把结果交出来,到 API server 写出去,这一段)
  服务端很快就写了          -> 慢在前端之后(写出去以后:内核的网络栈,和客户端自己的进程)

运行: python send_attr.py <调度记录 jsonl> <服务端写出时刻 json> <客户端结果 json> [<客户端结果 json> ...]
输出一张表,每轮一行;结果写到 <服务端写出时刻 json> 同目录的 *_attr.json。
"""
import bisect
import json
import sys


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


tus = {}
for line in open(sys.argv[1]):
    o = json.loads(line)
    if "tu" not in o:
        continue
    for rid, pre, dec in o.get("reqs", []):
        if dec > 0:
            tus.setdefault(rid, []).append((o["ts"], o["tu"]))   # 这一步排班的时刻、算完的时刻
for v in tus.values():
    v.sort()
sends = json.load(open(sys.argv[2]))
runs = [json.load(open(p))["runs"][0] for p in sys.argv[3:]]
print("%-4s | %s | %s | %s" % ("轮", "客户端收到: p50 / p99 / 最大(毫秒)", "服务端写出: p50 / p99 / p99.9 / 最大(毫秒)", "晚于 10 毫秒的 token:服务端也晚 / 写出后才晚"))
out = []
for i, r in enumerate(runs):
    t_lo = min(t for _, ts in r["req_ts"] for t in ts) - 1
    t_hi = max(t for _, ts in r["req_ts"] for t in ts) + 1
    by_id = {}
    for t, rid in sends:
        if t_lo <= t <= t_hi and rid:
            by_id.setdefault(rid, []).append(t)
    sd, cd, server_late, after, gaps = [], [], 0, 0, []
    for rid, ts in r["req_ts"]:
        s_ts = by_id.get(rid, [])
        # 第 0 个 token 来自 prefill 结束的那一步,之后每个 token 对一步
        for j, t in enumerate(ts):
            k = j - 1 if j >= 1 else 0
            if j >= 1 and k >= len(tus.get(rid, [])):
                break
            if j == 0:
                pre = [(ts_, tu) for ts_, tu in tus.get(rid, []) if True]
                first_pre = pre[0] if pre else None
                tu = first_pre[1] if first_pre else None
            else:
                tu = tus[rid][k][1]
            if tu is None:
                continue
            c = (t - tu) * 1e3
            if j < len(s_ts):
                sv = (s_ts[j] - tu) * 1e3
                sd.append(sv)
                gaps.append((t - s_ts[j]) * 1e3)
            else:
                sv = float("nan")
            cd.append(c)
            if c > 10.0 and sv == sv:
                if sv >= c - 3.0:
                    server_late += 1
                else:
                    after += 1
    row = {"run": i + 1, "tokens": len(cd), "client_p50": pct(cd, 0.5), "client_p99": pct(cd, 0.99), "client_max": max(cd),
           "server_p50": pct(sd, 0.5), "server_p99": pct(sd, 0.99), "server_p999": pct(sd, 0.999), "server_max": max(sd),
           "write_to_recv_p50": pct(gaps, 0.5), "write_to_recv_p99": pct(gaps, 0.99), "write_to_recv_max": max(gaps),
           "slow_tokens": server_late + after, "server_late": server_late, "after_write": after}
    out.append(row)
    print("%-4d | %5.1f / %5.1f / %6.1f | %5.1f / %5.1f / %5.1f / %6.1f | %d / %d   (写出之后到客户端收到:p50 %.1f p99 %.1f 最大 %.1f 毫秒)" % (
        i + 1, row["client_p50"], row["client_p99"], row["client_max"], row["server_p50"], row["server_p99"], row["server_p999"],
        row["server_max"], server_late, after, row["write_to_recv_p50"], row["write_to_recv_p99"], row["write_to_recv_max"]))
json.dump({"runs": out}, open(sys.argv[2].replace(".json", "_attr.json"), "w"), indent=1)
