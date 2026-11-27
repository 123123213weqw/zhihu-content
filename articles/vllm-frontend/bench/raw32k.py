"""裸 socket 客户端:发一条长提示词的流式请求,给每次 recv 打时间戳。不要 aiohttp,不要 curl。

  python raw32k.py [--quickack] [--tokens 32768]      --quickack:每次 recv 之前都让内核立刻发 ACK

输出:响应头、prefill 期间收到的空事件、第一个 token 各在"最后一块 prefill 算完"之后多少毫秒到达。
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, "/data/wangyue/code/zhihu-frontend-bench")
os.environ.setdefault("API_PID", "0")
os.environ.setdefault("SCHED_TRACE_FILE", "/data/wangyue/code/zhihu-frontend-bench/results/trace_base.jsonl")
import bench_client as bc

quickack = "--quickack" in sys.argv
n = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 32768
bc.load_corpus()
ids = bc.token_ids(818181 + int(time.time()) % 100000, n)
body = json.dumps({"model": "m", "prompt": ids, "max_tokens": 4, "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
req = b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body
s = socket.create_connection(("127.0.0.1", 8765))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
t0 = time.time()
s.sendall(req)
buf, recvs, rid, kinds = b"", [], None, []
while True:
    if quickack:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
    d = s.recv(65536)
    now = time.time()
    if not d:
        break
    recvs.append((round((now - t0) * 1e3, 1), len(d)))
    buf += d
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        if not line.startswith(b"data: "):
            continue
        p = line[6:].strip()
        if p == b"[DONE]":
            kinds.append(("done", round((now - t0) * 1e3, 1)))
            break
        o = json.loads(p)
        rid = o.get("id", rid)
        kinds.append(("token" if o["choices"][0].get("text") else "empty", round((now - t0) * 1e3, 1)))
    if kinds and kinds[-1][0] == "done":
        break
s.close()
time.sleep(0.3)
rows = bc.load_trace()
x = bc.span(rows, rid.split("-")[1][-6:])
tu, t_add = x["ready"]["tu"], (x["add"] or {"t": x["admit"]["ts"]})["t"]
empties = [t for k, t in kinds if k == "empty"]
first = next(t for k, t in kinds if k == "token")
rec = {"quickack": quickack, "prompt_tokens": n, "prefill_steps": x["prefill_steps"], "to_engine_ms": round((t_add - t0) * 1e3, 1),
       "empty_events": len(empties), "first_token_after_tu_ms": round((t0 + first / 1e3 - tu) * 1e3, 1)}
import json
path = "/data/wangyue/code/zhihu-frontend-bench/results/raw32k.json"
old = json.load(open(path)) if os.path.exists(path) else []
old.append(rec)
json.dump(old, open(path, "w"), indent=1)
print("发出 %.6f,最后一步算完 %.6f" % (t0, tu))
print("quickack=%s 提示词 %d 个 token,分 %d 步 prefill:到引擎 %.1f 毫秒,最后一步算完在发出后 %.1f 毫秒" % (
    quickack, n, x["prefill_steps"], (t_add - t0) * 1e3, (tu - t0) * 1e3))
print("  prefill 期间收到 %d 个空事件,最后两个在最后一步算完之前 %.1f / %.1f 毫秒" % (
    len(empties), (tu - t0) * 1e3 - empties[-2] if len(empties) >= 2 else float("nan"),
    (tu - t0) * 1e3 - empties[-1] if empties else float("nan")))
print("  第 1 个 token 在最后一步算完之后 %.1f 毫秒到达;之后的事件: %s" % (
    (t0 + first / 1e3 - tu) * 1e3, [k for k, t in kinds if t > first]))
