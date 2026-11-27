#!/usr/bin/env python3
"""一条 32k 请求的剖析结果(prof_delta.py 相邻两份快照相减),挑出正文要引用的几个函数,连同这一条请求实测的拆解一起存成 JSON。

用法: python prof_summary.py <之前的快照> <之后的快照> --to-engine-ms 44.0 --out-ms 54.0 --out results/prof_32k.json
(--to-engine-ms / --out-ms 是同一条请求在 one_req.py 里量到的数,见实测记录)
"""
import argparse
import json

import pstats


def load(path):
    return {f: (nc, tt, ct) for f, (cc, nc, tt, ct, callers) in pstats.Stats(path).stats.items()}


ap = argparse.ArgumentParser()
ap.add_argument("before")
ap.add_argument("after")
ap.add_argument("--to-engine-ms", type=float, required=True)
ap.add_argument("--out-ms", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
A, B = load(a.before), load(a.after)
diff = {}
for f in set(A) | set(B):
    n0, t0, c0 = A.get(f, (0, 0.0, 0.0))
    n1, t1, c1 = B.get(f, (0, 0.0, 0.0))
    if n1 - n0 > 0:
        diff[f] = (n1 - n0, (t1 - t0) * 1e3, (c1 - c0) * 1e3)


def find(*needles):
    for f, v in diff.items():
        hay = "%s:%s" % (f[0], f[2])
        if all(n in hay for n in needles):
            return v
    return None


name = lambda f: "%s:%d(%s)" % (f[0].split("site-packages/")[-1].split("lib/python3.12/")[-1], f[1], f[2])
keep = sorted(((v[2], v[1], v[0], f) for f, v in diff.items() if "vllm" in f[0]), reverse=True)[:60]
res = {"request": {"prompt_tokens": 32768, "kind": "ids", "to_engine_ms": a.to_engine_ms, "out_ms": a.out_ms}}
for key, needles in (("decode_stream_step", ("DecodeStream", "step")), ("json", ("decoder.py", "raw_decode")),
                     ("validate", ("_compat/v2.py", "validate")), ("packed_prompt", ("core_client.py", "_with_packed_prompt")),
                     ("output_handler", ("async_llm.py", "output_handler")), ("detokenizer_update", ("detokenizer.py", "update"))):
    v = find(*needles)
    res[key] = None if v is None else {"calls": v[0], "self_ms": round(v[1], 1), "cum_ms": round(v[2], 1)}
res["vllm_top"] = [{"func": name(f), "calls": n, "self_ms": round(tt, 1), "cum_ms": round(ct, 1)} for ct, tt, n, f in keep]
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "vllm_top"}, ensure_ascii=False, indent=1))
