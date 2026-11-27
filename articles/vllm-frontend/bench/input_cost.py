#!/usr/bin/env python3
"""一条"提示词直接给 token 编号"的请求,进引擎之前有几样固定的活,各花多久(随提示词长度)。纯 CPU。

  解析请求体(JSON)      /v1/completions 收到的 182 KB 请求体,32k 个整数
  校验参数(pydantic)    CompletionRequest.model_validate
  打包成 int32 数组      前端发给引擎之前做的 array("i", ...),免得每个 token 一个 Python 整数

运行: taskset -c 56-71 python input_cost.py --out results/input_cost.json
"""
import argparse
import json
import sys
import time
from array import array

from vllm.entrypoints.openai.completion.protocol import CompletionRequest  # noqa: E402

sys.path.insert(0, "/data/wangyue/code/zhihu-frontend-bench")
import bench_client as bc  # noqa: E402


def med(fn, repeat=9):
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return sorted(out)[len(out) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bc.load_corpus()
    rows = []
    for n in (128, 1024, 4096, 16384, 32768):
        ids = bc.token_ids(606060 + n, n)
        body = json.dumps({"model": "m", "prompt": ids, "max_tokens": 4, "temperature": 0.0, "ignore_eos": True,
                           "stream": True, "stream_options": {"include_usage": True}}).encode()
        obj = json.loads(body)
        row = {"tokens": n, "body_bytes": len(body),
               "json_ms": med(lambda: json.loads(body)),
               "validate_ms": med(lambda: CompletionRequest.model_validate(obj)),
               "pack_ms": med(lambda: array("i", obj["prompt"]))}
        rows.append(row)
        print("%6d 个 token,请求体 %7d 字节:解析 JSON %6.2f 毫秒,校验参数 %6.2f 毫秒,打包成 int32 数组 %5.2f 毫秒" % (
            n, len(body), row["json_ms"], row["validate_ms"], row["pack_ms"]), flush=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
