#!/usr/bin/env python3
"""在真实 checkpoint 上校验 W12:解码是不是逐 bit 等于原权重,escape 占多少。

对每个 bfloat16 的二维权重做一遍 w12_encode / w12_decode,统计:
  - 窗口内的权重解码后和原权重的 16 个 bit 是否完全一样(必须全部一样)
  - escape 的个数,分三种原因:0 或非规格数、低于窗口、高于窗口
  - 按这个口径算每个权重占多少 bit:12 + 每个 escape 8 字节(列号 int32 + 精确值 float32)+ 每行 8 字节(行首指针 int32 + 公因子 float32)
不是 bfloat16 的、或者不是二维的张量不编码,只记下字节数。

三个模型都是论文表 A2 里的:论文给的 escape 比例分别是 7.66e-5、1.21e-4、7.88e-5。
Run:  python w12_stats.py --out bench/w12_stats.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import time

import torch
from safetensors import safe_open

from w12_ref import w12_decode, w12_encode

PAPER = {  # 论文表 A2:escape 比例,每权重 bit 数
    "Llama-3.1-8B-Instruct": (7.66e-5, 12.015),
    "Mistral-7B-Instruct-v0.3": (1.21e-4, 12.018),
    "Qwen3.5-4B": (7.88e-5, 12.020),
}
EXAMPLE = "model.layers.16.mlp.down_proj.weight"  # 给文章出图用:这一层里一行权重的指数分布


def category(name: str) -> str:
    if ".visual." in name:                                       # Qwen3.5-4B 的 checkpoint 带一个视觉塔,和语言模型分开算
        return "visual tower"
    if "mtp." in name:
        return "MTP head"
    if "embed" in name or "lm_head" in name:
        return "embedding / lm_head"
    if re.search(r"mlp|gate_proj|up_proj|down_proj|ffn|feed_forward", name):
        return "MLP"
    if re.search(r"self_attn|attention|q_proj|k_proj|v_proj|o_proj|qkv", name):
        return "attention"
    if re.search(r"linear_attn|gdn|mamba|in_proj|out_proj|dt_proj|ba_proj", name):
        return "linear attention"
    return "other"


def new_bucket():
    return {"tensors": 0, "rows": 0, "weights": 0, "esc": 0, "esc_zero": 0, "esc_below": 0, "esc_above": 0,
            "bit_mismatch": 0, "max_esc_in_a_row": 0, "bytes_bf16": 0}


def encode_one(t: torch.Tensor, b: dict):
    u = t.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    hi, lo, E, esc = w12_encode(t)
    back = w12_decode(hi, lo, E).view(torch.int16).to(torch.int32) & 0xFFFF
    inside = ~esc
    b["bit_mismatch"] += int((back[inside] != u[inside]).sum())
    del back
    c = e - (E[:, None] - 15)
    b["tensors"] += 1
    b["rows"] += t.shape[0]
    b["weights"] += t.numel()
    b["bytes_bf16"] += t.numel() * 2
    b["esc"] += int(esc.sum())
    b["esc_zero"] += int((esc & (e == 0)).sum())
    b["esc_below"] += int((esc & (c < 0) & (e != 0)).sum())
    b["esc_above"] += int((esc & (c > 15)).sum())
    b["max_esc_in_a_row"] = max(b["max_esc_in_a_row"], int(esc.sum(1).max()))
    return E, esc, e, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="/data/wangyue/models")
    ap.add_argument("--out", default="w12_stats.json")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    result = {"torch": torch.__version__, "models": {}}
    for model, (paper_esc, paper_bits) in PAPER.items():
        if args.only and args.only not in model:
            continue
        d = os.path.join(args.models_root, model)
        wmap = json.load(open(glob.glob(os.path.join(d, "*.index.json"))[0]))["weight_map"]
        by_shard = collections.defaultdict(list)
        for tname, shard in wmap.items():
            by_shard[shard].append(tname)
        cats, skipped, pattern_rows = collections.defaultdict(new_bucket), collections.Counter(), {}
        example = None
        t0 = time.time()
        for shard in sorted(by_shard):
            with safe_open(os.path.join(d, shard), "pt") as f:
                for tname in sorted(by_shard[shard]):
                    shape = f.get_slice(tname).get_shape()
                    dtype = str(f.get_slice(tname).get_dtype())
                    if len(shape) != 2 or dtype != "BF16":
                        n = 1
                        for s in shape:
                            n *= s
                        skipped[f"{dtype} ndim={len(shape)}"] += n * {"BF16": 2, "F16": 2, "F32": 4}.get(dtype, 4)
                        continue
                    t = f.get_tensor(tname)
                    E, esc, e, c = encode_one(t, cats[category(tname)])
                    if tname == EXAMPLE:
                        per_row = esc.sum(1)
                        order = per_row.argsort()
                        rows = {"median": int(order[order.numel() // 2]), "worst": int(order[-1])}
                        example = {"tensor": tname, "shape": list(t.shape), "rows": {}}
                        for tag, r in rows.items():
                            cnt = torch.bincount(e[r], minlength=256)
                            example["rows"][tag] = {
                                "row": r, "E": int(E[r]), "escapes": int(per_row[r]),
                                "hist": {str(i): int(cnt[i]) for i in range(256) if cnt[i]},
                                "esc_cols": esc[r].nonzero()[:, 0][:64].tolist(),
                                "esc_vals": [float(v) for v in t[r][esc[r]][:64]],
                            }
                        example["rows_with_escape"] = int((per_row > 0).sum())
                        example["rows_total"] = t.shape[0]
                        example["esc_hist"] = {str(k): int(v) for k, v in sorted(collections.Counter(per_row.tolist()).items())}
                    del t, E, esc, e, c
        tot = new_bucket()
        for b in cats.values():
            for k, v in b.items():
                tot[k] = v if k == "max_esc_in_a_row" else tot[k] + v
        rate = tot["esc"] / tot["weights"]
        rows, weights = tot["rows"], tot["weights"]
        mine = 12 + 64 * rate + 64 * rows / weights
        result["models"][model] = {
            "paper_escape_rate": paper_esc, "paper_bits_per_weight": paper_bits,
            "weights": weights, "escapes": tot["esc"], "escape_rate": rate,
            "escape_reasons": {"zero_or_denormal": tot["esc_zero"], "below_window": tot["esc_below"], "above_window": tot["esc_above"]},
            "bit_mismatch": tot["bit_mismatch"], "max_escapes_in_a_row": tot["max_esc_in_a_row"],
            "rows": rows, "bits_per_weight": mine,
            "bytes_bf16": tot["bytes_bf16"],
            "bytes_w12": 12 * weights / 8 + 8 * tot["esc"] + 8 * rows,
            "by_category": {k: {kk: vv for kk, vv in v.items()} for k, v in cats.items()},
            "skipped_bytes": dict(skipped), "seconds": round(time.time() - t0, 1), "example": example,
        }
        r = result["models"][model]
        lm = [v for k, v in cats.items() if k not in ("visual tower", "MTP head")]
        if len(lm) < len(cats):                                   # 只算语言模型的那部分
            lw, le = sum(v["weights"] for v in lm), sum(v["esc"] for v in lm)
            r["language_model_only"] = {"weights": lw, "escapes": le, "escape_rate": le / lw, "bits_per_weight": 12 + 64 * le / lw + 64 * sum(v["rows"] for v in lm) / lw}
            print(f"    language model only: weights {lw:,} escapes {le:,} ({le / lw:.3e}) bits/weight {r['language_model_only']['bits_per_weight']:.3f}", flush=True)
        print(f"{model}: weights {weights:,} escapes {tot['esc']:,} ({rate:.3e}; paper {paper_esc:.2e}) "
              f"bit mismatch {tot['bit_mismatch']} bits/weight {mine:.3f} (paper {paper_bits:.3f}) {r['seconds']} s", flush=True)
        for k, v in cats.items():
            print(f"    {k:20s} weights {v['weights']:>15,} esc {v['esc']:>9,} ({v['esc'] / v['weights']:.2e}) max/row {v['max_esc_in_a_row']}", flush=True)
    # 只跑了其中几个模型时 (--only),保留输出文件里已有的其它模型
    if os.path.exists(args.out):
        kept = json.load(open(args.out))["models"]
        kept.update(result["models"])                           # 这次跑过的模型用新结果,没跑的保留
        result["models"] = kept
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
