#!/usr/bin/env python3
"""文章里由论文数字推出来的量,都在这里算一遍。

论文:SpecPilot (arXiv:2611.02501v1)。单请求的数字除非注明,都是 H100 SXM5、Qwen3.8-27B + DFlash2、BF16。
C500 的数字来自 JunkaiShi 的复现仓库 (gitlink.org.cn/JunkaiShi/c500-specpilot) 的 README,是对方的实测,这里只拿来推算。
自己的实测来自 bench/w12_stats.json (三个 checkpoint 的 W12 编码) 和 bench/w12_bench_4090d.json (RTX 4090 D),缺哪个就跳过哪一段。
Run:  python3 calc.py  ->  bench/calc_summary.json
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench")
out = {}


def load(name):
    p = os.path.join(BENCH, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- 1. 自回归解码的带宽账
# 论文表 A6:Qwen3.8-27B 的权重 BF16 51.1 GB,W12 39.8 GB;H100 上单请求的 TPOT 19.1 / 15.6 ms。
# H100 SXM5 的峰值带宽 3.35 TB/s (论文 3.1 节)。RTX PRO 6000 实测 1.51 TB/s (论文 5.1 节)。
W_BF16, W_W12 = 51.1e9, 39.8e9
H100_BW, PRO6000_BW = 3.35e12, 1.51e12
ar = {
    "h100": {"tpot_ms": 19.1, "tok_s": 49.9, "bw": H100_BW},                 # 表 A6,表 A1
    "pro6000": {"tpot_ms": 1000 / 26.9, "tok_s": 26.9, "bw": PRO6000_BW},    # 表 A1
    "c500": {"tpot_ms": 38.55, "tok_s": 25.82, "bw": None},                  # 复现仓库 README (vLLM 0.28,BF16)
}
bw_account = {}
for k, v in ar.items():
    eff = W_BF16 / (v["tpot_ms"] * 1e-3)
    bw_account[k] = {"tpot_ms": round(v["tpot_ms"], 2), "tok_s": v["tok_s"], "effective_gbps": round(eff / 1e9, 1),
                     "fraction_of_peak": None if v["bw"] is None else round(eff / v["bw"], 3)}
out["ar_bandwidth"] = {
    "per_platform": bw_account,
    "h100_floor_ms_bf16": round(W_BF16 / H100_BW * 1e3, 2),
    "h100_floor_ms_w12": round(W_W12 / H100_BW * 1e3, 2),
    "w12_ar_speedup_h100": round(19.1 / 15.6, 3),
    "weight_bytes_saved_gb": round((W_BF16 - W_W12) / 1e9, 2),
    "weight_bytes_saved_frac": round(1 - W_W12 / W_BF16, 3),
    "upper_bound_w12": round(16 / 12, 3),
    "c500_free_after_bf16_gb": round(64 - 51.1, 1),
    "c500_free_after_w12_gb": round(64 - 39.8, 1),
}

# ---------------------------------------------------------------- 2. 成本模型:S = tau / c,c = T_step / T_AR
T_AR = 1000 / 49.9                                   # ms,表 A1
stock = {"tau": 4.65, "tok_s": 183.9}                # 表 1,表 A1
final = {"tau": 13.08, "tok_s": 416.1}
c500 = {"ar_tok_s": 25.82, "stock": {"tau": 4.27, "tok_s": 67.79}, "final": {"tau": 9.36, "tok_s": 105.01}}
pro6000 = {"ar_tok_s": 26.9, "stock": {"tau": 5.23, "tok_s": 108.2}, "final": {"tau": 12.80, "tok_s": 272.3}}


def cost_model(t_ar, s, f):
    t_step, t_cycle = 1e3 * s["tau"] / s["tok_s"], 1e3 * f["tau"] / f["tok_s"]      # ms:每步时间 = 每步 token 数 / 每秒 token 数
    return {"t_step_ms": round(t_step, 2), "c_stock": round(t_step / t_ar, 3), "tau_stock": s["tau"], "tok_s_stock": s["tok_s"], "S_stock": round(s["tok_s"] / (1000 / t_ar), 2),
            "t_cycle_ms": round(t_cycle, 2), "c_final": round(t_cycle / t_ar, 3), "tau_final": f["tau"], "tok_s_final": f["tok_s"], "S_final": round(f["tok_s"] / (1000 / t_ar), 2),
            "S_final_over_S_stock": round(f["tok_s"] / s["tok_s"], 3), "t_ar_ms": round(t_ar, 2)}
out["cost_model"] = {
    "h100": cost_model(T_AR, stock, final),
    "pro6000": cost_model(1000 / pro6000["ar_tok_s"], pro6000["stock"], pro6000["final"]),
    "c500": cost_model(1000 / c500["ar_tok_s"], c500["stock"], c500["final"]),
    "note_paper_speedup_is_geomean": {"h100_stock": 3.95, "h100_final": 8.86, "ratio_of_rates": round(416.1 / 49.9, 2)},
}

# ---------------------------------------------------------------- 3. 多验证几行要不要钱
# 论文 3.1 节:H100 上,第一轮之前,GEMM 的总耗时从 1 行的 17.88 ms 到 64 行的 18.58 ms。
out["rows_are_nearly_free"] = {"gemm_ms_1_row": 17.88, "gemm_ms_64_rows": 18.58, "increase": round(18.58 / 17.88 - 1, 4),
                               "extra_per_row_us": round((18.58 - 17.88) * 1e3 / 63, 2)}
bench = load("w12_bench_4090d.json")
if bench:
    r = {}
    for tag, rows in bench["rows"].items():
        us = {int(m): v["us_median"] for m, v in rows.items()}
        n, k = bench["verify"][tag]["info"]["N"], bench["verify"][tag]["info"]["K"]
        floor = 2 * n * k / (bench["copy_bandwidth_gbps"] * 1e9) * 1e6      # us:权重字节数 ÷ 拷贝带宽
        r[tag] = {"shape": [n, k], "us": us, "ratio_16_over_1": round(us[16] / us[1], 3), "ratio_64_over_1": round(us[64] / us[1], 3),
                  "ratio_1024_over_1": round(us[1024] / us[1], 2),
                  "gbps_at_1": rows["1"]["gbps"], "copy_gbps": bench["copy_bandwidth_gbps"],
                  "byte_floor_us": round(floor, 1), "floor_fraction_at_1": round(floor / us[1], 3)}
    out["rows_are_nearly_free"]["rtx4090d"] = r

# ---------------------------------------------------------------- 4. 树:离线回放里每步接受多少 token (论文 6.4 节,图 10)
chain8 = 4.25
out["tree_replay"] = {
    "chain_8_positions": chain8,
    "best_first_8pos_32nodes": {"tau": 5.38, "gain": round(5.38 / chain8 - 1, 3)},
    "best_first_16pos_32nodes": {"tau": 5.85, "gain": round(5.85 / chain8 - 1, 3)},
    "best_first_16pos_63nodes": {"tau": 6.25, "gain": round(6.25 / chain8 - 1, 3)},
    "oracle_8pos": {"tau": 6.42, "gain": round(6.42 / chain8 - 1, 3)},
    "oracle_16pos": {"tau": 8.20, "gain": round(8.20 / chain8 - 1, 3)},
    "live_stock_chain": 4.65,
}

# ---------------------------------------------------------------- 5. 两级验证:一个周期的账 (论文附录 C,图 A3)
draft, q_pass, exact, glue = 1.05, 11.5, 14.9, 1.6
cycle = 2 * draft + q_pass + exact + glue
tok_tlv, tok_one, pair_gain = 11.67, 6.16, 1.037          # 每次精确验证接受的 token;TLV 的成对增益 (表 A3)
t_one = pair_gain * tok_one * cycle / tok_tlv              # 由 (tok_tlv / cycle) / (tok_one / T1) = pair_gain 反推
bytes_cycle = 0.5 * W_BF16 + W_W12                        # Q pass 读 8 bit 视图,精确那趟读 W12
out["two_level"] = {
    "cycle_ms": round(cycle, 2), "parts_ms": {"draft_x2": 2 * draft, "q_pass": q_pass, "exact_pass": exact, "glue": glue},
    "q_pass_over_exact": round(q_pass / exact, 3),
    "tokens_per_exact_pass": {"one_level": tok_one, "two_level": tok_tlv, "ratio": round(tok_tlv / tok_one, 3)},
    "one_level_step_ms_derived": round(t_one, 2),
    "one_level_tok_s_derived": round(tok_one / t_one * 1e3, 1), "two_level_tok_s": round(tok_tlv / cycle * 1e3, 1),
    "cycle_over_one_level_step": round(cycle / t_one, 3),
    "cycle_over_ar_step": round(cycle / T_AR, 3),
    "bytes_per_cycle_gb": round(bytes_cycle / 1e9, 2), "bytes_over_ar": round(bytes_cycle / W_BF16, 3),
    "h100_floor_ms": round(bytes_cycle / H100_BW * 1e3, 2), "fraction_of_peak": round(bytes_cycle / (cycle * 1e-3) / H100_BW, 3),
    # 批处理那边的盈亏平衡 (论文 6.5 节):Q pass 占一步的 0.80 到 0.88
    "breakeven_tokens_batching": [1.80, 1.88],
}

# ---------------------------------------------------------------- 6. H100 单请求那条线上的 15 次提升 (论文表 A3)
promotions = [  # (轮次, 改动, 类别, 成对增益)
    (1, "FP8 草稿,块长 12", "草稿模型", 1.160), (1, "树状验证 (16 节点)", "树", 1.160), (1, "宽树 (32 节点)", "树", 1.065),
    (2, "瘦 GEMM 与融合", "其它内核", 1.031), (2, "certified LM head", "其它内核", 1.017), (2, "内核预置", "其它内核", 1.023),
    (2, "GDN 树内核 (WY)", "其它内核", 1.030), (2, "W12", "W12", 1.076), (2, "W12 流水线调优", "W12", 1.063),
    (3, "尾步跳过", "其它内核", 1.010), (3, "INT4 草稿,快速接受", "草稿模型", 1.022), (3, "W12 布局与边界校准", "W12", 1.045),
    (3, "两级验证", "两级验证", 1.037), (3, "prefill 图与 Q-pass 门控", "其它内核", 1.015), (3, "草稿到精确验证的交接融合", "其它内核", 1.010),
]
groups, total, cum = {}, 1.0, []
for rnd, name, cat, g in promotions:
    total *= g
    groups.setdefault(cat, []).append(g)
    cum.append((name, cat, round(total, 3)))
out["promotions_h100"] = {
    "count": len(promotions), "product": round(total, 3), "sealed": 2.13,
    "by_group": {k: {"n": len(v), "product": round(math.prod(v), 3), "items": v} for k, v in groups.items()},
    "largest_single": max(promotions, key=lambda p: p[3])[1],
    "cumulative": cum,
    "tokens_per_s": {"stock": 183.9, "after_round_1": 270, "final": 416.1},
}

# ---------------------------------------------------------------- 7. 上限的三个因子 (附录 B)
out["ceiling_factors"] = {"before_round_1": {"tau": 1.83, "overhead": 1.01, "kernel": 1.39, "product": round(1.83 * 1.01 * 1.39, 3), "paper": 2.57},
                          "before_round_2": {"tau": 1.34, "step_cost": 1.45, "product": round(1.34 * 1.45, 3), "paper": 1.95},
                          "byte_floor": "取两个下限里大的那个:FLOP 时间按峰值稠密算力的 60%,字节时间按峰值带宽的 90%"}

# ---------------------------------------------------------------- 8. 批处理 (论文 3.1 节,表 2,附录 A)
out["batching"] = {
    "stock_fp8_speedup": {"1_request": 3.40, "24_requests": 1.89, "drop": round(1 - 1.89 / 3.40, 3)},
    "stock_qwen35_4b_speedup": {"1_request": 3.73, "32_requests": 1.49, "drop": round(1 - 1.49 / 3.73, 3)},
    "shared_node_budget": {"nodes": 192, "requests": 24, "nodes_per_request": 8},
    "per_request_rates_h100_fp8_24_requests": {"ar": 61.4, "stock": 115, "final": 215,
                                               "S_stock": round(115 / 61.4, 2), "S_final": round(215 / 61.4, 2)},
    "eagle3_gemma4_12b": {"stock": 0.68, "final": 2.09},
}

# ---------------------------------------------------------------- 9. 搜索的开销 (附录 D)
out["search_cost"] = {"rounds": 13, "hours": 164, "gpu_hours": 530, "gpus_average": round(530 / 164, 2),
                      "changes": 486, "promoted": 54, "promoted_frac": round(54 / 486, 3), "memory_entries": 140}

# ---------------------------------------------------------------- 10. W12 在三个 checkpoint 上
stats = load("w12_stats.json")
if stats:
    s = {}
    for model, m in stats["models"].items():
        s[model] = {"weights": m["weights"], "escape_rate": m["escape_rate"], "paper_escape_rate": m["paper_escape_rate"],
                    "ratio_to_paper": round(m["escape_rate"] / m["paper_escape_rate"], 3),
                    "bits_per_weight": round(m["bits_per_weight"], 3), "paper_bits_per_weight": m["paper_bits_per_weight"],
                    "bit_mismatch": m["bit_mismatch"], "escape_reasons": m["escape_reasons"],
                    "bytes_saved_frac": round(1 - m["bytes_w12"] / m["bytes_bf16"], 4),
                    "by_category": {k: round(v["esc"] / v["weights"], 8) for k, v in m["by_category"].items()}}
        if "language_model_only" in m:                         # 带视觉塔的 checkpoint:只算语言模型的那部分
            lm = m["language_model_only"]
            s[model]["language_model_only"] = {"weights": lm["weights"], "escape_rate": lm["escape_rate"],
                                               "ratio_to_paper": round(lm["escape_rate"] / m["paper_escape_rate"], 3),
                                               "bits_per_weight": round(lm["bits_per_weight"], 3)}
    out["w12_checkpoints"] = s

# ---------------------------------------------------------------- 11. 4090D 上的五种读法
if bench:
    v = {}
    for tag, d in bench["verify"].items():
        p, info = d["points"], d["info"]
        n, k = info["N"], info["K"]
        us16 = {kk: p["16"][kk]["us_median"] for kk in p["16"]}
        read_gbps = {kk: round(bench["bytes_per_weight"][kk] * n * k / (us16[kk] * 1e-6) / 1e9, 1)
                     for kk in ("bf16", "bf16_triton", "w12", "q8")}   # 按 GEMM 真正读的权重字节算;dequant 另有落盘和读回,不列
        v[tag] = {"shape": [n, k], "escapes": info["escapes"], "escape_rate": round(info["escapes"] / (n * k), 7),
                  "speedup_vs_bf16_at_1": info["speedup_vs_bf16"]["1"],
                  "speedup_vs_bf16_at_16": info["speedup_vs_bf16"]["16"],
                  "speedup_vs_bf16_at_64": info["speedup_vs_bf16"]["64"],
                  "us_at_1": {kk: p["1"][kk]["us_median"] for kk in p["1"]},
                  "us_at_16": us16, "us_at_64": {kk: p["64"][kk]["us_median"] for kk in p["64"]},
                  "gbps_at_16": {kk: p["16"][kk]["gbps"] for kk in p["16"]},
                  "effective_read_gbps_at_16": read_gbps,
                  "error": info["error"], "bitwise_equal_to_cublas": info.get("bitwise_equal_to_cublas"),
                  "gpu_during": info.get("gpu"), "gpu_after_warmup": info.get("gpu_after_warmup"),
                  "best_config": {kind: bench["configs"][f"{tag}:{kind}"]["best"] for kind in ("bf16_triton", "w12", "q8")}}
    out["rtx4090d"] = {"copy_gbps": bench["copy_bandwidth_gbps"], "gpu": bench["gpu"], "torch": bench["torch"], "triton": bench["triton"], "verify": v}

# ---------------------------------------------------------------- 12. 换一块卡
out["platforms"] = {
    "h100": {"ar": 49.9, "stock": 183.9, "final": 416.1, "source": "论文表 A1"},
    "pro6000": {"ar": 26.9, "stock": 108.2, "final": 272.3, "source": "论文表 A1"},
    "c500": {"ar": 25.82, "stock": 67.79, "final": 105.01, "source": "JunkaiShi 复现仓库 README"},
    "pro6000_porting": {"round_1_sealed_gain_over_stock": 1.92, "after_tlv": 2.44, "ceiling": 2.12, "gpu_hours_round_1": 24.3},
}

json.dump(out, open(os.path.join(BENCH, "calc_summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
