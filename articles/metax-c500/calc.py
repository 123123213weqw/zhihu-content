#!/usr/bin/env python3
"""文章里的数都在这里算一遍。

沐曦没有公布 C500 的算力峰值和显存带宽, 所以每个输入都标了出处和证据等级:

  【官方】 沐曦自己的材料
           - 招股说明书 (2025 年 4 月申报稿, 490 页), 页码写成 1-1-xxx
           - 首轮审核问询的回复 (上交所, 2025-08-27, 314 页), 页码写成 8-1-xx
             8-1-58 到 8-1-64 是核心技术形成及演变过程表 (列是 2020-2021 / 2022-2023 / 2024-2025, C500 在中间一列)
             8-1-86 到 8-1-93 是 MXMACA 怎么兼容 CUDA 生态
           - 官网产品页 metax-tech.com/prod.html, 开发者社区的《曦云系列通用计算GPU快速上手指南》
           - 《MXMACA-3.3.0.X 简要技术报告》(2025-12)
           - GitHub 组织 MetaX-MACA 下的公开代码
  【平台】 模力方舟 (moark.com) 的 C500 文档, 自称引自官方数据的理论峰值
  【实测A】 xpu-perf 的单卡算子级测试 (CSDN, 2026-10-14, MACA 3.8.2.6), 在 sGPU 切片上跑的
  【实测B】 SpecPilot 复现仓库的硬件档案 (gitlink.org.cn/JunkaiShi/c500-specpilot, 2026-11-20 记录), 整卡
  【对比】 NVIDIA A100 的官方规格页; H20 取自问询回复里的表, 带宽取自 TechPowerUp 的数据库
  推算     我的计算, 前提写在注释里

Run:  python3 calc.py  ->  bench/calc_summary.json
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---------------------------------------------------------------- 1. 身份 (官方)
out["identity"] = {
    "company_founded": "2020-09-14",                 # 招股书: 沐曦集成电路 (上海) 股份有限公司成立日期
    "n100_mass_production": "2023-04",               # 招股书: 智算推理 GPU 芯片曦思 N100 系列 2023 年 4 月正式量产
    "c500_tapeout_return": "2023-06",                # 问询回复 8-1-28: 曦云 C500 于 2023 年 6 月回片
    "c500_mass_production": "2024-02",               # 问询回复 8-1-28: 2024 年 2 月正式量产
    "architecture": "XCORE 1.0",
    "isa_instructions_xcore1": ">600",               # 招股书: XCORE1.0 计算 GPU 的指令集超过 600 条
    "hbm_gb": 64, "tdp_w_pcie_card": 350, "tdp_w_oam_module": 450,
    "hbm_phy_count": 4,                              # 招股书 1-1-122 的芯片框图: 左右各两个 HBM PHY
    "hbm_stacks": 4, "package": "2.5D CoWoS-S",      # 问询回复 8-1-61: 采用 4 颗 HBM2e, 所有 HBM2e 颗粒下是 CoWoS-S
    "vgpu_instances": 16,                            # 问询回复 8-1-62: 虚拟化技术可支持 16 个实例切分 (官网写最小 1% 颗粒度)
    "cache_levels": 2,                               # 问询回复 8-1-59: 两级缓存, L1 在流处理器内, L2 在流处理器之间共享
    "metaxlink_ports": 7,                            # 招股书 1-1-122: 单颗芯片 7 个高速 MetaXLink 互连接口
    "precisions_c500": ["FP64", "FP32", "TF32", "FP16", "BF16", "INT8"],   # 问询回复 8-1-27
}

# ---------------------------------------------------------------- 2. 显存带宽
# 问询回复 8-1-61 (芯粒架构与先进封装设计技术): "采用 4 颗 HBM2e, 在所有 HBM2e 颗粒下的 CoWoS-S 上实现单通道
# 3.6Gbps, 成功应用于曦云 C500"。堆栈数和每根线的速率都是官方的。
# 剩下 "每个堆栈 1024 根线" 是 HBM2e 的标准位宽 (JEDEC); 沐曦没有把这几个数乘出来公布过。
# 框图里有 4 个 HBM PHY (招股书 1-1-122), 和 "4 颗" 对得上。显存 64 GB -> 每颗 16 GB。
stack_bits, stacks, pin_gbps = 1024, 4, 3.6
bw = {rate: stacks * stack_bits * rate / 8 / 1e3 for rate in (3.2, pin_gbps)}      # TB/s; 3.2 Gbps 是 HBM2e 的下限速率, 用来排除
out["hbm"] = {
    "gb_per_stack": out["identity"]["hbm_gb"] / stacks,
    "stacks_official": stacks, "pin_rate_gbps_official": pin_gbps, "bits_per_stack_assumed": stack_bits,
    "tbps_derived": round(bw[pin_gbps], 3), "tbps_if_3p2_gbps": round(bw[3.2], 3),
    "third_party_claims_tbps": [1.8, 3.35],          # 1.8: flopper.io / ici-cn / CSDN 的分析文; 3.35: 一篇 CSDN 对比文
    # 实测的读带宽上限 (GB/s)
    "measured_gbps": {
        "gemm_read_m1_gate_up": 1523.9,              # 实测B: 17408×5120 的权重, 1 行, 117.0 us
        "gemm_read_m1_lm_head": 1654,                # 实测B: 248320×5120 的权重 (2.543 GB), 1 行, mcBLAS, 1536.7 us
        "elementwise": [1470, 1571],                 # 实测A: 逐元素算子和 reduce_sum
        "copy_d2d": [1427.4, 1443],                  # 实测B / 实测A: 1 GB 拷贝, 读加写
    },
}
m = out["hbm"]["measured_gbps"]
out["hbm"]["efficiency"] = {  # 实测读带宽 ÷ 推算的峰值
    "read_1654_over_1p64": round(1654 / (bw[3.2] * 1e3), 3), "read_1654_over_1p84": round(1654 / (bw[3.6] * 1e3), 3),
    "read_1654_over_3p35": round(1654 / 3350, 3),
    "copy_1427_over_1p84": round(1427.4 / (bw[3.6] * 1e3), 3),
}

# ---------------------------------------------------------------- 3. 解码一步的账 (同上一篇的算法)
# 实测B: vLLM 0.28.0 上 Qwen3.8-27B BF16 自回归解码, 单请求 25.71 tok/s, TPOT 38.69 ms
W27 = 51.1e9                                           # SpecPilot 论文表 A6: 权重 51.1 GB
tpot_ms = 38.69
out["ar_decode"] = {
    "tok_s": 25.71, "tpot_ms": tpot_ms,
    "effective_gbps": round(W27 / (tpot_ms * 1e-3) / 1e9, 1),
    "floor_ms_at_1p84": round(W27 / (bw[3.6] * 1e12) * 1e3, 1),
    "max_tok_s_at_1p84": round((bw[3.6] * 1e12) / W27, 1),
    "free_gb_after_bf16": round(64 - 51.1, 1), "free_gb_after_w12": round(64 - 39.8, 1),   # W12: 论文的 12 bit 格式, 39.8 GB
}

# ---------------------------------------------------------------- 4. 一个 CU 里有多少运算通路 (推算, 前提: 104 个 CU)
# 104 个 CU 来自实测B 的硬件档案, 只此一处出处。
# 平台文档给的峰值 (OAM 形态): FP32 (向量) 18, FP32 (矩阵) 36, TF32 140, FP16/BF16 280 TFLOPS, INT8 560 TOPS。
# 另一组流传的数 (8 卡服务器的规格): FP32 15, FP16 240 TFLOPS, INT8 512 TOPS (百科写 480)。
# 假设每个 CU 有 64 条 FP32 通路 (一个波前 64 个线程, 4 个 SIMD 各 16 位宽), 矩阵单元每个周期做 1024 次 BF16 乘加
# (一条 16×16×16 的 MMA 指令 4 个周期)。反推频率。
cu = 104
lanes, macc_per_cycle = 64, 1024
mma_mac = 16 * 16 * 16
def fma_clock(tflops):        # 让向量峰值成立的频率 (GHz)
    return tflops * 1e12 / 2 / (cu * lanes) / 1e9
def mma_clock(tflops):        # 让矩阵峰值成立的频率 (GHz)
    return tflops * 1e12 / 2 / (cu * macc_per_cycle) / 1e9
out["per_cu"] = {
    "cu": cu, "mma_mac_per_instruction": mma_mac, "accumulators_per_wavefront": 64 * 4,
    "mma_cycles_if_1024_mac": mma_mac / macc_per_cycle,
    "set_oam": {"fp32": 18, "bf16": 280, "int8": 560, "ghz_from_fp32": round(fma_clock(18), 3), "ghz_from_bf16": round(mma_clock(280), 3)},
    "set_pcie": {"fp32": 15, "bf16": 240, "int8": 480, "ghz_from_fp32": round(fma_clock(15), 3), "ghz_from_bf16": round(mma_clock(240), 3)},
    "measured_simt_fp32": 15,                        # 实测B: C500 FP32 SIMT 算力约 15 TFLOPS
    "measured_simt_over_set_pcie": round(15 / 15, 3),
    "mac_per_cycle_if_3p2ghz_240": 240e12 / 2 / cu / 1.125e9,   # 1.125 GHz 下每 CU 每周期多少次乘加 (校验)
}

# A100 (NVIDIA Ampere 白皮书): 108 个 SM, 每个 SM 64 个 FP32 核、每周期 1024 次 FP16/BF16 乘加 (4 个 Tensor Core), 加速频率 1.41 GHz
a100_sm, a100_f = 108, 1.41
c500_f = out["per_cu"]["set_oam"]["ghz_from_bf16"]
out["a100"] = {
    "sm": a100_sm, "boost_ghz": a100_f,
    "fp32_tflops": round(a100_sm * lanes * 2 * a100_f / 1e3, 1),          # 19.5
    "bf16_tflops": round(a100_sm * macc_per_cycle * 2 * a100_f / 1e3, 1),  # 311.8
}
out["nine_tenths"] = {
    "cu_ratio": round(cu / a100_sm, 3), "clock_ratio": round(c500_f / a100_f, 3),
    "product": round(cu / a100_sm * c500_f / a100_f, 3), "peak_ratio": round(280 / 312, 3),
    "per_thread_macs_nvidia_m16n8k16": 16 * 8 * 16 // 32, "per_thread_macs_c500": mma_mac // 64,
}

# ---------------------------------------------------------------- 5. 矩阵乘的实测效率 (实测A)
out["gemm"] = {
    "bf16_measured_tflops": 229.6, "bf16_spec": 280, "bf16_eff": round(229.6 / 280, 3),
    "bf16_eff_of_240": round(229.6 / 240, 3),
    "fp32_measured_tflops": 29.9, "fp32_matrix_spec": 36, "fp32_eff": round(29.9 / 36, 3),
    "fa_prefill_tflops": 139.5, "shape": "30720×8192×8192, torch (mcBLAS)",
    "int8_measured_tops": 112.3, "int8_note": "torch 兜底路径, 不是厂商内核",
}

# ---------------------------------------------------------------- 6. 拐点
ridge_spec = 280e12 / (bw[3.6] * 1e12)
ridge_meas = 229.6e12 / 1523.9e9
out["ridge"] = {
    "flop_per_byte_spec": round(ridge_spec, 1), "flop_per_byte_measured": round(ridge_meas, 1),
    "rows_at_ridge_spec": round(ridge_spec / 2),            # 验证 M 行时, 每个权重字节做 2M 次运算
    "util_at_m1": round(2 / ridge_spec, 4),                 # 自回归解码 M = 1 时, 矩阵单元用上的比例
}

# ---------------------------------------------------------------- 7. 矩阵乘的台阶 (实测B)
plateau_m = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
gate_up = {"shape": "17408×5120", "mb": 178.3, "us": [117.0, 133.6, 133.8, 135.8, 135.6, 137.7, 140.8, 168.6, 172.5, 192.9, 199.8, 271.2, 277.8],
           "gbps": [1523.9, 1335.7, 1335.1, 1316.9, 1319.6, 1302.0, 1275.8, 1069.8, 1050.0, 946.7, 920.9, 689.2, 683.3]}
o_proj = {"shape": "5120×6144", "mb": 62.9, "us": [52.8, 59.0, 65.4, 49.7, 50.0, 50.0, 49.0, 53.8, 58.3, 72.4, 79.9, 109.6, 113.5],
          "gbps": [1192.6, 1068.3, 964.7, 1271.4, 1266.0, 1269.0, 1300.0, 1188.8, 1104.5, 899.1, 823.1, 613.6, 605.3]}
def steps(d):
    u = d["us"]
    return {"m1_to_m32": round(u[6] / u[0] - 1, 3), "m32_to_m48": round(u[7] / u[6] - 1, 3), "m128_to_m192": round(u[11] / u[10] - 1, 3)}
out["plateau"] = {"m": plateau_m, "gate_up": gate_up, "o_proj": o_proj, "gate_up_steps": steps(gate_up), "o_proj_steps": steps(o_proj)}

# ---------------------------------------------------------------- 8. 互联
out["metaxlink"] = {
    "total_gbps_8card": 896, "ports": 7, "per_port_gbps": round(896 / 7, 1),
    "nvlink_a100_gbps": 600, "nvlink_h20_h200_gbps": 900, "infinity_fabric_gbps_as_cited": 896,
    "over_a100": round(896 / 600, 2), "over_h200": round(896 / 900, 3),
}

# ---------------------------------------------------------------- 9. 放在一起
cmp = {  # (C500 平台文档, A100 官方, H20)
    "bf16_tflops": (280, 312, 148), "fp32_tflops": (18, 19.5, None), "tf32_tflops": (140, 156, None), "int8_tops": (560, 624, 296),
    "memory_gb": (64, 80, 96), "hbm_tbps": (round(bw[pin_gbps], 3), 2.039, 4.0), "interconnect_gbps": (896, 600, 900), "tdp_w": (350, 400, 400),
}
out["compare"] = {k: {"c500": v[0], "a100": v[1], "h20": v[2], "c500_over_a100": (round(v[0] / v[1], 3) if v[0] and v[1] else None),
                      "c500_over_h20": (round(v[0] / v[2], 3) if v[0] and v[2] else None)} for k, v in cmp.items()}
out["compare"]["measured_c500"] = {"bf16_gemm_over_a100_spec": round(229.6 / 312, 3), "read_bw_over_a100_spec": round(1654 / 2039, 3),
                                   "gemm_read_over_a100_spec": round(1523.9 / 2039, 3)}

# ---------------------------------------------------------------- 10. 软件栈
out["software"] = {
    "repos_tested": 4490, "repos_direct": 4173, "repos_modified": 260, "repos_unusable": 57,   # MXMACA 3.3.0.X 技术报告
    "direct_rate": round(4173 / 4490 * 100, 2), "modified_rate": round(260 / 4490 * 100, 2),
    "pytorch_ops": 2650, "pytorch_gpu_ops": 2410, "frameworks": 40, "model_categories": 500,
    # 问询回复 (2025-08-27): 8-1-87 支持超过 6,000 个 CUDA 应用和超过 1,000 个开源模型; 8-1-62 C500 的编译器基于 LLVM 12,
    # 近 13,000 个编译器测试用例; 8-1-90 开发者社区截至 2025-07-31 注册用户超过 15,000 人
    "cuda_apps_supported": 6000, "open_source_models": 1000, "compiler_base_for_c500": "LLVM 12",
    "compiler_test_cases": 13000, "registered_dev_users_by_2025_07_31": 15000,
    "triton_w12_m1_over_blas": 0.49, "triton_w12_m32_over_blas": 0.11, "triton_q8_m1_over_blas": 0.64,     # 实测B
    "lds_fix_ms": 13.46, "lds_before_ms": 59.72, "lds_after_ms": 44.81,                             # 实测B: 64 层加 lm_head, M = 32
    "lds_gain": round(59.72 / 44.81, 3),
}

# ---------------------------------------------------------------- 11. 沐曦算子库里内建指令的用法 (bench/mcoplib_builtins.txt, 提交 39511c5)
# 逐类求和; "其中直接进共享内存的" 是 "带谓词的全局读写" 的子集, 不计入合计。
tally = {}
for line in open(os.path.join(HERE, "bench", "mcoplib_builtins.txt"), encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2 and parts[0].isdigit():
        tally[parts[1].replace("__builtin_mxc_", "")] = int(parts[0])


def used(prefixes):
    return sum(n for name, n in tally.items() if name.startswith(prefixes))


by_cat = {
    "矩阵乘加": used(("mma_",)), "倒数": used(("rcpf",)), "打包的 FP32 乘加": used(("pk_fma_f32",)),
    "类型转换": used(("cvt_", "b0_cast", "b1_cast", "b2_cast", "b3_cast", "i16_to_f16")),
    "带谓词的全局读写": used(("ldg_", "stg_")), "其中直接进共享内存的": used(("ldg_b32_bsm", "ldg_b64_bsm", "ldg_b128_bsm")),
    "位域提取": used(("ubfe",)), "同步": used(("barrier_ex", "arrive", "barrier_inst")),
    "字节重排": used(("byte_perm",)), "投票计数": used(("mbcnt_lo",)),
}
out["builtins"] = {"kinds": len(tally), "call_sites": sum(tally.values()), "by_category": by_cat,
                   "sum_of_disjoint_categories": sum(n for k, n in by_cat.items() if not k.startswith("其中"))}

json.dump(out, open(os.path.join(HERE, "bench", "calc_summary.json"), "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(f"== {k}\n{json.dumps(v, ensure_ascii=False)}")
