#!/usr/bin/env python3
"""文章里的数都在这里算一遍。

出处分四类, 写在每一段的注释里:

  【官方代码】 沐曦给 C500 写的 GDN kernel: GitHub 组织 MetaX-MACA 下 mcoplib 的 op/chunk_gated_delta_rule_fwd_h_native.cu (提交 39511c5)
              上游 vLLM v0.28.0 的 vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py 和 vllm/model_executor/custom_op.py
              vLLM-metax v0.28.0 (分支 releases/v0.28.0): pyproject.toml, vllm_metax/patch/, vllm_metax/registry/, requirements/maca_private.txt
  【官方文档】 沐曦开发者文档中心 MXC500 版本 3.10.1 的《运行时API编程指南》《mcTriton用户指南》《mcPyTorch用户指南》;
              上游 vLLM 的 docs/design/custom_op.md; vLLM-metax 的 vllm_metax/patch/README.md 和 AUDIT.md
  【第三方】   gitlink.org.cn/JunkaiShi/c500-specpilot 的实验日志: Qwen3.8-27B 的层结构, GDN 的耗时, CUDA Graph 的收益和踩过的坑
  推算        自己的计算, 前提写在注释里

Run:  python3 calc.py  ->  bench/calc_summary.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---------------------------------------------------------------- 1. Qwen3.8-27B 的 GDN 有多大
# 【第三方】 实验日志: 文本主干 64 层, 48 层 GDN (H=16, HV=48, K=128, V=128), 16 层全注意力 (24 Q / 4 KV heads, head_dim=256);
#            "单层 FP32 状态 3.15 MB, 48 层共 151.0 MB"
gdn_layers, full_layers = 48, 16
h_qk, h_v, k_dim, v_dim = 16, 48, 128, 128
attn_q_heads, attn_kv_heads, attn_head_dim = 24, 4, 256
state_elems = h_v * k_dim * v_dim                                  # 786,432
state_layer = state_elems * 4                                      # FP32
state_all = state_layer * gdn_layers
kv_per_token = full_layers * attn_kv_heads * attn_head_dim * 2 * 2  # K 和 V, BF16
step_ms, hbm_tbps = 38.69, 1.843                                   # 上一篇: BF16 权重 51.1 GB, 每个 token 38.69 ms; 显存带宽 1.84 TB/s
out["model"] = {
    "layers": gdn_layers + full_layers, "gdn_layers": gdn_layers, "full_attention_layers": full_layers,
    "gdn_heads_qk": h_qk, "gdn_heads_v": h_v, "k": k_dim, "v": v_dim,
    "state_elems_per_layer": state_elems,
    "state_fp32_layer_bytes": state_layer, "state_fp32_layer_mb": round(state_layer / 1e6, 2),        # 3.15, 和日志对得上
    "state_fp32_all_bytes": state_all, "state_fp32_all_mb": round(state_all / 1e6, 1),                # 151.0, 和日志对得上
    "state_bf16_all_mb": round(state_all / 2 / 1e6, 1),
    "kv_bytes_per_token": kv_per_token,
    "kv_mb_at_2048": round(kv_per_token * 2048 / 1e6, 1), "kv_gb_at_32768": round(kv_per_token * 32768 / 1e9, 2),
    "kv_over_state_at_32768": round(kv_per_token * 32768 / state_all, 1),
    # 解码一步: 每层把状态读一遍 (乘衰减, 加新贡献, 再乘 q), 再写回去
    "decode_state_bytes_per_step": 2 * state_all, "decode_state_ms_at_1p84": round(2 * state_all / (hbm_tbps * 1e12) * 1e3, 3),
    "decode_state_share_of_step": round(2 * state_all / (hbm_tbps * 1e12) * 1e3 / step_ms * 100, 2),
    # FLOPs: 衰减一次乘, 新贡献一次乘加, q 乘状态一次乘加 -> 每个状态元素 4 次
    "decode_flops_per_step": gdn_layers * state_elems * 4, "decode_intensity": round(gdn_layers * state_elems * 4 / (2 * state_all), 1),
}

# ---------------------------------------------------------------- 2. 沐曦的 GDN chunk kernel 怎么摆
# 【官方代码】 constexpr int kK = 128, kV = 128, kBT = 64, kBV = 64, kThreads = 256, kResStride = kBV + 4, kKStride = kK + 4
#            __shared__ bf16 state_bf16[kStateBf16Elems]   (kStateBf16Elems = max(Bv * kKStride, kBT * kKStride))
#            __shared__ bf16 residual_panel[Bv * kResStride]
#            __shared__ bf16 w_panel[kBT * kKStride]        (整块 W 面板的那条路径)
#            __shared__ float decay_panel[kK]
#            注释: "No FP32 state panel: the Bv x 128 state is register-resident."
kK, kV, kBT, kBV, kThreads, res_stride, k_stride = 128, 128, 64, 64, 256, 68, 132
state_elems_block = kBV * kK
smem = {"state_bf16": max(kBV * k_stride, kBT * k_stride) * 2, "residual_panel": kBV * res_stride * 2,
        "w_panel": kBT * k_stride * 2, "decay_panel": kK * 4}
smem_total = sum(smem.values())
out["kernel"] = {
    "chunk_tokens": kBT, "state_rows_per_block": kBV, "state_cols": kK, "threads_per_block": kThreads,
    "waves_per_block": kThreads // 64,                                  # 64: 线程束的宽度, 上一篇
    "state_regs_per_block": state_elems_block, "state_regs_per_thread": state_elems_block // kThreads,
    "update_fragments_per_thread": 2 * (kBV // 32), "update_regs_per_thread": 2 * (kBV // 32) * 4,
    "smem_bytes": smem, "smem_total_bytes": smem_total, "smem_total_kb": round(smem_total / 1024, 1),
    "smem_limit_bytes": 65536, "smem_share": round(smem_total / 65536 * 100, 1),
    "mtreg_limit_in_kernel_comment": 128,                              # GPTQ 那个文件头给 4 条线程束定的上限, 上一篇
    "res_stride_words": res_stride // 2, "res_stride_bank_shift": (res_stride // 2) % 32,
    "k_stride_words": k_stride // 2, "k_stride_bank_shift": (k_stride // 2) % 32,
}

# ---------------------------------------------------------------- 3. 第三方在 C500 上量 GDN
# 【第三方】 Triton-on-MACA, 48 层 GDN, H=16, HV=48, K=V=128; Path Verify 是逐个节点验证, TreeWY 是论文 §4.2 的树状算法
gdn = [  # (N, path_verify_ms, treewy_ms)
    (8, 3.55, 6.70), (16, 7.11, 6.72), (24, 12.52, 6.17), (32, 18.13, 6.19),
]
out["gdn_third_party"] = {
    "rows": [{"n": n, "path_ms": p, "treewy_ms": t, "speedup": round(p / t, 2), "path_us_per_layer": round(p * 1e3 / 48, 1),
              "treewy_us_per_layer": round(t * 1e3 / 48, 1)} for n, p, t in gdn],
    "commit_replay_ms": {4: 0.75, 8: 1.27, 12: 1.82},
    "crossover_n": 16,
}

# ---------------------------------------------------------------- 4. CUDA Graph 的账
# 【第三方】 eager 模式 64 层一步有 25 到 30 ms 的 Python 派发开销; 分桶 CUDA Graph 后 Draft 单步从约 25 ms 降到 7.0 到 9.2 ms;
#            @torch.library.custom_op 的 eager 派发: M=1 一步 46.01 ms, CUDAGraph 38.40 ms; 捕获 18 s 占 0.27 GiB (另一次 5 s, 0.10 GiB)
py_lo, py_hi = 25.0, 30.0
out["graph"] = {
    "layers": 64, "python_dispatch_ms_per_step": [py_lo, py_hi], "python_dispatch_ms_per_layer": [round(py_lo / 64, 2), round(py_hi / 64, 2)],
    "draft_before_ms": 25.0, "draft_after_ms": [7.0, 9.2], "draft_speedup": [round(25.0 / 9.2, 1), round(25.0 / 7.0, 1)],
    "draft_saved_ms": [round(25.0 - 9.2, 1), round(25.0 - 7.0, 1)],
    "verify_eager_custom_op_ms": 46.01, "verify_graph_ms": 38.40, "verify_saved_ms": round(46.01 - 38.40, 2),
    "verify_saved_share": round((46.01 - 38.40) / 46.01 * 100, 1),
    "capture_s": [5, 18], "capture_gib": [0.10, 0.27],
    "buckets_draft": [4, 8, 16], "buckets_verify": [16, 32, 64, 128],
}

# ---------------------------------------------------------------- 5. vLLM-metax v0.28.0 的口子
# 【官方代码】 逐个数出来的: vllm_metax/registry 下 register_oot 出现 25 次; vllm_metax/patch 下 @patch( 的安装点 bugfix 28 处, enhancement 8 处, performance 2 处;
#            其中 triton_support/rejection_sampler.py 的 3 处在 v0.28.0 的 __init__.py 里没有启用; 被补的模块 (vllm 和 transformers) 去重后 37 个
out["hooks"] = {
    "vllm_metax_version": "v0.28.0", "mcoplib_version": "0.4.13", "fla_wheel": "flash-linear-attention==0.5.0+metax3.8.2.2torch2.10",
    "registered_oot_ops": 25, "patches": {"bugfix": 28, "enhancement": 8, "performance": 2}, "patches_total": 38,
    "patches_not_activated": 3, "patches_active": 38 - 3, "patched_modules": 37,
    "gdn_registered": "ChunkGatedDeltaRule -> MacaChunkGatedDeltaRule (forward_oot 默认调 forward_native)",
    "gdn_layer_free": "QwenGatedDeltaNetAttention (PluggableLayer, 名字 qwen_gated_delta_net_attention), v0.28.0 没人登记",
    "decode_op_wired": False,                                           # fused_sigmoid_gating_delta_rule_update 在 vLLM-metax 里搜不到
    "mcoplib_for_vllm": {"0.28.0": "release/v0.4.13", "0.29.0": "release/v0.4.14", "0.30.0": "release/v0.4.15", "0.32.0": "release/v0.4.16"},
    "chunk_kernel_patch": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64 的 num_stages 砍成 [1], 整个文件只标了这一处改动",
    # RMSNorm 的 Triton kernel 要把一行装进一个块 (本专栏融合算子那篇). Qwen3.8-27B 的隐藏维是 5120, 块要取 2 的幂
    "hidden_size": 5120, "block_for_rmsnorm": 8192, "lanes_outside_mask_share": round(1 - 5120 / 8192, 3),      # 0.375
}

json.dump(out, open(os.path.join(HERE, "bench", "calc_summary.json"), "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(f"== {k}\n{json.dumps(v, ensure_ascii=False)}")
