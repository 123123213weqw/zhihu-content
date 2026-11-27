#!/usr/bin/env python3
"""文章里的数都在这里算一遍。

出处分四类, 写在每一段的注释里:

  【官方文档】 沐曦开发者文档中心 (developer.metax-tech.com/doc) 里 MXC500 的文档, 版本 3.10.1:
               《MXMACA C++编程指南》《运行时API编程指南》《MXMACA运行时API参考》《mxcc编译器用户指南》
               《MXMACA编译器内建函数编程指南》《mcPyTorch用户指南》《mcTriton用户指南》《快速上手指南》《mx-smi使用手册》
  【官方代码】 GitHub 组织 MetaX-MACA 下的 maca-samples 和 mcoplib (提交 39511c5)
  【N260】     曦思 N260 上 macainfo 的输出, 取自掘金专栏「沐曦异构并行计算软件栈」第 2 章 (2025-02-12)。
               芯片名 XCORE1020, 按《运行时API编程指南》计算能力 10.20 就是 N260, 和 C500 (10.00) 同一个大版本、同一条指令集
  【实测】     gitlink.org.cn/JunkaiShi/c500-specpilot 的实验日志 (整卡), ASKCOS 用 xpu-perf 做的算子级测试 (CSDN, 2026-10-14, 切片)
  【对比】     NVIDIA 的 H100 SM 取自本专栏《GPU 的心脏:SM 内部解剖》那篇; Ampere 的 mma.m16n8k16 取自 NVIDIA 的 PTX 文档
  推算        自己的计算, 前提写在注释里

Run:  python3 calc.py  ->  bench/calc_summary.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---------------------------------------------------------------- 1. 一个 AP 里能放多少线程
# 【N260】 macainfo: Accelerator Processors 56, PEUs per AP 4, Wavefront Size 64, Max Waves Per AP 32,
#         Max Work-items Per AP 2048, Workgroup Max Size 1024, Max fbarriers/Workgroups 32, GROUP 段 64 KB,
#         L1 32 KB, L2 8192 KB, Cacheline 128 B, Max Clock Freq 1600 MHz, Fast Float16 Operation FALSE
# 【官方文档】 《运行时API编程指南》2.2: 每 64 个线程为一组, 称为线程束 (wave); 硬件把线程分到线程束里, 在加速处理器 (AP) 上调度
n260_ap, peu_per_ap, waves_per_ap, lanes = 56, 4, 32, 64
workgroup_max = 1024
out["ap"] = {
    "n260_ap_count": n260_ap, "peu_per_ap": peu_per_ap, "waves_per_ap": waves_per_ap, "lanes_per_wave": lanes,
    "waves_per_peu": waves_per_ap // peu_per_ap,                 # 8
    "threads_per_peu": waves_per_ap // peu_per_ap * lanes,       # 512
    "threads_per_ap": waves_per_ap * lanes,                      # 2048, 和 macainfo 的 Max Work-items Per AP 对得上
    "workgroup_max": workgroup_max, "waves_in_max_workgroup": workgroup_max // lanes,      # 16
    "max_workgroup_spans_peus": (workgroup_max // lanes) / (waves_per_ap // peu_per_ap),   # 2 (推算: 16 条线程束要占两个 PEU)
    "named_barriers_per_workgroup": 32,
    "group_mem_kb": 64, "group_mem_bytes": 64 * 1024,
    "l1_kb_n260": 32, "l2_kb_n260": 8192, "cacheline_bytes": 128, "max_clock_mhz_n260": 1600,
    "lanes_per_peu_if_wave_is_64_wide": lanes // peu_per_ap,    # 16 (推算: 一条线程束的 64 个 lane 平铺在 4 个 PEU 上)
    "c500_ap_count_third_party": 104,                           # 【实测】 真卡上读 multiProcessorCount
}

# ---------------------------------------------------------------- 2. 共享内存的 bank
# 【官方文档】 《MXMACA运行时API参考》mcDeviceGetSharedMemConfig: 当前设备唯一支持的配置是每个 bank 4 个字节
# 【实测】   LDS (工作组共享内存) 32 个 bank; 每个工作组不超过 64 KB (mcoplib 用 sharedMemPerBlock > 65536 判断 C600)
banks, bank_bytes = 32, 4
out["smem"] = {
    "banks": banks, "bank_bytes": bank_bytes, "row_bytes": banks * bank_bytes,          # 128
    "rows_in_64kb": 64 * 1024 // (banks * bank_bytes),                                  # 512
    "case_stride_bytes": 144, "case_stride_words": 144 // bank_bytes,                   # 36 个字, 落在 bank 4, 8, 12 ... 上
    "case_ms_before": 63.02, "case_ms_after": 49.55, "case_saved_ms": round(63.02 - 49.55, 2),   # 【实测】 64 位加载拆成两条 32 位
    "kernel_comment_block_threads": 256, "kernel_comment_block_waves": 256 // lanes,    # 4  【官方代码】 mcoplib hgemm_gptq.h
    "kernel_comment_block_smem_kb": 16, "kernel_comment_mtreg": 128,
}

# ---------------------------------------------------------------- 3. 线程束内的结构
# 【官方文档】 《内建函数指南》__builtin_mxc_mov_shfl: 64 个线程里每 16 个连续线程叫一个 row, 共 4 个 row; bmsk 每 4 个 lane 一组
# 《C++编程指南》2.7/2.19/2.22: __syncwarp, __activemask, __shfl_sync 的掩码是 unsigned long (64 位), 默认 0xffffffffffffffff
out["wave"] = {
    "rows": 4, "lanes_per_row": 16, "quads": 16, "lanes_per_quad": 4,
    "mask_bits_mxac": 64, "mask_bits_cuda": 32, "lanes_covered_by_32bit_mask": 32,
    "share_of_wave_covered_by_32bit_mask": round(32 / 64, 2),      # 0.5
    "nvidia_warp_lanes": 32,
}

# ---------------------------------------------------------------- 4. 矩阵指令: 每个线程拿多少, 一条指令算多少
# 【官方文档】 《内建函数指南》3.6 的接口声明 (按 64 个线程的线程束算):
#   v4f32 mma_16x16x16f16  (v4f16 a, v4f16 b, v4f32 c)      v4f32 mma_16x16x16i8   (int a, int b, v4i32 c)      xcore1500 以下
#   v4f32 mma_16x16x16bf16 (v4f16 a, v4f16 b, v4f32 c)      v4i32 mma_16x16x32i8   (v2i32 a, v2i32 b, v4i32 c)  xcore1500 起
#   v4f32 mma_16x16x8tf32  (v2f32 a, v2f32 b, v4f32 c)      v4f32 mma_16x16x4f32   (float a, float b, v4f32 c)
#   v4f64 mma_16x16x4f64   (double a, double b, v4f64 c)
mma = {  # 名字: (k, 每线程 A 的 32 位寄存器数, 每线程 B 的, 每线程累加器的, 架构限制)
    "16x16x16f16": (16, 2, 2, 4, ""), "16x16x16bf16": (16, 2, 2, 4, ""),
    "16x16x16i8": (16, 1, 1, 4, "xcore1500 以下"), "16x16x32i8": (32, 2, 2, 4, "xcore1500 起"),
    "16x16x8tf32": (8, 2, 2, 4, ""), "16x16x4f32": (4, 1, 1, 4, ""), "16x16x4f64": (4, 2, 2, 8, ""),
}
out["mma"] = {
    name: {"k": k, "regs_a": ra, "regs_b": rb, "regs_acc": rc, "macs": 16 * 16 * k, "macs_per_thread": 16 * 16 * k // lanes, "arch": arch}
    for name, (k, ra, rb, rc, arch) in mma.items()
}
out["mma_compare"] = {
    "nvidia_m16n8k16_macs": 16 * 8 * 16, "nvidia_m16n8k16_lanes": 32, "nvidia_m16n8k16_macs_per_thread": 16 * 8 * 16 // 32,
    "mxmaca_m16n16k16_macs_per_thread": 16 * 16 * 16 // lanes,
    "int8_over_bf16_peak": round(560 / 280, 1), "tf32_over_bf16_peak": round(140 / 280, 2),     # 上一篇里的平台峰值
    "int8_mma_macs_ratio": (16 * 16 * 16) / (16 * 16 * 16), "tf32_mma_macs_ratio": (16 * 16 * 8) / (16 * 16 * 16),
}

# ---------------------------------------------------------------- 5. 寄存器预算: 一条线程束算多大的块, 每个线程要几个寄存器
# 只数矩阵乘主循环里的三样东西 (BF16): 累加器, A 的片段, B 的片段。地址, 循环变量, 流水线的缓冲都不算。
# 一条 64 线程的线程束负责 rows x cols 的输出块: 累加器要 (rows/16)*(cols/16) 组, 每组 4 个;
# A 要 rows/16 组, 每组 2 个; B 要 cols/16 组, 每组 2 个。
def budget(rows, cols):
    t_acc, t_a, t_b = (rows // 16) * (cols // 16), rows // 16, cols // 16
    return {"rows": rows, "cols": cols, "tiles": t_acc, "acc": 4 * t_acc, "a": 2 * t_a, "b": 2 * t_b,
            "total": 4 * t_acc + 2 * t_a + 2 * t_b}
out["budget"] = {f"{r}x{c}": budget(r, c) for r, c in ((32, 32), (64, 64), (128, 64), (64, 128), (128, 128))}
out["budget"]["mtreg_limit_in_kernel_comment"] = 128      # 【官方代码】 mcoplib hgemm_gptq.h: MTRegisters should be less than 128
out["budget"]["fits_under_128"] = {k: v["total"] < 128 for k, v in out["budget"].items() if isinstance(v, dict)}

# ---------------------------------------------------------------- 6. 占用率: 寄存器堆怎么分
# 寄存器堆的总大小沐曦没有公布。能查的接口: mcDevAttrMaxRegistersPerMultiprocessor (所有驻留的线程块一起分), mcDevAttrMaxRegistersPerBlock,
# MC_FUNC_ATTRIBUTE_NUM_REGS (每个线程用了几个), mxcc -resource-usage。
# 这里按《SM 解剖》那篇里 H100 的 64K 个 32 位寄存器做对照: 同样是 2048 个线程, 平均每个线程 32 个。
regs_h100 = 65536
out["registers"] = {
    "h100_regs_per_sm": regs_h100, "h100_bytes": regs_h100 * 4, "h100_threads_per_sm": 2048, "h100_regs_per_thread_at_full": regs_h100 // 2048,
    "h100_max_regs_per_thread": 255,
    "mxmaca_regs_per_ap": None,                     # 未公布
    "mxmaca_two_classes": {"mtreg": "-maxrregcount 限它的数量", "streg": "-maxsregcount 限它的数量; 内建函数指南写成 single thread register"},
    # 反过来问: 每个 AP 2048 个线程, 如果寄存器堆也是 64K, 一个用了 128 个寄存器的 kernel 最多驻留多少线程束 (仅作示例)
    "example_if_64k_waves_resident_at_128_regs": regs_h100 // (128 * lanes),
}

# ---------------------------------------------------------------- 7. 放在一起: AP 对 SM
# 【对比】 H100 的 SM: 4 个分区 (warp 调度器), 128 个 CUDA Core, 64 条线程束, 每条 32 个线程, 共 2048 个线程, 64K 个 32 位寄存器
out["compare"] = {
    "nvidia_h100_sm": {"sub_units": 4, "lanes_per_wave": 32, "waves": 64, "threads": 2048, "regs": regs_h100, "threads_per_block_max": 1024},
    "metax_ap_n260": {"sub_units": peu_per_ap, "lanes_per_wave": lanes, "waves": waves_per_ap, "threads": waves_per_ap * lanes,
                      "threads_per_block_max": workgroup_max, "regs": None},
    "same_threads_per_unit": waves_per_ap * lanes == 2048,
}

# ---------------------------------------------------------------- 8. 计算能力的两套编号
# 【官方文档】 《运行时API编程指南》3.1.1.1: 10.xx 是 MXC500 系列, 15.xx 是 MXC600 系列; 10.00 是 C500, 10.02 是 C550, 10.20 是 N260
# 《mcPyTorch用户指南》3.10.1: torch.cuda.get_device_capability 返回 (8, 0), get_arch_list 返回 ['sm_80'], 模拟 CUDA 能力 8.0
out["capability"] = {"mxmaca": {"C500": "10.00", "C550": "10.02", "N260": "10.20", "C600 系列": "15.xx"},
                     "offload_arch": {"通用": "xcore1000", "C500 专用": "xcore1000", "C550 专用": "xcore1002"},
                     "pytorch_reports": "8.0 (sm_80)"}

json.dump(out, open(os.path.join(HERE, "bench", "calc_summary.json"), "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(f"== {k}\n{json.dumps(v, ensure_ascii=False)}")
