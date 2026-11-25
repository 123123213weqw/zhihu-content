#!/usr/bin/env python3
"""导出 Inductor 为"加残差 + RMSNorm"生成的 Triton 代码(给文章当插图用)。
Run:  TORCH_LOGS=output_code python dump_inductor.py 2> inductor_output_code.log
"""
import torch


def fused_add_rms_norm_native(x, x_residual, weight, epsilon):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


if __name__ == "__main__":
    fn = torch.compile(fused_add_rms_norm_native, dynamic=False, backend="inductor")
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = fn(x, r, w, 1e-5)
    torch.cuda.synchronize()
    print("ok", out[0].shape, out[1].shape)
