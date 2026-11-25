#!/usr/bin/env python3
"""按 vLLM 0.34 的接口,从零登记一个融合算子,并确认它在三种情形下都对:
  1. 直接调用,走登记的 Triton 实现
  2. 放进 torch.compile,Inductor 把它当自定义算子(opaque)
  3. 放进 torch.compile,优先级设成 native,Inductor 自己融合
Run:  CUDA_VISIBLE_DEVICES=0 python my_op_recipe.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from vllm import ir
from vllm.config import get_current_vllm_config


# ---- 1. 定义算子:函数体就是 PyTorch 写的参考实现
@ir.register_op
def my_rms_norm(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


# ---- 2. 给一组示例输入,vLLM 用它自动核对各种实现
@my_rms_norm.register_input_generator
def _my_rms_norm_inputs(num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-5,
                        device=None):
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    return x, weight, epsilon


# ---- 3. 写 kernel
@triton.jit
def _my_rms_norm_kernel(x_ptr, w_ptr, y_ptr, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    ptr = row * D + cols
    x = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + ptr, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


# ---- 4. 把 kernel 包成一个 torch 自定义算子。
# 直接发射 Triton kernel 的函数没法被追踪:编译期"下降"那一步要用假张量把它追进图里,会报错。
# 包成 triton_op 之后能追踪了,Inductor 也认得它是一个 Triton kernel。
@triton_op("my::my_rms_norm_triton", mutates_args=())
def _my_rms_norm_launch(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    y = torch.empty_like(x)
    n, d = x.shape
    wrap_triton(_my_rms_norm_kernel)[(n,)](x, weight, y, epsilon, D=d, BLOCK=triton.next_power_of_2(d), num_warps=8)
    return y


# ---- 5. 把实现登记到算子上
@my_rms_norm.register_impl("my_triton", supported=torch.cuda.is_available())
def _my_rms_norm_triton(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    return _my_rms_norm_launch(x, weight, epsilon)


def short(names: list[str]) -> list[str]:
    return [n[:60] for n in names]


def kernels_of(fn) -> list[str]:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == DeviceType.CUDA and getattr(e, "self_device_time_total", 0) > 0]


def via_vllm_pipeline(x, w, ref) -> None:
    """把算子放进 vLLM 自己的编译流程(PostGradPassManager)里,两种优先级各试一次。
    配置、编译、调用都要在同一个"当前配置 + 编译区间"的上下文里。"""
    from vllm.compilation.passes.inductor_pass import pass_context
    from vllm.compilation.passes.pass_manager import PostGradPassManager
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.utils import Range

    with set_current_vllm_config(VllmConfig()), pass_context(Range(1, 8192)):
        pm = PostGradPassManager()
        pm.configure(get_current_vllm_config())
        for prio in (["native"], ["my_triton", "native"]):
            my_rms_norm.set_default(prio)
            torch._dynamo.reset()   # 下降发生在编译时,实现被写进了编译产物:换优先级必须重新编译
            fn = torch.compile(lambda x, w: my_rms_norm(x, w, 1e-5), dynamic=False, backend="inductor",
                               options={"post_grad_custom_post_pass": pm})
            report(f"5. vLLM 的编译流程,优先级 {prio[0]}", fn(x, w), ref, lambda: fn(x, w))
            torch._dynamo.reset()
            around = torch.compile(lambda x, w: my_rms_norm(x, w, 1e-5) * 2.0 + 1.0, dynamic=False, backend="inductor",
                                   options={"post_grad_custom_post_pass": pm})
            report(f"6. vLLM 的编译流程,前后各加一个算子,优先级 {prio[0]}", around(x, w), (ref.double() * 2 + 1),
                   lambda: around(x, w))


if __name__ == "__main__":
    dev = torch.device("cuda")
    x = torch.randn(64, 4096, device=dev, dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(4096, device=dev, dtype=torch.float32)).to(torch.bfloat16)
    ref = my_rms_norm.impls["native"].impl_fn(x, w, 1e-5)
    print("torch 算子的名字:", my_rms_norm.torch_op)
    print("登记了的实现:", sorted(my_rms_norm.impls))

    def report(tag: str, out: torch.Tensor, expect: torch.Tensor, fn) -> None:
        names = kernels_of(fn)
        err = float((out.double() - expect.double()).abs().max())
        print(f"{tag}:最大误差 {err},{len(names)} 个 kernel:{[n[:56] for n in names]}")

    # 1. 直接调用
    my_rms_norm.set_default(["my_triton", "native"])
    print("1. 优先级", my_rms_norm.get_priority(), "-> 选中的是", my_rms_norm.dispatch(x, w, 1e-5).provider)
    report("   直接调用", my_rms_norm(x, w, 1e-5), ref, lambda: my_rms_norm(x, w, 1e-5))

    # 2. 普通的 torch.compile,优先级里 Triton 实现在前:算子是黑盒,Inductor 不碰
    compiled = torch.compile(lambda x, w: my_rms_norm(x, w, 1e-5), dynamic=False, backend="inductor")
    report("2. 普通 torch.compile,my_triton 在前", compiled(x, w), ref, lambda: compiled(x, w))

    # 3. 普通的 torch.compile,优先级设成 native:黑盒里跑的还是那 8 个 PyTorch kernel
    my_rms_norm.set_default(["native"])
    compiled2 = torch.compile(lambda x, w: my_rms_norm(x, w, 1e-5), dynamic=False, backend="inductor")
    report("3. 普通 torch.compile,native 在前", compiled2(x, w), ref, lambda: compiled2(x, w))

    # 4. 前后各加一个算子:黑盒外面的能融,黑盒里面的融不进去
    for prio in (["native"], ["my_triton", "native"]):
        my_rms_norm.set_default(prio)
        fn = torch.compile(lambda x, w: my_rms_norm(x, w, 1e-5) * 2.0 + 1.0, dynamic=False, backend="inductor")
        report(f"4. 前后各加一个算子,优先级 {prio[0]}", fn(x, w), (ref.double() * 2 + 1), lambda: fn(x, w))

    via_vllm_pipeline(x, w, ref)
