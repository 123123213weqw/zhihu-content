#!/usr/bin/env python3
"""W12 的参考实现:把 bfloat16 权重无损地存成每个 12 bit。

格式取自 SpecPilot 论文(arXiv:2611.02501)第 4.1 节和它公开的编码函数 specpilot_d5_w12.py 里的 _encode_rows。
每个输出通道(权重的一行)选一个 16 个 binade 的指数窗口 [E-15, E]。窗口里的权重存成
符号 1 bit + 指数偏移 4 bit + 尾数 7 bit。窗口外的值、0、非规格数进一张按行的稀疏表。
12 bit 拆成两块放:高 8 bit(符号、指数偏移、尾数的前 3 位)一个平面,低 4 bit(尾数的后 4 位)一个平面。

这里只做正确性,不追求速度。想看怎么在矩阵乘里一边读一边解码,见 w12_bench.py 里的 Triton kernel。
"""
from __future__ import annotations

import torch

NEG15 = 2.0 ** -15  # escape 位置的占位值:符号 0、偏移 0、尾数 0 解出来就是 2^-15


def w12_encode(w: torch.Tensor):
    """w: [N, K] bfloat16,K 是偶数。

    返回 hi [N, K] uint8,lo [N, K/2] uint8(相邻两个权重的低 4 位合在一个字节里),E [N] int32,esc [N, K] bool。
    """
    u = w.view(torch.int16).to(torch.int32) & 0xFFFF       # 把 16 个 bit 当整数看
    e = (u >> 7) & 0xFF                                    # 指数域,带偏置 127;0 表示 0 或非规格数
    row = torch.arange(w.shape[0], device=w.device)[:, None] * 257
    hist = torch.bincount((row + e).flatten(), minlength=w.shape[0] * 257).view(w.shape[0], 257)
    hist[:, 0] = 0                                         # 0 和非规格数不进窗口
    cum = hist.cumsum(1)                                   # cum[:, t]:指数域不超过 t 的权重个数
    tops = torch.arange(15, 256, device=w.device)
    below = cum[:, (tops - 16).clamp(min=0)]
    below[:, tops < 16] = 0
    cover = cum[:, tops] - below                           # 窗口 [E-15, E] 能盖住多少个
    best = cover == cover.max(1, keepdim=True).values
    E = tops[best.int().cumsum(1).argmax(1)].int()         # 盖得最多的那个;并列时取最低的
    c = e - (E[:, None] - 15)                              # 窗口内的偏移,0 到 15
    esc = (c < 0) | (c > 15) | (e == 0)                    # 窗口外,或者是 0 和非规格数
    s, m = u >> 15, u & 0x7F
    hi = torch.where(esc, 0, (s << 7) | (c.clamp(0, 15) << 3) | (m >> 4)).to(torch.uint8)
    m4 = torch.where(esc, 0, m & 0xF)                      # escape 位置两块都清零,解出来是占位值 2^-15
    lo = (m4[:, 0::2] | (m4[:, 1::2] << 4)).to(torch.uint8)
    return hi, lo, E, esc


def w12_decode(hi: torch.Tensor, lo: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """还原成 bfloat16 权重 [N, K]。escape 位置解出来是占位值 2^-15 * 2^(E-127),要再用稀疏表盖回去。"""
    low = torch.stack([lo & 0xF, lo >> 4], dim=2).view(hi.shape).to(torch.int32)
    h = hi.to(torch.int32)
    wt = ((h << 8) & 0x8000) | ((h & 0x7F) << 4) | 0x3800 | low   # w~ = (-1)^s * 2^(c-15) * (1 + m/128)
    bits = wt + ((E[:, None] - 127) << 7)                         # 指数域加回 E - 127,就是原权重
    return bits.to(torch.int16).view(torch.bfloat16)


def w12_linear(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor, E: torch.Tensor,
               esc_col: torch.Tensor, esc_val: torch.Tensor) -> torch.Tensor:
    """y = x @ W^T。x: [M, K] bfloat16。

    esc_col: [N, R_max] int64,每行 escape 位置的列号,不足的补 -1;esc_val: 同形状,escape 处 w / 2^(E-127) 的精确值。
    """
    low = torch.stack([lo & 0xF, lo >> 4], dim=2).view(hi.shape).to(torch.int32)
    h = hi.to(torch.int32)
    wt = (((h << 8) & 0x8000) | ((h & 0x7F) << 4) | 0x3800 | low).to(torch.int16).view(torch.bfloat16)
    y = x.float() @ wt.float().T                                     # 先在约分后的那一边累加
    valid = esc_col >= 0
    xs = torch.where(valid, x.float()[:, esc_col.clamp(min=0)], torch.zeros(()))   # [M, N, R_max]
    y = y + (xs * torch.where(valid, esc_val - NEG15, torch.zeros(()))).sum(-1)     # 每个 escape:把占位值换成精确值
    return y * torch.exp2((E - 127).float())                           # 行内公因子是 2 的整数次幂,乘回去不丢位


def self_test(seed: int = 0) -> dict:
    """随机权重上走一遍:解码逐 bit 相等,矩阵乘和 float64 的参考对得上。"""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(512, 1024, generator=g) * torch.exp2(torch.randint(-6, 3, (512, 1), generator=g).float())).to(torch.bfloat16)
    w[0, :5] = 0.0
    w[1, 7] = 300.0
    hi, lo, E, esc = w12_encode(w)
    back = w12_decode(hi, lo, E)
    ok_mask = ~esc
    bit_exact = bool((back.view(torch.int16)[ok_mask] == w.view(torch.int16)[ok_mask]).all())
    rows, cols = esc.nonzero(as_tuple=True)
    rmax = int(torch.bincount(rows, minlength=w.shape[0]).max()) if rows.numel() else 0
    ec = torch.full((w.shape[0], max(rmax, 1)), -1, dtype=torch.int64)
    ev = torch.zeros_like(ec, dtype=torch.float32)
    if rows.numel():
        slot = torch.arange(rows.numel()) - torch.cumsum(torch.bincount(rows, minlength=w.shape[0]), 0)[rows] + torch.bincount(rows, minlength=w.shape[0])[rows]
        ec[rows, slot] = cols
        ev[rows, slot] = (w[rows, cols].double() * torch.exp2((127 - E[rows]).double())).float()
    x = torch.randn(4, 1024, generator=g).to(torch.bfloat16)
    y = w12_linear(x, hi, lo, E, ec, ev)
    ref = x.double() @ w.double().T
    base = x.float() @ w.float().T
    rel = (lambda a: float(((a.double() - ref).abs() / ref.abs().clamp(min=1e-6)).max()))
    return {"escapes": int(esc.sum()), "bit_exact": bit_exact,
            "max_rel_err_w12": rel(y), "max_rel_err_bf16_gemm": rel(base)}


if __name__ == "__main__":
    print(self_test())
