#!/usr/bin/env python3
"""Generate figures for the optimizer-evolution article.

Figures:
  fig1_canyon.png        — SGD zig-zag vs Momentum in an ill-conditioned valley
  fig2_adagrad_decay.png — AdaGrad effective lr decays to 0, RMSProp stays flat
  fig3_bias_correction.png — Adam bias correction: m vs m-hat at start of training
  fig4_trajectory.png    — 5 optimizers on the Beale surface

Run:  python gen_figures.py   (outputs to ./images/)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# ---- pick a CJK font if available ----
_all = {f.name for f in font_manager.fontManager.ttflist}
cjk = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                        "WenQuanYi Micro Hei"] if n in _all), None)
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("CJK font:", cjk)

COLORS = {"SGD": "#d62728", "Momentum": "#1f77b4", "AdaGrad": "#2ca02c",
          "RMSProp": "#9467bd", "Adam": "#ff7f0e"}


# ---------------------------------------------------------------- fig 1
def fig1_canyon():
    """Ill-conditioned quadratic: f = 0.5*(x^2 + 9y^2). SGD oscillates in x."""
    f = lambda p: 0.5 * (p[0] ** 2 + 9 * p[1] ** 2)
    grad = lambda p: np.array([p[0], 9 * p[1]])

    def run_sgd(p0, lr, n):
        p = np.array(p0, float)
        traj = [p.copy()]
        for _ in range(n):
            p = p - lr * grad(p)
            traj.append(p.copy())
        return np.array(traj)

    def run_mom(p0, lr, beta, n):
        p = np.array(p0, float)
        v = np.zeros(2)
        traj = [p.copy()]
        for _ in range(n):
            v = beta * v + grad(p)
            p = p - lr * v
            traj.append(p.copy())
        return np.array(traj)

    sgd = run_sgd((1.5, 0.4), 0.21, 28)      # just under 2/9 -> heavy zig-zag
    mom = run_mom((1.5, 0.4), 0.045, 0.9, 28)

    xs = np.linspace(-1.0, 1.8, 400)
    ys = np.linspace(-0.55, 0.55, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (X ** 2 + 9 * Y ** 2)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.contour(X, Y, Z, levels=np.logspace(-3, 0.3, 18), colors="0.82", linewidths=0.8)
    ax.plot(sgd[:, 0], sgd[:, 1], "-o", color=COLORS["SGD"], ms=3.5,
            lw=1.4, label="SGD(同一个学习率)")
    ax.plot(mom[:, 0], mom[:, 1], "-o", color=COLORS["Momentum"], ms=3.5,
            lw=1.4, label="Momentum(带惯性)")
    ax.plot(*sgd[0], "k*", ms=14, label="起点")
    ax.plot(0, 0, "k*", ms=14, mfc="white")
    ax.annotate("最优点", (0, 0), textcoords="offset points", xytext=(14, 12),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))
    ax.set_title("坑 1:狭长峡谷里,SGD 在陡峭方向来回震荡,Momentum 靠惯性平滑穿越")
    ax.set_xlabel("$x$(平坦方向)"); ax.set_ylabel("$y$(陡峭方向)")
    ax.legend(loc="upper left"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_canyon.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2_adagrad_decay():
    """Constant gradient g=1: AdaGrad effective lr ~ lr/sqrt(t) -> 0."""
    t = np.arange(1, 401)
    lr = 1.0
    adagrad = lr / np.sqrt(t)                       # G_t = t
    rho = 0.9
    e = np.empty_like(t, float)                     # E[g^2] warm-up then flat
    acc = 0.0
    for i, ti in enumerate(t):
        acc = rho * acc + (1 - rho) * 1.0
        e[i] = acc
    rmsprop = lr / np.sqrt(e)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, adagrad, color=COLORS["AdaGrad"], lw=2.2,
            label="AdaGrad:步长 = η/√G,G 只增不减 → 衰减到 0")
    ax.plot(t, rmsprop, color=COLORS["RMSProp"], lw=2.2,
            label="RMSProp:步长 = η/√E[g²],指数遗忘 → 稳定")
    ax.set_xlabel("训练步数 t(梯度恒为 1 的情形)")
    ax.set_ylabel("有效步长")
    ax.set_title("AdaGrad 的早衰 vs RMSProp 的终身稳定")
    ax.annotate("步长→0,训练提前熄火", (t[-1] * 0.78, adagrad[-1] + 0.035),
                color=COLORS["AdaGrad"], fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["AdaGrad"], alpha=0.85))
    ax.annotate("稳定在 η", (t[-1] * 0.8, 1.06),
                color=COLORS["RMSProp"], fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["RMSProp"], alpha=0.85))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_adagrad_decay.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig3_bias_correction():
    """Gradient with mean 1: m climbs from 0 slowly, m-hat is on target instantly."""
    rng = np.random.default_rng(0)
    n, beta = 60, 0.9
    g = 1.0 + 0.4 * rng.standard_normal(n)
    m = np.empty(n); acc = 0.0
    for i in range(n):
        acc = beta * acc + (1 - beta) * g[i]
        m[i] = acc
    t = np.arange(1, n + 1)
    mhat = m / (1 - beta ** t)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, g, ".", color="0.78", ms=4, alpha=0.6, label="梯度 g(均值=1,带噪声)")
    ax.plot(t, m, "-o", color=COLORS["Adam"], ms=3, lw=1.8,
            label="m(未修正:从 0 慢慢爬,开局梦游)")
    ax.plot(t, mhat, "-", color=COLORS["SGD"], lw=2.2,
            label=r"$\hat{m}=m/(1-\beta^t)$(修正:开局即贴住真实值)")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.annotate("真实梯度均值", (n * 0.82, 1.03))
    ax.set_xlabel("训练步数 t"); ax.set_ylabel("一阶矩估计")
    ax.set_title(r"Adam 的偏差修正:为什么训练开头必须除以 $(1-\beta^t)$")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_bias_correction.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig 4
BEALE = lambda x, y: (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

def beale_grad(p):
    x, y = p
    # analytic gradient of the Beale function
    a, b, c = 1.5, 2.25, 2.625
    ex1 = a - x + x * y
    ex2 = b - x + x * y ** 2
    ex3 = c - x + x * y ** 3
    dx = 2 * ex1 * (y - 1) + 2 * ex2 * (y ** 2 - 1) + 2 * ex3 * (y ** 3 - 1)
    dy = 2 * ex1 * x + 2 * ex2 * (2 * x * y) + 2 * ex3 * (3 * x * y ** 2)
    return np.array([dx, dy])


def fig4_trajectory():
    p0 = np.array([-0.6, -0.9])
    n = 220

    def run(name):
        p = p0.copy()
        traj = [p.copy()]
        if name == "SGD":
            lr = 0.012
            for _ in range(n):
                p = p - lr * beale_grad(p); traj.append(p.copy())
        elif name == "Momentum":
            lr, beta, v = 0.0014, 0.92, np.zeros(2)
            for _ in range(n):
                v = beta * v + beale_grad(p)
                p = p - lr * v; traj.append(p.copy())
        elif name == "AdaGrad":
            lr, G = 0.6, np.zeros(2)
            for _ in range(n):
                g = beale_grad(p)
                G += g ** 2
                p = p - lr * g / (np.sqrt(G) + 1e-8); traj.append(p.copy())
        elif name == "RMSProp":
            lr, rho, E = 0.035, 0.9, np.zeros(2)
            for _ in range(n):
                g = beale_grad(p)
                E = rho * E + (1 - rho) * g ** 2
                p = p - lr * g / (np.sqrt(E) + 1e-8); traj.append(p.copy())
        elif name == "Adam":
            lr, b1, b2, m, v = 0.14, 0.9, 0.999, np.zeros(2), np.zeros(2)
            for t in range(1, n + 1):
                g = beale_grad(p)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
                p = p - lr * mh / (np.sqrt(vh) + 1e-8); traj.append(p.copy())
        return np.array(traj)

    names = ["SGD", "Momentum", "AdaGrad", "RMSProp", "Adam"]
    trajs = {k: run(k) for k in names}

    xs = np.linspace(-1.6, 3.6, 500)
    ys = np.linspace(-1.2, 1.6, 500)
    X, Y = np.meshgrid(xs, ys)
    Z = np.log1p(BEALE(X, Y))
    levels = np.linspace(np.percentile(Z, 2), np.percentile(Z, 97), 24)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ax, name in zip(axes.flat, names):
        tr = trajs[name]
        ax.contour(X, Y, Z, levels=levels, cmap="Greys", linewidths=0.6, alpha=0.75)
        ax.plot(tr[:, 0], tr[:, 1], "-o", color=COLORS[name], ms=2.6, lw=1.4)
        ax.plot(*tr[0], "*", color="k", ms=11)
        ax.plot(3, 0.5, "*", color="crimson", ms=13)
        dist = np.hypot(*(tr[-1] - np.array([3.0, 0.5])))
        ax.set_title(f"{name}  (终点距最优 {dist:.2f})", fontsize=12)
        ax.set_aspect("equal")

    ax = axes.flat[5]
    ax.contour(X, Y, Z, levels=levels, cmap="Greys", linewidths=0.6, alpha=0.6)
    for name in names:
        tr = trajs[name]
        ax.plot(tr[:, 0], tr[:, 1], color=COLORS[name], lw=1.7, label=name)
    ax.plot(*p0, "k*", ms=11)
    ax.plot(3, 0.5, "*", color="crimson", ms=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("全部叠加", fontsize=12)
    ax.set_aspect("equal")

    fig.suptitle("五个优化器在 Beale 曲面上的轨迹(★起点 / ★最优点 (3, 0.5))", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/fig4_trajectory.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig1_canyon(); print("fig1 done")
    fig2_adagrad_decay(); print("fig2 done")
    fig3_bias_correction(); print("fig3 done")
    fig4_trajectory(); print("fig4 done")
    print("all figures ->", OUT)
