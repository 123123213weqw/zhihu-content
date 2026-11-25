#!/usr/bin/env python3
"""SpecPilot 里几个算法的最小实现,纯 Python,不用任何框架。

  greedy_accept_chain   朴素做法:草稿是一条链,逐个比,第一个不一样就停
  best_first_tree       在草稿模型的候选格上,按路径得分从高到低挑节点,建一棵树
  greedy_accept_tree    一趟验证之后,在树上找最长的那条被接受的路径
  accept_patch          控制器收不收一个改动:成对比较,看增益的 95% 置信下界

前三个对应论文 4.2 节。公开代码里建树和接受各是一个 Triton kernel (specpilot_tree.py 的
_tree_build_kernel 和 _tree_accept_kernel,一个请求一个 program),规则和这里一样。
Run:  python3 specpilot_toys.py
"""
from __future__ import annotations

import heapq
import itertools
import math
import random


def greedy_accept_chain(draft: list, target_pred: list) -> tuple[int, object]:
    """draft: 草稿 token,一条链。target_pred[i]:目标模型看完草稿的前 i 个 token 之后,自己想输出的下一个 token。

    返回 (接受个数, 白拿的那个 token)。第一个对不上的位置,目标模型的预测本身就是正确的下一个 token,所以每步都多拿一个。
    """
    n = 0
    while n < len(draft) and draft[n] == target_pred[n]:
        n += 1
    return n, target_pred[n]


def best_first_tree(root: list, edges: list, n_nodes: int) -> list:
    """root[j]:位置 0 选第 j 个候选的对数概率。edges[s][i][j]:位置 s-1 选了第 i 个候选时,位置 s 选第 j 个的条件对数概率。

    返回最多 n_nodes 条路径,按路径得分 (对数概率之和) 从高到低。一条路径的第 d 个元素是位置 d 选的候选下标。
    一个节点被取出来之后它的孩子才进堆,所以取出来的这批路径,凡是有一条的前缀也在里面,那条前缀一定也在。
    """
    heap, serial = [], itertools.count()
    for j, lp in enumerate(root):
        heapq.heappush(heap, (-lp, next(serial), (j,)))       # 堆顶是得分最高的:对数概率取相反号
    tree = []
    while heap and len(tree) < n_nodes:
        score, _, path = heapq.heappop(heap)
        tree.append(path)
        s = len(path) - 1                                       # path[-1] 在位置 s,它的孩子在位置 s+1
        if s < len(edges):
            for j, lp in enumerate(edges[s][path[-1]]):
                heapq.heappush(heap, (score - lp, next(serial), path + (j,)))
    return tree


def greedy_accept_tree(tokens: list, parents: list, target_pred: list):
    """tokens[i]:节点 i 的 token。parents[i]:节点 i 的父节点,根是 -1。target_pred[i]:目标模型看完"根到节点 i"之后的下一个 token。

    节点 i 被接受要同时满足两条:它的 token 等于目标模型在它父节点处给出的预测;它的父节点也被接受。
    根是上一步已经提交的 token。返回 (被接受路径上的 token, 白拿的 token)。父节点的下标总比自己小,从小到大扫一遍就够。
    """
    accepted, depth, leaf = [True], [0], 0
    for i in range(1, len(tokens)):
        p = parents[i]
        accepted.append(accepted[p] and tokens[i] == target_pred[p])
        depth.append(depth[p] + 1)
        if accepted[i] and depth[i] > depth[leaf]:
            leaf = i                                            # 被接受的节点里最深的那一个
    path, best = [], leaf
    while leaf > 0:                                             # 从它走回根
        path.append(leaf)
        leaf = parents[leaf]
    return [tokens[i] for i in reversed(path)], target_pred[best]


def accept_patch(before: list, after: list, noise: float, min_gain: float = 0.01, boot: int = 2000, seed: int = 0):
    """before[i], after[i]:同一道题在改动前、改动后的输出速度 (token/s),成对出现。

    增益是每道题速度比值的几何平均。置信区间用 bootstrap:有放回地重抽 boot 次,取 2.5% 分位作下界。
    下界超过 1 + min_gain + 2 * noise 才收。noise 是"什么都没改,同一套系统测两次"时增益的波动。
    """
    logs = [math.log(b / a) for a, b in zip(before, after)]
    rng = random.Random(seed)
    resampled = sorted(sum(rng.choice(logs) for _ in logs) / len(logs) for _ in range(boot))
    gain, lower = math.exp(sum(logs) / len(logs)), math.exp(resampled[int(0.025 * boot)])
    return lower > 1 + min_gain + 2 * noise, gain, lower


# ------------------------------------------------------------------ 一个手算得过来的例子
def demo_lattice():
    """一个手算得过来的候选格:三个位置,每个位置两个候选。条件概率只看上一个位置选的是哪个候选。

    cand[d][j] 是位置 d 第 j 个候选的 token;root[j] 是位置 0 的概率;cond[s][i][j] 是"位置 s 选了第 i 个候选时,
    位置 s+1 选第 j 个候选"的条件概率。truth 是目标模型从头往下的真实输出。
    """
    return {"cand": [["A", "B"], ["a", "b"], ["x", "y"]],
            "root": [0.55, 0.45],
            "cond": [[[0.6, 0.4], [0.2, 0.8]],                 # 位置 1:上一位置是 A / B 时,选 a / b
                     [[0.9, 0.1], [0.1, 0.9]]],               # 位置 2:上一位置是 a / b 时,选 x / y
            "truth": ["B", "b", "y", "x"]}


def demo():
    """链每一步取条件概率最大的那个;树按路径概率挑 6 个节点。

    目标模型从头往下的真实输出是 B、b、y、x。链的头一个 token 就猜错了,这一步只拿到白拿的那 1 个。
    树里有 B、B 后面的 b、b 后面的 y,这一步拿到 3 个。根是上一步白拿的 token,不重复计数。
    """
    lat = demo_lattice()
    cand, root_p, cond, truth = lat["cand"], lat["root"], lat["cond"], lat["truth"]

    def name(path):
        return "".join(cand[d][c] for d, c in enumerate(path))

    def prob(path):
        s, prev = root_p[path[0]], path[0]
        for d in range(1, len(path)):
            s *= cond[d - 1][prev][path[d]]                    # 条件概率只看上一个位置选的是哪个候选
            prev = path[d]
        return s

    def target_next(path):                                      # 目标模型看完一条路径后的下一个 token;走岔了就随便给一个
        return truth[len(path)] if name(path) == "".join(truth[:len(path)]) else "?"

    edges = [[[math.log(v) for v in row] for row in layer] for layer in cond]
    chain, node = [0, 0, 0], best_first_tree([math.log(v) for v in root_p], edges, 6)
    print("链 (每一步取条件概率最大的):", name(chain), f"{prob(chain):.3f}")
    print("树里挑出来的 6 个节点 (路径概率从高到低):", "  ".join(f"{name(p)} {prob(p):.2f}" for p in node))

    n, bonus = greedy_accept_chain([cand[d][c] for d, c in enumerate(chain)], [target_next(chain[:k]) for k in range(3)])
    print(f"链:接受 {n} 个,白拿 {bonus},这一步 {n + 1} 个 token")

    paths = [()] + node                                         # 节点 0 是根:上一步白拿的那个 token
    tokens = [truth[0]] + [cand[len(p) - 1][p[-1]] for p in node]
    parents = [-1] + [paths.index(p[:-1]) for p in node]
    got, bonus = greedy_accept_tree(tokens, parents, [target_next(p) for p in paths])
    print(f"树:接受 {'、'.join(got[1:])},白拿 {bonus},这一步 {len(got)} 个 token")


# ------------------------------------------------------------------ 自测
def self_test(trials: int = 300) -> dict:
    rng = random.Random(0)
    tree_ok = accept_ok = 0
    for _ in range(trials):
        # 1. 树:前缀都在里面;并且所有"父亲已在树里"的路径,得分都不超过树里最低的那个 (这就是按得分从高到低挑的定义)
        pos, k, n = rng.randint(1, 4), rng.randint(2, 4), rng.randint(1, 12)
        root = [math.log(rng.random()) for _ in range(k)]
        edges = [[[math.log(rng.random()) for _ in range(k)] for _ in range(k)] for _ in range(pos - 1)]
        tree = best_first_tree(root, edges, n)
        inside = set(tree)
        score = lambda p: sum([root[p[0]]] + [edges[d - 1][p[d - 1]][p[d]] for d in range(1, len(p))])
        prefix_closed = all(p[:-1] in inside for p in tree if len(p) > 1)
        fringe = [p + (j,) for p in tree if len(p) <= pos - 1 for j in range(k) if p + (j,) not in inside]
        exhaustive = len(fringe) == 0 or max(map(score, fringe)) <= min(map(score, tree))
        tree_ok += prefix_closed and exhaustive

        # 2. 树上的接受:和直接按定义算的比 (根算已接受,逐个节点看两个条件,取最深的)
        size = rng.randint(2, 9)
        parents = [-1] + [rng.randrange(i) for i in range(1, size)]
        tokens = [rng.choice("abc") for _ in range(size)]
        pred = [rng.choice("abc") for _ in range(size)]
        got, bonus = greedy_accept_tree(tokens, parents, pred)
        acc, depth, best = [True], [0], 0
        for i in range(1, size):
            acc.append(acc[parents[i]] and tokens[i] == pred[parents[i]])
            depth.append(depth[parents[i]] + 1)
            if acc[i] and depth[i] > depth[best]:
                best = i
        leaf, ref = best, []
        while best:
            ref.append(tokens[best])
            best = parents[best]
        accept_ok += got == ref[::-1] and bonus == pred[leaf]

    # 3. 控制器的规则:30 道题,每道题两次运行之间有 2% 的随机涨落,真实增益不同的改动各试 2000 次,数一数多少次过线
    rows = {}
    for true_gain in (0.0, 0.01, 0.015, 0.02, 0.03):
        passed = 0
        for t in range(2000):
            r = random.Random(t)
            before = [r.lognormvariate(0, 0.02) for _ in range(30)]
            after = [v * (1 + true_gain) * r.lognormvariate(0, 0.02) for v in before]
            passed += accept_patch(before, after, noise=0.001, boot=500, seed=t)[0]
        rows[true_gain] = passed / 2000
    return {"tree": tree_ok, "accept": accept_ok, "trials": trials, "pass_rate_by_true_gain": rows}


if __name__ == "__main__":
    demo()
    print()
    print(self_test())
