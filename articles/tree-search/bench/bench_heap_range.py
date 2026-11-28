#!/usr/bin/env python3
"""最值和区间:两种"每个节点只管一小片"的树。只吃 CPU。

  E  从 400 万个数里挑最大的 k 个:全部排一遍序,和只用一个装 k 个数的堆
  F  区间和:一段一段地求和,中间还会改里面的数。直接加,前缀和,树状数组

运行(单线程,绑在一个核上):
  taskset -c 56 /data/wangyue/venvs/vllm/bin/python bench_heap_range.py --out results/heap_range.json
"""
import argparse
import heapq
import json
import math
import statistics
import time

import numpy as np

from bench_common import warmup


def med(v):
    return statistics.median(v)


# ---------------------------------------------------------------- E. 最大的 k 个
def bounded_heap_topk(data, k, cnt, entered):
    """一个只装 k 个数的小顶堆:堆顶是"目前最大的 k 个"里最小的那个。后面的数比堆顶大,才进堆。"""
    heap = list(data[:k])
    heapq.heapify(heap)
    cnt[0] += k
    for x in data[k:]:
        cnt[0] += 1
        if x > heap[0]:
            heapq.heapreplace(heap, x)     # 弹出堆顶、放进新数、往下 sift:大约 log2(k) 次比较
            cnt[0] += len(heap).bit_length()
            entered[0] += 1
    return sorted(heap, reverse=True)


def section_e():
    rng = np.random.default_rng(5)
    n = 4_000_000
    data = rng.random(n).tolist()
    small = data[: n // 10]                        # 数据少 10 倍
    ks = (10, 100, 1000)
    t_sort, t_heap, t_small = [], {k: [] for k in ks}, {k: [] for k in ks}
    for _ in range(5):                             # 排序和堆轮流各测 5 轮,取中位数
        t0 = time.perf_counter()
        sorted(data, reverse=True)
        t_sort.append(time.perf_counter() - t0)
        for k in ks:
            t0 = time.perf_counter()
            heapq.nlargest(k, data)
            t_heap[k].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            heapq.nlargest(k, small)
            t_small[k].append(time.perf_counter() - t0)
    out = {"n": n, "sort_s": med(t_sort), "rows": []}
    for k in ks:
        cnt, entered = [0], [0]
        d = bounded_heap_topk(data, k, cnt, entered)
        assert d == sorted(data, reverse=True)[:k] == heapq.nlargest(k, data)
        out["rows"].append({"k": k, "nlargest_s": med(t_heap[k]), "nlargest_n_over_10_s": med(t_small[k]),
                            "heap_comparisons_est": cnt[0], "entered_heap": entered[0], "speedup": med(t_sort) / med(t_heap[k]),
                            "linear_in_n": med(t_heap[k]) / med(t_small[k])})
    # 排序到底比了几次:包一层会数数的壳,在 10 万个数上量,折成"相当于 n log2 n 的几倍"
    class Counted(float):
        def __lt__(self, other):
            cnt[0] += 1
            return float.__lt__(self, other)
    m = 100_000
    counted = [Counted(x) for x in data[:m]]
    cnt[0] = 0
    sorted(counted)
    out["sort_counted"] = {"n": m, "comparisons": cnt[0], "per_n_log2n": cnt[0] / (m * math.log2(m))}
    return out


# ---------------------------------------------------------------- F. 区间和
class Fenwick:
    """树状数组:下标 i 的格子,管原数组里下标从 i - lowbit(i) + 1 到 i 这一段的和。"""

    def __init__(self, a):
        self.n = len(a)
        self.t = [0] * (self.n + 1)
        for i, v in enumerate(a, 1):
            self.t[i] += v
            j = i + (i & -i)
            if j <= self.n:
                self.t[j] += self.t[i]

    def add(self, i, delta, cnt=None):
        i += 1
        while i <= self.n:
            self.t[i] += delta
            if cnt is not None:
                cnt[0] += 1
            i += i & -i

    def sum(self, i, cnt=None):
        """前 i 个数的和(下标 0 到 i-1)。"""
        s = 0
        while i > 0:
            s += self.t[i]
            if cnt is not None:
                cnt[0] += 1
            i -= i & -i
        return s


def check_correct():
    """先在一小段数组上把三种做法对一遍答案。"""
    rng = np.random.default_rng(66)
    n, a = 4096, rng.integers(0, 1000, 4096)
    fw = Fenwick(a.tolist())
    p = np.concatenate(([0], np.cumsum(a)))
    for _ in range(200):
        i, l = int(rng.integers(0, n)), int(rng.integers(0, n - 1))
        r = l + int(rng.integers(1, n - l))
        delta = int(rng.integers(-500, 500))
        a[i] += delta
        fw.add(i, delta)
        p = np.concatenate(([0], np.cumsum(a)))
        x, y, z = int(a[l:r].sum()), fw.sum(r) - fw.sum(l), int(p[r] - p[l])
        assert x == y == z, (x, y, z)


def section_f():
    check_correct()
    rng = np.random.default_rng(6)
    rows = []
    for n in (1 << 20, 1 << 22, 1 << 24):
        base = rng.integers(0, 1000, n)
        q = 1000
        ls = rng.integers(0, n - 1, q)
        rs = ls + rng.integers(1, n - ls)
        idx = rng.integers(0, n, q)
        deltas = rng.integers(-500, 500, q)

        # 直接加(numpy):改一个数 O(1),求一段的和要把这一段扫一遍
        a = base.copy()
        tq, tu = [], []
        for i in range(q):
            t0 = time.perf_counter()
            a[idx[i]] += deltas[i]
            tu.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            int(a[ls[i]:rs[i]].sum())
            tq.append(time.perf_counter() - t0)
        row = {"n": n, "avg_range": (rs - ls).mean(), "naive_query_us": med(tq) * 1e6, "naive_update_us": med(tu) * 1e6}

        # 树状数组:查和改都是 O(log n)
        fw = Fenwick(base.tolist())
        tq, tu = [], []
        for i in range(q):
            t0 = time.perf_counter()
            fw.add(int(idx[i]), int(deltas[i]))
            tu.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            v = fw.sum(int(rs[i])) - fw.sum(int(ls[i]))
            tq.append(time.perf_counter() - t0)
        assert v == int(a[ls[-1]:rs[-1]].sum())       # 同样 1,000 次操作之后,两边的答案要一样
        cq, cu = [0], [0]
        fw.sum(int(rs[0]), cq)
        fw.sum(int(ls[0]), cq)
        fw.add(int(idx[0]), int(deltas[0]), cu)
        row.update({"fenwick_query_us": med(tq) * 1e6, "fenwick_update_us": med(tu) * 1e6, "fenwick_query_steps": cq[0],
                    "fenwick_update_steps": cu[0], "fenwick_max_steps": 2 * (n.bit_length() - 1)})

        # 前缀和:只查不改的时候最快;改一个数,后面的前缀和全要重算
        a = base.copy()
        p = np.concatenate(([0], np.cumsum(a)))
        t0 = time.perf_counter()
        for i in range(q):
            int(p[rs[i]] - p[ls[i]])
        row["prefix_query_us"] = (time.perf_counter() - t0) / q * 1e6
        tu = []
        for i in range(40):
            t0 = time.perf_counter()
            a[idx[i]] += deltas[i]
            p = np.concatenate(([0], np.cumsum(a)))
            tu.append(time.perf_counter() - t0)
        row["prefix_update_us"] = med(tu) * 1e6
        row["prefix_update_over_fenwick"] = row["prefix_update_us"] / row["fenwick_update_us"]
        row["naive_over_fenwick"] = row["naive_query_us"] / row["fenwick_query_us"]
        rows.append(row)
    return {"ops": n and q, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warmup()
    res = {"e_topk": section_e(), "f_range": section_f()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:3500])


if __name__ == "__main__":
    main()
