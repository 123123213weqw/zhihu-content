#!/usr/bin/env python3
"""点查询:在一堆有序的东西里,找一个确切的值在哪。四组对照,只吃 CPU。

  A  顺序查找 对 二分查找:每找一次比了多少次,花了多久
  B  二叉搜索树的形状:按从小到大的顺序插入,长成一条链;随机插入;AVL 树,插入时自己转平衡
  C  B+ 树的形状:一个叶子装几个,一个内部节点分几个叉,决定一次查找要读几个节点
  D  SQLite:5,000,000 行的表,按主键找一行,按没有索引的列找一行,建了索引再找

运行(绑在同一个 NUMA 节点的核上;D 会在 --db 指定的位置建一个临时数据库,测完删掉):
  taskset -c 56-71 /data/wangyue/venvs/vllm/bin/python bench_point.py --out results/point.json
"""
import argparse
import bisect
import json
import math
import os
import random
import sqlite3
import statistics
import time

import numpy as np

from bench_common import warmup


def med(v):
    return statistics.median(v)


# ---------------------------------------------------------------- A. 顺序查找 对 二分查找
def linear_search(a, x, cnt):
    for i in range(len(a)):
        cnt[0] += 1
        if a[i] == x:
            return i
    return -1


def binary_search(a, x, cnt):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        cnt[0] += 1
        mid = (lo + hi) // 2
        v = a[mid]
        if v == x:
            return mid
        if v < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def binary_search_ids(n, x, cnt):
    """在 0 到 n-1 这串排好序的键里找 x。表不用真的建出来,只数比了几次。"""
    lo, hi = 0, n - 1
    while lo <= hi:
        cnt[0] += 1
        mid = (lo + hi) // 2
        if mid == x:
            return mid
        if mid < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def section_a():
    rng = random.Random(1)
    n = 1 << 20
    keys = rng.sample(range(1 << 40), n)      # 互不相同的键,排好序
    keys.sort()
    hits = [rng.choice(keys) for _ in range(300)]
    keyset = set(keys)
    misses = []
    while len(misses) < 150:
        x = rng.randrange(1 << 40)
        if x not in keyset:
            misses.append(x)
    out = {"n": n}
    ts = {"linear_hit": [], "binary_hit": [], "linear_miss": [], "binary_miss": []}
    # 顺序和二分交错着测,各 3 轮,每轮取中位数,最后再取中位数
    for rnd in range(3):
        for name, qs in (("hit", hits), ("miss", misses)):
            for how, fn in (("linear", linear_search), ("binary", binary_search)):
                c, t = [0], []
                for x in qs:
                    t0 = time.perf_counter()
                    fn(keys, x, c)
                    t.append(time.perf_counter() - t0)
                ts["%s_%s" % (how, name)].append(med(t) * 1e6)
                out["%s_%s" % (how, name)] = {"comparisons": c[0] / len(qs), "us": med(ts["%s_%s" % (how, name)])}
    # 再大得多的表:不建表,只数二分查找比了几次
    big = []
    for e in (10, 20, 30, 40):
        n2 = 1 << e
        cs = []
        for _ in range(5000):
            c = [0]
            binary_search_ids(n2, rng.randrange(n2), c)
            cs.append(c[0])
        big.append({"n": n2, "avg": sum(cs) / len(cs), "max": max(cs), "log2": math.log2(n2)})
    out["big"] = big
    # 标准库的 bisect,在 2 的 24 次方个元素上,每次要多久
    arr = np.sort(np.random.default_rng(2).integers(0, 1 << 40, 1 << 24)).tolist()
    q = [rng.choice(arr) for _ in range(20000)]
    t0 = time.perf_counter()
    for x in q:
        bisect.bisect_left(arr, x)
    out["bisect_2p24_us"] = (time.perf_counter() - t0) / len(q) * 1e6
    return out


# ---------------------------------------------------------------- B. 二叉搜索树的形状
class Node:
    __slots__ = ("k", "l", "r", "h")

    def __init__(self, k):
        self.k, self.l, self.r, self.h = k, None, None, 1


def bst_insert(root, k, cnt):
    if root is None:
        return Node(k)
    cur = root
    while True:
        cnt[0] += 1
        if k < cur.k:
            if cur.l is None:
                cur.l = Node(k)
                return root
            cur = cur.l
        else:
            if cur.r is None:
                cur.r = Node(k)
                return root
            cur = cur.r


def hgt(n):
    return n.h if n else 0


def rot_right(y):
    x = y.l
    y.l = x.r
    x.r = y
    y.h = 1 + max(hgt(y.l), hgt(y.r))
    x.h = 1 + max(hgt(x.l), hgt(x.r))
    return x


def rot_left(x):
    y = x.r
    x.r = y.l
    y.l = x
    x.h = 1 + max(hgt(x.l), hgt(x.r))
    y.h = 1 + max(hgt(y.l), hgt(y.r))
    return y


def avl_insert(n, k, cnt, rot):
    """AVL 树的插入:两边的高度差到 2 就转一下。返回新的子树的根。"""
    if n is None:
        return Node(k)
    cnt[0] += 1
    if k < n.k:
        n.l = avl_insert(n.l, k, cnt, rot)
    else:
        n.r = avl_insert(n.r, k, cnt, rot)
    n.h = 1 + max(hgt(n.l), hgt(n.r))
    bf = hgt(n.l) - hgt(n.r)
    if bf > 1:
        if hgt(n.l.l) < hgt(n.l.r):
            n.l = rot_left(n.l)
        rot[0] += 1
        return rot_right(n)
    if bf < -1:
        if hgt(n.r.r) < hgt(n.r.l):
            n.r = rot_right(n.r)
        rot[0] += 1
        return rot_left(n)
    return n


def tree_stats(root):
    """高度(根算第 1 层)和平均深度(从根走到每个节点要走几步,所有节点取平均)。"""
    stack = [(root, 0)]
    maxd, total, count = 0, 0, 0
    while stack:
        n, d = stack.pop()
        maxd = max(maxd, d)
        total += d
        count += 1
        if n.l:
            stack.append((n.l, d + 1))
        if n.r:
            stack.append((n.r, d + 1))
    return {"height": maxd + 1, "avg_depth": total / count, "nodes": count}


def section_b():
    n = 1 << 16
    rng = random.Random(3)
    order = list(range(n))
    rng.shuffle(order)
    out = {"n": n, "log2_n": math.log2(n), "two_ln_n": 2 * math.log(n),
           "avl_height_bound": math.floor(1.4405 * math.log2(n + 2) - 0.3277)}
    # 按从小到大插入普通的二叉搜索树:每个新键都比前面所有的键大,一路走到最右下,树成了一条链。
    # 第 i 个键路上要比 i 次,一共 n(n-1)/2 次,这里直接按公式记。
    root = Node(0)
    tail = root
    for k in range(1, n):
        tail.r = Node(k)
        tail = tail.r
    out["bst_sorted"] = {**tree_stats(root), "build_comparisons": n * (n - 1) // 2}
    del root, tail
    root, c = None, [0]
    for k in order:
        root = bst_insert(root, k, c)
    out["bst_random"] = {**tree_stats(root), "build_comparisons": c[0]}
    del root
    for name, keys in (("avl_random", order), ("avl_sorted", list(range(n)))):
        root, c, rot = None, [0], [0]
        for k in keys:
            root = avl_insert(root, k, c, rot)
        out[name] = {**tree_stats(root), "build_comparisons": c[0], "rotations": rot[0]}
        del root
    return out


# ---------------------------------------------------------------- C. B+ 树的形状
def bptree_shape(n, leaf, fan):
    """n 个键,一个叶子最多装 leaf 个,一个内部节点最多有 fan 个孩子。
    返回层数(叶子算一层),和一次查找在节点内部二分,大约要比几次。"""
    leaves = math.ceil(n / leaf)
    levels = 1
    parents = leaves
    while parents > 1:
        parents = math.ceil(parents / fan)
        levels += 1
    return {"n": n, "leaf": leaf, "fan": fan, "leaves": leaves, "levels": levels,
            "comparisons": (levels - 1) * math.ceil(math.log2(fan + 1)) + math.ceil(math.log2(leaf + 1))}


def section_c():
    return [bptree_shape(n, leaf, leaf) for n in (10**6, 10**9, 10**12) for leaf in (4, 16, 64, 256, 1024)]


# ---------------------------------------------------------------- D. SQLite
def section_d(db_path):
    n = 5_000_000
    rng = random.Random(4)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, w INTEGER, v TEXT)")
    probe = {}

    def rows():
        for i in range(1, n + 1):
            w = rng.getrandbits(62)            # 每一行的 w 都不一样
            if i == n // 2:
                probe["w"] = w
            yield (i, w, "x" * 40)

    t0 = time.perf_counter()
    con.executemany("INSERT INTO t VALUES(?,?,?)", rows())
    con.commit()
    build_s = time.perf_counter() - t0
    page_size = con.execute("PRAGMA page_size").fetchone()[0]
    page_count = con.execute("PRAGMA page_count").fetchone()[0]
    missing = probe["w"] + 1                  # 几乎可以肯定不在表里:按它找,只能把全表扫完

    def plan(sql, arg):
        return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql, (arg,))]

    def timeit(sql, arg, reps):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            con.execute(sql, (arg,)).fetchone()
            ts.append(time.perf_counter() - t0)
        return ts

    plans = {"pk": plan("SELECT v FROM t WHERE id=?", 123), "scan": plan("SELECT v FROM t WHERE w=?", missing)}
    by_pk = timeit("SELECT v FROM t WHERE id=?", rng.randrange(1, n + 1), 300)
    scan = timeit("SELECT v FROM t WHERE w=?", missing, 5)
    t0 = time.perf_counter()
    con.execute("CREATE INDEX idx_w ON t(w)")
    con.commit()
    index_s = time.perf_counter() - t0
    by_index = timeit("SELECT v FROM t WHERE w=?", probe["w"], 300)
    res = {"rows": n, "build_s": build_s, "page_size": page_size, "page_count": page_count,
           "page_count_with_index": con.execute("PRAGMA page_count").fetchone()[0],
           "plan_pk": plans["pk"], "plan_scan": plans["scan"],
           "plan_index": plan("SELECT v FROM t WHERE w=?", probe["w"]),
           "by_pk_us": med(by_pk) * 1e6, "by_pk_max_us": max(by_pk) * 1e6, "scan_s": med(scan), "scan_min_s": min(scan),
           "index_build_s": index_s, "by_index_us": med(by_index) * 1e6, "by_index_max_us": max(by_index) * 1e6,
           "sqlite_version": sqlite3.sqlite_version}
    con.close()
    os.remove(db_path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", default="/data/wangyue/tmp/tree_bench.sqlite")
    a = ap.parse_args()
    warmup()
    res = {"a_linear_vs_binary": section_a(), "b_bst": section_b(), "c_bptree": section_c(), "d_sqlite": section_d(a.db)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
