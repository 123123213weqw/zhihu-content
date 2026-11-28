#!/usr/bin/env python3
"""最近邻:N 个点里,离给定的点最近的是哪个。只吃 CPU。

  G1  自己搭一棵 kd 树,数一数一次查询要算多少个点的距离:维数从 2 到 64
  G2  scipy 的 cKDTree,和把全部距离算一遍比,一次查询各要多久(都是单线程)
  G3  维数高了树不行,换成图:hnswlib 的 HNSW。看它找到的是不是真的最近的 k 个(召回率),一次查询要多久。
      100 万个点,两种数据:均匀撒开的,和分成 1,000 团的(真实 embedding 常常是一团一团的)

运行(用单独的 /data/wangyue/venvs/trees 环境,里面装了 numpy、scipy、hnswlib;G1 G2 是单线程,建 HNSW 索引用 16 个线程,
所以绑同一个 NUMA 节点里的 16 个核,整个跑完要 20 分钟左右):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 36-51 /data/wangyue/venvs/trees/bin/python bench_nn.py --out results/nn.json
"""
import argparse
import json
import os
import time

# 先把 BLAS 的线程数定成 1,再导入 numpy,直接算全部距离那一方才是单线程
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402

from bench_common import warmup  # noqa: E402


# ---------------------------------------------------------------- G1. 自己搭的 kd 树,数距离算了几次
LEAF = 16


class KDNode:
    __slots__ = ("leaf", "idx", "axis", "split", "left", "right", "lo", "hi")


def kd_build(pts, idx, depth):
    node = KDNode()
    node.lo, node.hi = pts[idx].min(0), pts[idx].max(0)
    if len(idx) <= LEAF:
        node.leaf, node.idx = True, idx
        return node
    axis = depth % pts.shape[1]
    order = idx[np.argsort(pts[idx, axis], kind="stable")]
    m = len(idx) // 2
    node.leaf, node.axis, node.split = False, axis, pts[order[m], axis]
    node.left, node.right = kd_build(pts, order[:m], depth + 1), kd_build(pts, order[m:], depth + 1)
    return node


def kd_query(node, q, pts, st):
    """先看这个节点管的那块空间,离 q 最近的可能距离是不是已经比手里的答案远:是就整块跳过。"""
    st["nodes"] += 1
    d = 0.0
    for k in range(q.shape[0]):
        if q[k] < node.lo[k]:
            d += (node.lo[k] - q[k]) ** 2
        elif q[k] > node.hi[k]:
            d += (q[k] - node.hi[k]) ** 2
    if d >= st["best"]:
        st["pruned"] += 1
        return
    if node.leaf:
        diff = pts[node.idx] - q
        dist = np.einsum("ij,ij->i", diff, diff)
        st["computed"] += len(node.idx)
        i = int(dist.argmin())
        if dist[i] < st["best"]:
            st["best"], st["nn"] = float(dist[i]), int(node.idx[i])
        return
    near, far = (node.left, node.right) if q[node.axis] < node.split else (node.right, node.left)
    kd_query(near, q, pts, st)
    kd_query(far, q, pts, st)


def section_g1():
    rng = np.random.default_rng(7)
    n, nq = 20_000, 200
    rows = []
    for d in (2, 4, 8, 16, 32, 64):
        pts = rng.random((n, d))
        qs = rng.random((nq, d))
        t0 = time.perf_counter()
        root = kd_build(pts, np.arange(n), 0)
        build_s = time.perf_counter() - t0
        comp, pruned, nodes, wrong = [], [], [], 0
        for q in qs:
            st = {"best": float("inf"), "nn": -1, "computed": 0, "pruned": 0, "nodes": 0}
            kd_query(root, q, pts, st)
            comp.append(st["computed"])
            pruned.append(st["pruned"])
            nodes.append(st["nodes"])
            true = int(np.argmin(np.einsum("ij,ij->i", pts - q, pts - q)))
            wrong += st["nn"] != true
        rows.append({"d": d, "computed_mean": float(np.mean(comp)), "computed_max": int(max(comp)), "fraction": float(np.mean(comp)) / n,
                     "pruned_mean": float(np.mean(pruned)), "nodes_mean": float(np.mean(nodes)), "build_s": build_s, "wrong": wrong})
    return {"n": n, "queries": nq, "leaf": LEAF, "rows": rows}


# ---------------------------------------------------------------- G2. cKDTree 对 直接算全部距离
def brute_argmin(pts, pts_sq, q):
    return int(np.argmin(pts_sq - 2.0 * (pts @ q)))


def section_g2():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(8)
    n, nq = 100_000, 350
    rows = []
    for d in (2, 8, 32, 128, 512, 2048):
        pts = rng.random((n, d))
        qs = rng.random((nq, d))
        t0 = time.perf_counter()
        tree = cKDTree(pts, balanced_tree=True, compact_nodes=True)
        build_s = time.perf_counter() - t0
        pts_sq = np.einsum("ij,ij->i", pts, pts)
        warmup(1.0)
        tb, tt, disagree = [], [], 0
        for j, q in enumerate(qs):
            t0 = time.perf_counter()               # 每个查询,两种做法紧挨着各做一次;前 50 个只用来热身
            b = brute_argmin(pts, pts_sq, q)
            t1 = time.perf_counter()
            dist, t = tree.query(q, k=1, workers=1)
            t2 = time.perf_counter()
            disagree += int(t) != b
            if j >= 50:
                tb.append(t1 - t0)
                tt.append(t2 - t1)
        rows.append({"d": d, "brute_us": float(np.median(tb)) * 1e6, "tree_us": float(np.median(tt)) * 1e6,
                     "brute_min_us": float(np.min(tb)) * 1e6, "tree_min_us": float(np.min(tt)) * 1e6, "build_s": build_s, "disagree": disagree,
                     "speedup": float(np.median(tb)) / float(np.median(tt)), "speedup_min": float(np.min(tb)) / np.min(tt)})
    return {"n": n, "queries": nq, "warm": 50, "rows": rows}


# ---------------------------------------------------------------- G3. HNSW:图上的近似最近邻
def make_points(rng, n, d, kind):
    """均匀撒在单位立方体里,或者 1,000 团各占一角的正态团(标准差 0.05)。"""
    if kind == "uniform":
        return rng.random((n, d), dtype=np.float32)
    centers = rng.random((1000, d), dtype=np.float32)
    which = rng.integers(0, 1000, n)
    pts = centers[which] + rng.normal(0.0, 0.05, (n, d)).astype(np.float32)
    return np.clip(pts, 0.0, 1.0)


def section_g3(dims=(64, 768), kinds=("uniform", "clustered"), n=1_000_000, nq=200, k=10):
    import hnswlib
    rng = np.random.default_rng(9)
    rows = []
    for kind in kinds:
        for d in dims:
            pts = make_points(rng, n, d, kind)
            qs = make_points(rng, nq, d, kind)
            pts_sq = np.einsum("ij,ij->i", pts, pts)
            truth, tb = [], []
            warmup(1.0)
            for j, q in enumerate(qs):             # 老老实实算 100 万个距离,取最近的 10 个
                t0 = time.perf_counter()
                dist = pts_sq - 2.0 * (pts @ q)
                truth.append(set(np.argpartition(dist, k)[:k].tolist()))
                tb.append(time.perf_counter() - t0)
            brute_us = float(np.median(tb[10:])) * 1e6
            index = hnswlib.Index(space="l2", dim=d)
            index.init_index(max_elements=n, M=16, ef_construction=200)
            t0 = time.perf_counter()
            index.add_items(pts, num_threads=16)
            build_s = time.perf_counter() - t0
            for ef in (50, 400):
                index.set_ef(ef)
                index.knn_query(qs[:20], k, num_threads=1)          # 热身
                warmup(0.5)
                t0 = time.perf_counter()
                labels, _ = index.knn_query(qs, k, num_threads=1)
                qt = (time.perf_counter() - t0) / nq
                recall = float(np.mean([len(set(l.tolist()) & t) / k for l, t in zip(labels, truth)]))
                rows.append({"kind": kind, "d": d, "ef": ef, "brute_us": brute_us, "hnsw_us": qt * 1e6, "recall@10": recall,
                             "build_s": build_s, "speedup": brute_us / (qt * 1e6)})
            del pts, qs, index
    return {"n": n, "queries": nq, "k": k, "M": 16, "ef_construction": 200, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warmup()
    res = {"g1_kd_count": section_g1(), "g2_ckdtree_vs_brute": section_g2(), "g3_hnsw": section_g3()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
