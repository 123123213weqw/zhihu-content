#!/usr/bin/env python3
"""给文章里"三种二叉搜索树长什么样"那张图准备数据:23 个键,按从小到大插入,随机插入,AVL 树。
用的是 bench_point.py 里同一套插入代码。纯 CPU,不计时。

运行: python trees_small.py --out results/trees_small.json
"""
import argparse
import json
import random

from bench_point import Node, avl_insert, bst_insert, tree_stats


def layout(root):
    """给每个节点定坐标:横坐标按中序遍历的次序,纵坐标按深度。"""
    xs, stack, cur, out = [], [], root, []
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.l
        cur = stack.pop()
        xs.append(cur)
        cur = cur.r
    pos = {id(n): (float(i), 0) for i, n in enumerate(xs)}
    stack = [(root, 0)]
    edges = []
    while stack:
        n, d = stack.pop()
        x, _ = pos[id(n)]
        pos[id(n)] = (x, d)
        for c in (n.l, n.r):
            if c:
                edges.append([n.k, c.k])
                stack.append((c, d + 1))
    return {"points": [[n.k, pos[id(n)][0], pos[id(n)][1]] for n in xs], "edges": edges}   # points 里是 [键, 横坐标, 深度]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 23
    keys = list(range(1, n + 1))
    random.Random(42).shuffle(keys)
    out = {"n": n, "keys_random": keys}
    root, cnt = None, [0]
    for k in range(1, n + 1):
        root = bst_insert(root, k, cnt)
    out["sorted"] = {**layout(root), **tree_stats(root)}
    root, cnt = None, [0]
    for k in keys:
        root = bst_insert(root, k, cnt)
    out["random"] = {**layout(root), **tree_stats(root)}
    root, cnt, rot = None, [0], [0]
    for k in keys:
        root = avl_insert(root, k, cnt, rot)
    out["avl"] = {**layout(root), **tree_stats(root), "rotations": rot[0]}
    json.dump(out, open(a.out, "w"))
    for name in ("sorted", "random", "avl"):
        print(name, "高度", out[name]["height"], "平均深度", round(out[name]["avg_depth"], 2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
