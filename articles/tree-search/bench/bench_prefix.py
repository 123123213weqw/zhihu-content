#!/usr/bin/env python3
"""前缀查询:新来一条序列,在已经存着的很多条里,找和它开头相同的最长一段。

场景照着推理服务来:每条序列是同一个系统提示词(前 256 个 token,都一样)接一段各不相同的后文(256 个 token)。
四种找法:
  逐个 token 线性比   对每一条存着的序列,一个 token 一个 token 地比,比到不一样为止
  按块线性比          每 16 个 token 算一个指纹,一条序列切成 32 块,一块一块地比
  前缀树(trie)       一个 token 一层,顺着往下走,走不动为止
  压缩的前缀树(radix) 单链合并成一条边,边上存一串 token
  链式哈希表          第 i 块的指纹里混进第 i-1 块的指纹,一个指纹就代表从开头到这一块的整段。查字典就行(vLLM 0.34.0 的做法)

运行(单线程,绑在一个核上):
  taskset -c 56 /data/wangyue/venvs/vllm/bin/python bench_prefix.py --out results/prefix.json
"""
import argparse
import json
import statistics
import time

from bench_common import warmup

BLOCK = 16
SYS_LEN, PRIV_LEN, VOCAB = 256, 256, 32000


def med(v):
    return statistics.median(v)


def lcp_tokens(a, b, cnt):
    """a 和 b 开头相同的长度。"""
    n = min(len(a), len(b))
    for i in range(n):
        cnt[0] += 1
        if a[i] != b[i]:
            return i
    return n


def blocks(seq):
    return [tuple(seq[i:i + BLOCK]) for i in range(0, len(seq), BLOCK)]


def chained_hashes(seq):
    h, out = 0, []
    for i in range(0, len(seq), BLOCK):
        h = hash((h, tuple(seq[i:i + BLOCK])))
        out.append(h)
    return out


def trie_add(root, seq, nodes):
    node = root
    for t in seq:
        nxt = node.get(t)
        if nxt is None:
            nxt = node[t] = {}
            nodes[0] += 1
        node = nxt


def trie_lcp(root, q, cnt):
    node, depth = root, 0
    for t in q:
        cnt[0] += 1
        nxt = node.get(t)
        if nxt is None:
            break
        node, depth = nxt, depth + 1
    return depth


# 压缩的前缀树:节点是一个字典,{第一个 token: [边上的那串 token, 孩子]}。None 是"有条序列在这里结束"的记号。
def radix_add(node, seq, nodes):
    i, n = 0, len(seq)
    while True:
        if i == n:
            node[None] = True
            return
        key = seq[i]
        if key not in node:
            node[key] = [seq[i:], {}]
            nodes[0] += 1
            return
        label, child = node[key]
        lim = min(len(label), n - i)
        j = 0
        while j < lim and label[j] == seq[i + j]:
            j += 1
        if j == len(label):                     # 整条边都对上了,接着往下
            node, i = child, i + j
        elif j == lim:                          # 新序列在这条边的中间就完了:把边断成两截
            mid = {label[j]: [label[j + 1:], child], None: True}
            node[key] = [label[:j], mid]
            nodes[0] += 1
            return
        else:                                   # 在这条边的中间分了叉
            mid = {label[j]: [label[j + 1:], child], seq[i + j]: [seq[i + j:], {}]}
            node[key] = [label[:j], mid]
            nodes[0] += 3
            return


def radix_lcp(root, q, cnt):
    node, i, n = root, 0, len(q)
    while i < n:
        edge = node.get(q[i])
        if edge is None:
            break
        label, child = edge
        lim = min(len(label), n - i)
        j = 0
        while j < lim:
            cnt[0] += 1
            if label[j] != q[i + j]:
                return i + j
            j += 1
        if j < len(label):                      # 要查的序列在这条边的中间就完了
            return i + j
        node, i = child, i + j
    return i


NAMES = ("linear_token", "linear_block", "trie", "radix", "chained_hash")


def find(name, q, ctx, cnt):
    """用某一种找法,算 q 和存着的序列开头相同的最长长度。cnt 数走了几步。"""
    if name == "linear_token":
        best = 0
        for s in ctx["stored"]:
            best = max(best, lcp_tokens(s, q, cnt))
        return best
    if name == "linear_block":
        qb = blocks(q)
        best = 0
        for sb in ctx["stored_blocks"]:
            k = 0
            while k < len(qb):
                cnt[0] += 1
                if sb[k] != qb[k]:
                    break
                k += 1
            best = max(best, k)
        return best * BLOCK
    if name == "trie":
        return trie_lcp(ctx["trie"], q, cnt)
    if name == "radix":
        return radix_lcp(ctx["radix"], q, cnt)
    best, h = 0, 0
    for i in range(0, len(q), BLOCK):
        h = hash((h, tuple(q[i:i + BLOCK])))
        cnt[0] += 1
        if h not in ctx["table"]:
            break
        best += BLOCK
    return best


def run(n, rng):
    sys_ids = [rng.randrange(VOCAB) for _ in range(SYS_LEN)]
    stored = [sys_ids + [rng.randrange(VOCAB) for _ in range(PRIV_LEN)] for _ in range(n)]
    queries = [sys_ids + [rng.randrange(VOCAB) for _ in range(PRIV_LEN)] for _ in range(8)]
    t0 = time.perf_counter()
    table = set()
    for s in stored:
        table.update(chained_hashes(s))
    ctx = {"stored": stored, "stored_blocks": [blocks(s) for s in stored], "table": table}
    hash_build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    trie, nodes = {}, [0]
    for s in stored:
        trie_add(trie, s, nodes)
    trie_build_s, trie_nodes = time.perf_counter() - t0, nodes[0]
    t0 = time.perf_counter()
    radix, nodes = {}, [0]
    for s in stored:
        radix_add(radix, s, nodes)
    radix_build_s, radix_nodes = time.perf_counter() - t0, nodes[0]
    ctx["trie"], ctx["radix"] = trie, radix

    res = {"n": n, "sys_len": SYS_LEN, "priv_len": PRIV_LEN, "block": BLOCK, "trie_nodes": trie_nodes, "radix_nodes": radix_nodes,
           "trie_build_s": trie_build_s, "radix_build_s": radix_build_s, "hash_build_s": hash_build_s, "hash_table_size": len(table)}
    hits, steps, per_round = {}, {}, {name: [] for name in NAMES}
    for rnd in range(5):                           # 五种找法轮流各测 5 轮,每轮取中位数,最后再取中位数
        for name in NAMES:
            ts = []
            for q in queries:
                cnt = [0]
                t0 = time.perf_counter()
                best = find(name, q, ctx, cnt)
                ts.append(time.perf_counter() - t0)
                steps[name] = cnt[0]
                hits.setdefault(name, []).append(best) if rnd == 0 else None
            per_round[name].append(med(ts) * 1e3)
    for name in NAMES:
        res[name] = {"steps": steps[name], "ms": med(per_round[name]), "rounds_ms": per_round[name], "hit": hits[name][0]}
    res["steps_formula"] = {"linear_token": (SYS_LEN + 1) * n, "linear_block": (SYS_LEN // BLOCK + 1) * n, "trie": SYS_LEN + 1,
                            "radix": SYS_LEN, "chained_hash": SYS_LEN // BLOCK + 1}
    assert hits["trie"] == hits["radix"] == hits["linear_token"], hits        # 三种按 token 的找法,答案要一样
    for name in ("linear_block", "chained_hash"):                              # 按块的找法,答案是不超过它的最大的 16 的倍数
        assert all(h == t // BLOCK * BLOCK for h, t in zip(hits[name], hits["linear_token"])), (name, hits)
    for name, formula in res["steps_formula"].items():
        assert steps[name] == formula, (name, steps[name], formula)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import random
    warmup()
    out = {"rows": [run(n, random.Random(100 + n)) for n in (500, 4000)]}
    json.dump(out, open(a.out, "w"), indent=1)
    for r in out["rows"]:
        print("存了 %d 条,每条 %d 个 token(前 %d 个相同):前缀树 %d 个节点,压缩后 %d 个节点" % (
            r["n"], SYS_LEN + PRIV_LEN, SYS_LEN, r["trie_nodes"], r["radix_nodes"]))
        for name in ("linear_token", "linear_block", "trie", "radix", "chained_hash"):
            print("   %-13s 每次查 %8d 步,%9.3f 毫秒,命中 %d,五轮 %s" % (
                name, r[name]["steps"], r[name]["ms"], r[name]["hit"], [round(x, 3) for x in r[name]["rounds_ms"]]))


if __name__ == "__main__":
    main()
