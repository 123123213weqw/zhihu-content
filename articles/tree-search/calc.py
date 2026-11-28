#!/usr/bin/env python3
"""树搜索一篇里引用的每个数,都在这里算一遍。输出 bench/calc_summary.json,图和正文都从它读。

量的是一台 CPU 服务器(Xeon Platinum 8352V,单线程,绑在一个核上,先跑热再计时)上的 Python 3.12.4。
实测数据在 bench/results/ 里,由 bench/ 下的几个脚本生成:
  point.json         点查询(bench_point.py):顺序查找对二分查找,二叉搜索树的形状,B+ 树的形状,SQLite
  trees_small.json   23 个键的三棵小树,给图用(trees_small.py)
  prefix.json        前缀查询(bench_prefix.py)
  heap_range.json    最大的 k 个,区间和(bench_heap_range.py)
  nn.json            最近邻(bench_nn.py):自己搭的 kd 树,scipy 的 cKDTree,hnswlib 的 HNSW(100 万个均匀随机的点)
  nn_mnist.json      HNSW 换成真实数据再量一遍(bench_nn_mnist.py):6 万张手写数字图片
  game.json          井字棋(bench_game.py)
运行:python3 calc.py
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "bench", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def bptree_levels(n, leaf, fan):
    parents = math.ceil(n / leaf)
    levels = 1
    while parents > 1:
        parents = math.ceil(parents / fan)
        levels += 1
    return levels


point = load("point.json")
small = load("trees_small.json")
prefix = load("prefix.json")
hr = load("heap_range.json")
nn = load("nn.json")
mnist = load("nn_mnist.json")
game = load("game.json")

out = {}
out["env"] = {"python": "3.12.4", "cpu": "Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz", "sqlite": point["d_sqlite"]["sqlite_version"]}

# ---- 1. 顺序查找 对 二分查找 ------------------------------------------------------------------------
a = point["a_linear_vs_binary"]
out["linear_vs_binary"] = {
    "n": a["n"], "log2_n": math.log2(a["n"]),
    "linear_hit": a["linear_hit"], "binary_hit": a["binary_hit"], "linear_miss": a["linear_miss"], "binary_miss": a["binary_miss"],
    "speedup_hit": a["linear_hit"]["us"] / a["binary_hit"]["us"], "speedup_miss": a["linear_miss"]["us"] / a["binary_miss"]["us"],
    "big": a["big"], "bisect_2p24_us": a["bisect_2p24_us"],
}

# ---- 2. 二叉搜索树的形状 ----------------------------------------------------------------------------
b = point["b_bst"]
rows = [("bst_sorted", "按从小到大插入"), ("bst_random", "随机插入"), ("avl_random", "AVL 树,随机插入"), ("avl_sorted", "AVL 树,按从小到大插入")]
out["bst"] = {
    "n": b["n"], "log2_n": b["log2_n"], "two_ln_n": b["two_ln_n"], "avl_height_bound": b["avl_height_bound"],
    "rows": [{"name": key, "label": label, **b[key]} for key, label in rows],
    "sorted_build_years": b["bst_sorted"]["build_comparisons"] / 1e9,     # 每秒比 10 亿次的话,要几秒
    "random_depth_over_log2n": b["bst_random"]["avg_depth"] / b["log2_n"],
}
s = {k: small[k] for k in ("sorted", "random", "avl")}
out["trees_small"] = {"n": small["n"], **s}
out["ideal_height_23"] = math.ceil(math.log2(small["n"] + 1))

# ---- 3. B+ 树的形状,SQLite --------------------------------------------------------------------------
out["bptree"] = {"rows": point["c_bptree"]}
by_n = {}
for r in point["c_bptree"]:
    by_n.setdefault(r["n"], []).append(r)
out["bptree"]["by_n"] = by_n
d = point["d_sqlite"]
out["sqlite"] = {
    **d,
    "speedup": d["scan_s"] * 1e6 / d["by_pk_us"], "speedup_min": d["scan_min_s"] * 1e6 / d["by_pk_max_us"],
    "table_mb": d["page_count"] * d["page_size"] / 1e6, "index_mb": (d["page_count_with_index"] - d["page_count"]) * d["page_size"] / 1e6,
    "index_pages": d["page_count_with_index"] - d["page_count"],
    "scan_ns_per_row": d["scan_s"] / d["rows"] * 1e9, "rows_per_page": d["rows"] / d["page_count"],
    "leaves_if_64_per_page": math.ceil(d["rows"] / 64),
}

# ---- 4. 前缀 ----------------------------------------------------------------------------------------
names = ("linear_token", "linear_block", "trie", "radix", "chained_hash")
out["prefix"] = {"rows": prefix["rows"]}
r5, r40 = prefix["rows"]
out["prefix"]["scale"] = {
    "linear": r40["linear_token"]["ms"] / r5["linear_token"]["ms"], "linear_block": r40["linear_block"]["ms"] / r5["linear_block"]["ms"],
    "trie": r40["trie"]["ms"] / r5["trie"]["ms"], "radix": r40["radix"]["ms"] / r5["radix"]["ms"], "chained_hash": r40["chained_hash"]["ms"] / r5["chained_hash"]["ms"],
    "n_ratio": r40["n"] / r5["n"],
    "speedup_at_4000": r40["linear_token"]["ms"] / r40["trie"]["ms"],
    "speedup_hash_at_4000": r40["linear_token"]["ms"] / r40["chained_hash"]["ms"],
    "steps_ratio_4000": r40["linear_token"]["steps"] / r40["trie"]["steps"],
    "radix_nodes_shrink_4000": r40["trie_nodes"] / r40["radix_nodes"],
    "radix_nodes_shrink_500": r5["trie_nodes"] / r5["radix_nodes"],
    "trie_depth_vs_steps": (r5["sys_len"] + 1, r5["trie"]["steps"]),
}

# ---- 5. 最大的 k 个 ----------------------------------------------------------------------------------
e = hr["e_topk"]
out["topk"] = {
    "n": e["n"], "sort_s": e["sort_s"], "sort_counted": e["sort_counted"], "rows": e["rows"],
    "sort_comparisons_4m": e["sort_counted"]["per_n_log2n"] * e["n"] * math.log2(e["n"]),
    "entered_share": [r["entered_heap"] / e["n"] for r in e["rows"]],
}

# ---- 6. 区间和 --------------------------------------------------------------------------------------
f = hr["f_range"]
out["range"] = {"ops": f["ops"], "rows": f["rows"]}
r1, r4 = f["rows"][0], f["rows"][-1]
out["range"]["scale"] = {
    "n_ratio": r4["n"] / r1["n"], "naive_query": r4["naive_query_us"] / r1["naive_query_us"],
    "fenwick_query": r4["fenwick_query_us"] / r1["fenwick_query_us"], "fenwick_update": r4["fenwick_update_us"] / r1["fenwick_update_us"],
    "extra_query_steps": r4["fenwick_query_steps"] - r1["fenwick_query_steps"], "extra_max_steps": r4["fenwick_max_steps"] - r1["fenwick_max_steps"],
}

# ---- 7. 最近邻 --------------------------------------------------------------------------------------
g1 = nn["g1_kd_count"]
out["kd_count"] = {**{k: v for k, v in g1.items() if k != "rows"}, "rows": g1["rows"]}
out["kd_count"]["one_over_log"] = [1.0 / math.log2(g1["n"] / g1["leaf"]) for _ in g1["rows"]]       # 对照:节点数的对数,不是这样涨的
g2 = nn["g2_ckdtree_vs_brute"]
out["ckdtree"] = {**{k: v for k, v in g2.items() if k != "rows"}, "rows": g2["rows"]}
g3 = nn["g3_hnsw"]
out["hnsw"] = {**{k: v for k, v in g3.items() if k != "rows"}, "rows": [r for r in g3["rows"] if r["kind"] == "uniform"],
               "synthetic_clusters": [r for r in g3["rows"] if r["kind"] == "clustered"]}
out["hnsw_mnist"] = mnist
m5 = {r["ef"]: r for r in mnist["mnist"]}
u5 = {r["ef"]: r for r in mnist["uniform_same_shape"]}
out["hnsw_mnist"]["compare"] = {"recall_50": (m5[50]["recall@10"], u5[50]["recall@10"]), "recall_400": (m5[400]["recall@10"], u5[400]["recall@10"]),
                                "speedup_50": (m5[50]["speedup"], u5[50]["speedup"])}
c64 = [r for r in out["hnsw"]["synthetic_clusters"] if r["d"] == 64]
out["hnsw"]["clustered_64"] = {r["ef"]: r["recall@10"] for r in c64}
# SQLite 那 500 万行的表,按 B+ 树的算法估层数:叶子页数是读出来的,一个内部节点能放 300 个路标是假设
out["sqlite"]["levels_est"] = bptree_levels(out["sqlite"]["rows"], out["sqlite"]["rows_per_page"], 300)

# ---- 8. 井字棋 --------------------------------------------------------------------------------------
mm, ab, abc = game["minimax"], game["alphabeta_natural"], game["alphabeta_center_first"]
out["game"] = {
    "minimax": mm, "alphabeta_natural": ab, "alphabeta_center_first": abc,
    "after_two": {k: v for k, v in game.items() if k.startswith("alphabeta_after_two")},
    "uct": game["uct_vs_random"],
    "natural_share": ab["nodes"] / mm["nodes"], "center_share": abc["nodes"] / mm["nodes"],
    "natural_cut": mm["nodes"] / ab["nodes"], "center_cut": mm["nodes"] / abc["nodes"],
    "center_over_natural": ab["nodes"] / abc["nodes"],
    "games": mm["leaves"],
}
# 围棋和国际象棋的搜索空间:分支数和深度取自 Silver 等 2016 年那篇论文的引言(b ≈ 250, d ≈ 150;b ≈ 35, d ≈ 80)
out["game"]["go"] = {"b": 250, "d": 150, "log10_full": 150 * math.log10(250), "log10_sqrt": math.log10(2) + 75 * math.log10(250)}
out["game"]["chess"] = {"b": 35, "d": 80, "log10_full": 80 * math.log10(35)}
uct = {r["sims_per_move"]: r for r in game["uct_vs_random"]}
out["game"]["uct_sims_per_game"] = {k: v["simulations_per_game"] for k, v in uct.items()}
out["game"]["uct_win"] = {k: v["win_rate"] for k, v in uct.items()}
out["game"]["uct_no_loss"] = {k: v["no_loss_rate"] for k, v in uct.items()}

with open(os.path.join(HERE, "bench", "calc_summary.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)


def p(k, v):
    print("%-40s %s" % (k, json.dumps(v, ensure_ascii=False)[:330]))


p("顺序 对 二分(比较次数, 微秒)", {k: (round(a[k]["comparisons"], 1), round(a[k]["us"], 2)) for k in ("linear_hit", "binary_hit", "linear_miss", "binary_miss")})
p("快了多少倍", (round(out["linear_vs_binary"]["speedup_hit"]), round(out["linear_vs_binary"]["speedup_miss"])))
p("大表上二分比几次", [(int(r["n"]), round(r["avg"], 1), r["max"]) for r in a["big"]])
p("bisect 2^24(微秒)", round(a["bisect_2p24_us"], 2))
for r in out["bst"]["rows"]:
    p("树 " + r["label"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("name", "label")})
p("log2 n / 2 ln n / AVL 上界", (out["bst"]["log2_n"], round(out["bst"]["two_ln_n"], 2), out["bst"]["avl_height_bound"]))
p("小树 23 个键", {k: (v["height"], round(v["avg_depth"], 2)) for k, v in s.items()})
p("B+ 树 1e9 行", [(r["fan"], r["levels"], r["comparisons"]) for r in by_n[10**9]])
p("SQLite", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["sqlite"].items() if k not in ("plan_pk", "plan_scan", "plan_index", "sqlite_version")})
p("SQLite 计划", (out["sqlite"]["plan_pk"], out["sqlite"]["plan_scan"], out["sqlite"]["plan_index"]))
p("前缀", [{n: (r[n]["steps"], round(r[n]["ms"], 3)) for n in names} for r in prefix["rows"]])
p("前缀 节点数", [(r["trie_nodes"], r["radix_nodes"]) for r in prefix["rows"]])
p("前缀 涨幅", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["prefix"]["scale"].items()})
p("top-k", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["topk"].items() if k not in ("rows",)})
for r in out["topk"]["rows"]:
    p("  k=%d" % r["k"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "k"})
for r in out["range"]["rows"]:
    p("区间和 n=%d" % r["n"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "n"})
p("区间和 涨幅", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["range"]["scale"].items()})
p("kd 树看了百分之几", [(r["d"], round(r["fraction"] * 100, 2)) for r in g1["rows"]])
p("cKDTree 对直接算", [(r["d"], round(r["brute_us"]), round(r["tree_us"]), round(r["speedup"], 2)) for r in g2["rows"]])
p("HNSW 均匀", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in out["hnsw"]["rows"]])
p("HNSW 人工分团", out["hnsw"]["clustered_64"])
p("HNSW MNIST", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mnist.items() if k not in ("mnist", "uniform_same_shape")})
p("HNSW MNIST 结果", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in mnist["mnist"]])
p("HNSW 同形状均匀点", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in mnist["uniform_same_shape"]])
p("SQLite 估的层数", (out["sqlite"]["levels_est"], round(out["sqlite"]["rows_per_page"], 1)))
p("井字棋", (mm["nodes"], mm["leaves"], ab["nodes"], abc["nodes"], round(out["game"]["center_cut"], 1), round(out["game"]["natural_cut"], 1)))
p("走了两步之后", out["game"]["after_two"])
p("UCT", [(r["sims_per_move"], r["x_win"], r["draw"], r["x_loss"], r["games"], round(r["simulations_per_game"])) for r in game["uct_vs_random"]])
p("围棋 / 国际象棋的 10 的几次方", (round(out["game"]["go"]["log10_full"], 1), round(out["game"]["go"]["log10_sqrt"], 1), round(out["game"]["chess"]["log10_full"], 1)))
p("二叉树 10 亿个叶子几层", math.ceil(math.log2(10**9)))
p("cKDTree 慢了几倍", [round(1 / r["speedup"], 2) for r in g2["rows"] if r["speedup"] < 1])
