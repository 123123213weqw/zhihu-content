#!/usr/bin/env python3
"""最近邻换一组真实数据再量一遍 HNSW 的召回率:6 万张手写数字图片(MNIST),每张拉成一个 784 维向量。

均匀随机的点是对近似最近邻最不友好的数据。这里用真实数据对照:一样的 HNSW 参数(M = 16,建索引时 ef = 200),
看搜索时的 ef 取 50 和 400,各能找回真最近的 10 个里的几个,一次查询要多久。只吃 CPU。

数据第一次用会下载到 --data 指定的目录(两个文件,约 10 MB)。
运行(用 /data/wangyue/venvs/trees 环境;查询是单线程):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 /data/wangyue/venvs/trees/bin/python bench_nn_mnist.py --out results/nn_mnist.json
"""
import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.request

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402

from bench_common import warmup  # noqa: E402

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train": "train-images-idx3-ubyte.gz", "test": "t10k-images-idx3-ubyte.gz"}


def load_images(data_dir, split, threads):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, FILES[split])
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + FILES[split], path)
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":                    # 有的镜像存的是压缩的,有的没压
        raw = gzip.decompress(raw)
    magic, n, rows, cols = [int.from_bytes(raw[i:i + 4], "big") for i in range(0, 16, 4)]
    assert magic == 2051, magic
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/data/wangyue/datasets/mnist")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import hnswlib
    warmup()
    pts = load_images(a.data, "train", a.threads)
    qs = load_images(a.data, "test", a.threads)[: a.queries]
    n, d, k = pts.shape[0], pts.shape[1], a.k
    pts_sq = np.einsum("ij,ij->i", pts, pts)
    truth, tb = [], []
    for q in qs:                                   # 老老实实算 6 万个距离,取最近的 10 个
        t0 = time.perf_counter()
        dist = pts_sq - 2.0 * (pts @ q)
        truth.append(set(np.argpartition(dist, k)[:k].tolist()))
        tb.append(time.perf_counter() - t0)
    brute_us = float(np.median(tb[20:])) * 1e6
    index = hnswlib.Index(space="l2", dim=d)
    index.init_index(max_elements=n, M=16, ef_construction=200)
    t0 = time.perf_counter()
    index.add_items(pts, num_threads=a.threads)
    build_s = time.perf_counter() - t0
    rows = []
    for ef in (50, 400):
        index.set_ef(ef)
        index.knn_query(qs[:20], k, num_threads=1)           # 热身
        warmup(0.5)
        t0 = time.perf_counter()
        labels, _ = index.knn_query(qs, k, num_threads=1)
        qt = (time.perf_counter() - t0) / len(qs)
        recall = float(np.mean([len(set(l.tolist()) & t) / k for l, t in zip(labels, truth)]))
        rows.append({"ef": ef, "brute_us": brute_us, "hnsw_us": qt * 1e6, "recall@10": recall, "build_s": build_s, "speedup": brute_us / (qt * 1e6)})
    # 1 万个均匀随机点做对照:同样的维数、同样的查询数,只是数据没有结构
    rng = np.random.default_rng(10)
    uq = rng.random((len(qs), d), dtype=np.float32)
    uidx = hnswlib.Index(space="l2", dim=d)
    uidx.init_index(max_elements=n, M=16, ef_construction=200)
    up = rng.random((n, d), dtype=np.float32)
    uidx.add_items(up, num_threads=a.threads)
    up_sq = np.einsum("ij,ij->i", up, up)
    utruth, utb = [], []
    for q in uq:
        t0 = time.perf_counter()
        dist = up_sq - 2.0 * (up @ q)
        utruth.append(set(np.argpartition(dist, k)[:k].tolist()))
        utb.append(time.perf_counter() - t0)
    ubrute_us = float(np.median(utb[20:])) * 1e6
    uniform = []
    for ef in (50, 400):
        uidx.set_ef(ef)
        uidx.knn_query(uq[:20], k, num_threads=1)
        warmup(0.5)
        t0 = time.perf_counter()
        labels, _ = uidx.knn_query(uq, k, num_threads=1)
        qt = (time.perf_counter() - t0) / len(uq)
        recall = float(np.mean([len(set(l.tolist()) & t) / k for l, t in zip(labels, utruth)]))
        uniform.append({"ef": ef, "brute_us": ubrute_us, "hnsw_us": qt * 1e6, "recall@10": recall, "speedup": ubrute_us / (qt * 1e6)})
    res = {"n": n, "d": d, "queries": len(qs), "k": k, "M": 16, "ef_construction": 200, "mnist": rows, "uniform_same_shape": uniform}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
