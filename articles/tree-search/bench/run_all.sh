#!/bin/bash
# 把树搜索一篇的全部实测重跑一遍。在服务器本机跑,用法: bash run_all.sh
#
# 约定:单线程的脚本都绑在同一个 NUMA 节点(node 1)的同一个核上,一个跑完再跑下一个。
# 同时跑会抢内存带宽,线性扫描能慢一倍;放到另一个节点的核上更糟,同样的矩阵乘向量慢了两百倍。
# 每个脚本开头都先把核跑热(bench_common.py 的 warmup),因为这台机器空闲时核降到 800 MHz。
set -e
cd "$(dirname "$0")"
mkdir -p results
PY=/data/wangyue/venvs/vllm/bin/python        # numpy 2.3.5
PYN=/data/wangyue/venvs/trees/bin/python      # numpy 2.5.5,scipy 1.18.1,hnswlib 0.8.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

taskset -c 56 $PY bench_point.py      --out results/point.json
taskset -c 56 $PY trees_small.py      --out results/trees_small.json
taskset -c 56 $PY bench_prefix.py     --out results/prefix.json
taskset -c 56 $PY bench_heap_range.py --out results/heap_range.json
taskset -c 56 $PY bench_game.py       --out results/game.json
# 建索引用 16 个线程,绑同一个节点里的 16 个核;查询是单线程。100 万个点的索引要建 20 分钟左右。
taskset -c 36-51 $PYN bench_nn.py       --out results/nn.json
taskset -c 36-51 $PYN bench_nn_mnist.py --out results/nn_mnist.json
echo "都跑完了,结果在 results/。下一步: python3 ../calc.py && python3 ../gen_figs.py"
