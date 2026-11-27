#!/bin/bash
# 前端进程被抢 CPU 时会怎样:把 API server 的线程绑在 56-63 这 8 个核上,再分别在同样的核、
# 同一个 NUMA 节点的别的核、另一个节点的核上起 16 个死循环。用法: bash run_hog.sh base
TAG=${1:-base}
cd "$(dirname "$0")"
export SCHED_TRACE_FILE=$PWD/results/trace_$TAG.jsonl
export API_PID=$(cat server-$TAG.pid)
export BENCH_URL=http://127.0.0.1:8765      # 8000 被机器上别的服务占着
PY=/data/wangyue/venvs/vllm/bin/python
taskset -a -pc 56-63 "$API_PID" > /dev/null
LABEL=pinned_8cores       $PY bench_client.py laglong -b 32 --tokens 600  --out results/hog_0.json > results/hog_0.log 2>&1
HOG=56-63,16,40  LABEL=same_cores      $PY bench_client.py laglong -b 32 --tokens 1200 --out results/hog_1.json > results/hog_1.log 2>&1
HOG=36-51,16,40  LABEL=same_node_other $PY bench_client.py laglong -b 32 --tokens 1200 --out results/hog_2.json > results/hog_2.log 2>&1
HOG=0-15,16,40   LABEL=other_node       $PY bench_client.py laglong -b 32 --tokens 1200 --out results/hog_3.json > results/hog_3.log 2>&1
echo done > results/run_hog_$TAG.done
