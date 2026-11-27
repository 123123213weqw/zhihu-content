#!/bin/bash
# 第二轮:默认配置的服务,机器空的时候跑。用法: bash run2.sh base
TAG=${1:-base}
cd "$(dirname "$0")"
export SCHED_TRACE_FILE=$PWD/results/trace_$TAG.jsonl
export API_PID=$(cat server-$TAG.pid)
export BENCH_URL=http://127.0.0.1:8765      # 8000 被机器上别的服务占着
PY=/data/wangyue/venvs/vllm/bin/python
F='Token indices\|clean_up_tokenization'
$PY bench_client.py cold    --out results/cold.json          > results/cold.log    2>&1
$PY bench_client.py ttft    --out results/ttft.json          > results/ttft.log    2>&1
$PY bench_client.py lag     --out results/lag.json           > results/lag.log     2>&1
$PY bench_client.py laglong -b 16 --tokens 2400 --out results/laglong.json > results/laglong.log 2>&1
$PY bench_client.py abort   --out results/abort.json         > results/abort.log   2>&1
$PY bench_client.py tokq    --out results/tokq_w1.json       > results/tokq_w1.log 2>&1
echo done > results/run2_$TAG.done
