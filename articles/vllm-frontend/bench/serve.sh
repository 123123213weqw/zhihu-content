#!/bin/bash
# 起一个带调度记录器的 vLLM 服务。在服务器本机跑,用法:
#   bash serve.sh <GPU 编号> <标签> [端口] [vllm 的额外参数...]
# 记录写到 results/trace_<标签>.jsonl,日志在 server-<标签>.log,API server 的进程号写到 server-<标签>.pid
GPU=$1
TAG=$2
PORT=${3:-8765}   # 8000 被机器上别的服务占着
shift 3
cd "$(dirname "$0")"
mkdir -p results
export CUDA_VISIBLE_DEVICES=$GPU
export SCHED_TRACE_FILE=$PWD/results/trace_$TAG.jsonl
export SCHED_TRACE_REQS=1
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
export VLLM_CACHE_ROOT=/data/wangyue/cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"
rm -f "$SCHED_TRACE_FILE"
echo $$ > server-$TAG.pid
exec /data/wangyue/venvs/vllm/bin/vllm serve /data/wangyue/models/Llama-3.1-8B-Instruct \
  --served-model-name m --max-model-len 40000 --max-num-seqs 256 --port $PORT \
  --scheduler-cls trace_sched.TraceScheduler "$@"
