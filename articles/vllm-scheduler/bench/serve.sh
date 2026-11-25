#!/bin/bash
# 起一个带调度记录器的 vLLM 服务。在服务器本机跑,用法:
#   nohup ./serve.sh <GPU 编号> <标签> [vllm serve 的额外参数...] > server-<标签>.log 2>&1 &
# 记录写到 results/trace_<标签>.jsonl,日志在 server-<标签>.log,进程号写到 server-<标签>.pid
GPU=$1
TAG=$2
shift 2
cd "$(dirname "$0")"
mkdir -p results
export CUDA_VISIBLE_DEVICES=$GPU
export SCHED_TRACE_FILE=$PWD/results/trace_$TAG.jsonl
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
# 编译缓存放到数据盘,不写系统盘
export VLLM_CACHE_ROOT=/data/wangyue/cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"
echo $$ > server-$TAG.pid
exec /data/wangyue/venvs/vllm/bin/vllm serve /data/wangyue/models/Llama-3.1-8B-Instruct \
  --served-model-name m --max-model-len 16384 --max-num-seqs 256 \
  --scheduler-cls trace_sched.TraceScheduler "$@"
