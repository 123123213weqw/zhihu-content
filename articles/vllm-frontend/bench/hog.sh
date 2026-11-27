#!/bin/bin/bash
# 占住 CPU 的小工具:起 N 个死循环进程,轮流绑在给定的核上,跑 SECONDS 秒。用法: bash hog.sh 56-63 16 40
CPUS=$1
N=$2
SECONDS_=$3
for i in $(seq 1 "$N"); do
  cpu=$(echo "$CPUS" | tr ',' '\n' | tr '-' ' ' | awk -v i=$i 'NR==1{a=$1} NR==2{b=$2} END{print a + (i - 1) % (b - a + 1)}')
  taskset -c "$cpu" timeout "$SECONDS_" /data/wangyue/venvs/vllm/bin/python -c "while True: pass" &
done
wait
