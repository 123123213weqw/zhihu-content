#!/bin/bash
# 前端一篇的补充测量。每一段都要先按 serve.sh 起好服务(40 秒左右),所以这里只列出命令,不自动起停:
# 按注释里的方式起服务,等 /ping 通了,再跑下面对应的几行。在服务器本机跑,客户端一律走 8765 端口。
#
# 环境变量每个 shell 里设一次:
#   cd "$(dirname "$0")/.."/..   # 即 /data/wangyue/code/zhihu-frontend-bench
#   export BENCH_URL=http://127.0.0.1:8765 SCHED_TRACE_FILE=$PWD/results/trace_base.jsonl
#   export API_PID=$(cat server-base.pid)
#   PY=/data/wangyue/venvs/vllm/bin/python

# ---- 1. 占 CPU 的对照:API server 绑到 56-63 这 8 个核,16 个死循环占同样的核,重复三轮 -------------------
# bash serve.sh 0 base 8765
taskset -a -pc 56-63 "$API_PID"
HOG=56-63,16,45 LABEL=same_cores $PY bench_client.py laglong -b 32 --tokens 1200 --out results/hog_ab_2.json
HOG=56-63,16,45 LABEL=same_cores $PY bench_client.py laglong -b 32 --tokens 1200 --out results/hog_ab_4.json
# 没有死循环的那一组先测(hog_0),三个位置的对照(hog_1 到 hog_3)在 run_hog.sh 里。

# ---- 2. 垃圾回收:记下 API server 进程里每一次回收停了多久 ------------------------------------------------
# GC_LOG=$PWD/results/gc_server.json bash serve.sh 0 base 8765     第一次 SIGUSR1 开始记,第二次写出
kill -USR1 "$API_PID"
BENCH_GC_LOG=1 $PY bench_client.py laglong -b 32 --tokens 1200 --out results/gc_1.json
BENCH_GC_LOG=1 $PY bench_client.py laglong -b 32 --tokens 1200 --out results/gc_2.json
kill -USR1 "$API_PID"

# ---- 3. 尾部:客户端收到每个 token 的时刻,对上 API server 写出这一帧的时刻 ---------------------------------
# SEND_LOG=$PWD/results/send_srv.json bash serve.sh 0 base 8765    uvicorn 的默认设置;加 NODELAY=1 是对照
kill -USR1 "$API_PID"
for i in 1 2 3 4; do $PY bench_client.py laglong -b 32 --tokens 1200 --out results/send_$i.json; done
kill -USR1 "$API_PID"; sleep 3
mv results/send_srv.json results/send_srv_off.json
$PY send_attr.py results/trace_base.jsonl results/send_srv_off.json results/send_1.json results/send_2.json results/send_3.json results/send_4.json
# 开了 TCP_NODELAY 的那一组同样跑四轮,结果存成 send_srv_nd.json,得到的 *_attr.json 留在 results/,逐 token 的原始文件不进仓库。

# ---- 4. 分词线程 4 个时的排队 --------------------------------------------------------------------------
# bash serve.sh 0 w4 8765 --renderer-num-workers 4
export API_PID=$(cat server-w4.pid) SCHED_TRACE_FILE=$PWD/results/trace_w4.jsonl
$PY bench_client.py tokq --out results/tokq_w4.json
$PY tokq_overtake.py results/trace_w4.jsonl          # 结果复制成 results/tokq_overtake.json

# ---- 5. 一条 32k 请求的剖析:给 API server 的事件循环线程挂上 cProfile --------------------------------------
# PROF_OUT=$PWD/results/prof.pstats bash serve.sh 0 base 8765       每次 SIGUSR1 存一份快照:prof.pstats.1, .2 ...
$PY one_req.py ids 1024; $PY one_req.py ids 1024; kill -USR1 "$API_PID"; sleep 2
$PY one_req.py ids 32768 --no-warm; kill -USR1 "$API_PID"; sleep 2
$PY prof_summary.py results/prof.pstats.1 results/prof.pstats.2 --to-engine-ms 44.0 --out-ms 54.0 --out results/prof_32k.json

# ---- 6. 在 API server 自己的线程里跑同一个 DecodeStream 第一步,和离线对照 ------------------------------------
# INPROC_BENCH=$PWD/results/inproc.txt bash serve.sh 0 base 8765    第一次 SIGUSR1 挂钩子,第二次跑
kill -USR1 "$API_PID"; sleep 1
$PY one_req.py ids 32768 --no-warm; sleep 1; kill -USR1 "$API_PID"; sleep 18
# 离线:不绑核,先空转几秒再量
API_PID=0 SCHED_TRACE_FILE=/dev/null $PY decodestream.py

# ---- 7. 裸 socket 客户端的首 token,和不依赖显卡的极简流式服务对照 -----------------------------------------
# bash serve.sh 0 base 8765      不带任何钩子
$PY one_req.py ids 1024; rm -f results/raw32k.json
for i in 1 2 3 4; do $PY raw32k.py; done; for i in 1 2 3 4; do $PY raw32k.py --quickack; done
API_PID=0 SCHED_TRACE_FILE=/dev/null $PY sse_probe.py --out results/sse_probe.json

# ---- 8. 纯 CPU 的几组(不用起服务) --------------------------------------------------------------------------
API_PID=0 SCHED_TRACE_FILE=/dev/null taskset -c 56-71 $PY bench_cpu.py --out results/cpu_xeon8352v.json
API_PID=0 SCHED_TRACE_FILE=/dev/null taskset -c 56-71 $PY input_cost.py --out results/input_cost.json
