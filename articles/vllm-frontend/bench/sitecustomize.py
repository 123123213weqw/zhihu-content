"""通过 PYTHONPATH 注入,只在 API server 进程里生效(serve.sh 会把本目录放进 PYTHONPATH)。两个互不相关的开关:

PROF_OUT=快照路径   给事件循环线程挂上 cProfile。每收到一次 SIGUSR1 存一份快照到 路径.1、路径.2 ……
                    相邻两份相减就是中间那段时间的耗时(prof_delta.py)。
INPROC_BENCH=输出   在 API server 进程自己的主线程里跑一遍小基准,查"同一段代码为什么在服务里慢"。
                    第一次收到 SIGUSR1 时挂上钩子,记下第一条长请求用过的提示词和分词器;
                    第二次收到时:先睡 3 秒(和刚处理完一条长请求时一样,核是凉的),再计时。
                    结果写到 INPROC_BENCH 指定的文件。
GC_LOG=输出文件       在 API server 进程里记下每一次垃圾回收停了多久(和客户端的 BENCH_GC_LOG 配合,
                    bench_client.py 的 lag / laglong 结果里的 slow_delays 列出了每一个慢的 token)。
                    第一次收到 SIGUSR1 开始记,第二次写出。
SEND_LOG=输出文件     记下 API server 每往外写一个带文字的 SSE 事件的时刻和请求 id(挂在 uvicorn 的 send 上)。
                    第一次 SIGUSR1 挂上,第二次写出。和客户端收到 token 的时刻按请求、按第几个 token 对上
                    (send_attr.py),就知道慢的那一段在前端里还是在后面。
NODELAY=1            给 API server 接进来的连接打开 TCP_NODELAY(uvicorn 自己不开),做对照。
"""
import cProfile
import marshal
import os
import signal
import sys
import time


def _inproc_bench(*_):
    out_path = os.environ["INPROC_BENCH"]
    state = _inproc_bench.__dict__
    if "tok" not in state or state["tok"] is None:
        import vllm.v1.engine.detokenizer as detok

        orig = detok.FastIncrementalDetokenizer.__init__

        def init(self, tokenizer, request):
            orig(self, tokenizer, request)
            ids = request.prompt_token_ids
            if ids is not None and len(ids) >= 32000 and state.get("tok") is None:
                state["tok"], state["ids"] = self.tokenizer, list(ids)

        detok.FastIncrementalDetokenizer.__init__ = init
        return
    from tokenizers.decoders import DecodeStream

    lines = ["prompt_tokens=%d" % len(state["ids"])]
    for i in range(4):
        time.sleep(3.0)
        cpu = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[36])   # 第 39 个字段:当前所在的逻辑核
        freq = open("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % cpu).read().strip()
        t0 = time.perf_counter()
        DecodeStream(ids=state["ids"], skip_special_tokens=True).step(state["tok"], 791)
        d1 = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        DecodeStream(ids=state["ids"], skip_special_tokens=True).step(state["tok"], 791)
        d2 = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        sum(range(2_000_000))
        d3 = (time.perf_counter() - t2) * 1e3
        lines.append("第 %d 次:核 %3d,频率 %6s kHz;睡 3 秒后第一步 %6.2f 毫秒,紧接着 %6.2f 毫秒;纯 Python 循环 %5.2f 毫秒" % (
            i + 1, cpu, freq, d1, d2, d3))
    with open(out_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _gc_log(*_):
    """API server 进程里每一次垃圾回收:开始时刻、停了多久、第几代、回收了多少对象。"""
    state = _gc_log.__dict__
    if "pauses" not in state:
        import gc

        state["pauses"] = []

        def cb(phase, info):
            p = state["pauses"]
            if phase == "start":
                p.append([time.time(), None, info.get("generation")])
            elif p and p[-1][1] is None:
                p[-1][1] = round((time.time() - p[-1][0]) * 1e3, 3)
                p[-1].append(info.get("collected"))

        gc.callbacks.append(cb)
        return
    import json

    with open(os.environ["GC_LOG"], "w") as f:
        json.dump([p for p in state["pauses"] if p[1] is not None], f)


def _send_log(*_):
    """API server 进程每往外写一个带文字的 SSE 事件的时刻,和它是哪条请求的(挂在 uvicorn 的 send 上)。"""
    state = _send_log.__dict__
    if "recs" not in state:
        import re

        from uvicorn.protocols.http.httptools_impl import RequestResponseCycle

        recs = state["recs"] = []
        orig = RequestResponseCycle.send
        pat = re.compile(rb'"id":"(?:chat)?cmpl-([0-9a-f]+)"')

        async def send(self, message):
            if message["type"] == "http.response.body":
                body = message.get("body") or b""
                if b"data: " in body[:16]:
                    m = pat.search(body)
                    recs.append([time.time(), m.group(1)[-6:].decode() if m else ""])
            return await orig(self, message)

        RequestResponseCycle.send = send
        return
    import json

    with open(os.environ["SEND_LOG"], "w") as f:
        json.dump(state["recs"], f)


def _nodelay_patch() -> None:
    """给 API server 接进来的连接打开 TCP_NODELAY(对照用:uvicorn 自己不开)。"""
    import socket

    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

    orig = HttpToolsProtocol.connection_made

    def connection_made(self, transport):
        orig(self, transport)
        sock = transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    HttpToolsProtocol.connection_made = connection_made


def _install() -> None:
    try:
        import uvloop
    except Exception:
        return
    if getattr(uvloop.run, "_profiled", False):
        return
    orig = uvloop.run
    out = os.environ.get("PROF_OUT")

    def run(main, *a, **k):
        caller = sys._getframe(1)
        # 单个 API server 时它在启动器进程里(cli/serve.py 调 uvloop.run),多个时在 worker 进程里。
        is_api_server = "entrypoints/cli/serve.py" in caller.f_code.co_filename or caller.f_code.co_name == "run_api_server_worker_proc"
        if is_api_server and os.environ.get("NODELAY"):
            _nodelay_patch()
        if is_api_server and os.environ.get("INPROC_BENCH"):
            signal.signal(signal.SIGUSR1, _inproc_bench)
        elif is_api_server and os.environ.get("GC_LOG"):
            signal.signal(signal.SIGUSR1, _gc_log)
        elif is_api_server and os.environ.get("SEND_LOG"):
            signal.signal(signal.SIGUSR1, _send_log)
        if out and is_api_server:
            pr = cProfile.Profile()
            n = 0

            def dump(*_):
                # 不能用 dump_stats:它会顺手把 profiler 关掉。snapshot_stats 不关,取完接着记。
                nonlocal n
                n += 1
                pr.snapshot_stats()
                with open("%s.%d" % (out, n), "wb") as f:
                    marshal.dump(pr.stats, f)

            if not any(os.environ.get(k) for k in ("INPROC_BENCH", "GC_LOG", "SEND_LOG")):
                signal.signal(signal.SIGUSR1, dump)
            return pr.runcall(orig, main, *a, **k)
        return orig(main, *a, **k)

    run._profiled = True
    uvloop.run = run


if any(os.environ.get(k) for k in ("PROF_OUT", "INPROC_BENCH", "GC_LOG", "SEND_LOG", "NODELAY")):
    _install()
