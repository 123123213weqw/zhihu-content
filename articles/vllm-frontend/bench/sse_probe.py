#!/usr/bin/env python3
"""不依赖 GPU 的对照:一个只会流式输出的极简 HTTP 服务(asyncio,每 0.3 秒发一行保活,第 3 秒发"真正的 token"),
看三种客户端(裸 socket / aiohttp / curl)各在服务端 write 之后多久读到那一行。用来回答:长提示词的首 token
晚到的那几十毫秒,是不是客户端或者本机网络栈固有的延迟(结论:不是,这里都在 1 毫秒内)。

运行(不需要显卡): python sse_probe.py --out results/sse_probe.json
"""
import argparse
import asyncio
import json
import socket
import time

PORT = 8799
MARK = b'data: {"choices":[{"text":"TOKEN"}]}\n\n'
IDLE = b'data: {"choices":[{"text":""}]}\n\n'
writes = []


async def handle(reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
    await writer.drain()
    for i in range(10):
        await asyncio.sleep(0.3)
        # 不发保活的那一轮,前 9 次什么都不发
        payload = MARK if i == 9 else IDLE
        if i < 9:
            continue
        writer.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
        await writer.drain()
        writes.append(time.time())
    writer.write(b"0\r\n\r\n")
    await writer.drain()
    writer.close()


def raw_client():
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(b"POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n\r\n")
    s.settimeout(10)
    mark_t = None
    buf = b""
    while True:
        d = s.recv(65536)
        if not d:
            break
        buf += d
        if MARK in buf:
            mark_t = mark_t or time.time()
    s.close()
    return mark_t


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    server = await asyncio.start_server(handle, "127.0.0.1", PORT)
    res = {"keep_alive_lines": 0, "clients": {}}
    for kind in ("raw", "aiohttp", "curl"):
        res["clients"][kind] = []
    for _ in range(a.repeat):
        for kind in ("raw", "aiohttp", "curl"):
            writes.clear()
            if kind == "raw":
                mark_t = await asyncio.to_thread(raw_client)
            elif kind == "aiohttp":
                import aiohttp

                async with aiohttp.ClientSession() as sess:
                    async with sess.post("http://127.0.0.1:%d/x" % PORT, json={}) as resp:
                        async for line in resp.content:
                            if line.startswith(b"data: ") and b"TOKEN" in line:
                                mark_t = time.time()
                                break
            else:
                p = await asyncio.create_subprocess_exec(
                    "bash", "-c",
                    "curl -sN -X POST http://127.0.0.1:%d/x -d '' | while IFS= read -r line; do case \"$line\" in "
                    "*TOKEN*) date +%%s.%%6N;; esac; done" % PORT,
                    stdout=asyncio.subprocess.PIPE,
                )
                out, _ = await p.communicate()
                mark_t = float(out.decode().strip())
            res["clients"][kind].append(round((mark_t - writes[0]) * 1e3, 3))
    server.close()
    json.dump(res, open(a.out, "w"), indent=1)
    for kind, v in res["clients"].items():
        print("%-8s 服务端写出之后读到(毫秒):%s" % (kind, v))


asyncio.run(main())
