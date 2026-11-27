"""发一条流式请求,按首 token 的拆解打印。用法: python one_req.py ids|chat <token 数> [--no-warm]

不加 --no-warm 会先压热(16 条 512 token 的请求各生成 96 个 token)。"""""
import asyncio
import os
import sys
import time

sys.path.insert(0, "/data/wangyue/code/zhihu-frontend-bench")
import aiohttp
import bench_client as bc

kind, n = sys.argv[1], int(sys.argv[2])
no_warm = "--no-warm" in sys.argv
bc.load_corpus()
tag = int(time.time()) % 100000


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        if not no_warm:
            await bc.warm(s, rounds=1)
        seed = 500000 + tag
        prompt = bc.token_ids(seed, n) if kind == "ids" else bc.text_of(seed, n)
        r = await bc.stream(s, prompt, 4, "x")
        await asyncio.sleep(0.3)
        x = bc.span(bc.load_trace(), r["rid"])
        p = bc.parts(r, x)
        print("%s %d 个 token:到引擎 %.1f 毫秒,prefill %.1f 毫秒,出来 %.1f 毫秒,首 token %.1f 毫秒" % (
            kind, r["prompt_tokens"], p["to_engine_ms"], p["prefill_ms"], p["out_ms"], p["ttft_ms"]))


asyncio.run(main())
