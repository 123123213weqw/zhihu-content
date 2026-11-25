"""把 vLLM 每一步的排班结果记到 JSONL 里。

vLLM 自带扩展点: --scheduler-cls 模块名.类名。这里的 TraceScheduler 继承 AsyncScheduler
(默认调度器就是它),排班逻辑一行没改,只在 schedule() 前后和 update_from_output() 前后
多记几个数:

  ts    schedule() 开始的时刻(秒,time.time())
  sm    schedule() 自己花了多少毫秒
  um    update_from_output() 花了多少毫秒
  tu    update_from_output() 开始的时刻,同一步连续两个 tu 之差就是一步的时间
  run   这一步排完后 RUNNING 队列里有几条请求
  wait  这一步排完后 WAITING 队列里有几条请求
  pre   这一步排进去的 prefill token 数(位置还没算过的 token)
  dec   这一步排进去的 decode token 数
  n     这一步排了几条请求
  new   这一步新放进 RUNNING 的请求:[请求 id 后 6 位, 前缀缓存命中数, 这一步排的 token 数]
  preem 这一步被抢占的请求:[id 后 6 位, 被抢占时已算好的 token 数(这些要重算), 提示词加已生成的 token 数]
  请求 id 取接口返回的那一段(cmpl- 后 17 位十六进制)的末 6 位,和客户端记的对得上。
  fin   上一步结束时完成的请求
  kv    KV cache 用了多少(0 到 1)

环境变量:
  SCHED_TRACE_FILE   输出文件(追加写)
  SCHED_TRACE_REQS=1 每步额外记每条请求排了几个 token(reqs: [[id 后 6 位, token 数], ...])
"""
import json
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def short(rid: str) -> str:
    # 引擎内部的请求 id 形如 cmpl-<17 位十六进制>-0-<8 位>,接口返回的是前半段。取前半段末 6 位。
    return rid.split("-")[1][-6:]


class TraceScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.scheduler_config
        hdr = {
            "hdr": {
                "max_num_batched_tokens": cfg.max_num_batched_tokens,
                "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
                "max_num_seqs": cfg.max_num_seqs,
                "policy": cfg.policy,
                "async_scheduling": cfg.async_scheduling,
                "enable_chunked_prefill": cfg.enable_chunked_prefill,
                "long_prefill_token_threshold": cfg.long_prefill_token_threshold,
                "watermark": cfg.watermark,
                "scheduler_reserve_full_isl": cfg.scheduler_reserve_full_isl,
                "prefill_schedule_interval": cfg.prefill_schedule_interval,
                "block_size": self.kv_cache_manager.coordinator.single_type_managers[0].block_size,
                "num_blocks": self.kv_cache_manager.kv_cache_config.num_blocks,
                "max_model_len": self.max_model_len,
            }
        }
        self._k = 0
        self._pend: dict[int, dict] = {}
        self._known: dict[str, int] = {}
        self._reqs = os.environ.get("SCHED_TRACE_REQS") == "1"
        self._f = open(os.environ["SCHED_TRACE_FILE"], "a", buffering=1)
        self._f.write(json.dumps(hdr) + "\n")

    def schedule(self, throttle_prefills: bool = False):
        t0 = time.perf_counter()
        before = {
            r.request_id: (r.num_computed_tokens, r.num_tokens) for r in self.running
        }
        before.update(
            (r.request_id, (r.num_computed_tokens, r.num_tokens)) for r in self.waiting
        )
        for rid, (_, total) in before.items():
            self._known[rid] = total
        # 结束的请求已经不在 self.requests 里了,要算的 token 数从自己记的表里查。
        fin = [[short(rid), self._known.get(rid)] for rid in sorted(self.finished_req_ids)]
        out = super().schedule(throttle_prefills)
        t1 = time.perf_counter()

        pre = dec = 0
        reqs = []
        for rid, n in out.num_scheduled_tokens.items():
            c, total = before.get(rid, (0, 0))
            # 位置小于 num_tokens 的 token 是提示词或已经生成的 token,算 prefill;
            # 多出来的那一个是这一步要采样的新 token,算 decode。
            p = max(0, min(c + n, total) - c)
            pre += p
            dec += n - p
            if self._reqs:
                reqs.append([short(rid), p, n - p])

        rec = {
            "k": self._k,
            "ts": time.time(),
            "sm": round((t1 - t0) * 1e3, 3),
            "run": len(self.running),
            "wait": len(self.waiting),
            "pre": pre,
            "dec": dec,
            "n": len(out.num_scheduled_tokens),
            "new": [
                [short(d.req_id), d.num_computed_tokens, out.num_scheduled_tokens[d.req_id]]
                for d in out.scheduled_new_reqs
            ],
            "preem": [
                [short(rid), *before[rid]] for rid in sorted(out.preempted_req_ids)
            ],
            "fin": fin,
            "kv": round(self.kv_cache_manager.usage, 4),
        }
        if self._reqs:
            rec["reqs"] = reqs
        self._k += 1
        # schedule() 比 update_from_output() 早一步(异步调度,队列深度 2),按对象配对。
        self._pend[id(out)] = rec
        return out

    def update_from_output(self, scheduler_output, model_output):
        t0 = time.perf_counter()
        tu = time.time()
        res = super().update_from_output(scheduler_output, model_output)
        rec = self._pend.pop(id(scheduler_output), None)
        if rec is not None:
            rec["tu"] = tu
            rec["um"] = round((time.perf_counter() - t0) * 1e3, 3)
            self._f.write(json.dumps(rec) + "\n")
        return res
