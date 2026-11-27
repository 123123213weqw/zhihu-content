#!/usr/bin/env python3
"""前端一篇里纯 CPU 的几组测量:分词、聊天模板、反分词。不用 GPU,不用起服务。

  A  分词:一段文字越长,分词越久吗
  B  聊天模板:对话轮数越多,渲染越久吗
  C  反分词:每来一个 token 就把整段重新解码一遍(朴素),和 vLLM 的两种增量做法
  D  一个汉字被切成几个 token 时,增量反分词器怎么把它拼回来
  E  提示词是随机 token 时,增量反分词器每一步要多花多少时间
  G  前端为一条请求建状态、反分词第一个生成 token,各花多久(随提示词长度)

运行(绑在同一个 NUMA 节点的核上):
  taskset -c 56-71 /data/wangyue/venvs/vllm/bin/python bench_cpu.py --out results/cpu.json
"""
import argparse
import json
import logging
import os
import random
import statistics
import time
from types import SimpleNamespace

MODEL = "/data/wangyue/models/Llama-3.1-8B-Instruct"
CORPUS_FILE = os.environ.get("BENCH_CORPUS", "/data/wangyue/corpora/Isaac.Newton-Opticks.txt")
# 自己写的一段中文,用来产生"一个汉字跨几个 token"的序列
ZH = (
    "大模型的推理引擎把一条请求拆成两类活:先把提示词一次算完,再一个一个往外吐 token。"
    "前端不参与计算,它负责把文字变成编号,把编号送进引擎,再把引擎吐出来的编号变回文字。"
    "每一步只解码新来的那一个编号,这件事看起来简单,麻烦在于一个汉字常常被切成两三个编号,"
    "单个编号解码出来是半截字符,要等后面的编号到了才能拼成一个完整的字。"
)


def med(fn, repeat=7):
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


def make_request(prompt_ids, rid="r"):
    from vllm import SamplingParams

    return SimpleNamespace(
        request_id=rid,
        prompt_token_ids=list(prompt_ids),
        prompt_embeds=None,
        sampling_params=SamplingParams(max_tokens=1 << 20),
    )


class WarnCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0
        self.last = ""

    def emit(self, record):
        self.n += 1
        self.last = record.getMessage()[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    from vllm.tokenizers import get_tokenizer
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
        uses_fast_detokenizer,
    )

    t0 = time.perf_counter()
    tok = get_tokenizer(MODEL)
    load_s = time.perf_counter() - t0
    text = open(CORPUS_FILE, encoding="utf-8", errors="ignore").read()
    res = {
        "model": MODEL,
        "tokenizer": type(tok).__name__,
        "is_fast": uses_fast_detokenizer(tok),
        "tokenizer_load_s": load_s,
        "cpu": open("/proc/cpuinfo").read().split("model name\t: ")[1].split("\n")[0],
    }
    print("分词器:", res["tokenizer"], "快路径:", res["is_fast"], "加载 %.2f 秒" % load_s, flush=True)

    # ---- A. 分词时间随长度
    corpus_ids = tok.encode(text)
    res["corpus_tokens"] = len(corpus_ids)
    res["corpus_chars"] = len(text)
    res["tokenize"] = []
    for n in (256, 1024, 4096, 16384, 32768, 65536, 131072):
        if n > len(corpus_ids):
            continue
        piece = tok.decode(corpus_ids[:n])
        ids = tok.encode(piece)
        ms = med(lambda: tok.encode(piece))
        res["tokenize"].append({"tokens": len(ids), "chars": len(piece), "ms": ms, "us_per_token": ms * 1e3 / len(ids)})
        print("A 分词 %7d 个 token: %8.2f 毫秒,每个 %.3f 微秒" % (len(ids), ms, ms * 1e3 / len(ids)), flush=True)
    # 32 条短提示词:一条一条分,和一次批量分
    piece = tok.decode(corpus_ids[:32768])
    short = [tok.decode(corpus_ids[i * 128:(i + 1) * 128]) for i in range(32)]
    ms_one = med(lambda: [tok.encode(s) for s in short])
    ms_batch = med(lambda: tok(short)["input_ids"])
    res["tokenize_32x128"] = {"one_by_one_ms": ms_one, "batch_ms": ms_batch}
    print("A 32 条各 128 个 token:逐条 %.2f 毫秒,批量 %.2f 毫秒" % (ms_one, ms_batch), flush=True)

    # ---- B. 聊天模板
    user = "用两句话解释一下,为什么天空是蓝色的。"
    assistant = (
        "阳光穿过大气时,波长较短的蓝光被空气分子散射得比红光厉害得多,散射出来的蓝光从四面八方进入眼睛,所以抬头看到的天空是蓝色的。"
    )
    res["chat_template"] = []
    for turns in (1, 8, 32, 128, 512):
        msgs = []
        for i in range(turns):
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": user})
        ms = med(lambda: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        ntok = len(tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)))
        res["chat_template"].append({"turns": turns, "prompt_tokens": ntok, "ms": ms})
        print("B 聊天模板 %3d 轮(提示词 %6d 个 token):渲染 %7.2f 毫秒" % (turns, ntok, ms), flush=True)

    # ---- C. 反分词:朴素对增量
    gen_ids = corpus_ids[-8192:]
    prompt_ids = corpus_ids[-8192 - 64:-8192]
    out = {"naive": [], "slow": [], "fast": [], "single": []}
    naive_max = 1024 if a.quick else 4096
    for n in sorted(set([256, 512, 1024, 2048, naive_max])):
        ids = gen_ids[:n]
        acc, blocks = [], []
        t_start = t_block = time.perf_counter()
        for t in range(1, n + 1):
            acc.append(tok.decode(ids[:t]))
            if t % 256 == 0:
                now = time.perf_counter()
                blocks.append((now - t_block) / 256 * 1e6)
                t_block = now
        total = (time.perf_counter() - t_start) * 1e3
        out["naive"].append({"tokens": n, "total_ms": total, "us_per_token_last_block": blocks[-1], "blocks_us": blocks,
                             "ok": acc[-1] == tok.decode(ids)})
        print("C 朴素 %5d 个 token:一共 %9.1f 毫秒,最后一块每个 %8.1f 微秒" % (n, total, blocks[-1]), flush=True)
    for name, cls in (("slow", SlowIncrementalDetokenizer), ("fast", FastIncrementalDetokenizer)):
        for n in (256, 1024, 4096, 8192):
            ids = gen_ids[:n]
            d = cls(tok, make_request(prompt_ids))
            got, blocks = [], []
            t_start = t_block = time.perf_counter()
            for i, tid in enumerate(ids, 1):
                d.update([tid], False)
                got.append(d.get_next_output_text(finished=False, delta=True))
                if i % 256 == 0:
                    now = time.perf_counter()
                    blocks.append((now - t_block) / 256 * 1e6)
                    t_block = now
            total = (time.perf_counter() - t_start) * 1e3
            text_out = d.output_text
            out[name].append({"tokens": n, "total_ms": total, "us_per_token_last_block": blocks[-1], "blocks_us": blocks,
                              "ok": text_out == tok.decode(ids) and text_out == "".join(got)})
            print("C 增量(%s) %5d 个 token:一共 %8.2f 毫秒,最后一块每个 %6.2f 微秒,对得上 %s" % (
                name, n, total, blocks[-1], out[name][-1]["ok"]), flush=True)
    # 只解码新来的那一个 token:快,但汉字会被切成半截
    ids = gen_ids[:1024]
    t_start = time.perf_counter()
    pieces = [tok.decode([t]) for t in ids]
    out["single"].append({"tokens": 1024, "total_ms": (time.perf_counter() - t_start) * 1e3,
                          "ok_english": "".join(pieces) == tok.decode(ids)})
    zh_ids = tok.encode(ZH)
    single_zh = "".join(tok.decode([t]) for t in zh_ids)
    out["single"].append({"tokens": len(zh_ids), "ok_chinese": single_zh == ZH, "text": single_zh[:60]})
    res["detokenize"] = out
    print("C 只解码新 token:英文对得上 %s,中文对得上 %s" % (out["single"][0]["ok_english"], out["single"][1]["ok_chinese"]), flush=True)

    # ---- D. 汉字跨 token
    demo_text = "大模型的推理引擎"
    dids = tok.encode(demo_text)
    rows = []
    for name, cls in (("slow", SlowIncrementalDetokenizer), ("fast", FastIncrementalDetokenizer)):
        d = cls(tok, make_request(tok.encode("下面开始输出:")))
        seq = []
        for tid in dids:
            d.update([tid], False)
            seq.append(repr(d.get_next_output_text(finished=False, delta=True)))
        rows.append({"kind": name, "pieces": seq, "ok": d.output_text == demo_text})
    res["utf8_demo"] = {"text": demo_text, "token_pieces": [tok.decode([t]) for t in dids], "rows": rows}
    for r in rows:
        print("D", r["kind"], r["pieces"], r["ok"], flush=True)

    # ---- E. 随机 token
    wc = WarnCounter()
    logging.getLogger("vllm").addHandler(wc)
    rng = random.Random(0)
    vocab = len(tok)
    n = 2048
    zh_long = (ZH * 12)[:4000]
    cases = {
        "英文原文": gen_ids[:n],
        "中文原文": tok.encode(zh_long)[:n],
        "随机编号(全词表)": [rng.randrange(vocab) for _ in range(n)],
        "随机编号(前两万个)": [rng.randrange(100, 20000) for _ in range(n)],
    }
    res["random_tokens"] = {"vocab": vocab, "cases": []}
    for name, ids in cases.items():
        for kind, cls in (("slow", SlowIncrementalDetokenizer), ("fast", FastIncrementalDetokenizer)):
            wc.n = 0
            d = cls(tok, make_request(ids[:256]))
            times = []
            text_len = 0
            for tid in ids:
                t = time.perf_counter()
                d.update([tid], False)
                times.append((time.perf_counter() - t) * 1e6)
            times.sort()
            c = {"case": name, "kind": kind, "tokens": n, "total_ms": sum(times) / 1e3,
                 "us_p50": times[len(times) // 2], "us_p99": times[int(len(times) * 0.99)], "us_max": times[-1],
                 "warnings": wc.n, "out_chars": len(d.output_text)}
            res["random_tokens"]["cases"].append(c)
            print("E %-14s %s:一共 %7.1f 毫秒,每个 token p50 %6.1f / p99 %7.1f / 最大 %8.1f 微秒,警告 %d 条" % (
                name, kind, c["total_ms"], c["us_p50"], c["us_p99"], c["us_max"], c["warnings"]), flush=True)
    # ---- G. 前端为一条请求建状态、出第一个 token,各花多久(随提示词长度)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.engine.output_processor import RequestState

    res["frontend_state"] = []
    for n in (128, 1024, 4096, 16384, 32768):
        ids = corpus_ids[40000:40000 + n]
        sp = SamplingParams(max_tokens=256, output_kind=RequestOutputKind.DELTA, detokenize=True)

        def mkreq():
            return SimpleNamespace(request_id="r-%d" % n, external_req_id="r-%d" % n, lora_request=None,
                                    prompt_token_ids=list(ids), prompt_embeds=None, sampling_params=sp,
                                    arrival_time=0.0, resumable=False)
        t_detok = med(lambda: IncrementalDetokenizer.from_new_request(tok, mkreq()), repeat=5)
        t_lp = med(lambda: LogprobsProcessor.from_new_request(tokenizer=tok, request=mkreq()), repeat=5)
        req = mkreq()

        def new_state():
            return RequestState.from_new_request(tok, req, None, None, 0, None, True, 1)

        t_state = med(new_state, repeat=5)
        # 第一个生成 token 要单独计时:反分词器建好之后,第一次 update() 会把整段提示词重放一遍。
        # 每次都换一个新的状态来量,不然只有第一次才是"第一个 token"。
        t_first = med(lambda: new_state().detokenizer.update([gen_ids[0]], False), repeat=5)
        st = new_state()
        st.detokenizer.update([gen_ids[0]], False)
        t_next = med(lambda: st.detokenizer.update([gen_ids[1]], False), repeat=5)
        st2 = new_state()
        st2.detokenizer.update([gen_ids[0]], False)
        t_out = med(lambda: st2.make_request_output([gen_ids[0]], None, None, None), repeat=5)
        res["frontend_state"].append({"tokens": n, "detok_init_ms": t_detok, "logprobs_init_ms": t_lp,
                                      "state_init_ms": t_state, "first_token_ms": t_first, "second_token_ms": t_next,
                                      "make_output_ms": t_out})
        print("G %6d 个 token:建反分词器 %6.2f 毫秒,建 logprobs 处理器 %6.2f,建请求状态 %6.2f,"
              "第一个生成 token 的反分词 %7.2f 毫秒,第二个 %.3f 毫秒,拼一个输出对象 %.3f 毫秒" % (
                  n, t_detok, t_lp, t_state, t_first, t_next, t_out), flush=True)

    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
