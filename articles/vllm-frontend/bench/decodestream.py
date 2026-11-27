"""DecodeStream(tokenizers 库里的流式反分词器)在长提示词下的构造和第一步各花多久。纯 CPU。"""
import sys, time
sys.path.insert(0, "/data/wangyue/code/zhihu-frontend-bench")
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer
import bench_client as bc
bc.load_corpus()
tok = AutoTokenizer.from_pretrained(bc.MODEL)._tokenizer
for n in (128, 1024, 4096, 16384, 32768, 65536):
    ids = bc.token_ids(424242, n)
    c, s1, s2 = [], [], []
    for rep in range(7):
        t0 = time.perf_counter(); st = DecodeStream(ids=ids, skip_special_tokens=True); t1 = time.perf_counter()
        a = st.step(tok, 791); t2 = time.perf_counter(); b = st.step(tok, 279); t3 = time.perf_counter()
        c.append((t1 - t0) * 1e3); s1.append((t2 - t1) * 1e3); s2.append((t3 - t2) * 1e3)
    m = lambda x: sorted(x)[len(x) // 2]
    print("提示词 %6d 个 token:构造 %6.2f 毫秒,第 1 步(step)%7.2f 毫秒,第 2 步 %.3f 毫秒;第 1 步每 token %.2f 微秒" % (
        n, m(c), m(s1), m(s2), m(s1) * 1e3 / n), flush=True)
# 空 ids 构造,再手工 step 一遍提示词
ids = bc.token_ids(424242, 32768)
t0 = time.perf_counter(); st = DecodeStream(skip_special_tokens=True); t1 = time.perf_counter()
for i in ids: st.step(tok, i)
t2 = time.perf_counter(); a = st.step(tok, 791); t3 = time.perf_counter()
print("对照:空构造 %.2f 毫秒,逐个 step 32768 个提示词 token 共 %.1f 毫秒(每个 %.2f 微秒),之后第 1 个生成 token %.3f 毫秒" % (
    (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t1) * 1e6 / len(ids), (t3 - t2) * 1e3))
