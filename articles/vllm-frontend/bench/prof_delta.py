"""两次 cProfile 快照相减,看中间那段时间花在了哪些函数上。用法: python prof_delta.py a.pstats b.pstats"""
import sys

import pstats


def load(path):
    st = pstats.Stats(path)
    out = {}
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        out[func] = (nc, tt, ct)
    return out


def name(f):
    fn, line, fnc = f
    for mark in ("site-packages/", "lib/python3.12/", "vllm-034/src/"):
        i = fn.find(mark)
        if i >= 0:
            fn = fn[i + len(mark):]
            break
    return "%s:%d(%s)" % (fn, line, fnc)


args = [x for x in sys.argv[1:] if not x.startswith("--")]
json_out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
a, b = load(args[0]), load(args[1])
rows = []
for f in set(a) | set(b):
    n0, t0, c0 = a.get(f, (0, 0.0, 0.0))
    n1, t1, c1 = b.get(f, (0, 0.0, 0.0))
    if n1 - n0 > 0:
        rows.append((c1 - c0, t1 - t0, n1 - n0, f))
top_n = int(args[2]) if len(args) > 2 else 60
if json_out:
    keep = [r for r in rows if not any(x in name(r[3]) for x in ("sitecustomize", "cProfile.py", "_lsprof", "multiprocessing", "selectors.py"))]
    pick = lambda key, k: [{"func": name(f), "calls": n, "cum_ms": round(ct * 1e3, 2), "self_ms": round(tt * 1e3, 2)} for ct, tt, n, f in sorted(keep, key=key, reverse=True)[:k]]
    json.dump({"cum_top": pick(lambda r: r[0], 80), "self_top": pick(lambda r: r[1], 40)}, open(json_out, "w"), indent=1, ensure_ascii=False)
print("== 按累计时间(含子调用),前 %d 个 ==" % top_n)
for ct, tt, n, f in sorted(rows, reverse=True)[:top_n]:
    print("累计 %7.1f 毫秒,自身 %7.1f 毫秒,调用 %5d 次  %s" % (ct * 1e3, tt * 1e3, n, name(f)))
print("== 按自身时间,前 30 个 ==")
for ct, tt, n, f in sorted(rows, key=lambda r: r[1], reverse=True)[:30]:
    print("自身 %7.1f 毫秒,累计 %7.1f 毫秒,调用 %5d 次  %s" % (tt * 1e3, ct * 1e3, n, name(f)))
