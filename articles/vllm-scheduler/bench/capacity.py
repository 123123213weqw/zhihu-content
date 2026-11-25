"""第一次扫描里 batch 128 和 256 两组的专门汇总:每条请求 256+320=576 token,KV 装不下,发生抢占。"""
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{\"k\"")]
d = json.load(open(sys.argv[2]))
out = {}
for run in d["runs"]:
    if run["b"] not in (128, 256):
        continue
    win = [x for x in rows if run["t0"] - 0.5 <= x["ts"] <= run["t1"] + 0.5]
    t0 = win[0]["ts"]
    first_fin = next(i for i, x in enumerate(win) if x["fin"])
    drained = next((i for i, x in enumerate(win) if i > first_fin and x["wait"] == 0 and x["run"] > 0), len(win))
    steady = [x["run"] for x in win[first_fin:drained]]
    out.setdefault(str(run["b"]), []).append({
        "round": run["round"], "steps": len(win),
        "preemptions": sum(len(x["preem"]) for x in win),
        "max_run": max(x["run"] for x in win), "waiting_peak": max(x["wait"] for x in win),
        "steady_run_median": statistics.median(steady) if steady else None,
        "steady_run_min": min(steady) if steady else None,
        "steady_run_max": max(steady) if steady else None,
        "lost_tokens": sum(p[1] for x in win for p in x["preem"]),
        "wall_s": run["t1"] - run["t0"],
        "client_ttft_median_s": statistics.median(r["ttft"] for r in run["reqs"]),
        "client_ttft_max_s": max(r["ttft"] for r in run["reqs"]),
        "client_e2e_max_s": max(r["e2e"] for r in run["reqs"]),
        "series": [[round((x["ts"] - t0) * 1e3, 1), x["run"], x["wait"], x["dec"], x["pre"], x["kv"]] for x in win[::12]],
    })
json.dump({"note": "第一次扫描里 batch 128 和 256 两组:每条请求 256+320=576 token,KV 装不下,发生抢占", "runs": out}, open(sys.argv[3], "w"))
for b, rs in out.items():
    for r in rs:
        print("b", b, "round", r["round"], "steps", r["steps"], "preemptions", r["preemptions"], "lost", r["lost_tokens"],
              "max_run", r["max_run"], "waiting_peak", r["waiting_peak"], "steady run median/min/max",
              r["steady_run_median"], r["steady_run_min"], r["steady_run_max"],
              "ttft med %.2f max %.2f s" % (r["client_ttft_median_s"], r["client_ttft_max_s"]), "e2e max %.1f s" % r["client_e2e_max_s"])
