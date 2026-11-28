"""几个实测脚本共用的小工具。"""
import math
import time


def warmup(seconds=2.5):
    """先把所在的核跑热再计时。这台机器空闲时核降到 800 MHz,刚醒来的那几秒计时会慢好几倍。
    单线程的脚本绑在一个核上跑(taskset -c 56),开头调用一次,整个进程都待在热的核上。"""
    t_end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < t_end:
        for _ in range(20000):
            x += math.sqrt(x + 1.0) - math.sqrt(x + 0.5)
    return x
